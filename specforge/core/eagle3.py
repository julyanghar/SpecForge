# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in HuggingFace Transformers.
# Portions of this code are adapted from:
#   - https://github.com/EleutherAI/gpt-neox (Apache License 2.0)
#   - https://github.com/huggingface/transformers (Apache License 2.0)
#   - https://github.com/SafeAILab/EAGLE (Apache License 2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from specforge.core.compact_teacher import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    compute_target_p_padded_from_hidden,
)
from specforge.core.eagle3_adapters import BackendAdapter, SdpaLikeAdapter, UspAdapter
from specforge.core.lk_loss import compute_acceptance_rate, compute_lk_loss
from specforge.core.loss import LogSoftmaxLoss
from specforge.modeling.draft import Eagle3DraftModel
from specforge.utils import padding


class Eagle3Model(nn.Module):
    pass


def _compute_loss_and_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_p: torch.Tensor,
    target_p_on_draft: torch.Tensor,
    position_mask: torch.Tensor,
    lk_loss_type: Optional[str],
    kl_scale: float,
    kl_decay: float,
    reduce_metrics_fn: Optional[
        Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    reduce_loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute step loss and acceptance rate for KL/LK objectives.

    Args:
        logits: Draft model logits for current step.
        target_p: Renormalized target distribution over draft-vocab tokens (for KL).
        target_p_on_draft: Original target probabilities restricted to draft-vocab tokens (for acceptance terms).
        position_mask: Mask indicating valid tokens for loss/metric aggregation.
        lk_loss_type: LK objective mode (`None`, `"alpha"`, or `"lambda"`).
        kl_scale: Scale factor for lambda LK mixing weight.
        kl_decay: Decay factor for lambda LK mixing weight.
        reduce_metrics_fn: Optional distributed reducer for metric numer/denom.
        reduce_loss_fn: Optional distributed reducer for KL loss.
    """
    kl_loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
    if reduce_loss_fn is not None:
        kl_loss = reduce_loss_fn(kl_loss)

    with torch.set_grad_enabled(lk_loss_type is not None):
        acceptance_rate, log_acceptance_rate = compute_acceptance_rate(
            logits=logits,
            target_probs=target_p_on_draft,
            position_mask=position_mask,
            reduce_fn=reduce_metrics_fn,
        )

    if lk_loss_type is None:
        loss = kl_loss
    else:
        loss = compute_lk_loss(
            kl_loss=kl_loss,
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=log_acceptance_rate,
            lk_loss_type=lk_loss_type,
            kl_scale=kl_scale,
            kl_decay=kl_decay,
        )
    return acceptance_rate.detach(), loss


class OnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        draft_model: Eagle3DraftModel,
        length: int = 7,
        attention_backend="sdpa",
        target_model: Optional[Eagle3Model] = None,
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
        trim_loss_positions: bool = False,
        trim_prompt_rows: bool = False,
        trim_step1: bool = False,
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
            lk_loss_type: LK loss objective type. One of {"lambda", "alpha"}.
            kl_scale: Initial KL weight scale for lambda LK loss.
            kl_decay: Decay factor for adaptive KL weight in lambda LK loss.
            trim_loss_positions: A级裁剪——teacher/logits/loss 只在监督位置计算,
                数学等价(mean 分母重标定),默认关。batch>1 或 lk_loss 时自动回退全长。
        """
        super().__init__()
        self.draft_model = draft_model
        self.length = length
        self.attention_backend = attention_backend
        self.target_model = target_model
        self.lk_loss_type = lk_loss_type
        self.kl_scale = kl_scale
        self.kl_decay = kl_decay
        self.trim_loss_positions = trim_loss_positions
        self.trim_prompt_rows = trim_prompt_rows
        # B-ii: 步1 也短路 prompt 行(只算 K/V)。需 trim_prompt_rows。
        self.trim_step1 = trim_step1

    def _make_adapter(self) -> BackendAdapter:
        if self.attention_backend == "usp":
            return UspAdapter(self)
        return SdpaLikeAdapter(self)

    def _acc_and_loss(
        self,
        *,
        logits: torch.Tensor,
        target_p: torch.Tensor,
        target_p_on_draft: torch.Tensor,
        target_token_ids: torch.Tensor,
        position_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        adapter: BackendAdapter,
        loss_scale: float = 1.0,
        full_positions: Optional[int] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        with torch.no_grad():
            pred_draft_token_ids = logits.argmax(-1)
            pred_target_token_ids = (
                pred_draft_token_ids + self.draft_model.d2t[pred_draft_token_ids]
            )
            local_correct = (
                (pred_target_token_ids == target_token_ids) * loss_mask.squeeze(-1)
            ).sum()
            local_denom = loss_mask.sum().clamp_min(1e-6)
            local_correct, local_denom = adapter.reduce_metrics(
                local_correct=local_correct, local_denom=local_denom
            )
            acc = local_correct / local_denom

        acceptance_rate, loss = _compute_loss_and_acceptance_rate(
            logits=logits,
            target_p=target_p,
            target_p_on_draft=target_p_on_draft,
            position_mask=position_mask,
            lk_loss_type=self.lk_loss_type,
            kl_scale=self.kl_scale,
            kl_decay=self.kl_decay,
            reduce_metrics_fn=adapter.reduce_metrics,
            reduce_loss_fn=adapter.reduce_loss,
        )
        if loss_scale != 1.0:
            # MCTRIM(A级):紧凑 kernel 的 mean 分母是 n_sup,全长语义是 L —— 重标定。
            # 仅 lk_loss_type is None 时调用方可传(loss==kl_loss 才是线性可缩的)。
            loss = loss * loss_scale
        loss_denom = torch.tensor(
            logits.shape[0]
            * (full_positions if full_positions is not None else logits.shape[1]),
            device=logits.device,
            dtype=torch.float32,
        )
        return (
            acc,
            acceptance_rate,
            loss,
            local_correct,
            local_denom,
            loss.detach(),
            loss_denom,
        )

    def _prepare_position_ids(
        self,
        position_ids: Optional[torch.Tensor],
        *,
        seq_length: int,
        past_key_values_length: int,
        device: torch.device,
        is_vlm: bool,
        input_ids: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.attention_backend == "usp":
            return position_ids
        if position_ids is None:
            if is_vlm:
                mrope_positions_ids, _ = self.target_model.get_rope_index(
                    input_ids=input_ids, image_grid_thw=image_grid_thw
                )
                return mrope_positions_ids
            return (
                torch.arange(
                    past_key_values_length,
                    seq_length + past_key_values_length,
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .view(-1, seq_length)
            )

        position_ids = position_ids.long()
        return position_ids.view(-1, seq_length)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        is_vlm: bool = False,
        target_hidden_for_compact: Optional[torch.Tensor] = None,
        target_head_weight: Optional[torch.Tensor] = None,
        compact_teacher_chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
        **kwargs,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
            target_hidden_for_compact, target_head_weight, compact_teacher_chunk_size:
                when the first two are given, the padded teacher is built from hidden
                states in draft-vocab space and ``target`` is ignored.
        """
        # Step 1: handle vocab size
        if target_hidden_for_compact is not None:
            (
                target_p_padded,
                target_p_on_draft_padded,
                target_token_ids_padded,
                position_mask,
            ) = compute_target_p_padded_from_hidden(
                hidden=target_hidden_for_compact,
                lm_head_weight=target_head_weight,
                t2d=self.draft_model.t2d,
                loss_mask=loss_mask,
                length=self.length,
                chunk_size=compact_teacher_chunk_size,
            )
            del target_hidden_for_compact
            trim_pack = None
        else:
            # MCTRIM(A级):batch=1 且非 lk_loss 时,teacher 只在监督位置计算
            _trim_ok = (
                self.trim_loss_positions
                and self.lk_loss_type is None
                and loss_mask.shape[0] == 1
                and int(loss_mask.sum().item()) > 0
                and not is_vlm  # 非VL边界:VLM(mrope)不走 trim，全长回退
            )
            if _trim_ok:
                import os as _os4
                _sc = _os4.environ.get("MCTRIM_SELFCHECK", "0") == "1"
                trim_pack = _build_trim_pack(
                    target, self.draft_model.t2d, loss_mask, self.length
                )
                if _sc:
                    trim_pack["_ref"] = _compute_target_p_padded(
                        target=target, t2d=self.draft_model.t2d,
                        loss_mask=loss_mask, length=self.length,
                    )
                target_p_padded = None
                target_p_on_draft_padded = None
                target_token_ids_padded = None
                position_mask = trim_pack["position_mask_sup"]
            else:
                trim_pack = None
                (
                    target_p_padded,
                    target_p_on_draft_padded,
                    target_token_ids_padded,
                    position_mask,
                ) = _compute_target_p_padded(
                    target=target,
                    t2d=self.draft_model.t2d,
                    loss_mask=loss_mask,
                    length=self.length,
                )
            del target
        torch.cuda.empty_cache()

        import os as _os
        if _os.environ.get("MCDBG_NAN", "0") == "1":
            def _nanrep(tag, t):
                if t is None or not torch.is_tensor(t) or not t.is_floating_point():
                    return
                n = torch.isnan(t).sum().item()
                print(f"MCDBG_NAN {tag}: nan={n}/{t.numel()} shape={tuple(t.shape)} "
                      f"finite_min={t[~torch.isnan(t)].min().item() if n < t.numel() else 'ALL_NAN'}", flush=True)
            self._mcdbg = _nanrep
            _nanrep("input_target_hidden", hidden_states)
            _nanrep("target_p_padded", target_p_padded)
            _nanrep("position_mask", position_mask.float() if position_mask is not None else None)
        else:
            self._mcdbg = lambda *a: None

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)
        self._mcdbg("after_fc_projection", hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        position_ids = self._prepare_position_ids(
            position_ids=position_ids,
            seq_length=seq_length,
            past_key_values_length=past_key_values_length,
            device=hidden_states.device,
            is_vlm=is_vlm,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        acceptance_rates = []
        acces = []
        metric_corrects = []
        metric_denoms = []
        metric_losses = []
        metric_loss_denoms = []
        adapter = self._make_adapter()
        # for sequence paralle, position mask and input ids will split by sequence dim, need to keep origin for ttt shift
        global_input_ids = input_ids
        if self.attention_backend in ["sdpa", "fa", "usp"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        _trim_B = trim_pack is not None and self.trim_prompt_rows
        _trim_step1 = _trim_B and self.trim_step1  # B-ii:步1 也短路 prompt 行
        _sc_ref = None  # SELFCHECK 参考链(全长,no_grad,独立 cache)
        import os as _os5
        _sc_on = _os5.environ.get("MCTRIM_SELFCHECK", "0") == "1"
        for idx in range(self.length):
            _b_step1 = _trim_step1 and idx == 0  # B-ii 步1:q=[n_sup]/kv=[L] 拆分
            # _b_active = 本步 backbone 输出为紧凑形(步2..k,或 B-ii 的步1)
            _b_active = (_trim_B and idx >= 1) or _b_step1
            trim_ctx = None
            if _b_step1:
                # B-ii(步1):喂全长(prompt 行 K/V 需全行 embed+hidden),trim_ctx 标
                # step1_kv;attention 内部把 q 取 sup 行、k/v 保全长、rope 分别施。
                trim_ctx = {
                    "sup": trim_pack["sup"], "full_len": seq_length,
                    "step_idx": 0, "step1_kv": True,
                }
                step_input_ids = global_input_ids   # 全长(k/v 用全行 embed)
                step_hidden = hidden_states          # 全长
                step_attn = attention_mask
                step_pos = position_ids              # k 的全长位置(q 用 sup,在 attention 内取)
                if _sc_on:
                    # 自检:单独跑一遍全长 step1(no_grad,独立 cache)当参考 + seed _sc_ref
                    with torch.no_grad():
                        _re = self.draft_model.embed_input_ids(global_input_ids).to(
                            hidden_states.dtype
                        )
                        _rc = DynamicCache()
                        _ro = self.draft_model.backbone(
                            input_embeds=_re, hidden_states=hidden_states,
                            cache_hidden=None, attention_mask=attention_mask,
                            position_ids=position_ids, past_key_values=_rc,
                            use_cache=True, trim_ctx=None,  # 全长 step1 参考
                        )
                        _sc_ref = {"hidden": _ro, "cache": _rc}
            elif _b_active:
                # MCTRIM(B级 步 2..k):只跑监督行。rope 位置传绝对值 sup+idx。
                trim_ctx = {"sup": trim_pack["sup"], "full_len": seq_length, "step_idx": idx}
                if idx == 1 and _sc_on and not _trim_step1:
                    # B-i(无B-ii):idx==1 时 hidden 仍全长,用它 seed 参考链;
                    # B-ii 时 _sc_ref 已在 idx==0 seed 好,跳过。
                    _sc_ref = {
                        "hidden": hidden_states.detach().clone(),
                        "cache": DynamicCache(),
                    }
                    _sc_ref["cache"].update(
                        past_key_values.layers[0].keys.detach().clone(),
                        past_key_values.layers[0].values.detach().clone(),
                        layer_idx=0,
                    )
                if idx == 1 and not _trim_step1:
                    # B-i:step1 全长输出,idx==1 入口压紧;B-ii:step1 已返回紧凑,跳过。
                    hidden_states = hidden_states.index_select(1, trim_pack["sup"])
                step_input_ids = global_input_ids.index_select(1, trim_pack["sup"])
                step_hidden = hidden_states
                step_attn = attention_mask
                step_pos = trim_pack["sup"].unsqueeze(0) + idx
            elif trim_pack is not None:
                # MCTRIM(A级):teacher 已紧凑化,step_view 只为切 teacher 表;
                # SdpaLike 下 backbone 输入就是循环变量本身,直接用。
                step_input_ids = global_input_ids
                step_hidden = hidden_states
                step_attn = attention_mask
                step_pos = position_ids
            else:
                state = adapter.step_view(
                    idx=idx,
                    ttt_length=self.length,
                    global_input_ids=global_input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    target_p_padded=target_p_padded,
                    target_p_on_draft_padded=target_p_on_draft_padded,
                    target_token_ids_padded=target_token_ids_padded,
                    position_mask=position_mask,
                    seq_length=seq_length,
                )
                step_input_ids = state.input_ids
                step_hidden = state.hidden_states
                step_attn = state.attention_mask
                step_pos = state.position_ids
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            inputs_embeds = self.draft_model.embed_input_ids(step_input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=step_hidden,
                cache_hidden=cache_hidden,
                attention_mask=step_attn,
                position_ids=step_pos,
                past_key_values=past_key_values,
                use_cache=True,
                trim_ctx=trim_ctx,
            )
            if _b_step1 and _sc_ref is not None:
                # MCTRIM_SCB(B-ii 步0):紧凑 step1 输出 vs idx==0 已算好的全长 step1 参考
                _d = (
                    _sc_ref["hidden"].index_select(1, trim_pack["sup"]) - hidden_states_out
                ).abs().max().item()
                print(f"MCTRIM_SCB step=0 hid_maxdiff={_d:.3e}", flush=True)
            elif _b_active and _sc_ref is not None:
                # MCTRIM_SCB(步 2..k):全长参考链(no_grad,独立 cache)对拍监督行输出
                with torch.no_grad():
                    _ref_embeds = self.draft_model.embed_input_ids(global_input_ids)
                    _ref_embeds = _ref_embeds.to(_sc_ref["hidden"].dtype)
                    _ref_out = self.draft_model.backbone(
                        input_embeds=_ref_embeds,
                        hidden_states=_sc_ref["hidden"],
                        cache_hidden=None,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=_sc_ref["cache"],
                        use_cache=True,
                    )
                    _sc_ref["hidden"] = _ref_out
                    _d = (
                        _ref_out.index_select(1, trim_pack["sup"]) - hidden_states_out
                    ).abs().max().item()
                    print(f"MCTRIM_SCB step={idx} hid_maxdiff={_d:.3e}", flush=True)

            # update hidden states for next step
            hidden_states = hidden_states_out
            self._mcdbg(f"ttt{idx}_backbone_out", hidden_states)

            # Step 5.4 + 5.5 + 5.6: logits, metric and loss
            if trim_pack is not None:
                # MCTRIM(A级):只对监督行过 norm+lm_head;teacher 按滑窗索引紧凑取用
                sup = trim_pack["sup"]
                idx_j = trim_pack["idx_steps"][idx]
                if _b_active:
                    hidden_sup = hidden_states  # B级下已是紧凑形
                else:
                    hidden_sup = hidden_states.index_select(1, sup)
                logits = self.draft_model.compute_logits(hidden_sup)
                import os as _os3
                if _os3.environ.get("MCTRIM_SELFCHECK", "0") == "1" and trim_pack.get("_ref") is not None:
                    with torch.no_grad():
                        _ref = trim_pack["_ref"]
                        _sl = slice(idx, idx + seq_length)
                        _f_tp = _ref[0][:, _sl][:, sup]
                        _f_tpd = _ref[1][:, _sl][:, sup]
                        _f_tok = _ref[2][:, _sl][:, sup]
                        _f_pm = _ref[3][:, sup]
                        _t_tp = trim_pack["target_p_c"].index_select(1, idx_j)
                        _t_tpd = trim_pack["on_draft_c"].index_select(1, idx_j)
                        _t_tok = trim_pack["token_ids_c"].index_select(1, idx_j)
                        _full_hidden_src = (
                            _sc_ref["hidden"] if (_b_active and _sc_ref is not None)
                            else hidden_states
                        )
                        _full_logits_all = self.draft_model.compute_logits(_full_hidden_src)
                        _full_logits = _full_logits_all[:, sup]
                        from specforge.core.loss import LogSoftmaxLoss as _LSL
                        from specforge.core.loss import _compute_loss as _REF
                        _scale = sup.numel() / trim_pack["full_len"]
                        _tp_full = _ref[0][:, _sl].contiguous()
                        _ref_full = _REF(_full_logits_all, _tp_full, _ref[3])
                        _ref_trim = _REF(logits, _t_tp, trim_pack["position_mask_sup"]) * _scale
                        _ker_full = _LSL.apply(_full_logits_all, _tp_full, _ref[3])
                        _ker_trim = _LSL.apply(logits, _t_tp, trim_pack["position_mask_sup"]) * _scale
                        print(f"MCTRIM_SC4 step={idx} n_sup={sup.numel()} "
                              f"ref_full={_ref_full.double().item():.10f} ref_trim={_ref_trim.double().item():.10f} "
                              f"ker_full={_ker_full.double().item():.10f} ker_trim={_ker_trim.double().item():.10f}", flush=True)
                        print(f"MCTRIM_SC step={idx} tp={( _f_tp - _t_tp).abs().max().item():.3e} "
                              f"tpd={(_f_tpd - _t_tpd).abs().max().item():.3e} "
                              f"tok={(_f_tok != _t_tok).sum().item()} "
                              f"pm={(_f_pm - trim_pack['position_mask_sup']).abs().max().item():.3e} "
                              f"logits={(_full_logits - logits).abs().max().item():.3e}", flush=True)
                self._mcdbg(f"ttt{idx}_logits", logits)
                n_sup = sup.numel()
                (
                    acc,
                    acceptance_rate,
                    loss,
                    correct,
                    denom,
                    metric_loss,
                    loss_denom,
                ) = self._acc_and_loss(
                    logits=logits,
                    target_p=trim_pack["target_p_c"].index_select(1, idx_j),
                    target_p_on_draft=trim_pack["on_draft_c"].index_select(1, idx_j),
                    target_token_ids=trim_pack["token_ids_c"].index_select(1, idx_j),
                    position_mask=trim_pack["position_mask_sup"],
                    loss_mask=trim_pack["loss_mask_sup"],
                    adapter=adapter,
                    loss_scale=n_sup / trim_pack["full_len"],
                    full_positions=trim_pack["full_len"],
                )
            else:
                logits = self.draft_model.compute_logits(hidden_states)
                self._mcdbg(f"ttt{idx}_logits", logits)
                (
                    acc,
                    acceptance_rate,
                    loss,
                    correct,
                    denom,
                    metric_loss,
                    loss_denom,
                ) = self._acc_and_loss(
                    logits=logits,
                    target_p=state.target_p,
                    target_p_on_draft=state.target_p_on_draft,
                    target_token_ids=state.target_token_ids,
                    position_mask=state.position_mask,
                    loss_mask=state.loss_mask,
                    adapter=adapter,
                )
            acces.append(acc)
            acceptance_rates.append(acceptance_rate)
            self._mcdbg(f"ttt{idx}_loss", loss)
            import os as _os2
            if _os2.environ.get("MCTRIM_DBG", "0") == "1":
                _n_sup = (
                    trim_pack["sup"].numel() if trim_pack is not None
                    else int(loss_mask.sum().item())
                )
                print(
                    f"MCTRIM_loss step={idx} trim={'on' if trim_pack is not None else 'off'} "
                    f"loss={loss.double().item():.10f} acc={acc.double().item():.6f} "
                    f"ar={acceptance_rate.double().item():.6f} n_sup={_n_sup} L={seq_length}",
                    flush=True,
                )
            plosses.append(loss)
            metric_corrects.append(correct)
            metric_denoms.append(denom)
            metric_losses.append(metric_loss)
            metric_loss_denoms.append(loss_denom)

            if not is_last:
                # Step 5.7: we need to update the loss mask
                global_input_ids = padding(global_input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return (
            plosses,
            acceptance_rates,
            acces,
            metric_corrects,
            metric_denoms,
            metric_losses,
            metric_loss_denoms,
        )


class QwenVLOnlineEagle3Model(Eagle3Model):
    """
    In sgl-spec, we implement offline/online training.
    Online training means we have the target hidden_states available during training.
    Eagle3 using test time training technique (TTT) to train the draft model.
    1. We first extract the hidden states from the target model.
    2. Then concatenate the hidden states from 3 aux layers (layer 1, layer num_layers//2, layer num_layers-4).
    3. We project the concatenated hidden states to the target hidden size. from (batch, seq_len, 3*hidden_size) to (batch, seq_len, hidden_size)
    4. We concat the projected hidden states and embedding output as the input for the draft model.
    5. finally, we run TTT to train the draft model. input size is (batch, seq_len, hidden_size * 2)
    """

    def __init__(
        self,
        target_model,
        draft_model: Eagle3DraftModel,
        processor,
        length: int = 7,
        attention_backend: str = "sdpa",
        lk_loss_type: Optional[str] = None,
        kl_scale: float = 1.0,
        kl_decay: float = 1.0,
    ):
        """
        Args:
            target_model: the target model to extract hidden states.
            draft_model: the draft model to be trained.
            length: TTT length, it means how many turns to unroll during TTT.
            lk_loss_type: LK loss objective type. One of {"lambda", "alpha"}.
            kl_scale: Initial KL weight scale for lambda LK loss.
            kl_decay: Decay factor for adaptive KL weight in lambda LK loss.
        """
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        self.processor = processor
        self.length = length
        self.attention_backend = attention_backend
        self.lk_loss_type = lk_loss_type
        self.kl_scale = kl_scale
        self.kl_decay = kl_decay

    @torch.no_grad()
    def _prepare_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L692
        Extract the hidden states from the target model outputs.

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            device: the device to run the target model, if None, use the input_ids device
            pixel_values: image pixel values, used for VLM models
            image_grid_thw: image grid thw, used for VLM models

        Returns:
            hidden_states: (batch, seq_len, 3*hidden_size)
            target: (batch, seq_len, vocab_size)
            loss_mask: (batch, seq_len)
            input_ids: (batch, seq_len)
        """

        if device is None:
            device = input_ids.device

        # run the target model to get the hidden states
        outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            use_cache=False,
        )

        # extract the aux hidden states
        # output_hidden_states = True will return the embedding output as well
        # so we have an offset of 1
        num_hidden_states = len(outputs.hidden_states)
        offset = 1
        num_layers = num_hidden_states - 1

        # Eagle3 uses 3 aux layers from layer 1, num_layers//2, num_layers-4
        low_aux_layer = 1 + offset
        mid_aux_layer = num_layers // 2 - 1 + offset
        last_aux_layer = num_layers - 4 + offset

        hidden_states0 = outputs.hidden_states[low_aux_layer]
        hidden_states1 = outputs.hidden_states[mid_aux_layer]
        hidden_states2 = outputs.hidden_states[last_aux_layer]

        hidden_states = torch.cat(
            (hidden_states0, hidden_states1, hidden_states2), dim=-1
        )

        # apply pading
        target = outputs.logits
        target = padding(target, left=False)
        input_ids = padding(input_ids, left=False)

        if target is not None:
            target = target.to(device)
            loss_mask = loss_mask[..., None]
            loss_mask = loss_mask.to(device)

        return hidden_states, target, loss_mask, input_ids

    @torch.no_grad()
    def _get_input_embeds(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        # get input embeding with image
        # inputs_embeds = self.target_model.model.get_input_embeddings()(input_ids)
        inputs_embeds = self.draft_model.embed_input_ids(input_ids)
        image_embeds = self.target_model.model.get_image_features(
            pixel_values, image_grid_thw
        )
        image_embeds = torch.cat(image_embeds, dim=0)
        n_image_tokens = (
            input_ids == self.target_model.model.config.image_token_id
        ).sum()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == self.target_model.model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        """
        Online eagle model trainer, modified from: https://github.com/SafeAILab/EAGLE/blob/main/eagle/traineagle3/cnets.py#L711

        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            loss_mask: (batch, seq_len)
            past_key_values: We dont use this past_key_values in eagle3, but keep it for compatibility. We control kvcache by cache_hidden.
            position_ids: (batch, seq_len)
            pixel_values: batch image pixel values, used for VLM models
            image_grid_thw: (batch, 3), image grid thw, used for VLM models
        """
        # Step 0: prepare data with the target model
        hidden_states, target, loss_mask, input_ids = self._prepare_data(
            input_ids, attention_mask, loss_mask, pixel_values, image_grid_thw
        )

        # Step 1: handle vocab size
        (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        ) = _compute_target_p_padded(
            target=target,
            t2d=self.draft_model.t2d,
            loss_mask=loss_mask,
            length=self.length,
        )
        del target

        # basic info
        batch_size, seq_length, _ = hidden_states.shape
        seq_length_with_past = seq_length
        past_key_values_length = 0

        # Step 2: project the concatenated hidden states to the target hidden size
        hidden_states = self.draft_model.project_hidden_states(hidden_states)

        # Step 3: process kv cache, position ids and position ids
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                attention_mask_tensor = (
                    attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                )
                attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            position_ids, rope_deltas = self.target_model.model.get_rope_index(
                input_ids,
                image_grid_thw,
                None,
                second_per_grid_ts=None,
                attention_mask=attention_mask_tensor,
            )
            self.rope_deltas = rope_deltas
        else:
            position_ids = position_ids

        # Step 4: handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        if self.attention_backend == "sdpa":
            attention_mask = self.draft_model.prepare_decoder_attention_mask(
                attention_mask=attention_mask,
                hidden_states=hidden_states,
                batch_size=batch_size,
                seq_length=seq_length,
                past_key_values_length=past_key_values_length,
            )

        # Step 5: run TTT
        plosses = []
        acceptance_rates = []
        acces = []
        metric_corrects = []
        metric_denoms = []
        metric_losses = []
        metric_loss_denoms = []
        if self.attention_backend in ["sdpa", "fa"]:
            cache_hidden = [[], []]
            past_key_values = None
        elif self.attention_backend == "flex_attention":
            cache_hidden = None
            past_key_values = DynamicCache()
        else:
            raise ValueError(f"Unknown attention backend: {self.attention_backend}")

        for idx in range(self.length):
            target_p = target_p_padded[:, idx : idx + seq_length, :].contiguous()
            target_p_on_draft = target_p_on_draft_padded[
                :, idx : idx + seq_length, :
            ].contiguous()
            target_token_ids = target_token_ids_padded[
                :, idx : idx + seq_length
            ].contiguous()
            is_last = idx == self.length - 1

            # Step 5.1: embed the input ids
            # inputs_embeds = self._get_input_embeds(input_ids, pixel_values, image_grid_thw)
            inputs_embeds = self.draft_model.embed_input_ids(input_ids)
            inputs_embeds = inputs_embeds.to(hidden_states.dtype)

            # Step 5.2: run the draft model backbone
            hidden_states_out = self.draft_model.backbone(
                input_embeds=inputs_embeds,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # update hidden states for next step
            hidden_states = hidden_states_out

            # Step 5.4: get logits
            logits = self.draft_model.compute_logits(hidden_states)

            # Step 5.5: record metrics first as we in-place modify logits
            with torch.no_grad():
                correct, denom = _compute_metric_counts(
                    logits=logits,
                    target_token_ids=target_token_ids,
                    loss_mask=loss_mask,
                    d2t=self.draft_model.d2t,
                )
                acces.append(correct / denom)
                metric_corrects.append(correct)
                metric_denoms.append(denom)

            # Step 5.6: calculate loss, in-place modifies logits!
            acceptance_rate, loss = _compute_loss_and_acceptance_rate(
                logits=logits,
                target_p=target_p,
                target_p_on_draft=target_p_on_draft,
                position_mask=position_mask,
                lk_loss_type=self.lk_loss_type,
                kl_scale=self.kl_scale,
                kl_decay=self.kl_decay,
            )
            acceptance_rates.append(acceptance_rate)
            plosses.append(loss)
            metric_losses.append(loss.detach())
            metric_loss_denoms.append(
                torch.tensor(
                    logits.shape[0] * logits.shape[1],
                    device=logits.device,
                    dtype=torch.float32,
                )
            )

            if not is_last:
                # Step 5.7: we need to update the loss mask
                input_ids = padding(input_ids, left=False)
                position_mask = padding(position_mask, left=False)
                loss_mask = padding(loss_mask, left=False)
                # Flex attention mask shirnking is handled inside attention module
        return (
            plosses,
            acceptance_rates,
            acces,
            metric_corrects,
            metric_denoms,
            metric_losses,
            metric_loss_denoms,
        )


def _compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        (
            target_p,
            target_p_on_draft,
            target_token_ids,
            position_mask,
        ) = _compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )
        target_p_on_draft_padded = F.pad(
            target_p_on_draft,
            pad=(0, 0, 0, length),
            mode="constant",
            value=0.0,
        )
        target_token_ids_padded = F.pad(
            target_token_ids,
            pad=(0, length),
            mode="constant",
            value=0,
        )

        return (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        )



def _compute_target_p_eager(target, t2d, loss_mask, row_chunk=256):
    """MCTRIM: _compute_target_p 的非编译版(逐 batch 形状变化,套 compile 会重编译风暴)。
    数学与编译版一致;按行分块压全词表 fp32 瞬时(n_sup 大时原本可达数 GB)。"""
    tps, tpds, toks, pms = [], [], [], []
    n = target.shape[1]
    for s in range(0, n, row_chunk):
        t = target[:, s : s + row_chunk].float()
        ids = t.argmax(-1)
        tm = t2d[ids][..., None].int()
        pms.append(tm * loss_mask[:, s : s + row_chunk])
        dth = t[..., t2d]
        tps.append(nn.Softmax(dim=2)(dth).detach())
        lse = torch.logsumexp(t, dim=-1, keepdim=True)
        tpds.append(torch.exp(dth - lse).detach())
        toks.append(ids.detach())
    return (torch.cat(tps, 1), torch.cat(tpds, 1),
            torch.cat(toks, 1), torch.cat(pms, 1))


def _build_trim_pack(target, t2d, loss_mask, length):
    """A级裁剪(--trim-loss-positions):teacher 只在"监督行及其 k 步滑窗位置"上计算。

    语义对齐 _compute_target_p_padded + step_view 滑窗:
    - 监督行集合全 k 步共用(SpecForge 的 mask 不随步 shift,teacher 表滑窗);
    - 步 j 的 teacher = 位置 sup+j;越过 L 的位置 = pad 行(target_p 均匀 1/V,
      on_draft=0,token_id=0,与 F.pad 的 value 完全一致)。
    仅支持 batch=1(我们 online 每 rank batch=1);B>1 由调用方回退全长路径。
    返回 dict:sup[n_sup], 每步 teacher 取用的 gather 索引 idx_steps[k][n_sup],
    紧凑表 target_p_c/on_draft_c/token_ids_c([1, n_real+1, ...],末行=pad 行),
    position_mask_sup/loss_mask_sup([1, n_sup, 1])。
    """
    with torch.no_grad():
        B, L = loss_mask.shape[0], loss_mask.shape[1]
        assert B == 1, "trim path requires batch==1"
        sup = loss_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [n_sup]
        shifted = torch.cat([sup + j for j in range(length + 1)])     # 步 j=0..k(含 k:与 pad 长度对齐)
        uniq = torch.unique(shifted)
        real = uniq[uniq < L]                                          # 真实 teacher 位置
        n_real = real.numel()
        target_sel = target[:, real]                                   # [1, n_real, V_target] 唯一的全词表切片
        lm_sel = loss_mask[:, real]
        tp, tpd, tok, _ = _compute_target_p_eager(target_sel, t2d, lm_sel)
        V_d = tp.shape[-1]
        # 末行 append pad 行(与 F.pad 的 value 一致)
        pad_tp = torch.full((1, 1, V_d), 1.0 / V_d, dtype=tp.dtype, device=tp.device)
        pad_tpd = torch.zeros((1, 1, V_d), dtype=tpd.dtype, device=tpd.device)
        pad_tok = torch.zeros((1, 1), dtype=tok.dtype, device=tok.device)
        target_p_c = torch.cat([tp, pad_tp], dim=1)
        on_draft_c = torch.cat([tpd, pad_tpd], dim=1)
        token_ids_c = torch.cat([tok, pad_tok], dim=1)
        # remap:绝对位置 -> 紧凑表行号;未覆盖/越界位置 -> pad 行(n_real)
        full_map = torch.full((L + length + 1,), n_real, dtype=torch.long, device=sup.device)
        full_map[real] = torch.arange(n_real, device=sup.device)
        idx_steps = [full_map[sup + j] for j in range(length)]
        # position_mask 取链起点(sup)处——与全长路径对 step 不变的语义一致
        pm_sup = _compute_position_mask_at(target, t2d, loss_mask, sup)
        loss_mask_sup = loss_mask.view(-1)[sup].view(1, -1, 1)
        return dict(sup=sup, idx_steps=idx_steps, target_p_c=target_p_c,
                    on_draft_c=on_draft_c, token_ids_c=token_ids_c,
                    position_mask_sup=pm_sup, loss_mask_sup=loss_mask_sup,
                    full_len=L)


def _compute_position_mask_at(target, t2d, loss_mask, sup):
    """position_mask = t2d[argmax(target)] * loss_mask,只在 sup 位置上算。"""
    with torch.no_grad():
        tsel = target[:, sup]
        ids = tsel.float().argmax(-1)
        tm = t2d[ids][..., None].int()
        return tm * loss_mask[:, sup]


@torch.compile(dynamic=None)
def _compute_target_p(target, t2d, loss_mask):
    target_head = target.float()
    target_token_ids = target_head.argmax(-1)
    target_mask = t2d[target_token_ids]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    draft_target_head = target_head[..., t2d]
    target_p = nn.Softmax(dim=2)(draft_target_head)
    target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
    target_p_on_draft = torch.exp(draft_target_head - target_logsumexp)
    target_p = target_p.detach()
    target_p_on_draft = target_p_on_draft.detach()
    target_token_ids = target_token_ids.detach()
    return target_p, target_p_on_draft, target_token_ids, position_mask


@torch.compile(dynamic=None)
def _compute_metric_acc(logits, target_token_ids, loss_mask, d2t):
    correct, denom = _compute_metric_counts(logits, target_token_ids, loss_mask, d2t)
    return correct / denom


@torch.compile(dynamic=None)
def _compute_metric_counts(logits, target_token_ids, loss_mask, d2t):
    pred_draft_token_ids = logits.argmax(-1)
    pred_target_token_ids = pred_draft_token_ids + d2t[pred_draft_token_ids]
    correct = (
        (pred_target_token_ids == target_token_ids) * loss_mask.squeeze(-1)
    ).sum()
    denom = loss_mask.sum().clamp_min(1e-6)
    return correct, denom
