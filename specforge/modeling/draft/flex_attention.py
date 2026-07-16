import torch
import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    or_masks,
)
from transformers.utils import is_torchdynamo_compiling

dynamo.config.recompile_limit = 64


# Reference Implementation https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/flex_attention.py
class WrappedFlexAttention:
    """
    We are doing a singleton class so that flex attention is compiled once when it's first called.
    """

    _instance = None
    _is_flex_compiled = False
    _compiled_flex_attention = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            # Create a new instance if one doesn't already exist
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        """
        Initialize or update the singleton instance.
        """
        if not self._is_flex_compiled:
            # Enable dynamic shapes to handle different input sizes
            self._compiled_flex_attention = torch.compile(
                flex_attention,
                # mode="max-autotune-no-cudagraphs",
            )
            self._is_flex_compiled = True

    def __call__(self):
        return self._compiled_flex_attention


def compile_friendly_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    # First call initialise singleton wrapper object, second call invokes the object method to return compiled flex attention
    # Do not use compiled version if already compiling forward (it raises issues)
    flex_attention_compiled = (
        WrappedFlexAttention()() if not is_torchdynamo_compiling() else flex_attention
    )
    return flex_attention_compiled(
        query,
        key,
        value,
        **kwargs,
    )


class WrappedCreateBlockMask:
    _instance = None
    _is_create_block_mask_compiled = False
    _compiled_create_block_mask = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @torch.compiler.disable(recursive=False)
    def __init__(self):
        if not self._is_create_block_mask_compiled:
            self._compiled_create_block_mask = torch.compile(create_block_mask)
            self._is_create_block_mask_compiled = True

    def __call__(self):
        return self._compiled_create_block_mask


def compile_friendly_create_block_mask(
    mask_mod,
    B,
    H,
    Q_LEN,
    KV_LEN,
    device,
):
    create_block_mask_compiled = (
        WrappedCreateBlockMask()()
        if not is_torchdynamo_compiling()
        else create_block_mask
    )
    return create_block_mask_compiled(
        mask_mod,
        B,
        H,
        Q_LEN,
        KV_LEN,
        device,
    )


def generate_eagle3_mask(
    seq_lengths: torch.Tensor, Q_LEN: int, KV_LEN: int, lck: int = 0
):

    def causal_mask(b, h, q_idx, kv_idx):
        # Causal will keep shrinking by 1 diagnol due to appended suffix
        # Shirnk the causal by diagnol
        causal_mask = q_idx >= kv_idx
        padding_mask = (kv_idx < seq_lengths[b]) & (q_idx < seq_lengths[b])
        return causal_mask & padding_mask

    def suffix_mask(b, h, q_idx, kv_idx):
        suffix_mask = kv_idx >= Q_LEN
        padding_mask = kv_idx % Q_LEN < seq_lengths[b]
        diagnol_mask = (kv_idx - q_idx) % Q_LEN == 0
        return suffix_mask & padding_mask & diagnol_mask

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_Q_{Q_LEN}_KV_{KV_LEN}_lck_{lck}"
    return mask_mod


def generate_eagle3_mask_compact(
    sup: "torch.Tensor", seq_lengths: "torch.Tensor", full_len: int,
    Q_LEN: int, KV_LEN: int
):
    """MCTRIM(B级):TTT 步 2..k 只跑监督行时的 mask。

    q 行 r 对应绝对位置 sup[r];KV 布局 = [步1 全长块 full_len | 后续块 各 Q_LEN 宽]。
    - 步1 块(kv_idx < full_len):绝对 causal:sup[r] >= kv_idx,且双方在有效长度内;
    - 后续块:对角线私有:块内偏移 == r(与全长版 suffix_mask 的 (kv-q)%L==0 同语义)。
    """

    def causal_mask(b, h, q_idx, kv_idx):
        abs_q = sup[q_idx]
        in_prefix = kv_idx < full_len
        causal = abs_q >= kv_idx
        padding = (kv_idx < seq_lengths[b]) & (abs_q < seq_lengths[b])
        return in_prefix & causal & padding

    def suffix_mask(b, h, q_idx, kv_idx):
        in_suffix = kv_idx >= full_len
        diagonal = ((kv_idx - full_len) % Q_LEN) == q_idx
        padding = sup[q_idx] < seq_lengths[b]
        return in_suffix & diagonal & padding

    mask_mod = or_masks(causal_mask, suffix_mask)
    mask_mod.__name__ = f"eagle3_mask_compact_Q_{Q_LEN}_KV_{KV_LEN}"
    return mask_mod
