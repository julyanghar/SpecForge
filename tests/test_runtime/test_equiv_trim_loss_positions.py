# coding=utf-8
"""Equivalence: --trim-loss-positions must not change the training loss.

A-level position trimming (--trim-loss-positions) computes the teacher target_p,
the draft logits and the loss only at supervised (loss-masked) positions instead
of over the full sequence. It is mathematically equivalent to the full-length path
(the mean denominator is rescaled from n_sup back to the full length). This test
runs the identical online forward with trimming off and on and asserts the
per-step losses match within bf16 tolerance.

GPU-only, matching the other online EAGLE3 equivalence tests in this directory.
"""

import os
import shutil
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "trim-loss-positions equivalence requires CUDA")
class TestEquivTrimLossPositions(unittest.TestCase):
    def test_trim_loss_positions_matches_full(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29567")

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )

        H, V, SEQ, TTT = fx.H, fx.V, 16, 3
        workdir = tempfile.mkdtemp(prefix="equiv_trim_")
        self.addCleanup(shutil.rmtree, workdir, ignore_errors=True)
        target, _dir, _aux = fx.build_hf_target(workdir, hidden=H, layers=8, vocab=V)
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        draft = AutoEagle3DraftModel.from_config(
            AutoDraftModelConfig.from_file(cfg),
            attention_backend="flex_attention",
            torch_dtype=torch.bfloat16,
        ).cuda()
        draft.load_vocab_mapping(vocab_path)
        draft.freeze_embedding()

        # Prompt-heavy mask so trimming is non-trivial: the first half of the
        # sequence is unsupervised (loss_mask=0), the second half supervised (1).
        torch.manual_seed(11)
        input_ids = torch.randint(0, V, (1, SEQ), device="cuda")
        attn = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[:, SEQ // 2 :] = 1
        d = target.generate_eagle3_data(input_ids, attn, loss_mask)

        @torch.no_grad()
        def step_losses(trim: bool):
            model = OnlineEagle3Model(
                draft_model=draft,
                length=TTT,
                attention_backend="flex_attention",
                trim_loss_positions=trim,
            ).cuda()
            model.eval()
            plosses, *_ = model(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                loss_mask=d.loss_mask,
                target=d.target,
                hidden_states=d.hidden_states,
            )
            return [p.item() for p in plosses]

        full = step_losses(False)
        trimmed = step_losses(True)

        self.assertEqual(len(full), len(trimmed))
        for i, (a, b) in enumerate(zip(full, trimmed)):
            tol = 5e-3 * max(abs(a), abs(b)) + 1e-4
            self.assertLessEqual(
                abs(a - b),
                tol,
                msg=f"step {i}: full={a} trimmed={b} (tol={tol})",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
