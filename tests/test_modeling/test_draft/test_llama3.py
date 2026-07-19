import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaForCausalLMEagle3,
    LlamaMLP,
    LlamaRMSNorm,
)

# from model_module import LlamaForCausalLMEagle3


class TestLlamaForCausalLMEagle3Loading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()

        config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 128256,
            "draft_vocab_size": 32000,
        }

        self.config = LlamaConfig(**config_dict)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        model = LlamaForCausalLMEagle3(self.config)

        self.assertIsInstance(model.midlayer.self_attn, LlamaAttention)
        self.assertIsInstance(model.midlayer.mlp, LlamaMLP)
        self.assertIsInstance(model.midlayer.hidden_norm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.input_layernorm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.post_attention_layernorm, LlamaRMSNorm)
        self.assertEqual(model.midlayer.hidden_size, self.config.hidden_size)

    def test_save_pretrained(self):
        """Test the model's save_pretrained functionality."""
        model = LlamaForCausalLMEagle3(self.config)

        self.config.save_pretrained(self.temp_dir)

        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        """mock"""
        mock_model = LlamaForCausalLMEagle3(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, LlamaForCausalLMEagle3)

    def test_model_forward_pass(self):
        """forward"""
        model = LlamaForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 10

        input_emb = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=input_emb,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_state_dict_compatibility(self):
        model1 = LlamaForCausalLMEagle3(self.config)
        model2 = LlamaForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()

        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_rotary_buffers_absent_from_state_dict(self):
        """Root cause: rotary buffers are non-persistent, so they are not saved.

        inv_freq / cos_cached / sin_cached are registered with persistent=False, so a
        checkpoint's state_dict does not contain them. transformers' meta-device load
        path therefore leaves them uninitialized after from_pretrained (NaN on GPU),
        which is why build_draft_model must rebuild them via _init_rope.
        """
        model = LlamaForCausalLMEagle3(self.config)
        keys = model.state_dict().keys()
        for buf in ("inv_freq", "cos_cached", "sin_cached"):
            self.assertFalse(
                any(k.endswith(buf) for k in keys),
                f"{buf} unexpectedly present in state_dict (should be persistent=False)",
            )

    def test_rotary_buffers_rebuilt_after_from_pretrained(self):
        """Regression test for the persistent=False rotary-buffer warm-start NaN bug.

        After a real from_pretrained round-trip, re-running _init_rope (as
        build_draft_model does) must yield finite rotary buffers identical to a
        freshly-initialized reference. Equality is checked rather than relying on a
        NaN observation, since CPU-only CI does not reliably reproduce the
        uninitialized-memory NaN that the fix prevents on GPU.
        """
        ref = LlamaForCausalLMEagle3(self.config)
        ref_rot = ref.midlayer.self_attn.rotary_emb
        ref_bufs = {
            name: getattr(ref_rot, name).clone()
            for name in ("inv_freq", "cos_cached", "sin_cached")
        }

        ref.save_pretrained(self.temp_dir)
        loaded = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)

        # The fix in build_draft_model: rebuild non-persistent rotary buffers.
        rebuilt = False
        for module in loaded.modules():
            init_rope = getattr(module, "_init_rope", None)
            if callable(init_rope):
                init_rope()
                rebuilt = True
        self.assertTrue(rebuilt, "no module exposed a callable _init_rope()")

        got_rot = loaded.midlayer.self_attn.rotary_emb
        for name, ref_val in ref_bufs.items():
            val = getattr(got_rot, name)
            self.assertTrue(
                torch.isfinite(val).all(), f"{name} non-finite after rebuild"
            )
            self.assertTrue(
                torch.allclose(val.float(), ref_val.float()),
                f"{name} does not match freshly-initialized reference",
            )

    def test_config_validation(self):
        # A dimensionally-valid config (hidden_size divisible by num_attention_heads
        # so it passes transformers' strict config validation) that is still missing
        # the required `draft_vocab_size` attribute. Building the Eagle3 model from it
        # should raise AttributeError.
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        with self.assertRaises(AttributeError):
            LlamaForCausalLMEagle3(invalid_config)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestLlamaForCausalLMEagle3Loading))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
