"""Optional architecture test: run in the separate Gemma 4 environment, without downloads."""
import tempfile
import unittest

try:
    from transformers import AutoModelForMultimodalLM, Gemma4UnifiedConfig
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires the separate Gemma 4 Transformers environment")
class Gemma4RuntimeTest(unittest.TestCase):
    def test_two_image_generation_after_local_reload(self):
        import torch
        config = Gemma4UnifiedConfig(
            text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                             head_dim=16, layer_types=["sliding_attention", "full_attention"],
                             per_layer_config={}, max_position_embeddings=128, sliding_window=16,
                             rope_parameters={key: dict(rope_type="default", rope_theta=10000.)
                                              for key in ("full_attention", "sliding_attention")}),
            vision_config=dict(patch_size=2, pooling_kernel_size=2, mm_embed_dim=32,
                               output_proj_dims=32, mm_posemb_size=16),
            image_token_id=4, boi_token_id=5, eoi_token_id=6, audio_token_id=7, video_token_id=8)
        config._attn_implementation = "eager"
        model = AutoModelForMultimodalLM.from_config(config).eval()
        inputs = dict(input_ids=torch.tensor([[2, 4, 4, 9, 10]]),
                      attention_mask=torch.ones(1, 5, dtype=torch.long),
                      mm_token_type_ids=torch.tensor([[0, 1, 1, 0, 0]]),
                      pixel_values=torch.rand(2, 1, 48),
                      image_position_ids=torch.zeros(2, 1, 2, dtype=torch.long))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            model = AutoModelForMultimodalLM.from_pretrained(
                directory, torch_dtype=torch.float32, device_map={"": "cpu"},
                attn_implementation="eager")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                        disable_compile=True, eos_token_id=None)
        self.assertEqual(tuple(output.shape), (1, 7))


if __name__ == "__main__":
    unittest.main()
