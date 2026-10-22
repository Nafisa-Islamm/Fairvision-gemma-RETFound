"""Optional real Muse architecture check, without downloading pretrained weights."""
import tempfile
import unittest

try:
    from transformers import AutoModelForMultimodalLM, MuseGlimmerConfig
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


@unittest.skipUnless(AVAILABLE, "Requires the separate modern multimodal environment")
class MuseRuntimeTest(unittest.TestCase):
    def test_two_image_generation_after_reload(self):
        import torch
        config = MuseGlimmerConfig(
            text_config=dict(vocab_size=64, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             head_dim=8, max_position_embeddings=128, sliding_window=16,
                             pad_token_id=0, bos_token_id=1, eos_token_id=2,
                             layer_types=["sliding_attention", "full_attention"]),
            vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                               num_attention_heads=2, patch_size=2, patch_temporal=2,
                               merge_size=1, pos_emb_height=4, pos_emb_width=4,
                               max_position_embeddings=16,
                               layer_types=["window_attention", "full_attention"]),
            image_token_id=3, video_token_id=4, out_hidden_size=16, projector_hidden_size=16)
        config._attn_implementation = "eager"
        model = AutoModelForMultimodalLM.from_config(config).eval()
        inputs = dict(input_ids=torch.tensor([[1, 3, 3, 8, 9]]),
                      attention_mask=torch.ones(1, 5, dtype=torch.long),
                      pixel_values=torch.rand(2, 24), image_grid_thw=torch.ones(2, 3, dtype=torch.long))
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            model = AutoModelForMultimodalLM.from_pretrained(
                directory, torch_dtype=torch.float32, device_map={"": "cpu"}, attn_implementation="eager")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=2, do_sample=False,
                                        disable_compile=True, eos_token_id=None)
        self.assertEqual(tuple(output.shape), (1, 7))


if __name__ == "__main__":
    unittest.main()
