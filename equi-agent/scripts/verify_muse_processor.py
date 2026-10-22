"""CPU-only verification of the downloaded Muse processor; loads no model weights."""
import argparse
from pathlib import Path

from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForMultimodalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    path = args.model_path.expanduser().resolve()
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    if config.model_type != "muse_glimmer":
        raise ValueError(f"Expected muse_glimmer, got {config.model_type}")
    if type(config) not in AutoModelForMultimodalLM._model_mapping:
        raise ValueError("Installed Transformers has no multimodal Muse loader")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    if not processor.chat_template:
        raise ValueError("Downloaded Muse processor has no chat template; stop before weights download")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe the two supplied images."},
        {"type": "image", "image": Image.new("RGB", (128, 128), "white")},
        {"type": "image", "image": Image.new("RGB", (128, 128), "black")}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt")
    if "input_ids" not in inputs or "pixel_values" not in inputs:
        raise ValueError("Processor did not produce both image and text tensors")
    print("Architecture:", config.model_type)
    print("Chat template present:", bool(processor.chat_template))
    for name, tensor in inputs.items():
        print(name, tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor).__name__)
    print("Two-image processor check passed. No model weights were loaded.")


if __name__ == "__main__":
    main()
