"""CPU-only verification of the downloaded Muse processor; loads no model weights."""
import argparse
from pathlib import Path

from PIL import Image
import torchvision
from transformers import AutoConfig, AutoProcessor, AutoModelForMultimodalLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--require-native-lanczos", action="store_true",
                        help="Require CPU preprocessing and torchvision >=0.27, with LANCZOS configured")
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
    if args.require_native_lanczos:
        from packaging.version import Version
        if Version(torchvision.__version__.split("+")[0]) < Version("0.27"):
            raise ValueError("Native LANCZOS requires the separate updated Muse environment (torchvision >=0.27)")
        if processor.image_processor.resample != Image.Resampling.LANCZOS:
            raise ValueError("Checkpoint image processor does not specify LANCZOS; inspect before proceeding")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "Describe the two supplied images."},
        {"type": "image", "image": Image.new("RGB", (128, 128), "white")},
        {"type": "image", "image": Image.new("RGB", (128, 128), "black")}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt")
    if "input_ids" not in inputs or "pixel_values" not in inputs:
        raise ValueError("Processor did not produce both image and text tensors")
    if args.require_native_lanczos and inputs["pixel_values"].device.type != "cpu":
        raise ValueError("Native LANCZOS requires preprocessing on CPU")
    print("torchvision:", torchvision.__version__)
    print("Image backend:", processor.image_processor.backend)
    print("Image resample:", processor.image_processor.resample)
    print("Architecture:", config.model_type)
    print("Chat template present:", bool(processor.chat_template))
    for name, tensor in inputs.items():
        print(name, tuple(tensor.shape) if hasattr(tensor, "shape") else type(tensor).__name__)
    print("Two-image processor check passed. No model weights were loaded.")


if __name__ == "__main__":
    main()
