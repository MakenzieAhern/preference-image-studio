"""Opt-in text-mode smoke diagnostic using the real segmentation and language models."""
import argparse
import json
from pathlib import Path
import tempfile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "This can download Mask2Former and Qwen weights. The default Qwen 7B model uses "
        "substantial RAM on CPU; CUDA requires bitsandbytes for 4-bit loading. "
        "PIE_MODEL_LOCAL_FILES_ONLY=1 prevents downloads; PIE_LLM_MODEL explicitly "
        "overrides the language model. Use test_api_smoke for a bundled-CNN-only check."
    ))
    parser.add_argument("--with-models", action="store_true", help="Explicitly allow optional segmentation and language models")
    parser.add_argument("--max-side", type=int, default=256, help="Resize smoke input to this maximum side (default: 256)")
    args = parser.parse_args(argv)
    if not args.with_models:
        parser.error("Text diagnostics require --with-models")
    if not 1 <= args.max_side <= 1024:
        parser.error("--max-side must be between 1 and 1024")
    import torch
    from .image_io import load_rgb_image
    from .inference_service import InferenceService

    root = Path(__file__).resolve().parents[2]
    output = root / "demo_web" / "outputs" / "smoke_enhanced.png"
    original = root / "demo_web" / "frontend" / "public" / "examples" / "demo-person-wall.jpg"
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"gpu_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")
    with tempfile.TemporaryDirectory(prefix="pie-text-smoke-") as folder:
        source = Path(folder) / "source.png"
        image = load_rgb_image(original)
        image.thumbnail((args.max_side, args.max_side))
        image.save(source)
        service = InferenceService()
        result = service.enhance_text(source, "Make the person slightly warmer.", output)
        assert result["checkpoint_loaded"] and not result["has_nan"] and not result["has_inf"]
        assert 0 <= result["output_min"] <= result["output_max"] <= 1 and output.is_file()
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
