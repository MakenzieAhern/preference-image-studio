"""Opt-in diagnostic for real reference, text and manual models; not part of pytest."""
import argparse
import json
from pathlib import Path
import tempfile


def compact(result):
    keys = ("total_processing_seconds", "cnn_inference_seconds", "image_tensor_shape",
            "diffmap_tensor_shape", "combined_shape", "output_shape", "output_min", "output_max",
            "has_nan", "has_inf", "mode_used", "matched_regions", "global_adjustments",
            "final_global_adjustments", "regions", "region_json", "output_path")
    return {key: result[key] for key in keys if key in result}


def _checked(result):
    assert result["checkpoint_loaded"]
    assert not result["has_nan"] and not result["has_inf"]
    assert 0 <= result["output_min"] <= result["output_max"] <= 1
    assert Path(result["output_path"]).is_file()
    return compact(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Loads/downloads Mask2Former and Qwen (default: 7B). CPU execution of the default "
        "Qwen model requires substantial RAM; CUDA uses bitsandbytes 4-bit loading. "
        "Set PIE_MODEL_LOCAL_FILES_ONLY=1 for cached/local weights only, and PIE_LLM_MODEL "
        "to explicitly choose another model. A changed model is a different experiment."
    ))
    parser.add_argument("--with-models", action="store_true", help="Explicitly allow the optional segmentation and language models")
    parser.add_argument("--max-side", type=int, default=256, help="Resize smoke inputs to this maximum side (default: 256)")
    args = parser.parse_args(argv)
    if not args.with_models:
        parser.error("Full-model diagnostics require --with-models; use test_api_smoke for the bundled-CNN-only check")
    if not 1 <= args.max_side <= 1024:
        parser.error("--max-side must be between 1 and 1024")
    from .image_io import load_rgb_image
    from .inference_service import InferenceService

    root = Path(__file__).resolve().parents[2]
    outputs = root / "demo_web" / "outputs"
    examples = root / "demo_web" / "frontend" / "public" / "examples"
    with tempfile.TemporaryDirectory(prefix="pie-full-smoke-") as folder:
        source, reference = Path(folder) / "source.png", Path(folder) / "reference.png"
        for original, target in ((examples / "demo-person-wall.jpg", source), (examples / "demo.jpg", reference)):
            image = load_rgb_image(original)
            image.thumbnail((args.max_side, args.max_side))
            image.save(target)
        service = InferenceService()
        region = service.enhance_reference(source, reference, .7, True, outputs / "reference-region-smoke.png")
        from .z3_use import z2_llm_instruction
        z2_llm_instruction.DEBUG_LLM = False
        # "Cold" means the first LLM call in this process, not an empty disk cache.
        cold = service.enhance_text(source, "Make the person slightly warmer.", outputs / "text-cold-smoke.png")
        warm = service.enhance_text(source, "Make the person slightly warmer.", outputs / "text-warm-smoke.png")
        manual = service.enhance_manual(source, {"r": .08, "g": 0, "b": -.08, "contrast": .05}, .6,
                                        outputs / "manual-smoke.png")
        print(json.dumps({"reference": _checked(region), "text_cold": _checked(cold),
                          "text_warm": _checked(warm), "manual": _checked(manual)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
