"""Offline regressions: real bundled CNN; semantic model I/O is explicitly mocked."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock
import time

import numpy as np
from PIL import Image
import pytest
import torch

from demo_web.backend.image_io import load_rgb_array, load_rgb_image
from demo_web.backend.inference_service import InferenceService
from demo_web.backend.model.network import PreferenceControlledCNN
from demo_web.backend.preference_adapter import PreferenceAdapter
from demo_web.backend.reference_adapter import ReferenceAdapter
from demo_web.backend.z3_use.z1_segmentation import LABELS, present_labels, resize_for_processing
from demo_web.backend.z3_use.z2_llm_instruction import (
    _extract_json, _sanitize, build_skeleton, generate_region_json,
)
from demo_web.backend.z3_use.z3_diffmap import build_diffmap_from_json, diffmap_to_chw


@pytest.fixture(scope="module")
def service():
    torch.set_num_threads(2)
    return InferenceService(device="cpu")


@pytest.fixture
def rgb_image(tmp_path):
    pixels = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    path = tmp_path / "source.png"
    Image.fromarray(pixels).save(path)
    return path, pixels


def test_palette_indices_are_decoded_as_colours(tmp_path):
    image = Image.new("P", (2, 1))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * (768 - 6))
    image.putdata([0, 1])
    path = tmp_path / "palette.png"
    image.save(path)
    np.testing.assert_array_equal(load_rgb_array(path), [[[1, 0, 0], [0, 1, 0]]])


def test_cmyk_and_alpha_use_rgb_conversion(tmp_path):
    for mode, value in [("CMYK", (0, 255, 255, 0)), ("RGBA", (255, 0, 0, 128))]:
        path = tmp_path / (mode + ".tiff")
        Image.new(mode, (2, 2), value).save(path)
        np.testing.assert_allclose(load_rgb_array(path), np.broadcast_to([1, 0, 0], (2, 2, 3)))


def test_16bit_grayscale_scales_by_full_range_and_preview_agrees(tmp_path):
    path = tmp_path / "gray.tiff"
    Image.fromarray(np.array([[0, 32768, 65535]], dtype=np.uint16)).save(path)
    result = load_rgb_array(path)
    np.testing.assert_allclose(result[0, :, 0], [0, 32768 / 65535, 1])
    np.testing.assert_array_equal(np.array(load_rgb_image(path))[0, :, 0], [0, 128, 255])
    assert np.all(result[:, :, 0] == result[:, :, 2])


def test_exif_orientation_is_shared_with_semantic_input(tmp_path):
    image = Image.new("RGB", (3, 2), "red")
    exif = Image.Exif(); exif[274] = 6
    path = tmp_path / "rotated.jpg"; image.save(path, exif=exif)
    assert load_rgb_array(path).shape == (3, 2, 3)
    assert load_rgb_image(path).size == (2, 3)


def test_ambiguous_float_image_is_rejected(tmp_path):
    path = tmp_path / "float.tiff"
    Image.fromarray(np.ones((3, 3), dtype=np.float32)).save(path)
    with pytest.raises(ValueError, match="Floating-point"):
        load_rgb_array(path)


def test_extremely_narrow_segmentation_resize_never_becomes_zero():
    image, original = resize_for_processing(Image.new("RGB", (1, 5000)), 512)
    assert image.size == (1, 512) and original == (1, 5000)


def test_ade_class_127_is_a_real_bicycle_region():
    assert len(LABELS) == 150 and LABELS[127] == "bicycle"
    assert present_labels(np.array([[127, 255, -1, 150]])) == ["bicycle"]


def test_global_only_text_plan_applies_to_every_pixel_and_contrast():
    labels = np.array([[0, 127], [255, 1]], dtype=np.int32)
    result = build_diffmap_from_json(labels, {"global": {"r": .1, "contrast": .2}})
    np.testing.assert_allclose(result[:, :, 0], .1)
    np.testing.assert_allclose(result[:, :, 3], .2)
    assert diffmap_to_chw(result).shape == (4, 2, 2)


def test_text_regional_controls_add_to_global_then_clip():
    labels = np.array([[0, 1]])
    result = build_diffmap_from_json(labels, {"global": {"r": .25, "contrast": .2}, LABELS[0]: {"r": .2, "contrast": .2}})
    np.testing.assert_allclose(result[0, 0], [.3, 0, 0, .3])
    np.testing.assert_allclose(result[0, 1], [.25, 0, 0, .2])
    with pytest.raises(ValueError, match="finite"):
        build_diffmap_from_json(labels, {"global": {"r": float("nan")}})


def test_text_label_conversion_retains_global_contrast_and_ignores_unspecified_regions():
    result = _sanitize({"global": {"contrast": "more", "intensity": "slight"}}, build_skeleton(["sky"]))
    assert result["global"] == {"r": 0, "g": 0, "b": 0, "contrast": .08}
    assert all(value == 0 for value in result["sky"].values())
    warm = _sanitize({"sky": {"warmth": "warmer", "intensity": "normal"}}, build_skeleton(["sky"]))
    assert warm["sky"]["r"] == .13 and warm["sky"]["b"] == -.13


@pytest.mark.parametrize("bad", [{}, [], {"unknown": {}}, {"global": None}, {"global": {"contrast": "huge"}}])
def test_invalid_llm_plans_are_reported_instead_of_silent_zero(bad):
    with pytest.raises(ValueError):
        _sanitize(bad, build_skeleton(["sky"]))


def test_json_extraction_handles_fences_without_greedy_extra_objects():
    assert _extract_json('```json\n{"global":{"contrast":"more"}}\n``` trailing {}') == {"global": {"contrast": "more"}}
    with pytest.raises(ValueError):
        _extract_json("no JSON")


def test_generate_region_json_rejects_malformed_model_output():
    class Inputs(dict):
        def to(self, _device): return self
    tokenizer = Mock()
    tokenizer.return_value = Inputs(input_ids=torch.tensor([[1, 2]]))
    tokenizer.decode.return_value = "invalid generated output"
    model = Mock(device="cpu")
    model.generate.return_value = torch.tensor([[1, 2, 3]])
    with pytest.raises(RuntimeError, match="valid enhancement plan"):
        generate_region_json("more contrast", ["sky"], tokenizer, model)


def test_manual_region_feathering_and_zero_strength_leave_other_regions_unchanged():
    labels = np.zeros((40, 60), dtype=np.uint8); labels[:, 30:] = 1
    baseline, _ = PreferenceAdapter.from_manual_regions(labels, {"contrast": .1}, .5, [])
    result, applied = PreferenceAdapter.from_manual_regions(labels, {"contrast": .1}, .5, [{"region_id": 0, "r": .2, "strength": .5}])
    np.testing.assert_allclose(result[:, 20, 50], baseline[:, 20, 50])
    assert np.isclose(result[0, 20, 10], .1)
    assert 0 < result[0, 20, 30] < .1
    assert applied[0]["region_id"] == 0


@pytest.mark.parametrize("regions", [
    [None], [{}], [{"region_id": 1.5}], [{"region_id": True}],
    [{"region_id": 9}], [{"region_id": 0}, {"region_id": 0}],
    [{"region_id": 0, "r": float("inf")}],
    [{"region_id": 0, "strength": float("nan")}],
])
def test_invalid_manual_region_payloads_are_rejected(regions):
    with pytest.raises((ValueError, TypeError)):
        PreferenceAdapter.from_manual_regions(np.zeros((4, 4), dtype=np.uint8), {}, 1, regions)


@pytest.mark.parametrize("strength", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_global_strength_is_rejected(strength):
    with pytest.raises(ValueError):
        PreferenceAdapter.from_manual_regions(np.zeros((4, 4), dtype=np.uint8), {}, strength, [])


def test_reference_transfers_median_colour_with_different_dimensions():
    source = np.full((8, 10, 3), .3, dtype=np.float32)
    reference = np.broadcast_to(np.array([.5, .3, .1], dtype=np.float32), (12, 6, 3)).copy()
    result, details = ReferenceAdapter.build(source, reference, .5)
    np.testing.assert_allclose(result[:, 0, 0], [.1, 0, -.1, 0], atol=1e-7)
    assert details["mode_used"] == "global-fallback"


def test_tiny_reference_images_have_finite_global_statistics():
    source = np.array([[[0, 0, 0], [1, 1, 1]]], dtype=np.float32)
    reference = np.full((1, 1, 3), .5, dtype=np.float32)
    result, _ = ReferenceAdapter.build(source, reference, 1)
    assert result.shape == (4, 1, 2) and np.isfinite(result).all()
    np.testing.assert_allclose(result[:3], 0)
    np.testing.assert_allclose(result[3], -.3)


def test_reference_shared_regions_override_global_and_unmatched_keep_global():
    source = np.full((20, 20, 3), .3, dtype=np.float32)
    reference = np.full((20, 20, 3), .5, dtype=np.float32)
    source_labels = np.zeros((20, 20), dtype=np.uint8); source_labels[:, 10:] = 1
    ref_labels = np.zeros((20, 20), dtype=np.uint8); ref_labels[:, 10:] = 2
    reference[:, :10] = .4
    result, details = ReferenceAdapter.build(source, reference, 1, source_labels, ref_labels, ["sky", "tree", "wall"])
    assert details["mode_used"] == "region-aware" and [x["name"] for x in details["matched_regions"]] == ["sky"]
    np.testing.assert_allclose(result[:3, :, :10], .1, atol=1e-7)
    global_values = np.array(list(details["global_adjustments"].values()))
    np.testing.assert_allclose(result[:, 10, 15], global_values)


def test_reference_rejects_nonfinite_and_misaligned_input():
    image = np.zeros((10, 10, 3), dtype=np.float32)
    bad = image.copy(); bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ReferenceAdapter.build(image, bad)
    with pytest.raises(ValueError, match="align"):
        ReferenceAdapter.build(image, image, 1, np.zeros((2, 2)), np.zeros((10, 10)), ["sky"])


def test_real_checkpoint_manual_and_reference_work_without_semantic_loading(service, rgb_image, tmp_path, monkeypatch):
    source, pixels = rgb_image
    monkeypatch.setattr(service.adapter, "segment", Mock(side_effect=AssertionError("must remain offline")))
    first = tmp_path / "manual.png"; second = tmp_path / "ref.png"
    info = service.enhance_manual(source, {"r": .08, "g": 0, "b": -.08, "contrast": .05}, .6, first)
    neutral = service.enhance_reference(source, source, 0, False, second)
    assert first.exists() and second.exists()
    assert np.array(Image.open(first)).shape == pixels.shape
    assert not np.array_equal(np.array(Image.open(first)), np.array(Image.open(second)))
    assert info["checkpoint_loaded"] and info["combined_shape"] == [1, 7, 24, 32]
    assert info["checkpoint"] == "preference_controlled_cnn.pt"
    assert neutral["mode_used"] == "global-fallback" and not neutral["has_nan"]
    assert service.adapter.segmentation_ready is False and service.adapter.llm_ready is False


def test_checkpoint_loading_rejects_missing_or_incompatible_weights(tmp_path):
    with pytest.raises(FileNotFoundError): PreferenceControlledCNN(tmp_path / "missing.pt")
    invalid = tmp_path / "invalid.pt"; torch.save({"cnn": {}}, invalid)
    with pytest.raises(RuntimeError, match="incompatible"): PreferenceControlledCNN(invalid)


def test_cnn_checks_conditioning_dimensions(service):
    with pytest.raises(ValueError, match="spatial"):
        service.model(torch.zeros(1, 3, 2, 2), torch.zeros(1, 4, 3, 2))
    with pytest.raises(ValueError, match="4,H,W"):
        service.model(torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2))


def test_segment_cache_reuses_same_bytes_and_rejects_mismatch(service, tmp_path, monkeypatch):
    service._segment_cache.clear()
    first = tmp_path / "a"; first.write_bytes(b"a")
    copy = tmp_path / "copy"; copy.write_bytes(b"a")
    other = tmp_path / "b"; other.write_bytes(b"b")
    segment = Mock(return_value=(np.full((2, 3), 127, dtype=np.uint8), ["bicycle"]))
    monkeypatch.setattr(service.adapter, "segment", segment)
    key, _, _, hit = service.segment_image(first)
    assert not hit and service.segment_image(copy)[3] is True and segment.call_count == 1
    assert service.region_at(key, 1, 1) == {"region_id": 127, "label": "bicycle"}
    with pytest.raises(ValueError, match="match"): service.cached_segmentation(other, key)
    with pytest.raises(ValueError, match="outside"): service.region_at(key, 3, 0)


def test_concurrent_cache_requests_only_segment_once_and_capacity_is_bounded(service, tmp_path, monkeypatch):
    service._segment_cache.clear()
    path = tmp_path / "same"; path.write_bytes(b"same")
    def segment(_path):
        time.sleep(.01)
        return np.zeros((2, 2), dtype=np.uint8), ["wall"]
    spy = Mock(side_effect=segment); monkeypatch.setattr(service.adapter, "segment", spy)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(service.segment_image, [path] * 4))
    assert spy.call_count == 1 and sum(not result[3] for result in results) == 1
    for index in range(8):
        different = tmp_path / str(index); different.write_bytes(str(index).encode()); service.segment_image(different)
    assert len(service._segment_cache) == 8
    with pytest.raises(ValueError, match="expired"): service.region_at(results[0][0], 0, 0)
