"""API regressions use the supplied CNN; only optional semantic-model results are stubbed."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
import threading
import time

import numpy as np
from PIL import Image
import pytest
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from demo_web.backend import app as api


def image_bytes(mode="RGB", size=(32, 24), *, format="PNG", exif=None):
    width, height = size
    if mode == "RGB":
        grid = np.linspace(20, 220, width * height * 3, dtype=np.uint8).reshape(height, width, 3)
        image = Image.fromarray(grid)
    else:
        image = Image.new(mode, size, 64)
    output = BytesIO()
    options = {"exif": exif} if exif else {}
    image.save(output, format=format, **options)
    return output.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "OUTPUTS", tmp_path)
    monkeypatch.setattr(api.service, "_segment_cache", {})
    output_route = next(route for route in api.app.routes if route.name == "outputs")
    monkeypatch.setattr(output_route, "app", StaticFiles(directory=tmp_path))
    # Any accidental semantic inference in the normal suite must fail, not download weights.
    def unavailable(*args, **kwargs):
        raise AssertionError("This test must not download optional semantic models")
    monkeypatch.setattr(api.service.adapter, "segment", unavailable)
    monkeypatch.setattr(api.service.adapter, "from_text", unavailable)
    with TestClient(api.app) as test_client:
        yield test_client


def upload(content=None, filename="image.png"):
    return {"image": (filename, image_bytes() if content is None else content, "image/png")}


def test_health_reports_loaded_real_checkpoint(client):
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["backend_ready"] and health.json()["cnn_ready"]
    assert client.get("/").json()["docs"] == "/docs"
    assert api.service.model.checkpoint_path.is_file()


def test_manual_inference_downloads_real_png_and_changes_with_conditioning(client):
    results = []
    for red in (-0.15, 0.15):
        response = client.post("/api/enhance/manual", files=upload(), data={"red": red, "strength": .8})
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["checkpoint_loaded"]
        assert "/" not in result["checkpoint"] and "output_path" not in result
        assert result["combined_shape"] == [1, 7, 24, 32]
        assert not result["has_nan"] and not result["has_inf"]
        assert 0 <= result["output_min"] <= result["output_max"] <= 1
        download = client.get(result["enhanced_image_url"])
        assert download.status_code == 200
        with Image.open(BytesIO(download.content)) as image:
            assert image.format == "PNG" and image.size == (32, 24)
            results.append(np.asarray(image))
    assert not np.array_equal(*results), "The real checkpoint should respond to changed controls"


def test_reference_global_mode_does_not_load_semantic_models(client):
    response = client.post("/api/enhance/reference", files={
        "source": ("source.png", image_bytes(), "image/png"),
        "reference": ("reference.png", image_bytes(size=(19, 27)), "image/png"),
    }, data={"strength": .6, "region_aware": "false"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["checkpoint_loaded"] and body["mode_used"] == "global-fallback"
    assert body["source_dimensions"] == [32, 24]
    assert body["reference_dimensions"] == [19, 27]
    assert client.get(body["enhanced_image_url"]).status_code == 200


def test_preview_and_inference_apply_the_same_exif_orientation(client):
    exif = Image.Exif()
    exif[274] = 6
    content = image_bytes(size=(16, 9), format="JPEG", exif=exif)
    response = client.post("/api/preview", files=upload(content, "portrait.jpg"))
    assert response.status_code == 200, response.text
    assert response.json()["dimensions"] == [9, 16]
    with Image.open(BytesIO(client.get(response.json()["enhanced_image_url"]).content)) as image:
        assert image.size == (9, 16)
    response = client.post("/api/enhance/manual", files=upload(content, "portrait.jpg"))
    assert response.status_code == 200, response.text
    assert response.json()["original_dimensions"] == [9, 16]


def test_sixteen_bit_preview_retains_grayscale_range(client):
    image = Image.fromarray(np.tile(np.linspace(0, 65535, 32, dtype=np.uint16), (24, 1)))
    data = BytesIO()
    image.save(data, format="TIFF")
    response = client.post("/api/preview", files=upload(data.getvalue(), "gray.tiff"))
    assert response.status_code == 200, response.text
    with Image.open(BytesIO(client.get(response.json()["enhanced_image_url"]).content)) as preview:
        pixels = np.asarray(preview)
        assert pixels.min() < 10 and pixels.max() > 245
        assert 110 < pixels.mean() < 145


@pytest.mark.parametrize("pixels", [np.array([[0., .5]], dtype=np.float32),
                                     np.array([[-1, 70000]], dtype=np.int32)])
def test_ambiguous_or_invalid_tiff_intensities_are_client_errors(client, pixels):
    data = BytesIO()
    Image.fromarray(pixels).save(data, format="TIFF")
    response = client.post("/api/enhance/manual", files=upload(data.getvalue(), "data.tiff"))
    assert response.status_code == 400


@pytest.mark.parametrize("content,filename", [
    (b"", "empty.png"), (b"this is not a PNG", "photo.png"),
    (image_bytes(format="GIF"), "renamed.png"), (image_bytes(), "image.txt"),
    (image_bytes()[:40], "truncated.png"),
])
def test_bad_uploads_are_client_errors(client, content, filename):
    response = client.post("/api/enhance/manual", files=upload(content, filename))
    assert response.status_code == 400, response.text


def test_upload_byte_limit_is_enforced_before_decode(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 32)
    response = client.post("/api/preview", files=upload())
    assert response.status_code == 413


def test_upload_pixel_limit_is_enforced_before_inference(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/api/enhance/manual", files=upload())
    assert response.status_code == 413


def test_multiframe_tiff_is_rejected(client):
    output = BytesIO()
    Image.new("RGB", (8, 8)).save(output, format="TIFF", save_all=True,
                                  append_images=[Image.new("RGB", (8, 8), "red")])
    response = client.post("/api/preview", files=upload(output.getvalue(), "stack.tiff"))
    assert response.status_code == 400


@pytest.mark.parametrize("field,value", [("red", "nan"), ("green", "inf"), ("blue", -.31),
                                          ("contrast", .31), ("strength", -1), ("strength", "nan")])
def test_invalid_global_controls_are_rejected(client, field, value):
    response = client.post("/api/enhance/manual", files=upload(), data={field: value})
    assert response.status_code == 400, response.text


@pytest.mark.parametrize("regions", [
    "null", "{}", "broken json", "[null]", '[{"r":0.1}]',
    '[{"region_id":0,"r":"0.1"}]', '[{"region_id":0,"r":NaN}]',
    '[{"region_id":0,"strength":Infinity}]', '[{"region_id":0,"contrast":0.31}]',
    '[{"region_id":0,"extra":1}]', '[{"region_id":false}]',
    '[{"region_id":1.5}]', '[{"region_id":-1}]',
    '[{"region_id":0},{"region_id":"0"}]',
])
def test_invalid_regional_json_is_rejected_before_model_work(client, regions):
    response = client.post("/api/enhance/manual-regions", files=upload(),
                           data={"segment_id": "unused", "regions_json": regions})
    assert response.status_code == 400, response.text


def test_expired_segmentation_and_outside_click_are_client_errors(client, monkeypatch):
    response = client.post("/api/enhance/manual-regions", files=upload(),
                           data={"segment_id": "expired", "regions_json": "[]"})
    assert response.status_code == 400
    assert "segmentation session" in response.json()["detail"]
    content = image_bytes()
    # Cache lookup checks are exercised against a real array, without loading Mask2Former.
    segment_id = __import__("hashlib").sha256(content).hexdigest()
    monkeypatch.setitem(api.service._segment_cache, segment_id, (np.zeros((24, 32), dtype=np.uint8), ["wall"]))
    response = client.get(f"/api/segment/{segment_id}/region?x=32&y=0")
    assert response.status_code == 400
    response = client.get(f"/api/segment/{segment_id}/region?x=0&y=0")
    assert response.status_code == 200
    assert response.json()["region"]["region_id"] == 0



def test_segment_returns_aligned_overlays_and_reuses_the_source_cache(client, monkeypatch):
    # Only the semantic prediction is substituted; the API, cache and PNG overlays are real.
    labels = np.zeros((24, 32), dtype=np.uint8)
    labels[:, 16:] = 127  # ADE20K's bicycle class is valid; it is not the ignore label.
    labels[0, :] = 255
    calls = []
    def segment_image(path):
        calls.append(path)
        return labels, ["wall", "bicycle"]
    monkeypatch.setattr(api.service.adapter, "segment", segment_image)
    first = client.post("/api/segment", files=upload())
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["dimensions"] == [32, 24] and not body["cache_hit"]
    assert {region["region_id"] for region in body["regions"]} == {0, 127}
    assert "enhanced_image_url" not in body
    for region in body["regions"]:
        response = client.get(region["overlay_url"])
        assert response.status_code == 200
        with Image.open(BytesIO(response.content)) as overlay:
            assert overlay.size == (32, 24) and overlay.mode == "RGBA"
            np.testing.assert_array_equal(np.asarray(overlay)[:, :, 3] > 0,
                                          labels == region["region_id"])
        assert region["coverage"] == pytest.approx((labels == region["region_id"]).mean())
    second = client.post("/api/segment", files=upload())
    assert second.status_code == 200 and second.json()["cache_hit"]
    assert second.json()["segment_id"] == body["segment_id"] and len(calls) == 1

def test_regional_request_accepts_ui_string_ids_and_runs_real_cnn(client, monkeypatch):
    content = image_bytes()
    segment_id = __import__("hashlib").sha256(content).hexdigest()
    labels = np.zeros((24, 32), dtype=np.uint8)
    labels[:, 16:] = 1
    monkeypatch.setitem(api.service._segment_cache, segment_id, (labels, ["wall", "building"]))
    response = client.post("/api/enhance/manual-regions", files=upload(content), data={
        "segment_id": segment_id,
        "regions_json": json.dumps([{"region_id": "0", "r": .15, "strength": .7}]),
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["checkpoint_loaded"] and body["segmentation_cache_hit"]
    assert body["regional_conditioning"][0]["region_id"] == 0
    assert client.get(body["enhanced_image_url"]).status_code == 200
    response = client.post("/api/enhance/manual-regions", files=upload(content), data={
        "segment_id": segment_id, "regions_json": '[{"region_id":149,"r":0.1}]',
    })
    assert response.status_code == 400


def test_text_input_bounds_and_internal_errors_are_distinct(client, monkeypatch):
    for text in ("   ", "x" * 2001):
        assert client.post("/api/enhance/text", files=upload(), data={"preference": text}).status_code == 400
    def invalid_model_output(*args):
        raise ValueError("Model returned malformed JSON containing internal details")
    monkeypatch.setattr(api.service, "enhance_text", invalid_model_output)
    response = client.post("/api/enhance/text", files=upload(), data={"preference": "warm colours"})
    assert response.status_code == 500
    assert response.json()["detail"] == "Enhancement failed. Please try again."


def test_service_work_is_serialized_for_shared_models():
    active = 0
    peak = 0
    counter_lock = threading.Lock()
    def work():
        nonlocal active, peak
        with counter_lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.015)
        with counter_lock:
            active -= 1
        return {}, ""
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda _: api._run(work), range(3)))
    assert peak == 1
