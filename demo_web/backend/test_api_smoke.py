"""CLI API smoke check; optional segmentation requires --with-segmentation."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def _image_bytes(path: Path, max_side: int) -> tuple[bytes, tuple[int, int]]:
    from .image_io import load_rgb_image
    image = load_rgb_image(path)
    image.thumbnail((max_side, max_side))
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue(), image.size


def _segmentation_smoke(client, source, dimensions):
    response = client.post("/api/segment", files={"image": ("source.png", source, "image/png")})
    assert response.status_code == 200, response.text
    segmentation = response.json()
    assert segmentation["regions"] and segmentation["dimensions"] == list(dimensions)
    first_region = segmentation["regions"][0]
    overlay_response = client.get(first_region["overlay_url"])
    assert overlay_response.status_code == 200
    with Image.open(BytesIO(overlay_response.content)) as overlay:
        y, x = np.argwhere(np.asarray(overlay)[:, :, 3] > 0)[0]
    lookup = client.get(f"/api/segment/{segmentation['segment_id']}/region?x={x}&y={y}")
    assert lookup.status_code == 200 and lookup.json()["region"]["region_id"] == first_region["region_id"]
    assert client.get(f"/api/segment/{segmentation['segment_id']}/region?x=-1&y=0").status_code == 400
    regional = client.post("/api/enhance/manual-regions", files={
        "image": ("source.png", source, "image/png"),
    }, data={
        "segment_id": segmentation["segment_id"],
        "global_contrast": .03,
        "regions_json": json.dumps([{
            "region_id": first_region["region_id"], "r": .1, "g": 0, "b": -.05,
            "contrast": .08, "strength": 1,
        }]),
    })
    assert regional.status_code == 200, regional.text
    result = regional.json()
    assert result["diffmap_tensor_shape"] == [1, 4, dimensions[1], dimensions[0]]
    assert result["segmentation_cache_hit"] and not result["has_nan"] and not result["has_inf"]
    assert client.get(result["enhanced_image_url"]).status_code == 200
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=(
        "Default: bundled CNN only, no model downloads. --with-segmentation can download "
        "Mask2Former weights and needs additional RAM. Set PIE_MODEL_LOCAL_FILES_ONLY=1 "
        "to require cached/local weights. Run from the repository root with python -m."
    ))
    parser.add_argument("--with-segmentation", action="store_true", help="Load/download Mask2Former and test regional editing")
    parser.add_argument("--max-side", type=int, default=256, help="Resize smoke inputs to this maximum side (default: 256)")
    args = parser.parse_args(argv)
    if not 1 <= args.max_side <= 1024:
        parser.error("--max-side must be between 1 and 1024")

    # Parsing --help never loads the CNN or optional models.
    from fastapi.testclient import TestClient
    from .app import app

    examples = ROOT / "demo_web" / "frontend" / "public" / "examples"
    source, dimensions = _image_bytes(examples / "demo-person-wall.jpg", args.max_side)
    reference, _ = _image_bytes(examples / "demo.jpg", args.max_side)
    with TestClient(app) as client:
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json()["cnn_ready"]
        assert client.get("/").json()["status"] == "running"
        response = client.post("/api/enhance/manual", files={"image": ("source.png", source, "image/png")},
                               data={"red": .08, "green": 0, "blue": -.08, "contrast": .05, "strength": .6})
        assert response.status_code == 200, response.text
        manual = response.json()
        assert manual["diffmap_tensor_shape"] == [1, 4, dimensions[1], dimensions[0]]
        assert not manual["has_nan"] and not manual["has_inf"]
        assert client.get(manual["enhanced_image_url"]).status_code == 200
        response = client.post("/api/enhance/reference", files={
            "source": ("source.png", source, "image/png"),
            "reference": ("reference.png", reference, "image/png"),
        }, data={"strength": .7, "region_aware": "false"})
        assert response.status_code == 200, response.text
        reference_result = response.json()
        assert reference_result["mode_used"] == "global-fallback"
        assert reference_result["diffmap_tensor_shape"][1] == 4
        assert 0 <= reference_result["output_min"] <= reference_result["output_max"] <= 1
        assert client.get(reference_result["enhanced_image_url"]).status_code == 200

        def run_zero(endpoint, files, data):
            response = client.post(endpoint, files=files, data=data)
            assert response.status_code == 200, response.text
            downloaded = client.get(response.json()["enhanced_image_url"])
            assert downloaded.status_code == 200
            with Image.open(BytesIO(downloaded.content)) as image:
                return np.asarray(image).copy()

        manual_zero = run_zero("/api/enhance/manual", {"image": ("source.png", source, "image/png")}, {"strength": 1})
        reference_zero = run_zero("/api/enhance/reference", {
            "source": ("source.png", source, "image/png"),
            "reference": ("reference.png", reference, "image/png"),
        }, {"strength": 0, "region_aware": "false"})
        assert np.array_equal(manual_zero, reference_zero)
        assert client.post("/api/enhance/manual", files={"image": ("bad.txt", b"bad", "text/plain")}).status_code == 400
        assert client.post("/api/enhance/manual", files={"image": ("source.png", source, "image/png")},
                           data={"red": .31}).status_code == 400
        regional = _segmentation_smoke(client, source, dimensions) if args.with_segmentation else "skipped (opt in with --with-segmentation)"
        print(json.dumps({"health": health.json(), "manual": manual, "reference": reference_result,
                          "regional": regional}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
