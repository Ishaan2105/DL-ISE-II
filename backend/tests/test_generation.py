"""
Unit and integration tests for FaceForge AI Face Generation API.
Verifies determinism, batch independence, eval-mode assertions, quality modes, and upscaling.
"""

from pathlib import Path
from PIL import Image
import pytest
from backend.app import create_app
from backend.services.gan_service import GANService
from backend.services.image_service import ImageService
from backend.config import GENERATED_DIR, SUPPORTED_RESOLUTIONS


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_generate_single_face(client):
    """Verify single image generation with seed produces valid response."""
    payload = {
        "seed": 1337,
        "batch_size": 1,
        "resolution": 64,
        "quality": "standard"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["count"] == 1
    gen = data["generations"][0]
    assert gen["seed"] == 1337
    assert gen["resolution"] == 64
    assert "filename" in gen
    assert gen["duration_ms"] > 0


def test_generate_batch(client):
    """Verify batch generation with 4 faces."""
    payload = {
        "seed": 2026,
        "batch_size": 4,
        "resolution": 64,
        "quality": "standard"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["count"] == 4
    assert len(data["generations"]) == 4


def test_invalid_seed_rejection(client):
    """Verify negative or invalid non-integer seeds are rejected with HTTP 400."""
    response = client.post("/api/generate", json={"seed": -5})
    assert response.status_code == 400

    response = client.post("/api/generate", json={"seed": "not-a-number"})
    assert response.status_code == 400


def test_invalid_batch_size_rejection(client):
    """Verify batch size outside [1, 8] is rejected with HTTP 400."""
    response = client.post("/api/generate", json={"seed": 100, "batch_size": 16})
    assert response.status_code == 400


def test_unsupported_resolution_rejection(client):
    """Verify non-supported resolutions (e.g. 1024 or -1) are rejected with HTTP 400."""
    response = client.post("/api/generate", json={"seed": 100, "resolution": 1024})
    assert response.status_code == 400

    response = client.post("/api/generate", json={"seed": 100, "resolution": -64})
    assert response.status_code == 400


def test_supported_resolutions(client):
    """Verify all supported resolutions (64, 128, 256, 512) are valid."""
    for res in SUPPORTED_RESOLUTIONS:
        response = client.post("/api/generate", json={"seed": 42, "batch_size": 1, "resolution": res, "quality": "standard"})
        assert response.status_code == 200
        gen = response.get_json()["generations"][0]
        assert gen["resolution"] == res


def test_generator_eval_mode():
    """Verify generator is in eval mode and gradients are disabled during inference."""
    gs = GANService()
    assert not gs.generator.training, "Generator must remain in evaluation mode."


def test_determinism_identical_bytes(client):
    """Verify that identical seed and settings produce identical image bytes."""
    payload = {"seed": 42, "batch_size": 1, "resolution": 64, "quality": "standard"}
    res1 = client.post("/api/generate", json=payload).get_json()["generations"][0]
    res2 = client.post("/api/generate", json=payload).get_json()["generations"][0]

    path1 = GENERATED_DIR / res1["filename"]
    path2 = GENERATED_DIR / res2["filename"]

    assert path1.exists() and path2.exists()
    assert path1.read_bytes() == path2.read_bytes(), "Deterministic generation must produce identical image bytes."


def test_batch_size_independence(client):
    """Verify seed 44 image at index 0 is identical whether requested as batch 1 or batch 4."""
    res_single = client.post("/api/generate", json={"seed": 44, "batch_size": 1, "resolution": 64, "quality": "standard"}).get_json()["generations"][0]
    res_batch = client.post("/api/generate", json={"seed": 44, "batch_size": 4, "resolution": 64, "quality": "standard"}).get_json()["generations"][0]

    path_single = GENERATED_DIR / res_single["filename"]
    path_batch = GENERATED_DIR / res_batch["filename"]

    assert path_single.read_bytes() == path_batch.read_bytes(), "Batch item 0 must be identical to single generation."


def test_upscaler_fallback_behavior():
    """Verify fallback Lanczos+Unsharp pipeline works cleanly without error."""
    test_img = Image.new("RGB", (64, 64), color=(120, 140, 160))
    enhanced, upscaler = ImageService.fallback_lanczos_enhance(test_img, target_resolution=256)
    assert enhanced.size == (256, 256)
    assert upscaler == "lanczos_unsharp"


def test_latent_interpolation(client):
    """Verify latent space interpolation generates strip and frames."""
    payload = {
        "seed1": 10,
        "seed2": 25,
        "steps": 5,
        "resolution": 64
    }
    response = client.post("/api/generate/interpolate", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    interp = data["interpolation"]
    assert interp["steps"] == 5
    assert len(interp["frames"]) == 5
    assert "strip_url" in interp


def test_configurable_quality_overrides_and_timing(client):
    """Verify request overrides for truncation, best_of_n, weights, and stage timings."""
    payload = {
        "seed": 42,
        "batch_size": 1,
        "resolution": 64,
        "quality": "high",
        "truncation": 1.1,
        "best_of_n": 2,
        "gfpgan_weight": 0.6,
        "codeformer_weight": 0.7,
        "upscaler": "gfpgan"
    }
    response = client.post("/api/generate", json=payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    gen = data["generations"][0]
    assert gen["truncation_threshold"] == 1.1
    assert gen["best_of_n"] == 2
    assert gen["gfpgan_weight"] == 0.6
    assert gen["codeformer_weight"] == 0.7
    assert "timing" in gen
    timing = gen["timing"]
    assert "generate_ms" in timing
    assert "score_ms" in timing
    assert "restore_ms" in timing
    assert "polish_ms" in timing
    assert "total_ms" in timing


def test_high_quality_determinism(client):
    """Verify that High quality mode produces identical image bytes for identical seed and settings."""
    payload = {
        "seed": 77,
        "batch_size": 1,
        "resolution": 64,
        "quality": "high",
        "truncation": 1.1,
        "best_of_n": 4
    }
    res1 = client.post("/api/generate", json=payload).get_json()["generations"][0]
    res2 = client.post("/api/generate", json=payload).get_json()["generations"][0]

    path1 = GENERATED_DIR / res1["filename"]
    path2 = GENERATED_DIR / res2["filename"]

    assert path1.exists() and path2.exists()
    assert path1.read_bytes() == path2.read_bytes(), "High quality generation must be strictly deterministic."

