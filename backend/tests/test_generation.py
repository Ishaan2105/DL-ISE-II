"""
Unit and integration tests for FaceForge AI Face Generation API.
"""

import pytest
from backend.app import create_app

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
        "resolution": 64
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
        "resolution": 64
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
    """Verify non-supported resolutions (e.g. 512) are rejected with HTTP 400."""
    response = client.post("/api/generate", json={"seed": 100, "resolution": 512})
    assert response.status_code == 400

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
