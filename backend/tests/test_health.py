"""
Unit and integration tests for FaceForge AI Health API.
"""

import pytest
from backend.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    """Verify /api/health returns HTTP 200 with proper payload."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_ready" in data
    assert data["model_ready"] is True
    assert "checkpoint_loaded" in data

def test_model_info_endpoint(client):
    """Verify /api/model/info returns architectural parameters."""
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    model_info = data["model"]
    assert model_info["architecture"] == "Deep Convolutional GAN (DCGAN)"
    assert model_info["latent_dim"] == 100
    assert 64 in model_info["supported_resolutions"]
    assert "generator_summary" in model_info
    assert "discriminator_summary" in model_info
