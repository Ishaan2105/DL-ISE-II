"""
Unit and integration tests for FaceForge AI Gallery & Image Persistence API.
"""

import pytest
from backend.app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_gallery_listing(client):
    """Verify /api/gallery returns paginated generations."""
    response = client.get("/api/gallery")
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert "items" in data["data"]
    assert "total" in data["data"]

def test_save_and_unsave_gallery(client):
    """Verify toggling saved_to_gallery status."""
    # First generate a test face
    gen_res = client.post("/api/generate", json={"seed": 777, "batch_size": 1})
    gen_id = gen_res.get_json()["generations"][0]["id"]
    
    # Save to gallery
    save_res = client.post("/api/gallery/save", json={"id": gen_id, "saved": True})
    assert save_res.status_code == 200
    assert save_res.get_json()["saved"] is True
    
    # Verify in saved_only filter
    gal_res = client.get("/api/gallery?saved_only=true")
    ids = [item["id"] for item in gal_res.get_json()["data"]["items"]]
    assert gen_id in ids
    
    # Unsave
    unsave_res = client.post("/api/gallery/save", json={"id": gen_id, "saved": False})
    assert unsave_res.status_code == 200
    assert unsave_res.get_json()["saved"] is False

def test_dashboard_stats(client):
    """Verify dashboard stats returns numerical metrics."""
    response = client.get("/api/dashboard/stats")
    assert response.status_code == 200
    stats = response.get_json()["stats"]
    assert stats["total_generations"] >= 1
    assert "avg_duration_ms" in stats
    assert isinstance(stats["recent_generations"], list)

def test_path_traversal_protection(client):
    """Verify path traversal attacks on /api/images are rejected with 404."""
    response = client.get("/api/images/../../etc/passwd")
    assert response.status_code == 404
    
    response = client.get("/api/images/..%2F..%2Fwindows%2Fsystem32")
    assert response.status_code == 404
