"""
FaceForge AI - Gallery, Image Retrieval, and Dashboard Statistics Endpoints.
"""

from flask import Blueprint, request, jsonify, send_file
from backend.services.image_service import ImageService
from backend.database import (
    list_generations,
    get_generation_by_id,
    set_saved_status,
    delete_generation_record,
    get_stats
)
from backend.config import GENERATED_DIR

gallery_bp = Blueprint("gallery_bp", __name__)

@gallery_bp.route("/api/gallery", methods=["GET"])
def get_gallery():
    """Retrieve paginated gallery images with search, sort, and filtering."""
    limit = min(max(int(request.args.get("limit", 24)), 1), 100)
    offset = max(int(request.args.get("offset", 0)), 0)
    saved_only = request.args.get("saved_only", "false").lower() == "true"
    sort_order = request.args.get("sort", "newest")
    search = request.args.get("search", None)
    model = request.args.get("model", None)
    
    data = list_generations(
        limit=limit,
        offset=offset,
        saved_only=saved_only,
        sort_order=sort_order,
        search=search,
        model=model
    )
    
    # Enhance items with accessible image URLs
    for item in data["items"]:
        item["image_url"] = f"/api/images/{item['filename']}"
        
    return jsonify({"success": True, "data": data}), 200

@gallery_bp.route("/api/images/<filename>", methods=["GET"])
def serve_image(filename):
    """Safely stream generated image file with MIME headers."""
    resolved_path = ImageService.resolve_safe_image_path(filename, GENERATED_DIR)
    if not resolved_path:
        return jsonify({"success": False, "error": "Image not found or access denied."}), 404
        
    return send_file(resolved_path, mimetype="image/png")

@gallery_bp.route("/api/gallery/save", methods=["POST"])
def toggle_save_image():
    """Toggle or set saved_to_gallery status."""
    data = request.get_json(silent=True) or {}
    gen_id = data.get("id")
    saved = data.get("saved", True)
    
    if not gen_id:
        return jsonify({"success": False, "error": "Field 'id' is required."}), 400
        
    record = get_generation_by_id(gen_id)
    if not record:
        return jsonify({"success": False, "error": "Generation record not found."}), 404
        
    success = set_saved_status(gen_id, saved)
    return jsonify({"success": success, "saved": saved, "id": gen_id}), 200

@gallery_bp.route("/api/gallery/<gen_id>", methods=["DELETE"])
def delete_image(gen_id):
    """Delete a generated image and its metadata safely."""
    record = get_generation_by_id(gen_id)
    if not record:
        return jsonify({"success": False, "error": "Image not found."}), 404
        
    filename = delete_generation_record(gen_id)
    if filename:
        ImageService.delete_image_file(filename)
        return jsonify({"success": True, "message": f"Deleted generation {gen_id}."}), 200
        
    return jsonify({"success": False, "error": "Deletion failed."}), 500

@gallery_bp.route("/api/dashboard/stats", methods=["GET"])
def dashboard_stats():
    """Retrieve real generation count, avg generation duration, and recent records."""
    stats = get_stats()
    # Format image URLs for recent generations
    for item in stats["recent_generations"]:
        item["image_url"] = f"/api/images/{item['filename']}"
    return jsonify({"success": True, "stats": stats}), 200
