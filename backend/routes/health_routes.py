"""
FaceForge AI - Health check and system readiness endpoints.
"""

from flask import Blueprint, jsonify
from backend.config import APP_NAME, APP_VERSION, APP_SUBTITLE
from backend.services.gan_service import GANService

health_bp = Blueprint("health_bp", __name__)

@health_bp.route("/api/health", methods=["GET"])
def health_check():
    """Returns application status, PyTorch device, and model readiness."""
    gan_service = GANService()
    
    return jsonify({
        "status": "healthy",
        "application": APP_NAME,
        "version": APP_VERSION,
        "subtitle": APP_SUBTITLE,
        "device": gan_service.device_name,
        "device_type": str(gan_service.device),
        "cuda_available": gan_service.get_info()["cuda_available"],
        "model_ready": True,
        "checkpoint_loaded": gan_service.is_checkpoint_loaded,
        "restoration_ready": gan_service.is_restoration_ready()
    }), 200
