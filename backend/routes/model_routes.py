"""
FaceForge AI - Model information, checkpoint management, and layer inspection endpoints.
"""

from flask import Blueprint, jsonify
from backend.services.gan_service import GANService

model_bp = Blueprint("model_bp", __name__)

@model_bp.route("/api/model/info", methods=["GET"])
def model_info():
    """Retrieve full technical specifications of the GAN model."""
    gan_service = GANService()
    info = gan_service.get_info()
    return jsonify({"success": True, "model": info}), 200

@model_bp.route("/api/model/download-checkpoint", methods=["POST"])
def download_checkpoint():
    """Download genuine CelebA DCGAN checkpoint from Hugging Face."""
    gan_service = GANService()
    success, message = gan_service.download_celeba_checkpoint()
    status_code = 200 if success else 500
    return jsonify({
        "success": success,
        "message": message,
        "checkpoint_loaded": gan_service.is_checkpoint_loaded
    }), status_code
