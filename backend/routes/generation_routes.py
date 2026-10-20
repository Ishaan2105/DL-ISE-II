"""
FaceForge AI - Synthetic Face Generation & Latent Space Exploration Endpoints.
"""

from flask import Blueprint, request, jsonify
from backend.services.gan_service import GANService
from backend.config import (
    NATIVE_RESOLUTION,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    SUPPORTED_RESOLUTIONS,
    SUPPORTED_QUALITIES,
    DEFAULT_QUALITY
)

generation_bp = Blueprint("generation_bp", __name__)


@generation_bp.route("/api/generate", methods=["POST"])
def generate_face():
    """
    Generate synthetic faces from random latent seed.
    Payload:
      seed: int (required)
      batch_size: int (optional, default 1, range 1-8)
      resolution: int (optional, options [64, 128, 256, 512])
      quality: str (optional, options ["standard", "high"])
    """
    data = request.get_json(silent=True) or {}

    # Validate seed
    if "seed" not in data:
        return jsonify({"success": False, "error": "Field 'seed' is required."}), 400

    try:
        seed = int(data["seed"])
        if seed < 0 or seed > 2147483647:
            return jsonify({"success": False, "error": "Seed must be a positive integer between 0 and 2,147,483,647."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid seed format. Must be an integer."}), 400

    # Validate batch size
    batch_size = data.get("batch_size", 1)
    try:
        batch_size = int(batch_size)
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            return jsonify({"success": False, "error": f"Batch size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid batch_size format."}), 400

    # Validate quality
    quality = str(data.get("quality", DEFAULT_QUALITY)).strip().lower()
    if quality not in SUPPORTED_QUALITIES:
        return jsonify({"success": False, "error": f"Quality must be one of {SUPPORTED_QUALITIES}."}), 400

    # Validate resolution (default 256 for high quality, 64 for standard)
    if "resolution" in data:
        try:
            resolution = int(data["resolution"])
            if resolution not in SUPPORTED_RESOLUTIONS:
                return jsonify({"success": False, "error": f"Resolution must be one of {SUPPORTED_RESOLUTIONS}."}), 400
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid resolution format."}), 400
    else:
        resolution = 256 if quality == "high" else NATIVE_RESOLUTION

    try:
        gan_service = GANService()
        generations = gan_service.generate(
            seed=seed,
            batch_size=batch_size,
            resolution=resolution,
            quality=quality,
            upscaler=data.get("upscaler")
        )
        return jsonify({
            "success": True,
            "count": len(generations),
            "generations": generations
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Generation failed: {str(e)}"}), 500


@generation_bp.route("/api/generate/interpolate", methods=["POST"])
def interpolate_faces():
    """
    Interpolate smoothly between two seeds in the latent space: z(t) = (1-t)z1 + t*z2.
    Payload:
      seed1: int
      seed2: int
      steps: int (default 7, range 3-15)
      resolution: int (default 64, options [64, 128, 256, 512])
      quality: str (default "standard")
    """
    data = request.get_json(silent=True) or {}

    try:
        seed1 = int(data.get("seed1", 42))
        seed2 = int(data.get("seed2", 99))
        steps = int(data.get("steps", 7))
        quality = str(data.get("quality", DEFAULT_QUALITY)).strip().lower()
        if quality not in SUPPORTED_QUALITIES:
            quality = DEFAULT_QUALITY

        if "resolution" in data:
            resolution = int(data["resolution"])
        else:
            resolution = 256 if quality == "high" else NATIVE_RESOLUTION

        if resolution not in SUPPORTED_RESOLUTIONS:
            return jsonify({"success": False, "error": f"Resolution must be one of {SUPPORTED_RESOLUTIONS}."}), 400
    except (ValueError, TypeError):
        return jsonify({"success": False, "error": "Invalid numerical parameters."}), 400

    if steps < 3 or steps > 15:
        return jsonify({"success": False, "error": "Steps must be between 3 and 15."}), 400

    try:
        gan_service = GANService()
        result = gan_service.interpolate_latent_space(
            seed1=seed1,
            seed2=seed2,
            steps=steps,
            resolution=resolution,
            quality=quality
        )
        return jsonify({"success": True, "interpolation": result}), 200
    except Exception as e:
        return jsonify({"success": False, "error": f"Interpolation failed: {str(e)}"}), 500
