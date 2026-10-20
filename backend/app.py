"""
FaceForge AI - Flask Application Factory & Entry Point
Integrates PyTorch deep learning GAN services with REST API and static frontend views.
Focused strictly on DCGAN face synthesis, latent morphing, and gallery management.
"""

import os
import sys
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

from backend.config import (
    APP_NAME,
    APP_VERSION,
    FRONTEND_DIR,
    DEBUG,
    HOST,
    PORT,
    SECRET_KEY
)
from backend.database import init_db
from backend.routes.health_routes import health_bp
from backend.routes.model_routes import model_bp
from backend.routes.generation_routes import generation_bp
from backend.routes.gallery_routes import gallery_bp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("faceforge")

def create_app() -> Flask:
    """Application factory for FaceForge AI."""
    app = Flask(
        __name__,
        static_folder=str(FRONTEND_DIR),
        static_url_path=""
    )
    
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["JSON_SORT_KEYS"] = False
    
    # Configure CORS for local development
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Initialize database
    init_db()
    
    # Register API blueprints
    app.register_blueprint(health_bp)
    app.register_blueprint(model_bp)
    app.register_blueprint(generation_bp)
    app.register_blueprint(gallery_bp)
    
    # HTML View Routes
    @app.route("/")
    @app.route("/dashboard")
    def serve_dashboard():
        return send_from_directory(str(FRONTEND_DIR), "index.html")

    @app.route("/studio")
    def serve_studio():
        return send_from_directory(str(FRONTEND_DIR), "studio.html")

    @app.route("/gallery")
    def serve_gallery():
        return send_from_directory(str(FRONTEND_DIR), "gallery.html")

    @app.route("/model-explorer")
    def serve_model_explorer():
        return send_from_directory(str(FRONTEND_DIR), "model-explorer.html")
        
    # Static asset fallback
    @app.route("/<path:path>")
    def serve_static(path):
        target = FRONTEND_DIR / path
        if target.exists() and target.is_file():
            return send_from_directory(str(FRONTEND_DIR), path)
        return send_from_directory(str(FRONTEND_DIR), "index.html")

    # Global Error Handlers (Security & Stability)
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"success": False, "error": "Bad request", "message": str(error)}), 400

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"success": False, "error": "Resource not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({"success": False, "error": "Method not allowed"}), 405

    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal server error: {error}", exc_info=True)
        return jsonify({"success": False, "error": "An internal server error occurred"}), 500

    return app

app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting {APP_NAME} v{APP_VERSION} on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
