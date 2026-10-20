"""
FaceForge AI - Configuration Module
Defines global configurations, directory structures, model defaults, and security boundaries.
Focused purely on DCGAN facial synthesis inference and gallery persistence.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
STORAGE_DIR = BASE_DIR / "storage"
GENERATED_DIR = STORAGE_DIR / "generated"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
COMPARE_DIR = GENERATED_DIR / "compare"
DATABASE_PATH = STORAGE_DIR / "faceforge.db"

# Ensure runtime directories exist
for path in [STORAGE_DIR, GENERATED_DIR, CHECKPOINTS_DIR, COMPARE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Application Metadata
APP_NAME = "FaceForge AI"
APP_VERSION = "1.0.0"
APP_SUBTITLE = "Face Image Generation Using Generative Adversarial Networks"

# Model & GAN Architecture Settings
DEFAULT_MODEL_NAME = "DCGAN-CelebA"
DEFAULT_LATENT_DIM = 100
DEFAULT_NGF = 64  # Generator feature map base
DEFAULT_NDF = 64  # Discriminator feature map base
DEFAULT_IMAGE_CHANNELS = 3
NATIVE_RESOLUTION = 64
SUPPORTED_RESOLUTIONS = [64, 128, 256, 512]
MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 8

# Quality & Upscaling Settings (Environment configurable with request overrides)
DEFAULT_QUALITY = "high"
SUPPORTED_QUALITIES = ["standard", "high"]
TRUNCATION_THRESHOLD_STANDARD = float(os.environ.get("FACEFORGE_TRUNCATION_STANDARD", 1.5))
TRUNCATION_THRESHOLD_HIGH = float(os.environ.get("FACEFORGE_TRUNCATION_THRESHOLD", 1.1))
BEST_OF_N = int(os.environ.get("FACEFORGE_BEST_OF_N", 16))
HIGH_QUALITY_CANDIDATES = BEST_OF_N  # Backwards compatibility alias
GFPGAN_WEIGHT = float(os.environ.get("FACEFORGE_GFPGAN_WEIGHT", 0.6))
CODEFORMER_WEIGHT = float(os.environ.get("FACEFORGE_CODEFORMER_WEIGHT", 0.7))
DEFAULT_UPSCALER = os.environ.get("FACEFORGE_UPSCALER", "gfpgan").lower()
SUPPORTED_UPSCALERS = ["gfpgan", "codeformer", "both"]

# Server Settings
HOST = os.environ.get("FACEFORGE_HOST", "127.0.0.1")
PORT = int(os.environ.get("FACEFORGE_PORT", 5000))
DEBUG = os.environ.get("FACEFORGE_DEBUG", "True").lower() in ("true", "1")
SECRET_KEY = os.environ.get("FACEFORGE_SECRET", "faceforge-dev-secret-key-cse-capstone-2026")

# Checkpoint Files
DEFAULT_CHECKPOINT_FILE = CHECKPOINTS_DIR / "dcgan_celeba_generator.pth"
GFPGAN_CHECKPOINT_FILE = CHECKPOINTS_DIR / "GFPGANv1.4.pth"
REALESRGAN_CHECKPOINT_FILE = CHECKPOINTS_DIR / "RealESRGAN_x2plus.pth"
CODEFORMER_CHECKPOINT_FILE = CHECKPOINTS_DIR / "codeformer.pth"

