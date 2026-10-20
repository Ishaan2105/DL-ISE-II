"""
FaceForge AI - GAN Model Inference & Latent Space Exploration Service
Manages DCGAN model lifecycle, GPU/CPU acceleration, deterministic seeding,
batch generation, and latent space vector interpolation.
"""

import time
import urllib.request
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from backend.config import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT_FILE,
    DEFAULT_LATENT_DIM,
    DEFAULT_NGF,
    DEFAULT_NDF,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_MODEL_NAME,
    NATIVE_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE
)
from backend.models.dcgan import Generator, Discriminator, weights_init, get_model_summary
from backend.services.image_service import ImageService
from backend.database import insert_generation

logger = logging.getLogger(__name__)

# Hugging Face direct link for genuine DCGAN CelebA Generator checkpoint
CELEBA_CHECKPOINT_URL = "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth"

class GANService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GANService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.latent_dim = DEFAULT_LATENT_DIM
        self.model_name = DEFAULT_MODEL_NAME
        
        # Instantiate networks
        self.generator = Generator(latent_dim=self.latent_dim, ngf=DEFAULT_NGF, nc=DEFAULT_IMAGE_CHANNELS).to(self.device)
        self.discriminator = Discriminator(nc=DEFAULT_IMAGE_CHANNELS, ndf=DEFAULT_NDF).to(self.device)
        
        self.is_checkpoint_loaded = False
        self.checkpoint_path = None
        self._load_or_init_generator()
        
        self._initialized = True
        logger.info(f"GANService initialized on device: {self.device_name} ({self.device})")

    def _load_or_init_generator(self):
        """Load pretrained checkpoint if present, or initialize with DCGAN normal distribution."""
        target_path = DEFAULT_CHECKPOINT_FILE
        
        if target_path.exists() and target_path.is_file():
            try:
                state_dict = torch.load(target_path, map_location=self.device, weights_only=True)
                
                # Support state dict variations (e.g. wrapped in 'state_dict' or 'generator' key or raw dict)
                if isinstance(state_dict, dict):
                    if "generator" in state_dict:
                        state_dict = state_dict["generator"]
                    elif "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                        
                # Handle possible prefix mismatch (e.g. 'main.' or module.)
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    k_clean = k.replace("module.", "")
                    # Ensure prefix matches self.generator.main
                    if not k_clean.startswith("main.") and not any(k_clean.startswith(f"{i}.") for i in range(10)):
                        k_clean = f"main.{k_clean}"
                    cleaned_state_dict[k_clean] = v
                    
                self.generator.load_state_dict(cleaned_state_dict, strict=False)
                self.generator.eval()
                self.is_checkpoint_loaded = True
                self.checkpoint_path = str(target_path)
                logger.info(f"Loaded pretrained Generator checkpoint from: {target_path}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {target_path}: {e}. Initializing standard weights.")
                self.generator.apply(weights_init)
                self.generator.eval()
                self.is_checkpoint_loaded = False
        else:
            logger.info("No checkpoint found at startup. Initializing standard DCGAN weights.")
            self.generator.apply(weights_init)
            self.generator.eval()
            self.is_checkpoint_loaded = False
            
        # Discriminator initialization for architectural analysis
        self.discriminator.apply(weights_init)
        self.discriminator.eval()

    def download_celeba_checkpoint(self) -> Tuple[bool, str]:
        """Download genuine CelebA DCGAN checkpoint from Hugging Face."""
        target_path = DEFAULT_CHECKPOINT_FILE
        try:
            logger.info(f"Downloading CelebA checkpoint from {CELEBA_CHECKPOINT_URL}...")
            urllib.request.urlretrieve(CELEBA_CHECKPOINT_URL, target_path)
            self._load_or_init_generator()
            return True, f"Successfully downloaded and loaded checkpoint ({round(target_path.stat().st_size / (1024*1024), 2)} MB)."
        except Exception as e:
            logger.error(f"Failed to download checkpoint: {e}")
            return False, f"Failed to download checkpoint: {str(e)}"

    def get_info(self) -> Dict[str, Any]:
        """Return model metadata, hardware readiness, and layer specs."""
        g_summary = get_model_summary(self.generator, "generator")
        d_summary = get_model_summary(self.discriminator, "discriminator")
        
        return {
            "model_name": self.model_name,
            "architecture": "Deep Convolutional GAN (DCGAN)",
            "device": str(self.device),
            "device_name": self.device_name,
            "cuda_available": torch.cuda.is_available(),
            "latent_dim": self.latent_dim,
            "native_resolution": NATIVE_RESOLUTION,
            "supported_resolutions": SUPPORTED_RESOLUTIONS,
            "is_checkpoint_loaded": self.is_checkpoint_loaded,
            "checkpoint_path": self.checkpoint_path,
            "generator_summary": g_summary,
            "discriminator_summary": d_summary,
            "training_dataset": "CelebA (CelebFaces Attributes Dataset - 202,599 face images)",
            "loss_function": "Binary Cross-Entropy (Minimax V(D, G))",
            "notes": "Native 64x64 synthesis with optional high-fidelity bicubic 128x128 post-resampling."
        }

    def generate(
        self,
        seed: int,
        batch_size: int = 1,
        resolution: int = NATIVE_RESOLUTION
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic face images deterministically using seed.
        Returns list of metadata dicts for each generated image.
        """
        # Validate inputs
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}.")
            
        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}.")
            
        start_time = time.perf_counter()
        
        # Set deterministic seeds
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Sample latent vectors z ~ N(0, I)
        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
        
        # Run inference in eval mode without gradients
        self.generator.eval()
        with torch.no_grad():
            output_tensors = self.generator(z)
            
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        
        results = []
        for b_idx in range(batch_size):
            img_tensor = output_tensors[b_idx]
            pil_img = ImageService.tensor_to_pil(img_tensor, target_resolution=resolution)
            
            # Save to disk
            filename, file_path = ImageService.save_generated_image(pil_img, seed, batch_idx=b_idx)
            gen_id = Path(filename).stem
            
            # Save record in database
            record = insert_generation(
                gen_id=gen_id,
                filename=filename,
                seed=seed,
                batch_index=b_idx,
                model_name=self.model_name,
                resolution=resolution,
                duration_ms=duration_ms,
                device=self.device_name,
                saved_to_gallery=False
            )
            
            results.append({
                "id": gen_id,
                "filename": filename,
                "image_url": f"/api/images/{filename}",
                "seed": seed,
                "batch_index": b_idx,
                "batch_size": batch_size,
                "resolution": resolution,
                "native_resolution": NATIVE_RESOLUTION,
                "is_upscaled": resolution != NATIVE_RESOLUTION,
                "model_name": self.model_name,
                "device": self.device_name,
                "duration_ms": duration_ms,
                "created_at": record["created_at"]
            })
            
        return results

    def interpolate_latent_space(
        self,
        seed1: int,
        seed2: int,
        steps: int = 7,
        resolution: int = NATIVE_RESOLUTION
    ) -> Dict[str, Any]:
        """
        Generate continuous linear interpolation z(t) = (1-t)*z1 + t*z2 between two seeds.
        """
        if steps < 3 or steps > 15:
            raise ValueError("Interpolation steps must be between 3 and 15.")
            
        # Seed 1 vector
        torch.manual_seed(seed1)
        z1 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        
        # Seed 2 vector
        torch.manual_seed(seed2)
        z2 = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
        
        start_time = time.perf_counter()
        
        # Generate interpolated latent vectors
        t_values = torch.linspace(0.0, 1.0, steps, device=self.device)
        interpolated_z = torch.cat([(1.0 - t) * z1 + t * z2 for t in t_values], dim=0)
        
        self.generator.eval()
        with torch.no_grad():
            output_tensors = self.generator(interpolated_z)
            
        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        
        # Process individual frames and composite strip
        frame_images = []
        frame_items = []
        for idx in range(steps):
            t_val = round(float(t_values[idx].item()), 3)
            pil_img = ImageService.tensor_to_pil(output_tensors[idx], target_resolution=resolution)
            frame_images.append(pil_img)
            
            filename, _ = ImageService.save_generated_image(pil_img, seed=seed1, batch_idx=idx)
            frame_items.append({
                "step": idx + 1,
                "t": t_val,
                "filename": filename,
                "image_url": f"/api/images/{filename}"
            })
            
        # Stitch into strip
        strip_img = ImageService.create_interpolation_strip(frame_images)
        strip_filename = f"morph_s{seed1}_s{seed2}_{steps}steps.png"
        strip_path = CHECKPOINTS_DIR.parent / "storage" / "generated" / strip_filename
        strip_img.save(strip_path, format="PNG")
        
        return {
            "seed1": seed1,
            "seed2": seed2,
            "steps": steps,
            "duration_ms": duration_ms,
            "strip_filename": strip_filename,
            "strip_url": f"/api/images/{strip_filename}",
            "frames": frame_items
        }

    def score_image_with_discriminator(self, img_tensor: torch.Tensor) -> float:
        """Score an input image tensor (1, 3, 64, 64) with Discriminator, returning D(x) in [0, 1]."""
        self.discriminator.eval()
        with torch.no_grad():
            score = self.discriminator(img_tensor.to(self.device)).item()
        return round(score, 4)
