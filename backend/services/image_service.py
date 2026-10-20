"""
FaceForge AI - Image Processing & Enhancement Service
Handles tensor-to-image conversion, denormalization, face restoration upscaling (GFPGAN),
robust Lanczos+Unsharp fallback, safe file persistence, and path traversal validation.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageFilter
import cv2
import torch

from backend.config import GENERATED_DIR, NATIVE_RESOLUTION, GFPGAN_CHECKPOINT_FILE

logger = logging.getLogger(__name__)

# Ensure compatibility patch for torchvision and basicsr
import backend.services.compat

_GFPGAN_RESTORER = None
_GFPGAN_INITIALIZED = False


def _get_gfpgan_restorer():
    """Lazy-load and cache the GFPGAN face restoration model on CPU."""
    global _GFPGAN_RESTORER, _GFPGAN_INITIALIZED
    if _GFPGAN_INITIALIZED:
        return _GFPGAN_RESTORER

    _GFPGAN_INITIALIZED = True
    if not GFPGAN_CHECKPOINT_FILE.exists() or GFPGAN_CHECKPOINT_FILE.stat().st_size < 10_000_000:
        logger.warning(
            "GFPGAN checkpoint not found or incomplete at %s (%s bytes). Restoration not available.",
            GFPGAN_CHECKPOINT_FILE,
            GFPGAN_CHECKPOINT_FILE.stat().st_size if GFPGAN_CHECKPOINT_FILE.exists() else 0
        )
        return None

    try:
        from gfpgan import GFPGANer
        logger.info("Initializing GFPGANer from %s on CPU...", GFPGAN_CHECKPOINT_FILE)
        _GFPGAN_RESTORER = GFPGANer(
            model_path=str(GFPGAN_CHECKPOINT_FILE),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device="cpu"
        )
        logger.info("GFPGANer initialized successfully.")
        return _GFPGAN_RESTORER
    except Exception as e:
        logger.error("Failed to initialize GFPGANer from %s: %s", GFPGAN_CHECKPOINT_FILE, e, exc_info=True)
        _GFPGAN_RESTORER = None
        return None


def is_restoration_ready() -> bool:
    """Check if GFPGAN weights exist and model restorer initializes cleanly."""
    try:
        restorer = _get_gfpgan_restorer()
        return restorer is not None
    except Exception:
        return False


class ImageService:
    @staticmethod
    def is_restoration_ready() -> bool:
        return is_restoration_ready()

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, target_resolution: Optional[int] = None) -> Image.Image:
        """
        Convert a PyTorch tensor (C, H, W) normalized in [-1, 1] to a PIL Image (RGB, 0-255).
        Applies mathematical denormalization: (x + 1) / 2 * 255.
        """
        img_tensor = tensor.detach().cpu().clone()
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)

        # Denormalize [-1, 1] -> [0, 1]
        img_tensor = (img_tensor + 1.0) / 2.0
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Convert (C, H, W) -> (H, W, C) numpy array
        np_img = img_tensor.permute(1, 2, 0).numpy()
        np_img = np.round(np_img * 255.0).astype(np.uint8)

        # Create native PIL Image (64x64)
        pil_img = Image.fromarray(np_img, mode="RGB")

        if target_resolution and target_resolution != pil_img.width:
            enhanced_pil, _, _ = ImageService.enhance_image(pil_img, target_resolution=target_resolution)
            return enhanced_pil

        return pil_img

    @staticmethod
    def enhance_image(
        pil_img: Image.Image,
        target_resolution: int = 64,
        requested_upscaler: Optional[str] = None,
        quality: str = "high",
        seed: Optional[int] = None
    ) -> Tuple[Image.Image, str, Optional[str]]:
        """
        Enhance and upscale a native 64x64 face image to target_resolution.
        Uses GFPGAN face restoration when available; falls back to Lanczos + Unsharp Mask.
        Pre-upscales 64->256 with Lanczos, then GFPGAN (only_center_face=True, has_aligned=False, paste_back=True),
        then resizes to requested resolution.
        Returns: (enhanced_pil, upscaler_used, fallback_reason)
        """
        if target_resolution <= NATIVE_RESOLUTION:
            if pil_img.size != (NATIVE_RESOLUTION, NATIVE_RESOLUTION):
                return pil_img.resize((NATIVE_RESOLUTION, NATIVE_RESOLUTION), Image.Resampling.LANCZOS), "native", None
            return pil_img, "native", None

        # Determine requested upscaler
        if requested_upscaler is None:
            requested = "gfpgan_v1.4" if quality == "high" else "lanczos_unsharp"
        else:
            requested = requested_upscaler.lower()

        # 1. If GFPGAN requested
        if "gfpgan" in requested:
            # Check weights
            if not GFPGAN_CHECKPOINT_FILE.exists() or GFPGAN_CHECKPOINT_FILE.stat().st_size < 10_000_000:
                reason = f"GFPGAN checkpoint weights missing or incomplete at {GFPGAN_CHECKPOINT_FILE}"
                if quality == "high":
                    logger.error("High quality restoration failed: %s. Using Lanczos fallback.", reason)
                enhanced, used = ImageService.fallback_lanczos_enhance(pil_img, target_resolution)
                return enhanced, used, reason

            restorer = _get_gfpgan_restorer()
            if restorer is None:
                reason = "GFPGAN model failed to initialize."
                if quality == "high":
                    logger.error("High quality restoration failed: %s. Using Lanczos fallback.", reason)
                enhanced, used = ImageService.fallback_lanczos_enhance(pil_img, target_resolution)
                return enhanced, used, reason

            try:
                # Ensure deterministic noise sampling inside GFPGAN / StyleGAN generator
                if seed is not None:
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                # Pre-upscale 64->256 with Lanczos for optimal face landmark detection
                pre_upscaled = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
                rgb_np = np.array(pre_upscaled)
                bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

                _, _, restored_bgr = restorer.enhance(
                    bgr_np,
                    has_aligned=False,
                    only_center_face=True,
                    paste_back=True
                )
                if restored_bgr is not None:
                    restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
                    restored_pil = Image.fromarray(restored_rgb)
                    # Output at requested resolution
                    if restored_pil.size != (target_resolution, target_resolution):
                        restored_pil = restored_pil.resize(
                            (target_resolution, target_resolution),
                            Image.Resampling.LANCZOS
                        )
                    return restored_pil, "gfpgan_v1.4", None
                else:
                    raise RuntimeError("GFPGANer.enhance returned None for restored_bgr")
            except Exception as e:
                reason = f"GFPGAN enhancement error: {str(e)}"
                if quality == "high":
                    logger.error("High quality restoration failed: %s. Using Lanczos fallback.", reason, exc_info=True)
                enhanced, used = ImageService.fallback_lanczos_enhance(pil_img, target_resolution)
                return enhanced, used, reason

        # 2. Direct Lanczos requested
        enhanced, used = ImageService.fallback_lanczos_enhance(pil_img, target_resolution)
        return enhanced, used, None

    @staticmethod
    def fallback_lanczos_enhance(pil_img: Image.Image, target_resolution: int) -> Tuple[Image.Image, str]:
        """
        Fallback upscaling: High-fidelity Lanczos resampling followed by
        mild unsharp mask (radius 1.2, percent 120, threshold 2) and bilateral filtering.
        """
        # Step A: Lanczos upscale
        upscaled = pil_img.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)

        # Step B: Mild unsharp mask
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1.2, percent=120, threshold=2))

        # Step C: Edge-preserving bilateral filter denoising
        np_rgb = np.array(sharpened)
        np_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)
        denoised_bgr = cv2.bilateralFilter(np_bgr, d=5, sigmaColor=25, sigmaSpace=25)
        final_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(final_rgb), "lanczos_unsharp"

    @staticmethod
    def save_generation_pair(
        native_pil: Image.Image,
        enhanced_pil: Image.Image,
        seed: int,
        batch_idx: int = 0
    ) -> Tuple[str, str, Path, Path]:
        """
        Save both enhanced image and native 64x64 image to storage/generated/.
        Returns (enhanced_filename, native_filename, enhanced_path, native_path).
        """
        unique_suffix = uuid.uuid4().hex[:8]
        enhanced_filename = f"faceforge_seed_{seed}_b{batch_idx}_{unique_suffix}.png"
        enhanced_path = GENERATED_DIR / enhanced_filename
        enhanced_pil.save(enhanced_path, format="PNG", optimize=True)

        if enhanced_pil.size == native_pil.size:
            # Same 64x64 image
            return enhanced_filename, enhanced_filename, enhanced_path, enhanced_path

        native_filename = f"faceforge_seed_{seed}_b{batch_idx}_{unique_suffix}_native64.png"
        native_path = GENERATED_DIR / native_filename
        native_pil.save(native_path, format="PNG", optimize=True)

        return enhanced_filename, native_filename, enhanced_path, native_path

    @staticmethod
    def save_generated_image(
        pil_img: Image.Image,
        seed: int,
        batch_idx: int = 0
    ) -> Tuple[str, Path]:
        """
        Save single PIL image to storage/generated/.
        Maintains backward compatibility with legacy calls and tests.
        """
        unique_suffix = uuid.uuid4().hex[:8]
        filename = f"faceforge_seed_{seed}_b{batch_idx}_{unique_suffix}.png"
        file_path = GENERATED_DIR / filename
        pil_img.save(file_path, format="PNG", optimize=True)
        return filename, file_path

    @staticmethod
    def create_interpolation_strip(images: List[Image.Image]) -> Image.Image:
        """
        Stitch a list of interpolation frames into a single horizontal strip.
        """
        if not images:
            raise ValueError("No images provided for strip creation.")

        width, height = images[0].size
        strip = Image.new("RGB", (width * len(images), height))

        for idx, img in enumerate(images):
            # Ensure all frames match the first frame's dimensions
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            strip.paste(img, (idx * width, 0))

        return strip

    @staticmethod
    def resolve_safe_image_path(filename: str, directory: Path = GENERATED_DIR) -> Optional[Path]:
        """
        Ensure requested file path stays strictly within permitted directory (prevent path traversal).
        """
        clean_name = os.path.basename(filename)
        resolved_path = (directory / clean_name).resolve()

        try:
            resolved_path.relative_to(directory.resolve())
        except ValueError:
            return None

        if resolved_path.exists() and resolved_path.is_file():
            return resolved_path
        return None

    @staticmethod
    def delete_image_file(filename: str) -> bool:
        """Safely delete an image file and any associated native64 pair from storage."""
        path = ImageService.resolve_safe_image_path(filename, GENERATED_DIR)
        success = False
        if path and path.exists():
            try:
                os.remove(path)
                success = True
            except OSError:
                success = False

        # Check for matching _native64 pair
        if not filename.endswith("_native64.png"):
            native_name = filename.replace(".png", "_native64.png")
            native_path = ImageService.resolve_safe_image_path(native_name, GENERATED_DIR)
            if native_path and native_path.exists():
                try:
                    os.remove(native_path)
                except OSError:
                    pass

        return success
