"""
FaceForge AI - Image Processing Service
Handles tensor-to-image conversion, denormalization, safe file persistence,
bicubic post-resizing, and path traversal validation.
"""

import os
import uuid
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image
import torch
from backend.config import GENERATED_DIR, NATIVE_RESOLUTION

class ImageService:
    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor, target_resolution: Optional[int] = None) -> Image.Image:
        """
        Convert a PyTorch tensor (C, H, W) normalized in [-1, 1] to a PIL Image (RGB, 0-255).
        Applies mathematical denormalization: (x + 1) / 2 * 255.
        """
        # Detach and move to CPU
        img_tensor = tensor.detach().cpu().clone()
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.squeeze(0)
            
        # Denormalize [-1, 1] -> [0, 1]
        img_tensor = (img_tensor + 1.0) / 2.0
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        
        # Convert (C, H, W) -> (H, W, C) numpy array
        np_img = img_tensor.permute(1, 2, 0).numpy()
        np_img = (np_img * 255.0).astype(np.uint8)
        
        # Create PIL Image
        pil_img = Image.fromarray(np_img, mode="RGB")
        
        # High quality bicubic post-processing if upscale requested
        if target_resolution and target_resolution != pil_img.width:
            pil_img = pil_img.resize((target_resolution, target_resolution), Image.Resampling.BICUBIC)
            
        return pil_img

    @staticmethod
    def save_generated_image(
        pil_img: Image.Image,
        seed: int,
        batch_idx: int = 0
    ) -> Tuple[str, Path]:
        """
        Save PIL image to storage/generated/ with safe descriptive filename.
        Returns (filename, absolute_path).
        """
        unique_suffix = uuid.uuid4().hex[:8]
        filename = f"faceforge_seed_{seed}_b{batch_idx}_{unique_suffix}.png"
        file_path = GENERATED_DIR / filename
        
        # Save as optimized PNG with metadata preserved
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
            strip.paste(img, (idx * width, 0))
            
        return strip

    @staticmethod
    def resolve_safe_image_path(filename: str, directory: Path = GENERATED_DIR) -> Optional[Path]:
        """
        Ensure requested file path stays strictly within permitted directory (prevent path traversal).
        """
        # Disallow directory separators in input filename
        clean_name = os.path.basename(filename)
        resolved_path = (directory / clean_name).resolve()
        
        # Verify within expected directory
        try:
            resolved_path.relative_to(directory.resolve())
        except ValueError:
            return None
            
        if resolved_path.exists() and resolved_path.is_file():
            return resolved_path
        return None

    @staticmethod
    def delete_image_file(filename: str) -> bool:
        """Safely delete an image file from storage."""
        path = ImageService.resolve_safe_image_path(filename, GENERATED_DIR)
        if path and path.exists():
            try:
                os.remove(path)
                return True
            except OSError:
                return False
        return False
