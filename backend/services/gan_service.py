"""
FaceForge AI - GAN Model Inference & Latent Space Exploration Service
Manages DCGAN model lifecycle, GPU/CPU acceleration, deterministic seeding,
truncation trick, best-of-N candidate rejection, batch generation, and latent space vector interpolation.
"""

import time
import urllib.request
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn

from backend.config import (
    CHECKPOINTS_DIR,
    DEFAULT_CHECKPOINT_FILE,
    GFPGAN_CHECKPOINT_FILE,
    DEFAULT_LATENT_DIM,
    DEFAULT_NGF,
    DEFAULT_NDF,
    DEFAULT_IMAGE_CHANNELS,
    DEFAULT_MODEL_NAME,
    NATIVE_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    MIN_BATCH_SIZE,
    MAX_BATCH_SIZE,
    DEFAULT_QUALITY,
    SUPPORTED_QUALITIES,
    TRUNCATION_THRESHOLD_STANDARD,
    TRUNCATION_THRESHOLD_HIGH,
    HIGH_QUALITY_CANDIDATES
)
from backend.models.dcgan import Generator, Discriminator, weights_init, get_model_summary
from backend.services.image_service import ImageService
from backend.database import insert_generation

logger = logging.getLogger(__name__)

# Hugging Face direct link for genuine DCGAN CelebA Generator checkpoint
CELEBA_CHECKPOINT_URL = "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth"
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

_FACE_CASCADE = None


def get_face_cascade():
    """Lazy load OpenCV Haar cascade classifier for candidate scoring."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


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

                if isinstance(state_dict, dict):
                    if "generator" in state_dict:
                        state_dict = state_dict["generator"]
                    elif "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]

                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    k_clean = k.replace("module.", "")
                    if not k_clean.startswith("main.") and not any(k_clean.startswith(f"{i}.") for i in range(10)):
                        k_clean = f"main.{k_clean}"
                    cleaned_state_dict[k_clean] = v

                missing, unexpected = self.generator.load_state_dict(cleaned_state_dict, strict=False)
                if missing:
                    logger.warning("Missing checkpoint keys: %s", missing)
                if unexpected:
                    logger.warning("Unexpected checkpoint keys: %s", unexpected)
                if not missing and not unexpected:
                    logger.info("Loaded pretrained Generator checkpoint strictly (100% key match).")

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

    def download_gfpgan_checkpoint(self) -> Tuple[bool, str]:
        """Download GFPGANv1.4 face restoration weights."""
        target_path = GFPGAN_CHECKPOINT_FILE
        try:
            logger.info(f"Downloading GFPGAN checkpoint from {GFPGAN_URL}...")
            urllib.request.urlretrieve(GFPGAN_URL, target_path)
            return True, f"Successfully downloaded GFPGAN checkpoint ({round(target_path.stat().st_size / (1024*1024), 2)} MB)."
        except Exception as e:
            logger.error(f"Failed to download GFPGAN checkpoint: {e}")
            return False, f"Failed to download GFPGAN checkpoint: {str(e)}"

    def is_restoration_ready(self) -> bool:
        """Return True if face restoration pipeline (GFPGAN) is operational."""
        return ImageService.is_restoration_ready()

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
            "supported_qualities": SUPPORTED_QUALITIES,
            "is_checkpoint_loaded": self.is_checkpoint_loaded,
            "gfpgan_loaded": GFPGAN_CHECKPOINT_FILE.exists() and GFPGAN_CHECKPOINT_FILE.stat().st_size > 10_000_000,
            "restoration_ready": self.is_restoration_ready(),
            "checkpoint_path": self.checkpoint_path,
            "generator_summary": g_summary,
            "discriminator_summary": d_summary,
            "training_dataset": "CelebA (CelebFaces Attributes Dataset - 202,599 face images)",
            "loss_function": "Binary Cross-Entropy (Minimax V(D, G))",
            "notes": "Native 64x64 synthesis with best-of-N candidate rejection and high-fidelity face restoration up to 512x512."
        }

    def _sample_truncated_z(
        self,
        seed: int,
        threshold: float = TRUNCATION_THRESHOLD_STANDARD
    ) -> torch.Tensor:
        """
        Sample a 100-dim latent vector from N(0, I) with the truncation trick:
        Any component with |z_i| > threshold is resampled deterministically.
        """
        gen = torch.Generator(device="cpu")
        gen.manual_seed(seed)
        z = torch.randn(1, self.latent_dim, 1, 1, dtype=torch.float32, generator=gen)

        if threshold and threshold > 0:
            mask = torch.abs(z) > threshold
            # Deterministically resample truncated dimensions
            while mask.any():
                new_z = torch.randn(1, self.latent_dim, 1, 1, dtype=torch.float32, generator=gen)
                z[mask] = new_z[mask]
                mask = torch.abs(z) > threshold

        return z.to(self.device)

    def _score_candidate_image(self, img_bgr: np.ndarray) -> float:
        """
        Score a 64x64 candidate face for realism, centering, sharpness, and clean color.
        Uses:
          (a) Haar cascade face detection (bonus for detection, centering, size)
          (b) Sharpness score: variance of Laplacian
          (c) Penalty for extreme color saturation / high-frequency noise
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = get_face_cascade()

        # Face detection (scale to 128 for reliable Haar detection)
        gray_128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        faces = face_cascade.detectMultiScale(gray_128, scaleFactor=1.08, minNeighbors=3, minSize=(32, 32))

        if len(faces) == 0:
            face_score = -50.0  # Penalty if no face detected
        else:
            best_f = -100.0
            for (x, y, w, h) in faces:
                cx = x + w / 2.0
                cy = y + h / 2.0
                dist = np.sqrt((cx - 64.0)**2 + (cy - 64.0)**2)
                centering = max(0.0, 30.0 - dist * 0.5)
                size_score = min(20.0, (w * h) / (128.0 * 128.0) * 40.0)
                score = centering + size_score
                if score > best_f:
                    best_f = score
            face_score = best_f

        # Sharpness score (Laplacian variance)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(30.0, lap_var / 15.0)

        # Penalty for extreme oversaturation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_excess = np.mean(hsv[:, :, 1] > 225) * 80.0

        # Contrast / dynamic range score
        contrast_score = min(15.0, np.std(gray) / 3.0)

        total_score = face_score + sharpness_score + contrast_score - sat_excess
        return float(total_score)

    def generate(
        self,
        seed: int,
        batch_size: int = 1,
        resolution: int = NATIVE_RESOLUTION,
        quality: str = DEFAULT_QUALITY,
        upscaler: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesize face images deterministically.
        Supports Standard mode (1 candidate, truncation threshold 1.5) and
        High quality mode (best-of-8 candidates, truncation threshold 1.5, candidate rejection & scoring).
        Outputs are enhanced using GFPGAN / Lanczos upscaling to requested resolution.
        """
        # Validate inputs
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}.")

        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}.")

        quality = quality.lower()
        if quality not in SUPPORTED_QUALITIES:
            quality = DEFAULT_QUALITY

        truncation_thresh = TRUNCATION_THRESHOLD_HIGH if quality == "high" else TRUNCATION_THRESHOLD_STANDARD
        candidate_count = HIGH_QUALITY_CANDIDATES if quality == "high" else 1

        # Determine requested upscaler
        if upscaler is not None:
            upscaler_requested = upscaler.lower()
        elif resolution <= NATIVE_RESOLUTION:
            upscaler_requested = "native"
        elif quality == "high":
            upscaler_requested = "gfpgan_v1.4"
        else:
            upscaler_requested = "lanczos_unsharp"

        # Guarantee generator is in evaluation mode
        self.generator.eval()
        assert not self.generator.training, "Generator must be in eval mode during inference."

        results = []

        for b_idx in range(batch_size):
            item_start = time.perf_counter()
            gen_ms_total = 0.0
            score_ms_total = 0.0

            if candidate_count == 1:
                # Standard mode: single deterministic candidate
                c_seed = (seed + b_idx) % 2147483647
                gen_t0 = time.perf_counter()
                z = self._sample_truncated_z(c_seed, threshold=truncation_thresh)
                with torch.no_grad():
                    best_tensor = self.generator(z)[0]
                gen_ms_total = (time.perf_counter() - gen_t0) * 1000.0
                score_ms_total = 0.0
                total_evaluated = 1
                total_rejected = 0
            else:
                # High quality mode: Step 4 deterministic candidate rejection
                MAX_BLOCKS = 3
                total_evaluated = 0
                total_rejected = 0
                best_tensor = None
                fallback_tensor = None
                fallback_score = -1e9
                face_cascade = get_face_cascade()

                for block in range(MAX_BLOCKS):
                    gen_t0 = time.perf_counter()
                    candidate_latents = []
                    for c in range(candidate_count):
                        # Batch-size independent deterministic seed formula
                        c_seed = (seed + b_idx * 100000 + block * 10000 + c * 7919) % 2147483647
                        z_c = self._sample_truncated_z(c_seed, threshold=truncation_thresh)
                        candidate_latents.append(z_c)

                    z_batch = torch.cat(candidate_latents, dim=0)
                    with torch.no_grad():
                        out_tensors = self.generator(z_batch)
                    gen_ms_total += (time.perf_counter() - gen_t0) * 1000.0

                    score_t0 = time.perf_counter()
                    block_valid = []

                    for c in range(candidate_count):
                        total_evaluated += 1
                        t = (out_tensors[c] + 1.0) / 2.0
                        t = torch.clamp(t, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                        bgr = cv2.cvtColor(np.round(t * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                        gray_128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)

                        rects, _, weights = face_cascade.detectMultiScale3(
                            gray_128, scaleFactor=1.08, minNeighbors=3, minSize=(32, 32), outputRejectLevels=True
                        )

                        is_valid = False
                        cand_score = -1e9
                        if len(rects) > 0:
                            for idx, (x, y, w, h) in enumerate(rects):
                                cx = x + w / 2.0
                                cy = y + h / 2.0
                                dist = np.sqrt((cx - 64.0)**2 + (cy - 64.0)**2)
                                # Discard if face is too small or far off-center
                                if min(w, h) >= 52 and dist <= 32.0:
                                    is_valid = True
                                    conf = float(weights[idx]) if len(weights) > idx else 10.0
                                    centering = max(0.0, 30.0 - dist * 0.5)
                                    size_score = (w * h) / (128.0 * 128.0) * 40.0
                                    score = conf + centering + size_score
                                    if score > cand_score:
                                        cand_score = score

                        if is_valid:
                            block_valid.append((c, cand_score, out_tensors[c]))
                        else:
                            total_rejected += 1
                            if cand_score > fallback_score:
                                fallback_score = cand_score
                                fallback_tensor = out_tensors[c]

                    score_ms_total += (time.perf_counter() - score_t0) * 1000.0

                    if block_valid:
                        best_candidate = max(block_valid, key=lambda item: item[1])
                        best_tensor = best_candidate[2]
                        break
                    else:
                        logger.warning(
                            "All %d candidates rejected in block %d for seed %d (batch_idx %d). Deterministically moving to candidate block %d.",
                            candidate_count, block, seed, b_idx, block + 1
                        )

                if best_tensor is None:
                    logger.warning("All candidates rejected across %d blocks for seed %d. Using fallback candidate.", MAX_BLOCKS, seed)
                    best_tensor = fallback_tensor if fallback_tensor is not None else out_tensors[0]

            # Convert best candidate to native 64x64 PIL image
            native_pil = ImageService.tensor_to_pil(best_tensor, target_resolution=NATIVE_RESOLUTION)

            item_seed = (seed + b_idx) % 2147483647
            restore_t0 = time.perf_counter()
            enhanced_pil, upscaler_used, fallback_reason = ImageService.enhance_image(
                native_pil,
                target_resolution=resolution,
                requested_upscaler=upscaler_requested,
                quality=quality,
                seed=item_seed
            )
            restore_ms = round((time.perf_counter() - restore_t0) * 1000.0, 2)
            total_duration_ms = round((time.perf_counter() - item_start) * 1000.0, 2)
            gen_ms = round(gen_ms_total, 2)
            score_ms = round(score_ms_total, 2)

            # Persist image files (both enhanced and native 64x64)
            filename, native_filename, _, _ = ImageService.save_generation_pair(
                native_pil=native_pil,
                enhanced_pil=enhanced_pil,
                seed=seed,
                batch_idx=b_idx
            )
            gen_id = Path(filename).stem

            # Store in SQLite database
            record = insert_generation(
                gen_id=gen_id,
                filename=filename,
                seed=seed,
                batch_index=b_idx,
                model_name=self.model_name,
                resolution=resolution,
                duration_ms=total_duration_ms,
                device=self.device_name,
                saved_to_gallery=False,
                native_filename=native_filename,
                quality=quality,
                truncation=truncation_thresh,
                upscaler=upscaler_used,
                candidate_count=candidate_count
            )

            results.append({
                "id": gen_id,
                "filename": filename,
                "native_filename": native_filename,
                "image_url": f"/api/images/{filename}",
                "native_url": f"/api/images/{native_filename}",
                "seed": seed,
                "batch_index": b_idx,
                "batch_size": batch_size,
                "resolution": resolution,
                "native_resolution": NATIVE_RESOLUTION,
                "is_upscaled": resolution != NATIVE_RESOLUTION,
                "quality": quality,
                "truncation": truncation_thresh,
                "upscaler": upscaler_used,
                "upscaler_requested": upscaler_requested,
                "upscaler_used": upscaler_used,
                "fallback_reason": fallback_reason,
                "candidate_count": candidate_count,
                "candidates_evaluated": total_evaluated,
                "candidates_rejected": total_rejected,
                "timing": {
                    "generate_ms": gen_ms,
                    "score_ms": score_ms,
                    "restore_ms": restore_ms,
                    "total_ms": total_duration_ms
                },
                "model_name": self.model_name,
                "device": self.device_name,
                "duration_ms": total_duration_ms,
                "created_at": record["created_at"]
            })

        return results

    def interpolate_latent_space(
        self,
        seed1: int,
        seed2: int,
        steps: int = 7,
        resolution: int = NATIVE_RESOLUTION,
        quality: str = DEFAULT_QUALITY
    ) -> Dict[str, Any]:
        """
        Generate continuous linear interpolation z(t) = (1-t)*z1 + t*z2 between two seeds.
        Frames are enhanced with the upscaler pipeline so morph animations are crisp and high-quality.
        """
        if steps < 3 or steps > 15:
            raise ValueError("Interpolation steps must be between 3 and 15.")

        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}.")

        truncation_thresh = TRUNCATION_THRESHOLD_HIGH if quality == "high" else TRUNCATION_THRESHOLD_STANDARD

        # Ensure generator in eval mode
        self.generator.eval()
        assert not self.generator.training, "Generator must be in eval mode."

        # Sample z1 and z2 with truncation trick
        z1 = self._sample_truncated_z(seed1, threshold=truncation_thresh)
        z2 = self._sample_truncated_z(seed2, threshold=truncation_thresh)

        start_time = time.perf_counter()

        # Interpolate along geodesic
        t_values = torch.linspace(0.0, 1.0, steps, device=self.device)
        interpolated_z = torch.cat([(1.0 - t) * z1 + t * z2 for t in t_values], dim=0)

        with torch.no_grad():
            output_tensors = self.generator(interpolated_z)

        duration_ms = round((time.perf_counter() - start_time) * 1000.0, 2)

        # Enhance each frame and stitch into strip
        frame_images = []
        frame_items = []
        for idx in range(steps):
            t_val = round(float(t_values[idx].item()), 3)
            native_frame = ImageService.tensor_to_pil(output_tensors[idx], target_resolution=NATIVE_RESOLUTION)
            enhanced_frame, upscaler_used, _ = ImageService.enhance_image(
                native_frame, target_resolution=resolution, quality=quality
            )
            frame_images.append(enhanced_frame)

            filename, _ = ImageService.save_generated_image(enhanced_frame, seed=seed1, batch_idx=idx)
            frame_items.append({
                "step": idx + 1,
                "t": t_val,
                "filename": filename,
                "image_url": f"/api/images/{filename}",
                "upscaler": upscaler_used
            })

        # Stitch into high-resolution strip
        strip_img = ImageService.create_interpolation_strip(frame_images)
        strip_filename = f"morph_s{seed1}_s{seed2}_{steps}steps_{resolution}px.png"
        strip_path = CHECKPOINTS_DIR.parent / "storage" / "generated" / strip_filename
        strip_img.save(strip_path, format="PNG")

        return {
            "seed1": seed1,
            "seed2": seed2,
            "steps": steps,
            "resolution": resolution,
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
