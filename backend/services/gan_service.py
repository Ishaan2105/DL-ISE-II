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
    REALESRGAN_CHECKPOINT_FILE,
    CODEFORMER_CHECKPOINT_FILE,
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
    BEST_OF_N,
    HIGH_QUALITY_CANDIDATES,
    GFPGAN_WEIGHT,
    CODEFORMER_WEIGHT,
    DEFAULT_UPSCALER,
    SUPPORTED_UPSCALERS
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
        self._warmed_up = False
        self._load_or_init_generator()

        self._initialized = True
        logger.info(f"GANService initialized on device: {self.device_name} ({self.device})")

    def warmup_models(self):
        """
        Pre-load and warm up all deep learning models in memory so subsequent requests pay no load cost.
        Warms up: DCGAN Generator, Real-ESRGAN x2 background upsampler, GFPGANv1.4, and CodeFormer.
        """
        if self._warmed_up:
            return
        self._warmed_up = True
        logger.info("Initiating server-start model warmup sequence...")
        t_start = time.perf_counter()

        # 1. DCGAN Generator warmup
        try:
            self.generator.eval()
            dummy_z = torch.randn(1, self.latent_dim, 1, 1, device=self.device)
            with torch.no_grad():
                _ = self.generator(dummy_z)
            logger.info("DCGAN Generator warmed up in memory.")
        except Exception as e:
            logger.warning("DCGAN warmup encountered issue (%s)", e)

        # 2. Restoration & upscaling models warmup
        try:
            from backend.services.image_service import _get_realesrgan_upsampler, _get_gfpgan_restorer, _get_codeformer_restorer
            _ = _get_realesrgan_upsampler()
            _ = _get_gfpgan_restorer()
            _ = _get_codeformer_restorer()
            logger.info("Real-ESRGAN x2, GFPGAN v1.4, and CodeFormer models preloaded and cached.")
        except Exception as e:
            logger.warning("Restoration model warmup encountered issue (%s)", e)

        warmup_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.info("Model warmup sequence completed in %.2fms.", warmup_ms)

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
            "supported_upscalers": SUPPORTED_UPSCALERS,
            "default_upscaler": DEFAULT_UPSCALER,
            "is_checkpoint_loaded": self.is_checkpoint_loaded,
            "gfpgan_loaded": GFPGAN_CHECKPOINT_FILE.exists() and GFPGAN_CHECKPOINT_FILE.stat().st_size > 10_000_000,
            "realesrgan_loaded": REALESRGAN_CHECKPOINT_FILE.exists() and REALESRGAN_CHECKPOINT_FILE.stat().st_size > 10_000_000,
            "codeformer_loaded": CODEFORMER_CHECKPOINT_FILE.exists() and CODEFORMER_CHECKPOINT_FILE.stat().st_size > 10_000_000,
            "checkpoint_path": self.checkpoint_path,
            "generator_summary": g_summary,
            "discriminator_summary": d_summary,
            "training_dataset": "CelebA (CelebFaces Attributes Dataset - 202,599 face images)",
            "loss_function": "Binary Cross-Entropy (Minimax V(D, G))",
            "notes": "Native 64x64 synthesis with best-of-N candidate rejection, dual GFPGAN/CodeFormer restoration, and feathered face polish."
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
            while mask.any():
                new_z = torch.randn(1, self.latent_dim, 1, 1, dtype=torch.float32, generator=gen)
                z[mask] = new_z[mask]
                mask = torch.abs(z) > threshold

        return z.to(self.device)

    def _score_candidate_image(self, img_bgr: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Score a 64x64 candidate face for realism, centering, face-region sharpness,
        with penalties for high background edge noise and color blotches.
        Returns: (total_score, face_sharpness, bg_noise_penalty, blotch_penalty)
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        face_cascade = get_face_cascade()

        # Face detection (scale to 128 for reliable Haar detection)
        gray_128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        rects, reject_levels, level_weights = face_cascade.detectMultiScale3(
            gray_128,
            scaleFactor=1.08,
            minNeighbors=3,
            minSize=(28, 28),
            outputRejectLevels=True
        )

        if len(rects) == 0:
            # Heavy penalty if no face detected
            face_score = -60.0
            center_crop = gray[16:48, 16:48]
            face_sharpness = float(cv2.Laplacian(center_crop, cv2.CV_64F).var())
            sharpness_score = min(15.0, face_sharpness / 15.0)
            bg_noise_pen = 15.0
            blotch_pen = 10.0
            total_score = face_score + sharpness_score - bg_noise_pen - blotch_pen
            return float(total_score), face_sharpness, bg_noise_pen, blotch_pen

        # Find best detected face by area and centering
        best_idx = 0
        best_rank = -999.0
        for idx, (x, y, w, h) in enumerate(rects):
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = np.sqrt((cx - 64.0)**2 + (cy - 64.0)**2)
            rank = (w * h) - dist * 10.0
            if rank > best_rank:
                best_rank = rank
                best_idx = idx

        x, y, w, h = rects[best_idx]
        conf = float(level_weights[best_idx]) if len(level_weights) > best_idx else 2.0
        conf_score = min(20.0, max(0.0, conf * 3.5))

        cx = x + w / 2.0
        cy = y + h / 2.0
        dist = np.sqrt((cx - 64.0)**2 + (cy - 64.0)**2)
        centering_score = max(0.0, 25.0 - dist * 0.4)
        size_ratio = (w * h) / (128.0 * 128.0)
        size_score = min(20.0, size_ratio * 50.0)
        face_score = 25.0 + conf_score + centering_score + size_score

        # Map face rectangle back to native 64x64 coordinates
        fx = max(0, int(x // 2))
        fy = max(0, int(y // 2))
        fw = min(64 - fx, int(w // 2))
        fh = min(64 - fy, int(h // 2))

        # 1. Laplacian sharpness ON THE FACE REGION ONLY
        face_gray = gray[fy:fy + fh, fx:fx + fw]
        face_sharpness = float(cv2.Laplacian(face_gray, cv2.CV_64F).var()) if face_gray.size > 0 else 0.0
        sharpness_score = min(25.0, face_sharpness / 12.0)

        # 2. Background edge noise penalty (high-frequency artifacts outside face)
        bg_mask = np.ones((64, 64), dtype=bool)
        bg_mask[fy:fy + fh, fx:fx + fw] = False
        lap_full = cv2.Laplacian(gray, cv2.CV_64F)
        bg_edge_var = float(np.var(lap_full[bg_mask])) if np.sum(bg_mask) > 0 else 0.0
        bg_noise_pen = min(20.0, bg_edge_var / 15.0)

        # 3. Penalty for high background color blotches / extreme saturation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        bg_sat = hsv[:, :, 1][bg_mask] if np.sum(bg_mask) > 0 else hsv[:, :, 1]
        sat_excess = float(np.mean(bg_sat > 215) * 50.0)
        sat_std_pen = float(min(15.0, np.std(bg_sat) / 4.0))
        blotch_pen = sat_excess + sat_std_pen

        total_score = face_score + sharpness_score - bg_noise_pen - blotch_pen
        return float(total_score), face_sharpness, bg_noise_pen, blotch_pen

    def generate(
        self,
        seed: int,
        batch_size: int = 1,
        resolution: int = NATIVE_RESOLUTION,
        quality: str = DEFAULT_QUALITY,
        truncation: Optional[float] = None,
        best_of_n: Optional[int] = None,
        upscaler: Optional[str] = None,
        gfpgan_weight: Optional[float] = None,
        codeformer_weight: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Synthesize face images deterministically.
        Supports Standard mode (1 candidate, truncation 1.5) and
        High quality mode (best-of-16 candidates, truncation 1.1, face-only candidate scoring).
        Outputs are enhanced using GFPGAN / CodeFormer with Real-ESRGAN x2 background upsampler.
        """
        # Validate inputs
        if batch_size < MIN_BATCH_SIZE or batch_size > MAX_BATCH_SIZE:
            raise ValueError(f"Batch size must be between {MIN_BATCH_SIZE} and {MAX_BATCH_SIZE}.")

        if resolution not in SUPPORTED_RESOLUTIONS:
            raise ValueError(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}.")

        quality = quality.lower()
        if quality not in SUPPORTED_QUALITIES:
            quality = DEFAULT_QUALITY

        truncation_thresh = float(truncation) if truncation is not None else (
            TRUNCATION_THRESHOLD_HIGH if quality == "high" else TRUNCATION_THRESHOLD_STANDARD
        )
        candidate_count = int(best_of_n) if best_of_n is not None else (
            BEST_OF_N if quality == "high" else 1
        )
        w_gfp = GFPGAN_WEIGHT if gfpgan_weight is None else float(gfpgan_weight)
        w_cf = CODEFORMER_WEIGHT if codeformer_weight is None else float(codeformer_weight)
        upscaler_mode = (upscaler or DEFAULT_UPSCALER).strip().lower()
        if upscaler_mode not in SUPPORTED_UPSCALERS:
            upscaler_mode = DEFAULT_UPSCALER

        # Guarantee generator is in evaluation mode
        self.generator.eval()
        assert not self.generator.training, "Generator must be in eval mode during inference."

        overall_start = time.perf_counter()
        results = []

        for b_idx in range(batch_size):
            item_start = time.perf_counter()

            # Stage 1: Generate candidates
            gen_start = time.perf_counter()
            candidate_latents = []
            for c in range(candidate_count):
                if c == 0:
                    c_seed = (seed + b_idx) % 2147483647
                else:
                    c_seed = (seed + b_idx * 10000 + c * 7919) % 2147483647

                z_c = self._sample_truncated_z(c_seed, threshold=truncation_thresh)
                candidate_latents.append(z_c)

            z_batch = torch.cat(candidate_latents, dim=0)

            with torch.no_grad():
                out_tensors = self.generator(z_batch)

            generate_ms = round((time.perf_counter() - gen_start) * 1000.0, 2)

            # Stage 2: Score candidates (Face-region sharpness + detection + noise penalty)
            score_start = time.perf_counter()
            best_tensor = out_tensors[0]
            best_c_idx = 0
            best_score = 0.0

            if candidate_count > 1:
                candidate_scores = []
                for c in range(candidate_count):
                    t = (out_tensors[c] + 1.0) / 2.0
                    t = torch.clamp(t, 0.0, 1.0).permute(1, 2, 0).cpu().numpy()
                    bgr = cv2.cvtColor(np.round(t * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    score, face_sharp, bg_noise_pen, blotch_pen = self._score_candidate_image(bgr)
                    logger.info(
                        "Seed %d (batch %d) Candidate %2d/%2d: score=%6.2f [face_sharp=%5.1f, bg_noise_pen=%4.1f, blotch_pen=%4.1f]",
                        seed, b_idx, c + 1, candidate_count, score, face_sharp, bg_noise_pen, blotch_pen
                    )
                    candidate_scores.append((c, score))

                best_c_idx, best_score = max(candidate_scores, key=lambda x: x[1])
                best_tensor = out_tensors[best_c_idx]
                logger.info("Seed %d (batch %d) Winner Candidate: #%d with score %.2f", seed, b_idx, best_c_idx + 1, best_score)

            score_ms = round((time.perf_counter() - score_start) * 1000.0, 2)

            # Convert best candidate to native 64x64 PIL image
            native_pil = ImageService.tensor_to_pil(best_tensor, target_resolution=NATIVE_RESOLUTION)

            # Stage 3 & 4: Restoration & Face-only Polish
            enhanced_pil, upscaler_used, restore_ms, polish_ms = ImageService.enhance_image(
                pil_img=native_pil,
                target_resolution=resolution,
                upscaler=upscaler_mode,
                gfpgan_weight=w_gfp,
                codeformer_weight=w_cf
            )

            # Persist image files (both enhanced and native 64x64)
            filename, native_filename, _, _ = ImageService.save_generation_pair(
                native_pil=native_pil,
                enhanced_pil=enhanced_pil,
                seed=seed,
                batch_idx=b_idx
            )
            gen_id = Path(filename).stem
            total_item_ms = round((time.perf_counter() - item_start) * 1000.0, 2)

            timing_data = {
                "generate_ms": generate_ms,
                "score_ms": score_ms,
                "restore_ms": restore_ms,
                "polish_ms": polish_ms,
                "total_ms": total_item_ms
            }

            logger.info(
                "Generation timing [seed %d b%d]: generate=%.1fms, score=%.1fms, restore=%.1fms, polish=%.1fms, total=%.1fms",
                seed, b_idx, generate_ms, score_ms, restore_ms, polish_ms, total_item_ms
            )

            # Store in SQLite database
            record = insert_generation(
                gen_id=gen_id,
                filename=filename,
                seed=seed,
                batch_index=b_idx,
                model_name=self.model_name,
                resolution=resolution,
                duration_ms=total_item_ms,
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
                "truncation_threshold": truncation_thresh,
                "best_of_n": candidate_count,
                "candidate_count": candidate_count,
                "upscaler": upscaler_used,
                "gfpgan_weight": w_gfp,
                "codeformer_weight": w_cf,
                "timing": timing_data,
                "model_name": self.model_name,
                "device": self.device_name,
                "duration_ms": total_item_ms,
                "created_at": record["created_at"]
            })

        return results

    def interpolate_latent_space(
        self,
        seed1: int,
        seed2: int,
        steps: int = 7,
        resolution: int = NATIVE_RESOLUTION,
        quality: str = DEFAULT_QUALITY,
        upscaler: Optional[str] = None,
        gfpgan_weight: Optional[float] = None,
        codeformer_weight: Optional[float] = None
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
        w_gfp = GFPGAN_WEIGHT if gfpgan_weight is None else float(gfpgan_weight)
        w_cf = CODEFORMER_WEIGHT if codeformer_weight is None else float(codeformer_weight)
        upscaler_mode = upscaler or DEFAULT_UPSCALER

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
            enhanced_frame, upscaler_used, _, _ = ImageService.enhance_image(
                pil_img=native_frame,
                target_resolution=resolution,
                upscaler=upscaler_mode,
                gfpgan_weight=w_gfp,
                codeformer_weight=w_cf
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
