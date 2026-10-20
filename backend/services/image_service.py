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

import time
from backend.config import (
    GENERATED_DIR,
    NATIVE_RESOLUTION,
    GFPGAN_CHECKPOINT_FILE,
    REALESRGAN_CHECKPOINT_FILE,
    CODEFORMER_CHECKPOINT_FILE,
    GFPGAN_WEIGHT,
    CODEFORMER_WEIGHT,
    DEFAULT_UPSCALER,
    SUPPORTED_UPSCALERS
)

logger = logging.getLogger(__name__)

# Ensure torchvision compatibility patch for basicsr / gfpgan / codeformer
try:
    import torchvision.transforms.functional as F
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        sys.modules["torchvision.transforms.functional_tensor"] = F
except Exception:
    pass

_REALESRGAN_UPSAMPLER = None
_REALESRGAN_INITIALIZED = False
_GFPGAN_RESTORER = None
_GFPGAN_INITIALIZED = False
_CODEFORMER_RESTORER = None
_CODEFORMER_INITIALIZED = False


def _get_realesrgan_upsampler():
    """Lazy-load and cache the Real-ESRGAN x2 background upsampler on CPU."""
    global _REALESRGAN_UPSAMPLER, _REALESRGAN_INITIALIZED
    if _REALESRGAN_INITIALIZED:
        return _REALESRGAN_UPSAMPLER

    _REALESRGAN_INITIALIZED = True
    if not REALESRGAN_CHECKPOINT_FILE.exists() or REALESRGAN_CHECKPOINT_FILE.stat().st_size < 10_000_000:
        logger.info("RealESRGAN checkpoint not found or incomplete at %s.", REALESRGAN_CHECKPOINT_FILE)
        return None

    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        logger.info("Initializing RealESRGANer from %s on CPU...", REALESRGAN_CHECKPOINT_FILE)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        _REALESRGAN_UPSAMPLER = RealESRGANer(
            scale=2,
            model_path=str(REALESRGAN_CHECKPOINT_FILE),
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=False,
            device="cpu"
        )
        logger.info("RealESRGANer background upsampler initialized successfully.")
        return _REALESRGAN_UPSAMPLER
    except Exception as e:
        logger.warning("Failed to initialize RealESRGANer (%s).", e)
        _REALESRGAN_UPSAMPLER = None
        return None


def _get_gfpgan_restorer():
    """Lazy-load and cache the GFPGAN face restoration model with Real-ESRGAN bg_upsampler."""
    global _GFPGAN_RESTORER, _GFPGAN_INITIALIZED
    if _GFPGAN_INITIALIZED:
        return _GFPGAN_RESTORER

    _GFPGAN_INITIALIZED = True
    if not GFPGAN_CHECKPOINT_FILE.exists() or GFPGAN_CHECKPOINT_FILE.stat().st_size < 10_000_000:
        logger.info("GFPGAN checkpoint not found or incomplete at %s. Using Lanczos+Unsharp fallback.", GFPGAN_CHECKPOINT_FILE)
        return None

    try:
        from gfpgan import GFPGANer
        bg_upsampler = _get_realesrgan_upsampler()
        logger.info("Initializing GFPGANer from %s on CPU (bg_upsampler=%s)...", GFPGAN_CHECKPOINT_FILE, bool(bg_upsampler))
        _GFPGAN_RESTORER = GFPGANer(
            model_path=str(GFPGAN_CHECKPOINT_FILE),
            upscale=2,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=bg_upsampler,
            device="cpu"
        )
        logger.info("GFPGANer initialized successfully.")
        return _GFPGAN_RESTORER
    except Exception as e:
        logger.warning("Failed to initialize GFPGANer (%s). Falling back to Lanczos+Unsharp.", e)
        _GFPGAN_RESTORER = None
        return None


class CodeFormerRestorer:
    """Wrapper for CodeFormer face restoration with Real-ESRGAN x2 background upsampler."""
    def __init__(self, model_path: Path, bg_upsampler=None, device="cpu"):
        self.device = torch.device(device)
        self.bg_upsampler = bg_upsampler
        self.upscale = 2

        import codeformer.basicsr.archs.codeformer_arch  # Register CodeFormer in ARCH_REGISTRY
        from basicsr.utils.registry import ARCH_REGISTRY
        self.model = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=["32", "64", "128", "256"],
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["params_ema"] if "params_ema" in checkpoint else checkpoint.get("params", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        self.face_helper = FaceRestoreHelper(
            2,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device
        )

    @torch.no_grad()
    def enhance(
        self,
        img_bgr: np.ndarray,
        has_aligned: bool = False,
        only_center_face: bool = True,
        paste_back: bool = True,
        weight: float = 0.7
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        self.face_helper.clean_all()

        if has_aligned:
            img = cv2.resize(img_bgr, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img_bgr)
            self.face_helper.get_face_landmarks_5(
                only_center_face=only_center_face,
                resize=640,
                eye_dist_threshold=5
            )
            self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.model(cropped_face_t, w=weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            except Exception as e:
                logger.warning("CodeFormer inference error (%s). Falling back to cropped face.", e)
                restored_face = cropped_face

            restored_face = restored_face.astype("uint8")
            self.face_helper.add_restored_face(restored_face)

        if not has_aligned and paste_back:
            if self.bg_upsampler is not None:
                bg_img = self.bg_upsampler.enhance(img_bgr, outscale=self.upscale)[0]
            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            restored_bgr = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_bgr
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None


def _get_codeformer_restorer():
    """Lazy-load and cache CodeFormer restorer on CPU."""
    global _CODEFORMER_RESTORER, _CODEFORMER_INITIALIZED
    if _CODEFORMER_INITIALIZED:
        return _CODEFORMER_RESTORER

    _CODEFORMER_INITIALIZED = True
    if not CODEFORMER_CHECKPOINT_FILE.exists() or CODEFORMER_CHECKPOINT_FILE.stat().st_size < 10_000_000:
        logger.info("CodeFormer checkpoint not found or incomplete at %s.", CODEFORMER_CHECKPOINT_FILE)
        return None

    try:
        bg_upsampler = _get_realesrgan_upsampler()
        logger.info("Initializing CodeFormerRestorer from %s on CPU...", CODEFORMER_CHECKPOINT_FILE)
        _CODEFORMER_RESTORER = CodeFormerRestorer(
            model_path=CODEFORMER_CHECKPOINT_FILE,
            bg_upsampler=bg_upsampler,
            device="cpu"
        )
        logger.info("CodeFormerRestorer initialized successfully.")
        return _CODEFORMER_RESTORER
    except Exception as e:
        logger.warning("Failed to initialize CodeFormerRestorer (%s).", e)
        _CODEFORMER_RESTORER = None
        return None


class ImageService:
    @staticmethod
    def measure_face_sharpness(img_bgr: np.ndarray, landmarks: Optional[np.ndarray] = None) -> float:
        """Measure Laplacian variance sharpness strictly on the face region."""
        H, W = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if landmarks is not None and len(landmarks) > 0:
            try:
                lm = np.array(landmarks).reshape(-1, 2)
                x1 = max(0, int(np.min(lm[:, 0])))
                y1 = max(0, int(np.min(lm[:, 1])))
                x2 = min(W, int(np.max(lm[:, 0])))
                y2 = min(H, int(np.max(lm[:, 1])))
                if (x2 - x1) >= 15 and (y2 - y1) >= 15:
                    face_crop = gray[y1:y2, x1:x2]
                    return float(cv2.Laplacian(face_crop, cv2.CV_64F).var())
            except Exception:
                pass

        # Fallback: center 40% region
        cy, cx = H // 2, W // 2
        dy, dx = int(H * 0.2), int(W * 0.2)
        center_crop = gray[cy - dy:cy + dy, cx - dx:cx + dx]
        return float(cv2.Laplacian(center_crop, cv2.CV_64F).var())

    @staticmethod
    def polish_face_unsharp(
        restored_bgr: np.ndarray,
        landmarks: Optional[np.ndarray] = None,
        input_size: Tuple[int, int] = (256, 256)
    ) -> np.ndarray:
        """
        Apply a mild unsharp mask to the face region only using detected face landmarks
        with feathered edges, leaving the background completely untouched.
        """
        H, W = restored_bgr.shape[:2]
        mask = np.zeros((H, W), dtype=np.float32)

        found_geometry = False
        if landmarks is not None and len(landmarks) > 0:
            try:
                lm = np.array(landmarks).reshape(-1, 2).astype(np.float32)
                scale_x = W / float(input_size[0])
                scale_y = H / float(input_size[1])
                lm[:, 0] *= scale_x
                lm[:, 1] *= scale_y

                cx = int(np.mean(lm[:, 0]))
                cy = int(np.mean(lm[:, 1]))
                eye_dist = float(np.hypot(lm[1, 0] - lm[0, 0], lm[1, 1] - lm[0, 1])) if len(lm) >= 2 else 50.0
                rx = int(max(20.0, eye_dist * 1.35))
                ry = int(max(25.0, eye_dist * 1.65))

                cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
                ksize = max(21, (int(eye_dist * 0.45) // 2) * 2 + 1)
                mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)
                found_geometry = True
            except Exception as e:
                logger.debug("Landmark mask calculation fallback (%s)", e)

        if not found_geometry:
            cx, cy = W // 2, int(H * 0.48)
            rx, ry = int(W * 0.25), int(H * 0.32)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            ksize = max(21, (int(W * 0.1) // 2) * 2 + 1)
            mask = cv2.GaussianBlur(mask, (ksize, ksize), ksize / 3.0)

        # Mild unsharp mask
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        pil_full = Image.fromarray(restored_rgb)
        sharpened_pil = pil_full.filter(ImageFilter.UnsharpMask(radius=1.2, percent=125, threshold=1))
        sharpened_rgb = np.array(sharpened_pil, dtype=np.float32)

        restored_rgb_f = restored_rgb.astype(np.float32)
        mask_3d = np.expand_dims(mask, axis=-1)

        # Background untouched where mask == 0.0, face sharpened where mask > 0.0
        polished_rgb = restored_rgb_f * (1.0 - mask_3d) + sharpened_rgb * mask_3d
        polished_rgb = np.clip(polished_rgb, 0, 255).astype(np.uint8)

        return cv2.cvtColor(polished_rgb, cv2.COLOR_RGB2BGR)

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
            enhanced_pil, _, _, _ = ImageService.enhance_image(pil_img, target_resolution=target_resolution)
            return enhanced_pil

        return pil_img

    @staticmethod
    def enhance_image(
        pil_img: Image.Image,
        target_resolution: int = 64,
        upscaler: Optional[str] = None,
        gfpgan_weight: Optional[float] = None,
        codeformer_weight: Optional[float] = None
    ) -> Tuple[Image.Image, str, float, float]:
        """
        Enhance and upscale a native 64x64 face image to target_resolution.
        Workflow:
          1. Pre-upscale 64x64 -> 256x256 using Lanczos for reliable face detection.
          2. Run GFPGAN, CodeFormer, or both (selecting higher face-sharpness score).
          3. Polish face region with feathered unsharp mask (background untouched).
          4. Resize to exact target resolution.
        Returns: (enhanced_pil, upscaler_used, restore_ms, polish_ms)
        """
        if target_resolution <= NATIVE_RESOLUTION:
            if pil_img.size != (NATIVE_RESOLUTION, NATIVE_RESOLUTION):
                return pil_img.resize((NATIVE_RESOLUTION, NATIVE_RESOLUTION), Image.Resampling.LANCZOS), "native", 0.0, 0.0
            return pil_img, "native", 0.0, 0.0

        upscaler_mode = (upscaler or DEFAULT_UPSCALER).strip().lower()
        if upscaler_mode not in SUPPORTED_UPSCALERS:
            upscaler_mode = DEFAULT_UPSCALER

        w_gfp = GFPGAN_WEIGHT if gfpgan_weight is None else float(gfpgan_weight)
        w_cf = CODEFORMER_WEIGHT if codeformer_weight is None else float(codeformer_weight)

        # Step 1: Pre-upscale 64x64 to 256x256 via Lanczos before restoration
        pre_256 = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
        bgr_256 = cv2.cvtColor(np.array(pre_256), cv2.COLOR_RGB2BGR)

        restore_start = time.perf_counter()
        restored_bgr = None
        winning_landmarks = None
        upscaler_used = "lanczos_unsharp"

        # Step 2: Face restoration pass(es)
        if upscaler_mode == "both":
            gfp_restorer = _get_gfpgan_restorer()
            cf_restorer = _get_codeformer_restorer()

            res_gfp = None
            lm_gfp = None
            score_gfp = -1.0
            if gfp_restorer is not None:
                try:
                    _, _, res_gfp = gfp_restorer.enhance(
                        bgr_256,
                        has_aligned=False,
                        only_center_face=True,
                        paste_back=True,
                        weight=w_gfp
                    )
                    lm_gfp = gfp_restorer.face_helper.all_landmarks_5[0] if getattr(gfp_restorer.face_helper, "all_landmarks_5", None) else None
                    if res_gfp is not None:
                        score_gfp = ImageService.measure_face_sharpness(res_gfp, lm_gfp)
                except Exception as e:
                    logger.warning("GFPGAN in 'both' failed (%s)", e)

            res_cf = None
            lm_cf = None
            score_cf = -1.0
            if cf_restorer is not None:
                try:
                    _, _, res_cf = cf_restorer.enhance(
                        bgr_256,
                        has_aligned=False,
                        only_center_face=True,
                        paste_back=True,
                        weight=w_cf
                    )
                    lm_cf = cf_restorer.face_helper.all_landmarks_5[0] if getattr(cf_restorer.face_helper, "all_landmarks_5", None) else None
                    if res_cf is not None:
                        score_cf = ImageService.measure_face_sharpness(res_cf, lm_cf)
                except Exception as e:
                    logger.warning("CodeFormer in 'both' failed (%s)", e)

            logger.info("Upscaler 'both' comparison -> GFPGAN sharpness: %.2f | CodeFormer sharpness: %.2f", score_gfp, score_cf)
            if score_cf > score_gfp and res_cf is not None:
                restored_bgr = res_cf
                winning_landmarks = lm_cf
                upscaler_used = "codeformer (both)"
            elif res_gfp is not None:
                restored_bgr = res_gfp
                winning_landmarks = lm_gfp
                upscaler_used = "gfpgan (both)"
            elif res_cf is not None:
                restored_bgr = res_cf
                winning_landmarks = lm_cf
                upscaler_used = "codeformer (both)"

        elif upscaler_mode == "codeformer":
            cf_restorer = _get_codeformer_restorer()
            if cf_restorer is not None:
                try:
                    _, _, restored_bgr = cf_restorer.enhance(
                        bgr_256,
                        has_aligned=False,
                        only_center_face=True,
                        paste_back=True,
                        weight=w_cf
                    )
                    winning_landmarks = cf_restorer.face_helper.all_landmarks_5[0] if getattr(cf_restorer.face_helper, "all_landmarks_5", None) else None
                    upscaler_used = "codeformer"
                except Exception as e:
                    logger.warning("CodeFormer restoration failed (%s). Falling back.", e)

        else:  # default: gfpgan
            gfp_restorer = _get_gfpgan_restorer()
            if gfp_restorer is not None:
                try:
                    _, _, restored_bgr = gfp_restorer.enhance(
                        bgr_256,
                        has_aligned=False,
                        only_center_face=True,
                        paste_back=True,
                        weight=w_gfp
                    )
                    winning_landmarks = gfp_restorer.face_helper.all_landmarks_5[0] if getattr(gfp_restorer.face_helper, "all_landmarks_5", None) else None
                    upscaler_used = "gfpgan_v1.4"
                except Exception as e:
                    logger.warning("GFPGAN restoration failed (%s). Falling back.", e)

        restore_ms = round((time.perf_counter() - restore_start) * 1000.0, 2)

        # Fallback to high-quality Lanczos if restorers were unavailable or produced None
        if restored_bgr is None:
            fallback_pil, fallback_name = ImageService.fallback_lanczos_enhance(pil_img, target_resolution)
            return fallback_pil, fallback_name, restore_ms, 0.0

        # Step 3: Polish face region only with feathered unsharp mask
        polish_start = time.perf_counter()
        polished_bgr = ImageService.polish_face_unsharp(
            restored_bgr=restored_bgr,
            landmarks=winning_landmarks,
            input_size=(256, 256)
        )
        polish_ms = round((time.perf_counter() - polish_start) * 1000.0, 2)

        # Step 4: Resize to requested resolution if needed
        polished_rgb = cv2.cvtColor(polished_bgr, cv2.COLOR_BGR2RGB)
        final_pil = Image.fromarray(polished_rgb)
        if final_pil.size != (target_resolution, target_resolution):
            final_pil = final_pil.resize((target_resolution, target_resolution), Image.Resampling.LANCZOS)

        return final_pil, upscaler_used, restore_ms, polish_ms

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
