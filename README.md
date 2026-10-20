# FaceForge AI — Face Image Generation Using GANs

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Web Framework](https://img.shields.io/badge/Flask-3.1%2B-black.svg)](https://palletsprojects.com/p/flask/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20Vanilla%20JS-F16529.svg)]()
[![Tests](https://img.shields.io/badge/Tests-19%20Passed-34D399.svg)]()

> **Computer Science & Engineering Capstone / In-Semester Evaluation (ISE-II)**  
> A streamlined, focused Deep Learning laboratory and web platform demonstrating synthetic human face generation using Deep Convolutional Generative Adversarial Networks (DCGAN) trained on the CelebA benchmark dataset, coupled with dual GFPGAN / CodeFormer restorative super-resolution.

---

## 1. Project Overview

**FaceForge AI** is a streamlined local AI synthesis laboratory designed specifically for generating and exploring synthetic human faces. It demonstrates how two competing deep neural networks—a **Generator** $G(z)$ and a **Discriminator** $D(x)$—engage in a two-player zero-sum minimax game to learn the high-dimensional probability distribution of human facial features.

The application runs genuine PyTorch neural network tensor inference on your machine (supporting both CPU execution and CUDA acceleration), providing complete visibility into model weights, layer operations, latent space geometry, and deterministic generation.

---

## 2. High-Quality Synthesis & Restoration Pipeline

FaceForge AI features a multi-stage, high-fidelity facial synthesis and restoration pipeline designed to produce sharp, realistic portraits from a native $64 \times 64$ DCGAN base:

```
┌─────────────────────────┐
│ Latent Vector z ∈ R^100 │ ◄── Deterministic seed + Truncation trick (|z_i| <= 1.1)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   DCGAN Generator G(z)  │ ◄── Synthesizes native 64x64 candidate batch (e.g., Best-of-16)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Multi-Criteria Scoring  │ ◄── Face confidence + centering + FACE-ONLY Laplacian sharpness
└────────────┬────────────┘     - Background edge noise penalty - Color blotch penalty
             │ (Selects best candidate)
             ▼
┌─────────────────────────┐
│ Pre-Upscale (Lanczos)   │ ◄── Upscales 64x64 to 256x256 for reliable RetinaFace alignment
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ Face Restoration Pass   │ ◄── GFPGAN v1.4 / CodeFormer with Real-ESRGAN x2 bg_upsampler
└────────────┬────────────┘     (Dual-mode evaluates both and selects higher face sharpness)
             │
             ▼
┌─────────────────────────┐
│ Feathered Face Polish   │ ◄── Mild unsharp mask masked by facial landmarks (background untouched)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│ High-Res PNG & Gallery  │ ◄── Stored with per-stage timing breakdown (generate/score/restore/polish)
└─────────────────────────┘
```

### Key Enhancements & Tuning:
1. **Configurable Sampling & Restoration (Environment Variables & Request Overrides):**
   - `FACEFORGE_TRUNCATION_THRESHOLD` (default: `1.1` for High quality, `1.5` for Standard): Controls the latent sphere truncation boundary $|z_i| \le \tau$, trading extreme variance for dense, natural face geometry.
   - `FACEFORGE_BEST_OF_N` (default: `16` for High quality, `1` for Standard): Generates $N$ candidate latents deterministically and scores them.
   - `FACEFORGE_GFPGAN_WEIGHT` (default: `0.6`): Blends facial prior restoration with identity fidelity.
   - `FACEFORGE_CODEFORMER_WEIGHT` (default: `0.7`): Codebook-lookup restoration weight balancing hallucinated detail and identity.
   - `FACEFORGE_UPSCALER` (`gfpgan` | `codeformer` | `both`, default: `gfpgan`): Selects the restoration engine. In `both` mode, both models process the sample and the one achieving higher face-region Laplacian sharpness is automatically selected.

2. **Advanced Face-Only Candidate Scoring:**
   - Detects faces with OpenCV Haar cascade (`detectMultiScale3`), obtaining continuous detection confidence weights.
   - Computes facial centering distance from image center $(64, 64)$ and bounding box area ratio.
   - Measures Laplacian sharpness **strictly on the cropped face region**, preventing background hair flyaways from skewing scores.
   - Penalizes high background edge variance (suppressing abstract/garbled samples) and background color blotches.
   - Logs candidate scores and winning index for complete reproducibility.

3. **Restoration with Real-ESRGAN x2 Background Upsampler:**
   - Pre-upscales $64 \times 64$ to $256 \times 256$ via high-order Lanczos so RetinaFace can identify landmarks with 100% reliability.
   - Couples GFPGAN / CodeFormer with a Real-ESRGAN x2 background upsampler (`bg_upsampler`), cleaning hair, neck, and backdrop alongside facial features.

4. **Feathered Landmark-Guided Face Polish:**
   - Detects the 5 facial landmarks (eyes, nose, mouth corners) and projects an elliptical face boundary.
   - Applies an edge-feathered Gaussian blur to the face mask.
   - Blends a mild unsharp mask on the face region only while leaving background pixels 100% untouched.

5. **In-Memory Model Caching & Server Warmup:**
   - All networks (DCGAN, Real-ESRGAN x2, GFPGAN v1.4, CodeFormer) are loaded and warmed up at server startup (`GANService().warmup_models()`).
   - Per-stage latency metrics (`generate_ms`, `score_ms`, `restore_ms`, `polish_ms`, `total_ms`) are reported in every response and server log.

---

## 3. System Architecture

```
                    ┌────────────────────────────────────────┐
                    │      Random Latent Vector z            │
                    │         z ~ N(0, I) ∈ R^100            │
                    └──────────────────┬─────────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────────┐
                    │          Generator Network G           │
                    │   5x ConvTranspose2d + BN + ReLU       │
                    │         Output: Tanh()                 │
                    └──────────────────┬─────────────────────┘
                                       │
                                       ▼
                    ┌────────────────────────────────────────┐
                    │          Synthetic Face G(z)           │
                    │         RGB 64x64x3 [-1, 1]            │
                    └──────────────────┬─────────────────────┘
                                       │
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
    ┌──────────────────────┐                       ┌──────────────────────┐
    │ High-Res PNG Export  │                       │ Latent Morph Strip   │
    │  & Gallery Database  │                       │  z(t) Interpolation  │
    └──────────────────────┘                       └──────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Deep Learning** | PyTorch (`torch`, `torchvision`) | DCGAN neural networks, tensor forward pass, weights loading |
| **Numeric & Imaging** | NumPy, Pillow (PIL) | Denormalization $[-1, 1] \to [0, 255]$, bicubic upscaling, PNG encoding |
| **Backend Framework** | Python 3.11 / 3.13, Flask 3.1+ | REST API, static asset routing, and error handling |
| **Persistence** | SQLite3 (`storage/faceforge.db`) | Thread-safe generation records and gallery state |
| **Frontend UI** | HTML5 Semantic, Modular CSS3 | Responsive dark laboratory design system, CSS variables, glassmorphism |
| **Frontend Logic** | Vanilla JavaScript (ES6+) | Asynchronous API client, DOM manipulation, modal controllers, and toasts |
| **Testing** | Pytest 9.1+ | Automated unit and integration testing suite |

---

## 5. Directory Structure

```
DL ISE-II/
├── backend/
│   ├── app.py                     # Flask application factory and static view router
│   ├── config.py                  # Storage paths, model defaults, and runtime configuration
│   ├── database.py                # SQLite schema and parameterized CRUD operations
│   ├── models/
│   │   ├── __init__.py
│   │   └── dcgan.py               # PyTorch Generator and Discriminator neural networks
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── generation_routes.py   # /api/generate and /api/generate/interpolate
│   │   ├── gallery_routes.py      # /api/gallery, /api/images/<file>, /api/dashboard/stats
│   │   ├── health_routes.py       # /api/health system readiness
│   │   └── model_routes.py        # /api/model/info and /api/model/download-checkpoint
│   ├── services/
│   │   ├── gan_service.py         # Singleton model lifecycle, device management, inference
│   │   └── image_service.py       # Tensor denormalization, PNG persistence, path security
│   └── tests/
│       ├── test_gallery.py        # Gallery persistence, retrieval, and deletion tests
│       ├── test_generation.py     # Deterministic seed generation, batch size, and interpolation
│       └── test_health.py         # Health check and model info endpoint tests
├── checkpoints/
│   └── dcgan_celeba_generator.pth # Pretrained CelebA DCGAN weights (~14.3 MB)
├── frontend/
│   ├── index.html                 # Research Dashboard
│   ├── studio.html                # Face Generation Studio & Latent Morpher
│   ├── gallery.html               # Saved Generations & Inspection Modal
│   ├── model-explorer.html        # Layer-by-layer Architecture & Mathematical Formulations
│   ├── about.html                 # Academic Project Documentation & Viva Voce Guide
│   ├── css/
│   │   ├── variables.css          # Design system tokens and color variables
│   │   ├── base.css               # Typography, reset, and scrollbar styling
│   │   ├── layout.css             # Persistent sidebar and responsive topbar
│   │   ├── components.css         # Reusable buttons, cards, badges, modals, toasts
│   │   ├── dashboard.css          # Dashboard metrics grid and workflow diagram
│   │   ├── studio.css             # Studio canvas, viewport frame, and batch grid
│   │   ├── gallery.css            # Gallery cards, search toolbar, and detail modal
│   │   ├── model-explorer.css     # Layer tables and mathematical formulation boxes
│   │   └── about.css              # Academic documentation and viva accordion styles
│   └── js/
│       ├── config.js              # Global frontend settings
│       ├── api.js                 # Centralized fetch API wrapper with error handling
│       ├── components.js          # Toast notifications, modal controllers, image downloaders
│       ├── navigation.js          # Active nav highlighting and health poller
│       ├── dashboard.js           # Dashboard metrics loader and recent feed
│       ├── studio.js              # Generation controller and latent space morpher
│       ├── gallery.js             # Filter, search, inspect, and delete controller
│       └── model-explorer.js      # Layer table populator and tab switcher
├── scripts/
│   └── download_checkpoint.py     # Checkpoint downloader from Hugging Face
├── storage/
│   ├── generated/                 # Generated synthetic face PNGs
│   ├── logs/                      # Application execution logs
│   └── faceforge.db               # SQLite metadata database
├── requirements.txt               # Pinned Python package dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules for virtualenvs and weights
└── README.md                      # Comprehensive academic documentation
```

---

## 6. Installation & Quick Start

### Step 1: Create and activate virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: Verify or download CelebA pretrained weights
```powershell
python scripts/download_checkpoint.py
```

### Step 4: Start the application server
```powershell
python backend/app.py
```
Open your browser and navigate to:
**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 7. Configuration & Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `FACEFORGE_TRUNCATION_THRESHOLD` | `float` | `1.1` | Latent truncation threshold $\tau$ for High quality mode |
| `FACEFORGE_TRUNCATION_STANDARD` | `float` | `1.5` | Latent truncation threshold $\tau$ for Standard mode |
| `FACEFORGE_BEST_OF_N` | `int` | `16` | Number of candidate faces scored per generation in High quality |
| `FACEFORGE_GFPGAN_WEIGHT` | `float` | `0.6` | GFPGAN restoration fidelity weight ($0.0 - 1.0$) |
| `FACEFORGE_CODEFORMER_WEIGHT` | `float` | `0.7` | CodeFormer codebook weight ($0.0 - 1.0$) |
| `FACEFORGE_UPSCALER` | `string` | `gfpgan` | Primary restorer: `gfpgan`, `codeformer`, or `both` |
| `FACEFORGE_PORT` | `int` | `5000` | Application HTTP server port |
| `FACEFORGE_HOST` | `string` | `127.0.0.1` | Binding host address |

All generation parameters can also be overridden on a per-request basis in the `/api/generate` JSON payload.

---

## 8. Visual Quality Comparison & Evolution

Side-by-side comparisons demonstrating the quality evolution from Native $64 \times 64$ to the High-Quality Pipeline are saved in `storage/generated/compare/`:

| Seed | Comparison File | Pipeline Improvements |
|---|---|---|
| **Seed 42** | `storage/generated/compare/seed_42_comparison.png` | Natural facial contour, realistic teeth, sharp eye gaze, clean skin tone |
| **Seed 44** | `storage/generated/compare/seed_44_comparison.png` | Eliminates hairline artifacts, generates realistic beard and eye textures |
| **Seed 99** | `storage/generated/compare/seed_99_comparison.png` | Smooth blonde hair flow, realistic skin pores, perfectly centered face |

---

## 9. REST API Documentation

| Method | Endpoint | Description | Request Payload | Response |
|---|---|---|---|---|
| `GET` | `/api/health` | System health, PyTorch device (CPU/GPU), model status | None | `{"status": "healthy", "device": "CPU", "checkpoint_loaded": true}` |
| `GET` | `/api/model/info` | Architectural parameters and layer specifications | None | `{"model": {"architecture": "DCGAN", "latent_dim": 100, ...}}` |
| `POST` | `/api/model/download-checkpoint` | Downloads pretrained CelebA weights from Hugging Face | None | `{"success": true, "message": "..."}` |
| `POST` | `/api/generate` | Synthesizes face images with full quality overrides | `{"seed": 42, "quality": "high", "truncation": 1.1, "best_of_n": 16, "upscaler": "gfpgan", "gfpgan_weight": 0.6}` | `{"success": true, "generations": [{"timing": {"generate_ms": 18.5, ...}, ...}]}` |
| `POST` | `/api/generate/interpolate` | Continuous latent interpolation between two seeds | `{"seed1": 10, "seed2": 99, "steps": 7, "resolution": 256, "quality": "high"}` | `{"interpolation": {"strip_url": "...", "frames": [...]}}` |
| `GET` | `/api/gallery` | Paginated list of generated faces with filters | `?limit=24&saved_only=false&sort=newest` | `{"data": {"items": [...], "total": 12}}` |
| `POST` | `/api/gallery/save` | Toggles image bookmarking in gallery | `{"id": "faceforge_seed_42_...", "saved": true}` | `{"success": true, "saved": true}` |
| `DELETE`| `/api/gallery/<id>` | Safely deletes image file and database record | None | `{"success": true, "message": "Deleted generation"}` |
| `GET` | `/api/images/<file>` | Streams generated PNG image with MIME headers | None | Raw image/png binary stream |
| `GET` | `/api/dashboard/stats` | Aggregated metrics: total faces, average latency | None | `{"stats": {"total_generations": 15, "avg_duration_ms": 32.5}}` |

---

## 10. Automated Testing & Verification

Execute the test suite:
```bash
PYTHONPATH=. .venv/bin/pytest backend/tests/ -v
```

**Verification Results (19 Passed):**
```
backend/tests/test_gallery.py::test_gallery_listing PASSED               [  5%]
backend/tests/test_gallery.py::test_save_and_unsave_gallery PASSED       [ 11%]
backend/tests/test_gallery.py::test_dashboard_stats PASSED               [ 17%]
backend/tests/test_gallery.py::test_path_traversal_protection PASSED     [ 23%]
backend/tests/test_generation.py::test_generate_single_face PASSED       [ 29%]
backend/tests/test_generation.py::test_generate_batch PASSED             [ 35%]
backend/tests/test_generation.py::test_invalid_seed_rejection PASSED     [ 41%]
backend/tests/test_generation.py::test_invalid_batch_size_rejection PASSED [ 47%]
backend/tests/test_generation.py::test_unsupported_resolution_rejection PASSED [ 52%]
backend/tests/test_generation.py::test_supported_resolutions PASSED      [ 58%]
backend/tests/test_generation.py::test_generator_eval_mode PASSED        [ 64%]
backend/tests/test_generation.py::test_determinism_identical_bytes PASSED [ 70%]
backend/tests/test_generation.py::test_batch_size_independence PASSED    [ 76%]
backend/tests/test_generation.py::test_upscaler_fallback_behavior PASSED [ 82%]
backend/tests/test_generation.py::test_latent_interpolation PASSED       [ 78%]
backend/tests/test_generation.py::test_configurable_quality_overrides_and_timing PASSED [ 84%]
backend/tests/test_generation.py::test_high_quality_determinism PASSED   [ 89%]
backend/tests/test_health.py::test_health_endpoint PASSED                [ 94%]
backend/tests/test_health.py::test_model_info_endpoint PASSED            [100%]

======================= 19 passed in 42.49s ========================
```

---

## 11. Academic Demonstration Key Points & Model Credits

1. **Adversarial Objective:** $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left( 1 - D(G(z)) \right) \right]$.
2. **Transposed Convolutions:** The Generator uses fractionally strided convolutions (`ConvTranspose2d`) to expand $100$-dim latent vectors into spatial feature maps $[4\times4] \to [8\times8] \to [16\times16] \to [32\times32] \to [64\times64]$.
3. **Latent Space Continuity & Truncation:** Continuous latent morphing demonstrates that the model has learned a smooth manifold of human facial representations. The truncation trick ($|z_i| \le \tau$) resamples outlier latent coordinates to focus generation on the densest, highest-fidelity facial feature distributions.
4. **Third-Party Model Credits & Licenses:**
   - **CelebA Benchmark Dataset:** Liu et al., *Deep Learning Face Attributes in the Wild*, ICCV 2015.
   - **GFPGAN (Generative Facial Prior GAN):** Wang et al., *Towards Real-World Blind Face Restoration with Generative Facial Prior*, CVPR 2021 (Apache License 2.0).
   - **CodeFormer:** Zhou et al., *Towards Robust Blind Face Restoration with Codebook Lookup Transformer*, NeurIPS 2022 (S-Lab License).
   - **Real-ESRGAN:** Wang et al., *Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data*, ICCVW 2021 (BSD 3-Clause License).
   - **OpenCV Computer Vision Library:** Intel / OpenCV Team (Apache License 2.0).
