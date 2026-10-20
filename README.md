# FaceForge AI — Face Image Generation Using GANs

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/PyTorch-2.14%2B-ee4c2c.svg)](https://pytorch.org/)
[![Web Framework](https://img.shields.io/badge/Flask-3.1%2B-black.svg)](https://palletsprojects.com/p/flask/)
[![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%7C%20CSS3%20%7C%20Vanilla%20JS-F16529.svg)]()
[![Tests](https://img.shields.io/badge/Tests-12%20Passed-34D399.svg)]()

> **Computer Science & Engineering Capstone / In-Semester Evaluation (ISE-II)**  
> A streamlined, focused Deep Learning laboratory and web platform demonstrating synthetic human face generation using Deep Convolutional Generative Adversarial Networks (DCGAN) trained on the CelebA benchmark dataset.

---

## 1. Project Overview

**FaceForge AI** is a streamlined local AI synthesis laboratory designed specifically for generating and exploring synthetic human faces. It demonstrates how two competing deep neural networks—a **Generator** $G(z)$ and a **Discriminator** $D(x)$—engage in a two-player zero-sum minimax game to learn the high-dimensional probability distribution of human facial features.

The application runs genuine PyTorch neural network tensor inference on your machine (supporting both CPU execution and CUDA acceleration), providing complete visibility into model weights, layer operations, latent space geometry, and deterministic generation.

---

## 2. Streamlined Core Tasks

1. **Deterministic Face Synthesis (Face Studio):**
   - 100-dimensional Gaussian latent prior $z \sim \mathcal{N}(0, I)$.
   - 5-stage fractionally-strided transposed convolutional generator producing $64 \times 64 \times 3$ RGB synthetic faces.
   - Pretrained CelebA checkpoint integration with high-quality facial synthesis.
   - Seed input and randomizer dice button for exploring diverse facial identities.
   - Batch generation ($1, 2, 4, 8$ images).
   - High-resolution PNG download and gallery bookmarking.
2. **Latent-Space Vector Interpolation (Morphing):**
   - Continuous spherical/linear interpolation between two independent latent seeds:
     $$z(t) = (1 - t)z_1 + t z_2, \quad t \in [0, 1]$$
   - Generates multi-frame transition sequences and downloadable composite morph strips.
3. **Local Gallery Management:**
   - SQLite database (`faceforge.db`) storing generation timestamps, latency (ms), seeds, and parameters.
   - Real-time search by seed, generation ID, or filename.
   - Filter by model architecture and sort by newest/oldest.
   - Detailed image inspection modal and safe deletion.
4. **Model Architecture Inspection:**
   - Layer-by-layer architectural inspector for both Generator (3.57M parameters) and Discriminator (2.76M parameters).
   - Exact tensor shape tracking: $[B, 100, 1, 1] \to [B, 512, 4, 4] \to [B, 256, 8, 8] \to [B, 128, 16, 16] \to [B, 64, 32, 32] \to [B, 3, 64, 64]$.
   - Mathematical minimax formulation: $\min_G \max_D V(D, G)$.
5. **Academic Documentation & Viva Voce Guide:**
   - Comprehensive theoretical documentation: Abstract, Problem Statement, Objectives, CelebA dataset specifications, Limitations, and Ethical guardrails.
   - Interactive Viva Voce accordion guide with high-yield questions and detailed answers.

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

## 7. REST API Documentation

| Method | Endpoint | Description | Request Payload | Response |
|---|---|---|---|---|
| `GET` | `/api/health` | System health, PyTorch device (CPU/GPU), model status | None | `{"status": "healthy", "device": "CPU", "checkpoint_loaded": true}` |
| `GET` | `/api/model/info` | Architectural parameters and layer specifications | None | `{"model": {"architecture": "DCGAN", "latent_dim": 100, ...}}` |
| `POST` | `/api/model/download-checkpoint` | Downloads pretrained CelebA weights from Hugging Face | None | `{"success": true, "message": "..."}` |
| `POST` | `/api/generate` | Synthesizes face images from random seed | `{"seed": 42, "batch_size": 1, "resolution": 64}` | `{"success": true, "generations": [...]}` |
| `POST` | `/api/generate/interpolate` | Continuous latent interpolation between two seeds | `{"seed1": 10, "seed2": 99, "steps": 7}` | `{"interpolation": {"strip_url": "...", "frames": [...]}}` |
| `GET` | `/api/gallery` | Paginated list of generated faces with filters | `?limit=24&saved_only=false&sort=newest` | `{"data": {"items": [...], "total": 12}}` |
| `POST` | `/api/gallery/save` | Toggles image bookmarking in gallery | `{"id": "faceforge_seed_42_...", "saved": true}` | `{"success": true, "saved": true}` |
| `DELETE`| `/api/gallery/<id>` | Safely deletes image file and database record | None | `{"success": true, "message": "Deleted generation"}` |
| `GET` | `/api/images/<file>` | Streams generated PNG image with MIME headers | None | Raw image/png binary stream |
| `GET` | `/api/dashboard/stats` | Aggregated metrics: total faces, average latency | None | `{"stats": {"total_generations": 15, "avg_duration_ms": 32.5}}` |

---

## 8. Automated Testing & Verification

Execute the test suite:
```powershell
.venv\Scripts\python -m pytest backend/tests/ -v
```

**Verification Results:**
```
backend/tests/test_gallery.py::test_gallery_listing PASSED               [  8%]
backend/tests/test_gallery.py::test_save_and_unsave_gallery PASSED       [ 16%]
backend/tests/test_gallery.py::test_dashboard_stats PASSED               [ 25%]
backend/tests/test_gallery.py::test_path_traversal_protection PASSED     [ 33%]
backend/tests/test_generation.py::test_generate_single_face PASSED       [ 41%]
backend/tests/test_generation.py::test_generate_batch PASSED             [ 50%]
backend/tests/test_generation.py::test_invalid_seed_rejection PASSED     [ 58%]
backend/tests/test_generation.py::test_invalid_batch_size_rejection PASSED [ 66%]
backend/tests/test_generation.py::test_unsupported_resolution_rejection PASSED [ 75%]
backend/tests/test_generation.py::test_latent_interpolation PASSED       [ 83%]
backend/tests/test_health.py::test_health_endpoint PASSED                [ 91%]
backend/tests/test_health.py::test_model_info_endpoint PASSED            [100%]

============================= 12 passed in 4.87s ==============================
```

---

## 9. Academic Demonstration Key Points

1. **Adversarial Objective:** $\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)} \left[ \log D(x) \right] + \mathbb{E}_{z \sim p_z(z)} \left[ \log \left( 1 - D(G(z)) \right) \right]$.
2. **Transposed Convolutions:** The Generator uses fractionally strided convolutions (`ConvTranspose2d`) to expand $100$-dim latent vectors into spatial feature maps $[4\times4] \to [8\times8] \to [16\times16] \to [32\times32] \to [64\times64]$.
3. **Latent Space Continuity:** Continuous latent morphing demonstrates that the model has learned a smooth manifold of human facial representations rather than memorizing discrete training images.
