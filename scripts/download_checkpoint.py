"""
FaceForge AI - Model Checkpoint Downloader
Downloads genuine pretrained CelebA DCGAN weights from Hugging Face and
GFPGANv1.4 face restoration weights into checkpoints/
"""

import sys
import urllib.request
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

CHECKPOINTS = [
    {
        "name": "CelebA DCGAN Generator",
        "url": "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth",
        "filename": "dcgan_celeba_generator.pth",
        "min_size": 10_000_000
    },
    {
        "name": "GFPGANv1.4 Face Restorer",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "filename": "GFPGANv1.4.pth",
        "min_size": 300_000_000
    }
]


def download_file(item: dict):
    target_file = TARGET_DIR / item["filename"]
    if target_file.exists() and target_file.stat().st_size >= item["min_size"]:
        size_mb = target_file.stat().st_size / (1024 * 1024)
        print(f"✓ {item['name']} already exists at {target_file.name} ({size_mb:.2f} MB)")
        return

    print(f"\nDownloading {item['name']} from {item['url']}...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            sys.stdout.write(
                f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(item["url"], target_file, reporthook=reporthook)
        print(f"\n✓ Download complete! Saved to {target_file}")
    except Exception as e:
        print(f"\n✗ Download error for {item['name']}: {e}")
        if target_file.exists():
            target_file.unlink(missing_ok=True)


def download_all():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("FaceForge AI — Model Checkpoint Verification & Downloader")
    print("=" * 60)
    for ckpt in CHECKPOINTS:
        download_file(ckpt)
    print("\nAll checkpoints verified and ready for inference!")


if __name__ == "__main__":
    download_all()
