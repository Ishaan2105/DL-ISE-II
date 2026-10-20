"""
FaceForge AI - Checkpoint Downloader
Downloads genuine pretrained CelebA DCGAN weights from Hugging Face into checkpoints/
"""

import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = "https://huggingface.co/hussamalafandi/DCGAN_CelebA/resolve/main/generator.pth"
TARGET_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
TARGET_FILE = TARGET_DIR / "dcgan_celeba_generator.pth"

def download():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET_FILE.exists() and TARGET_FILE.stat().st_size > 10_000_000:
        print(f"Checkpoint already exists at {TARGET_FILE} ({TARGET_FILE.stat().st_size / (1024*1024):.2f} MB)")
        return
        
    print(f"Downloading genuine CelebA DCGAN Generator checkpoint from {CHECKPOINT_URL}...")
    try:
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100.0, downloaded * 100.0 / total_size)
                sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(CHECKPOINT_URL, TARGET_FILE, reporthook=reporthook)
        print(f"\nDownload complete! Saved to {TARGET_FILE}")
    except Exception as e:
        print(f"\nDownload error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download()
