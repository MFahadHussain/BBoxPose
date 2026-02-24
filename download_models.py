#!/usr/bin/env python3
"""
Download VRG Prague models from HuggingFace.
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download


def download_vrg_models():
    """Download all required models."""
    
    repo_id = "vrg-prague/BBoxMaskPose"
    local_dir = Path("checkpoints")
    local_dir.mkdir(exist_ok=True)
    
    print(f"Downloading from: {repo_id}")
    print(f"Target: {local_dir.absolute()}")
    print("=" * 60)
    
    # Required files from the actual repo (including small versions for faster testing)
    files = [
        "rtmdet-ins-l-mask.pth",
        "MaskPose/MaskPose-s-1.1.0.pth",
        "MaskPose/MaskPose-l-1.1.0.pth",
        "PMPose/PMPose-s-1.0.0.pth",
        "PMPose/PMPose-l-1.0.0.pth",
        "SAM-pose2seg_hiera_b+.pt",
        "README.md"
    ]
    
    for filename in files:
        try:
            print(f"Downloading {filename}...")
            # Handle subdirectories
            local_name = filename.split("/")[-1]
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            # The hf_hub_download will maintain the folder structure if local_dir is used correctly
            # or it might put it in a cache. Let's make sure it's in checkpoints.
            size = Path(path).stat().st_size / (1024**2)
            print(f"  ✓ {size:.1f} MB")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Done! Models saved to:", local_dir.absolute())


if __name__ == "__main__":
    download_vrg_models()
