# VRG Prague BBoxMaskPose

Official implementation using models from  
**https://huggingface.co/vrg-prague/BBoxMaskPose**

## Overview

Complete **BBox-Mask-Pose** iterative framework for high-precision person analytics.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Models
```bash
python download_models.py
```
Or manually from HuggingFace.

### 3. Run Pipeline
```bash
python vrg_bboxmaskpose.py -i your_video.mp4 --display
```

## Features
- **Detection**: RTMDet (Large)
- **Tracking**: IoU-based persistent tracking
- **Pose**: RTMPose (Large)
- **Segmentation**: SAM (ViT-H)
- **Refinement**: Iterative BBox-Mask-Pose loop for pixel-perfect results.
# BBoxPose
# BBoxPose
