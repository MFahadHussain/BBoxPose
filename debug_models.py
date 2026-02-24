import cv2
import torch
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model, inference_topdown
from mmengine.registry import init_default_scope

def inspect_frame():
    video = "Karate.mp4"
    cap = cv2.VideoCapture(video)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read video")
        return

    print(f"Frame shape: {frame.shape}")

    # Detector
    init_default_scope('mmdet')
    det_config = ".venv/lib/python3.12/site-packages/mmdet/.mim/configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py"
    det_ckpt = "checkpoints/rtmdet-ins-l-mask.pth"
    detector = init_detector(det_config, det_ckpt, device='cuda')
    
    det_result = inference_detector(detector, frame)
    
    # Pose
    init_default_scope('mmpose')
    pose_config = "checkpoints/rtmpose-23k.py"
    pose_ckpt = "checkpoints/MaskPose/MaskPose-s-1.1.0.pth"
    pose_model = init_model(pose_config, pose_ckpt, device='cuda')
    
    # Try one detection
    if len(det_result.pred_instances.scores) > 0:
        bbox = det_result.pred_instances.bboxes[0].cpu().numpy()
        print(f"\nTesting pose on bbox: {bbox}")
        try:
            results = inference_topdown(pose_model, frame, bbox[None, :])
            inst = results[0].pred_instances
            kpts = inst.keypoints[0]
            scores = inst.keypoint_scores[0]
            print(f"Pose KPTs: {len(kpts)}")
            print(f"KPT range: X({kpts[:, 0].min()}-{kpts[:, 0].max()}), Y({kpts[:, 1].min()}-{kpts[:, 1].max()})")
            print(f"Avg confidence: {scores.mean()}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Pose failed: {e}")

if __name__ == "__main__":
    inspect_frame()
