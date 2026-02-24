#!/usr/bin/env python3
"""
VRG Prague BBoxMaskPose Implementation
======================================

Complete pipeline using models from:
https://huggingface.co/vrg-prague/BBoxMaskPose

Pipeline: Detect (RTMDet) → Track (ByteTrack) → Pose (RTMPose) → Mask (SAM) → Refine
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VRGConfig:
    """VRG BBoxMaskPose configuration."""
    # Model paths from HuggingFace
    det_config: str = None  # Auto-found if None
    det_checkpoint: str = "checkpoints/rtmdet-ins-l-mask.pth"
    pose_config: str = None  # Auto-found if None
    pose_checkpoint: str = "checkpoints/MaskPose/MaskPose-s-1.1.0.pth"
    sam_checkpoint: str = "checkpoints/SAM-pose2seg_hiera_b+.pt"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Pipeline parameters
    refine_iterations: int = 3
    mask_dilation: int = 10
    track_buffer: int = 30
    match_threshold: float = 0.3
    
    # Output
    output_dir: str = "outputs"


# =============================================================================
# BBoxMaskPose Pipeline
# =============================================================================

class VRGBBoxMaskPose:
    """
    VRG Prague BBoxMaskPose complete pipeline.
    
    Iterative refinement:
        BBox → Pose → Mask → Refined BBox (repeat 3x)
    """
    
    def __init__(self, config: VRGConfig):
        self.config = config
        self._ensure_config_paths()
        self.frame_id = 0
        
        # Initialize models
        self._init_models()
        
        # Tracking state
        self.tracks: Dict[int, dict] = {}
        self.next_id = 1
        
        # Output setup
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.last_frame = None
        logger.info("VRG BBoxMaskPose initialized")
    
    def _ensure_config_paths(self):
        """Find standard MMLab configs if not provided."""
        import mmdet
        import mmpose
        import os
        
        mmdet_path = os.path.dirname(mmdet.__file__)
        mmpose_path = os.path.dirname(mmpose.__file__)
        
        if self.config.det_config is None:
            # Use RTMDet-Ins-L-Mask config
            path = os.path.join(mmdet_path, ".mim/configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py")
            if os.path.exists(path):
                self.config.det_config = path
                logger.info(f"Using RTMDet-Ins config: {path}")
            else:
                logger.warning("Could not find standard RTMDet-Ins config")
        
        if self.config.pose_config is None:
            # Use custom 23k config for research weights
            path = os.path.join(os.getcwd(), "checkpoints/rtmpose-23k.py")
            if os.path.exists(path):
                self.config.pose_config = path
                logger.info(f"Using custom 23K RTMPose config: {path}")
            else:
                # Fallback to standard if custom not found (unlikely)
                size = 'l'
                if 'pose-s' in self.config.pose_checkpoint.lower():
                    size = 's'
                path = os.path.join(mmpose_path, f".mim/configs/body_2d_keypoint/rtmpose/coco/rtmpose-{size}_8xb256-420e_coco-256x192.py")
                self.config.pose_config = path
                logger.warning(f"Custom 23K config not found, falling back to: {path}")

    def _init_models(self):
        """Initialize all models from VRG checkpoint."""
        
        # 1. RTMDet - Detection
        logger.info("Loading RTMDet...")
        from mmdet.apis import init_detector
        self.detector = init_detector(
            self.config.det_config,
            self.config.det_checkpoint,
            device=self.config.device
        )
        
        # 2. RTMPose - Pose Estimation
        logger.info("Loading RTMPose...")
        from mmpose.apis import init_model
        self.pose_model = init_model(
            self.config.pose_config,
            self.config.pose_checkpoint,
            device=self.config.device
        )
        self.pose_model.eval()
        
        # Disable flip testing to avoid index mismatch errors
        if hasattr(self.pose_model, 'cfg'):
            self.pose_model.cfg.test_cfg['flip_test'] = False
        if hasattr(self.pose_model, 'test_cfg'):
            self.pose_model.test_cfg['flip_test'] = False
        
        # 3. SAM - Segmentation
        logger.info("Loading SAM...")
        try:
            from segment_anything import sam_model_registry, SamPredictor
            sam = sam_model_registry["vit_h"](checkpoint=self.config.sam_checkpoint)
            sam.to(device=self.config.device)
            self.sam_predictor = SamPredictor(sam)
        except Exception as e:
            logger.warning(f"SAM failed: {e}")
            self.sam_predictor = None
        
        logger.info("All models loaded")
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Detect persons using RTMDet-Ins.
        
        Returns:
            dets: [N, 5] array of [x1, y1, x2, y2, score]
            masks: [N, H, W] boolean masks or None
        """
        from mmdet.apis import inference_detector
        from mmengine.registry import init_default_scope
        
        init_default_scope('mmdet')
        result = inference_detector(self.detector, frame)
        
        # Extract instances
        if not hasattr(result, 'pred_instances'):
            return np.empty((0, 5)), None
            
        bboxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()
        labels = result.pred_instances.labels.cpu().numpy()
        
        # Filter person class (COCO 0) and confidence
        keep = (labels == 0) & (scores > 0.3)
        
        if not np.any(keep):
            return np.empty((0, 5)), None
            
        dets = np.hstack([bboxes[keep], scores[keep].reshape(-1, 1)])
        
        masks = None
        if hasattr(result.pred_instances, 'masks'):
            masks = result.pred_instances.masks[keep].cpu().numpy()
            
        return dets, masks
    
    def track(self, detections: np.ndarray, masks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
        """
        Simple IoU-based tracking with mask support.
        
        Returns:
            tracks: [M, 6] array of [x1, y1, x2, y2, track_id, score]
            track_masks: List of masks matching tracks
        """
        if len(detections) == 0:
            # Advance lost tracks
            lost = []
            for tid, track in self.tracks.items():
                track['lost'] += 1
                if track['lost'] > self.config.track_buffer:
                    lost.append(tid)
            for tid in lost:
                del self.tracks[tid]
            return self._get_active_tracks(), []
        
        if not self.tracks:
            # All new
            for i, det in enumerate(detections):
                m = masks[i] if masks is not None else None
                self._create_track(det[:4], det[4], m)
            return self._get_active_tracks(), [masks[i] if masks is not None else None for i in range(len(detections))]
        
        # Match by IoU
        active_ids = list(self.tracks.keys())
        active_boxes = np.array([self.tracks[tid]['bbox'] for tid in active_ids])
        
        iou_matrix = self._compute_iou(active_boxes, detections[:, :4])
        
        matched_det = set()
        matched_track = set()
        
        # Greedy matching
        while iou_matrix.max() > self.config.match_threshold:
            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = active_ids[t_idx]
            
            # Update track
            self.tracks[tid]['bbox'] = detections[d_idx, :4]
            self.tracks[tid]['score'] = detections[d_idx, 4]
            self.tracks[tid]['lost'] = 0
            if masks is not None:
                self.tracks[tid]['mask'] = masks[d_idx]
            
            # Velocity
            prev_center = self.tracks[tid].get('center', np.zeros(2))
            new_center = np.array([
                (detections[d_idx, 0] + detections[d_idx, 2]) / 2,
                (detections[d_idx, 1] + detections[d_idx, 3]) / 2
            ])
            self.tracks[tid]['velocity'] = new_center - prev_center
            self.tracks[tid]['center'] = new_center
            self.tracks[tid]['history'].append(detections[d_idx, :4].copy())
            
            matched_track.add(t_idx)
            matched_det.add(d_idx)
            iou_matrix[t_idx, :] = 0
            iou_matrix[:, d_idx] = 0
        
        # Create new tracks
        for i in range(len(detections)):
            if i not in matched_det:
                self._create_track(detections[i, :4], detections[i, 4], masks[i] if masks is not None else None)
        
        # Mark lost
        for i in range(len(active_ids)):
            if i not in matched_track:
                tid = active_ids[i]
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.config.track_buffer:
                    del self.tracks[tid]
        
        active_tracks = self._get_active_tracks()
        active_masks = [self.tracks[int(t[4])].get('mask') for t in active_tracks]
        
        return active_tracks, active_masks
    
    def _create_track(self, bbox: np.ndarray, score: float, mask: Optional[np.ndarray] = None):
        """Create new track."""
        self.tracks[self.next_id] = {
            'bbox': bbox,
            'score': score,
            'lost': 0,
            'velocity': np.zeros(2),
            'center': np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]),
            'history': deque(maxlen=30),
            'keypoints': None,
            'mask': mask
        }
        self.tracks[self.next_id]['history'].append(bbox.copy())
        self.next_id += 1
    
    def _get_active_tracks(self) -> np.ndarray:
        """Get active tracks as array."""
        active = []
        for tid, t in self.tracks.items():
            if t['lost'] == 0:
                active.append([*t['bbox'], tid, t['score']])
        return np.array(active) if active else np.empty((0, 6))
    
    def _compute_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix."""
        iou = np.zeros((len(boxes1), len(boxes2)))
        for i, b1 in enumerate(boxes1):
            for j, b2 in enumerate(boxes2):
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                inter = max(0, x2-x1) * max(0, y2-y1)
                area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                union = area1 + area2 - inter
                iou[i, j] = inter / (union + 1e-6)
        return iou
    
    def estimate_pose(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """
        Estimate pose using RTMPose.
        
        Returns:
            keypoints: [17, 3] array [x, y, confidence]
        """
        from mmpose.apis import inference_topdown
        from mmengine.registry import init_default_scope
        
        init_default_scope('mmpose')
        bbox_input = np.array([[bbox[0], bbox[1], bbox[2], bbox[3]]])
        
        with torch.no_grad():
            results = inference_topdown(self.pose_model, frame, bbox_input)
        
        inst = results[0].pred_instances
        if self.frame_id == 1:
            logger.debug(f"Pose results: {inst.keys()}")
            logger.debug(f"KPT example: {inst.keypoints[0, 0]}")
            logger.debug(f"Score example: {inst.keypoint_scores[0, 0]}")
            
        kpts = inst.keypoints[0]
        scores = inst.keypoint_scores[0]
        
        return np.hstack([kpts, scores.reshape(-1, 1)])
    
    def segment(self, frame: np.ndarray, keypoints: np.ndarray, bbox: np.ndarray, initial_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate mask using SAM or Detector fallback.
        """
        # If we have an initial mask from the detector, use it as a base
        if initial_mask is not None:
            return initial_mask.astype(np.uint8)
            
        if self.sam_predictor is None:
            # Fallback to bbox mask if no SAM and no detector mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)] = 1
            return mask
        
        # SAM 1 logic (if someone provides a compatible SAM checkpoint)
        try:
            self.sam_predictor.set_image(frame)
            valid = keypoints[:, 2] > 0.3
            if valid.sum() < 5:
                points = bbox[None, :]
                labels = np.array([1])
            else:
                points = keypoints[valid, :2]
                labels = np.ones(len(points))
            
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=points,
                point_labels=labels,
                box=bbox[None, :],
                multimask_output=True
            )
            return masks[np.argmax(scores)].astype(np.uint8)
        except Exception as e:
            logger.warning(f"SAM predict failed: {e}")
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            x1, y1, x2, y2 = map(int, bbox)
            mask[max(0,y1):min(frame.shape[0],y2), max(0,x1):min(frame.shape[1],x2)] = 1
            return mask
    
    def refine_bbox(self, mask: np.ndarray, orig_bbox: np.ndarray) -> np.ndarray:
        """Refine bbox from mask."""
        coords = np.where(mask > 0)
        if len(coords[0]) == 0:
            return orig_bbox
        
        y1, y2 = coords[0].min(), coords[0].max()
        x1, x2 = coords[1].min(), coords[1].max()
        
        pad = self.config.mask_dilation
        h, w = mask.shape
        
        refined = np.array([
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(w, x2 + pad),
            min(h, y2 + pad)
        ], dtype=np.float32)
        
        # Blend with original
        return 0.7 * refined + 0.3 * orig_bbox
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Process single frame with iterative refinement."""
        self.frame_id += 1
        h, w = frame.shape[:2]
        
        # Stage 1: Detect
        dets, initial_masks = self.detect(frame)
        
        # Stage 2: Track
        tracks, track_masks = self.track(dets, initial_masks)
        
        # Stage 3: BBox-Mask-Pose Iteration
        outputs = []
        
        # Research mapping for 23 keypoints
        kpt_names = {
            0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
            5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow',
            9: 'left_wrist', 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip',
            13: 'left_knee', 14: 'right_knee', 15: 'left_ankle', 16: 'right_ankle',
            17: 'thorax_mpii', 18: 'pelvis_mpii', 19: 'neck_mpii', 
            20: 'head_top_mpii', 21: 'neck_aic', 22: 'head_top_aic'
        }
        
        for i, track in enumerate(tracks):
            tid = int(track[4])
            bbox = track[:4].astype(np.float32)
            init_mask = track_masks[i]
            
            # Iterative refinement
            current_bbox = bbox.copy()
            best_kpts = None
            best_mask = init_mask
            
            for _ in range(self.config.refine_iterations):
                # Pose
                kpts = self.estimate_pose(frame, current_bbox)
                
                # Mask (use detector mask in first iteration or if SAM fails)
                mask = self.segment(frame, kpts, current_bbox, initial_mask=best_mask)
                
                # Refine
                current_bbox = self.refine_bbox(mask, current_bbox)
                
                # Keep best
                if best_kpts is None or np.mean(kpts[:, 2]) > np.mean(best_kpts[:, 2]):
                    best_kpts = kpts
                    best_mask = mask
            
            # Store
            self.tracks[tid]['keypoints'] = best_kpts
            self.tracks[tid]['mask'] = best_mask
            
            # Output for all 23 keypoints
            outputs.append({
                'track_id': tid,
                'frame': self.frame_id,
                'timestamp': round(self.frame_id / 30.0, 3),
                'bbox': {
                    'initial': bbox.tolist(),
                    'refined': current_bbox.tolist(),
                    'normalized': [
                        float(current_bbox[0]/w),
                        float(current_bbox[1]/h),
                        float(current_bbox[2]/w),
                        float(current_bbox[3]/h)
                    ]
                },
                'keypoints': {
                    kpt_names[idx]: {
                        'x': float(best_kpts[idx, 0]),
                        'y': float(best_kpts[idx, 1]),
                        'confidence': float(best_kpts[idx, 2])
                    }
                    for idx in range(len(best_kpts))
                    if idx in kpt_names and best_kpts[idx, 2] > 0.1
                },
                'mask_area': int(best_mask.sum()) if best_mask is not None else 0
            })
        
        # Visualize
        vis = self._visualize(frame, tracks)
        
        self.last_frame = vis
        return vis, outputs
    
    def _visualize(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """Draw results."""
        vis = frame.copy()
        
        for track in tracks:
            tid = int(track[4])
            t = self.tracks.get(tid, {})
            
            bbox = t.get('bbox', track[:4]).astype(int)
            kpts = t.get('keypoints')
            mask = t.get('mask')
            
            # Color by ID
            colors = [(0,255,0), (0,0,255), (255,0,0), (0,255,255), (255,0,255)]
            color = colors[tid % len(colors)]
            
            # BBox
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(vis, f"ID:{tid}", (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Pose
            if kpts is not None:
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.3:
                        cv2.circle(vis, (int(x), int(y)), 3, color, -1)
                
                # Skeleton (including extended points)
                skeleton = [
                    [16,14], [14,12], [17,15], [15,13], [12,13], [6,12], [7,13],
                    [6,7], [6,8], [7,9], [8,10], [10,11], [2,3], [1,2], [1,3],
                    [2,4], [3,5], [4,6], [5,7],
                    [18,20], [11,12], [19,20] # MPII/AIC extensions (rough)
                ]
                for conn in skeleton:
                    i1, i2 = conn[0]-1, conn[1]-1
                    if kpts[i1,2] > 0.3 and kpts[i2,2] > 0.3:
                        pt1 = tuple(kpts[i1,:2].astype(int))
                        pt2 = tuple(kpts[i2,:2].astype(int))
                        cv2.line(vis, pt1, pt2, color, 2)
            
            # Mask
            if mask is not None:
                overlay = vis.copy()
                overlay[mask > 0] = color
                vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        cv2.putText(vis, f"Frame:{self.frame_id} Tracks:{len(tracks)}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return vis
    
    def process_video(self, video_path: str, display: bool = False, progress_callback=None) -> Dict:
        """Process video."""
        video_path = Path(video_path)
        name = video_path.stem
        
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out_path = self.output_dir / f"{name}_vrg.mp4"
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        
        all_out = []
        times = []
        
        pbar = tqdm(total=total, desc=f"Processing {name}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                t0 = time.time()
                vis, outs = self.process_frame(frame)
                times.append(time.time() - t0)
                
                all_out.extend(outs)
                writer.write(vis)
                
                if display:
                    cv2.imshow("VRG BBoxMaskPose", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
                
                if progress_callback:
                    progress_callback(self.frame_id, total)
                
                if self.frame_id % 100 == 0:
                    avg_fps = 1.0 / np.mean(times[-100:]) if times else 0
                    pbar.set_postfix({"fps": f"{avg_fps:.1f}"})
        
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            pbar.close()
        
        # Save JSON
        json_path = self.output_dir / f"{name}_vrg.json"
        with open(json_path, "w") as f:
            json.dump({
                "video": str(video_path),
                "config": {
                    "det": self.config.det_checkpoint,
                    "pose": self.config.pose_checkpoint,
                    "sam": self.config.sam_checkpoint,
                    "iterations": self.config.refine_iterations
                },
                "frames": self.frame_id,
                "fps": fps,
                "tracks": all_out
            }, f, indent=2)
        
        return {
            "video": str(video_path),
            "frames": self.frame_id,
            "fps": 1.0 / np.mean(times) if times else 0,
            "output_video": str(out_path),
            "output_json": str(json_path)
        }


# =============================================================================
# Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VRG Prague BBoxMaskPose")
    parser.add_argument("-i", "--input", required=True, help="Video file")
    parser.add_argument("--det-config", default=None)
    parser.add_argument("--det-ckpt", default="checkpoints/rtmdet-ins-l-mask.pth")
    parser.add_argument("--pose-config", default=None)
    parser.add_argument("--pose-ckpt", default="checkpoints/MaskPose/MaskPose-s-1.1.0.pth")
    parser.add_argument("--sam-ckpt", default="checkpoints/SAM-pose2seg_hiera_b+.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("-o", "--output", default="outputs")
    
    args = parser.parse_args()
    
    config = VRGConfig(
        det_config=args.det_config,
        det_checkpoint=args.det_ckpt,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_ckpt,
        sam_checkpoint=args.sam_ckpt,
        device=args.device,
        refine_iterations=args.iterations,
        output_dir=args.output
    )
    
    pipeline = VRGBBoxMaskPose(config)
    
    try:
        stats = pipeline.process_video(args.input, args.display)
        
        logger.info("\n" + "="*60)
        logger.info("VRG BBoxMaskPose Complete")
        logger.info(f"Frames: {stats['frames']}")
        logger.info(f"FPS: {stats['fps']:.1f}")
        logger.info(f"Video: {stats['output_video']}")
        logger.info(f"JSON: {stats['output_json']}")
        logger.info("="*60)
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
