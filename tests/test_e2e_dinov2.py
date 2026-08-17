import argparse
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports ---
from src.stage1_segmentation.config import SegmentationConfig
from src.stage1_segmentation.processor import MirrorSegmentationProcessor
from src.stage1_segmentation.utils import ImageUtils
from src.stage2_feature_extraction.detectors.dinov2 import DINOv2Detector
from src.stage3_feature_matching.matchers.mutual_nn import MutualNNMatcher
from src.stage4_analysis.vanishing_point import VanishingPointAnalyzer
from src.stage4_analysis.scorers.factory import get_scorer

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Mirror Validation Pipeline (DINOv2)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument("--object_names", type=str, required=True, help="Comma-separated object names (e.g. 'teddy bear, chair').")
    parser.add_argument("--output_dir", type=str, default="output_dinov2", help="Output directory.")
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], help='DINOv2 model size.')
    parser.add_argument('--top_k', type=int, default=50, help='Maximum number of strongest matches to visualize per object.')
    parser.add_argument('--show_centroid', action='store_true', help='Visualize the centroid.')

    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Initialization ---
    print(f"[Init] Loading models on {device}...")
    seg_config = SegmentationConfig(device=device)
    seg_processor = MirrorSegmentationProcessor(seg_config)
    
    feature_detector = DINOv2Detector(model_type=args.model_size, input_size=448, device=device)
    feature_matcher = MutualNNMatcher(similarity_threshold=0.45, use_cuda=(device=='cuda'))

    object_list = [obj.strip() for obj in args.object_names.split(',') if obj.strip()]
    print(f"--- Pipeline Started for: {object_list} ---")

    pil_image = seg_processor.load_image(args.image_path)
    final_vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    lines_global = []
    accumulated_errors = []

    # --- Process Each Object ---
    for object_name in object_list:
        print(f"\n>>> Processing Object: '{object_name}'")
        
        # --- Stage 1: Segmentation ---
        print("  Stage 1: Segmenting...")
        seg_results = seg_processor.process_image(pil_image, object_name)
        masks = seg_results.get("masks", [])
        scores = seg_results.get("scores", [])

        if len(masks) < 2:
            print(f"  [!] Error: Found only {len(masks)} masks. Skipping.")
            accumulated_errors.append(f"[{object_name}: < 2 masks]")
            continue

        scored_masks = list(zip(scores, masks))
        scored_masks.sort(key=lambda x: x[0], reverse=True)
        top_2_masks = scored_masks[:2]
        
        cutouts = []
        bboxes = []
        for i, (score, mask) in enumerate(top_2_masks):
            cutout, bbox = ImageUtils.extract_object_with_mask(pil_image, mask)
            if bbox is None: continue
            cutouts.append(cutout)
            bboxes.append(bbox)

        if len(cutouts) != 2:
            print("  [!] Error: Could not generate 2 valid cutouts. Skipping.")
            accumulated_errors.append(f"[{object_name}: cutout failed]")
            continue

        # --- Stage 2: Feature Extraction ---
        print("  Stage 2: DINOv2 Feature Extraction...")
        img1_np = np.array(cutouts[0])
        img2_np = np.array(cutouts[1])

        kp1_raw, desc1_raw = feature_detector.detect_and_compute(img1_np)
        kp2_raw, desc2_raw = feature_detector.detect_and_compute(img2_np)
        
        print("  Applying Mask Filtering...")
        kp1, desc1 = ImageUtils.filter_features_by_mask(kp1_raw, desc1_raw, cutouts[0])
        kp2, desc2 = ImageUtils.filter_features_by_mask(kp2_raw, desc2_raw, cutouts[1])

        if len(kp1) == 0 or len(kp2) == 0:
            print("  [!] Error: No keypoints left on the object after masking. Skipping.")
            accumulated_errors.append(f"[{object_name}: no features]")
            continue

        # --- Stage 3: Matching ---
        print("  Stage 3: Mutual NN Matching...")
        raw_matches = feature_matcher.match(desc1, desc2)
        print(f"  Found {len(raw_matches)} raw mutual matches.")

        if len(raw_matches) == 0:
            print("  [!] Error: No matches found. Skipping.")
            accumulated_errors.append(f"[{object_name}: no matches]")
            continue

        # --- Stage 4: Local to Global Mapping ---
        # Extract points and convert distances to scores (1.0 - distance)
        scores_matches = np.float32([1.0 - m.distance for m in raw_matches])
        src_pts = np.float32([kp1[m.queryIdx].pt for m in raw_matches])
        tgt_pts = np.float32([kp2[m.trainIdx].pt for m in raw_matches])

        # Sort and Limit Matches
        sorted_indices = np.argsort(scores_matches)[::-1]
        limit = min(len(scores_matches), args.top_k)
        top_indices = sorted_indices[:limit]

        # Prepare Global Lines for this object
        for idx in top_indices:
            pt1_local = src_pts[idx]
            pt2_local = tgt_pts[idx]
            pt1_global = np.array([pt1_local[0] + bboxes[0][0], pt1_local[1] + bboxes[0][1]])
            pt2_global = np.array([pt2_local[0] + bboxes[1][0], pt2_local[1] + bboxes[1][1]])
            lines_global.append((pt1_global, pt2_global))

    # --- Final Analysis & Scoring ---
    print("\n" + "=" * 40)
    if len(lines_global) == 0:
        print("PIPELINE FAILED: No valid lines extracted across any objects.")
        print(f"Errors encountered: {accumulated_errors}")
        return

    print("Stage 4: Global Centroid Analysis...")
    scorer = get_scorer('centroid')
    metrics = scorer.calculate_score(lines_global)
    vp = metrics.get('centroid', np.array([0, 0]))

    print(f"EVALUATION RESULTS:")
    print(f" > Total Valid Matches: {len(lines_global)}")
    print(f" > Centroid (Est. VP): ({vp[0]:.1f}, {vp[1]:.1f})")
    print(f" > Sum Distances: {metrics.get('sum_distance', 0):.2f}")
    print(f" > Mean Spread: {metrics.get('mean_distance', 0):.2f} px")
    print(f" > NORMALIZED SCORE: {metrics['final_score']:.4f}")
    if accumulated_errors:
        print(f" > Partial Errors: {accumulated_errors}")
    print("=" * 40)

    # --- Visualization ---
    h, w, _ = final_vis_image.shape
    
    # Draw Lines (Yellow) 
    for l in lines_global:
        pt1, pt2 = (int(l[0][0]), int(l[0][1])), (int(l[1][0]), int(l[1][1]))
        cv2.line(final_vis_image, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(final_vis_image, pt1, 3, (0, 0, 255), -1)
        cv2.circle(final_vis_image, pt2, 3, (0, 0, 255), -1)

    # Draw Centroid/VP (Green)
    if args.show_centroid:
        vx, vy = int(vp[0]), int(vp[1])
        if -w < vx < 2*w and -h < vy < 2*h:
            cv2.circle(final_vis_image, (vx, vy), 8, (0, 255, 0), -1)

    filename = os.path.basename(args.image_path)
    save_path = os.path.join(args.output_dir, f"viz_multi_{filename}")
    cv2.imwrite(save_path, final_vis_image)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    main()