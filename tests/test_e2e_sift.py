import argparse
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stage1_segmentation.config import SegmentationConfig
from src.stage1_segmentation.processor import MirrorSegmentationProcessor
from src.stage1_segmentation.utils import ImageUtils
from src.stage2_feature_extraction.detectors.sift import SIFTFeatureDetector
from src.stage3_feature_matching.matchers.knn import KNNFeatureMatcher

def parse_args():
    parser = argparse.ArgumentParser(description="End-to-End Mirror Validation Pipeline")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument("--object_name", type=str, required=True, help="Object name (e.g. 'coffee mug').")
    parser.add_argument("--output_dir", type=str, default="output_e2e", help="Output directory.")
    parser.add_argument('--n_features', type=int, default=2000, help='Number of features to detect.')
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_config = SegmentationConfig(device=device)
    
    seg_processor = MirrorSegmentationProcessor(seg_config)
    feature_detector = SIFTFeatureDetector(n_features=args.n_features)
    feature_matcher = KNNFeatureMatcher(ratio_threshold=0.75)

    print(f"--- Pipeline Started for '{args.object_name}' ---")
    print("Stage 1: Segmenting...")
    pil_image = seg_processor.load_image(args.image_path)
    final_vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    seg_results = seg_processor.process_image(pil_image, args.object_name)
    
    masks = seg_results.get("masks", [])
    scores = seg_results.get("scores", [])

    if len(masks) < 2:
        print(f"Error: Stage 1 failed. Needed at least 2 masks (object + reflection), found {len(masks)}.")
        return

    # Zip masks with scores, sort descending by score, and take top 2
    scored_masks = list(zip(scores, masks))
    scored_masks.sort(key=lambda x: x[0], reverse=True)
    top_2_masks = scored_masks[:2]

    print(f"Selected top 2 masks with scores: {top_2_masks[0][0]:.2f}, {top_2_masks[1][0]:.2f}")

    cutouts = []
    bboxes = []

    for i, (score, mask) in enumerate(top_2_masks):
        cutout, bbox = ImageUtils.extract_object_with_mask(pil_image, mask)
        
        if bbox is None:
            print(f"Warning: Mask {i} was empty.")
            continue
            
        cutouts.append(cutout)
        bboxes.append(bbox) # bbox is (x_min, y_min, x_max, y_max)
        
        cutout.save(os.path.join(args.output_dir, f"cutout_{i}.png"))

    if len(cutouts) != 2:
        print("Error: Failed to generate 2 valid cutouts.")
        return

    print("Stage 2: Detecting Keypoints...")
    
    # cutout_1 -> Object (likely), cutout_2 -> Reflection (likely)
    # Note: We need to convert PIL cutouts to OpenCV format for SIFT
    img1_cv = cv2.cvtColor(np.array(cutouts[0]), cv2.COLOR_RGBA2BGR)
    img2_cv = cv2.cvtColor(np.array(cutouts[1]), cv2.COLOR_RGBA2BGR)

    kp1, desc1 = feature_detector.detect_and_compute(img1_cv)
    kp2, desc2 = feature_detector.detect_and_compute(img2_cv)

    print(f"  Object 1: {len(kp1)} keypoints")
    print(f"  Object 2: {len(kp2)} keypoints")

    if len(kp1) == 0 or len(kp2) == 0:
        print("Error: No keypoints found in one of the objects.")
        return

    print("Stage 3: Matching & Visualization...")

    matches = feature_matcher.match(desc1, desc2)
    print(f"  Found {len(matches)} matches.")
    
    line_color = (0, 255, 0) # Green
    point_color = (0, 0, 255) # Red
    
    for m in matches:
        pt1_local = kp1[m.queryIdx].pt # (x, y)
        pt2_local = kp2[m.trainIdx].pt # (x, y)

        pt1_global = ImageUtils.map_point_to_original(pt1_local, bboxes[0])
        pt2_global = ImageUtils.map_point_to_original(pt2_local, bboxes[1])

        pt1_int = (int(pt1_global[0]), int(pt1_global[1]))
        pt2_int = (int(pt2_global[0]), int(pt2_global[1]))

        # Draw Line
        cv2.line(final_vis_image, pt1_int, pt2_int, line_color, 1)
        
        # Draw Keypoints
        cv2.circle(final_vis_image, pt1_int, 3, point_color, -1)
        cv2.circle(final_vis_image, pt2_int, 3, point_color, -1)

    # Save Final Result
    save_path = os.path.join(args.output_dir, "final_e2e_result.jpg")
    cv2.imwrite(save_path, final_vis_image)
    
    print(f"--- Success! ---")
    print(f"Result saved to: {save_path}")

if __name__ == "__main__":
    main()