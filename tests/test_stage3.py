import argparse
import os
import sys
import cv2
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stage2_feature_extraction.detectors.sift import SIFTFeatureDetector
from src.stage3_feature_matching.matchers.knn import KNNFeatureMatcher
from src.stage3_feature_matching.visualizer import MatchVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 3 Test: Keypoint Matching")
    parser.add_argument("--image1", type=str, required=True, help="Path to first cutout (Real Object).")
    parser.add_argument("--image2", type=str, required=True, help="Path to second cutout (Reflection).")
    parser.add_argument("--output_dir", type=str, default="output_stage3", help="Output directory.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.image1) or not os.path.exists(args.image2):
        print("Error: One or both images not found.")
        return

    # 1. Load Images (IMREAD_UNCHANGED to handle Alpha if present)
    img1 = cv2.imread(args.image1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(args.image2, cv2.IMREAD_UNCHANGED)

    # 2. Detect Features (Stage 2)
    # nfeatures=0 means 'all features', giving KNN more data to work with
    detector = SIFTFeatureDetector(nfeatures=1000)
    
    print("Detecting features...")
    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)
    
    print(f"Image 1: {len(kp1)} features")
    print(f"Image 2: {len(kp2)} features")

    # 3. Match Features (Stage 3)
    # Using L2 norm because SIFT descriptors are float vectors
    matcher = KNNFeatureMatcher(norm_type=cv2.NORM_L2, ratio_threshold=0.75)
    
    print("Matching features...")
    matches = matcher.match(desc1, desc2)
    
    print(f"Found {len(matches)} good matches after Lowe's Ratio Test.")

    # 4. Visualize
    os.makedirs(args.output_dir, exist_ok=True)
    
    if len(matches) > 0:
        vis_img = MatchVisualizer.draw_matches(img1, kp1, img2, kp2, matches)
        
        save_path = os.path.join(args.output_dir, "matches.jpg")
        cv2.imwrite(save_path, vis_img)
        print(f"Saved visualization to {save_path}")
    else:
        print("No matches found. Try adjusting thresholds.")

if __name__ == "__main__":
    main()