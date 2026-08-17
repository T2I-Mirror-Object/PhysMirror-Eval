import argparse
import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stage1_segmentation.utils import ImageUtils
from src.stage2_feature_extraction.detectors.dinov2 import DINOv2Detector
from src.stage2_feature_extraction.visualizer import KeypointVisualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 Test: Keypoint Detection (DINOv2 with Masking)")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing stage 1 cutouts (must have alpha channels)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="output_stage2_dinov2", 
        help="Directory to save visualized keypoints."
    )
    parser.add_argument(
        '--model_size', 
        type=str, 
        default='large', 
        choices=['small', 'base', 'large'], 
        help='DINOv2 model size.'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading DINOv2 model on {device}...")

    detector = DINOv2Detector(model_type=args.model_size, input_size=448, device=device)
    
    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.webp'))]
    
    if not image_files:
        print(f"No transparent images (PNG/WEBP) found in {args.input_dir}")
        return

    print(f"Found {len(image_files)} images. Starting detection...")

    for img_file in image_files:
        img_path = os.path.join(args.input_dir, img_file)
        
        # 1. Load with PIL to preserve Alpha channel perfectly for ImageUtils
        try:
            pil_image = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"Skipping corrupt file {img_file}: {e}")
            continue

        # 2. Convert to RGB Numpy array for DINOv2
        img_rgb_np = np.array(pil_image.convert("RGB"))

        # --- A. Detect ---
        kp_raw, desc_raw = detector.detect_and_compute(img_rgb_np)
        print(f"[{img_file}] Detected {len(kp_raw)} raw keypoints.")

        # --- B. Filter ---
        # Apply the alpha mask to kill background features
        kp_filtered, desc_filtered = ImageUtils.filter_features_by_mask(kp_raw, desc_raw, pil_image)
        print(f"  > Kept {len(kp_filtered)} keypoints after mask filtering.")

        if len(kp_filtered) > 0:
            # --- C. Visualize ---
            # Convert PIL RGBA to OpenCV BGRA for drawing
            img_bgra = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGRA)
            
            # Draw only the filtered keypoints
            
            vis_img = KeypointVisualizer.draw_keypoints(img_bgra, kp_filtered)
            
            save_path = os.path.join(args.output_dir, f"dinov2_masked_{img_file}")
            cv2.imwrite(save_path, vis_img)
            print(f"  > Saved visualization to {save_path}")
            
        else:
            print("  > No keypoints found on the foreground object.")

if __name__ == "__main__":
    main()