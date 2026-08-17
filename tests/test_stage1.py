import argparse
import os
import sys
import numpy as np
from PIL import Image

# Add project root to python path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.stage1_segmentation.config import SegmentationConfig
from src.stage1_segmentation.processor import MirrorSegmentationProcessor
from src.stage1_segmentation.utils import ImageUtils

def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1 Test: Segment, Draw Boxes, and Crop")
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image.")
    parser.add_argument("--object_name", type=str, required=True, help="Text prompt.")
    parser.add_argument("--output_dir", type=str, default="output_stage1", help="Output directory.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found at {args.image_path}")
        return

    # 1. Setup
    # Ensure CUDA is available for the model
    import torch
    config = SegmentationConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    processor = MirrorSegmentationProcessor(config)
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 2. Process
    print(f"Processing: {args.image_path} | Object: {args.object_name}")
    image = processor.load_image(args.image_path)
    result = processor.process_image(image, args.object_name)
    
    # Extract data from result dict
    masks = result.get("masks", [])
    boxes = result.get("boxes", []) # These are [x1, y1, x2, y2]
    scores = result.get("scores", [])

    print(f"\n--- Results ---")
    print(f"Detections: {len(boxes)}")

    if len(boxes) > 0:
        # --- A. Save Binary Masks (Existing Logic) ---
        mask_dir = os.path.join(args.output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)
        
        for i, mask_np in enumerate(masks):
            mask_np = np.squeeze(mask_np)
            mask_uint8 = (mask_np * 255).astype(np.uint8)
            Image.fromarray(mask_uint8).save(os.path.join(mask_dir, f"mask_{i}.png"))
        print(f"Saved {len(masks)} binary masks to {mask_dir}")

        # --- B. Draw Boxes on Original Image ---
        annotated_img = ImageUtils.draw_boxes(image, boxes, scores)
        annotated_path = os.path.join(args.output_dir, "annotated_scene.jpg")
        annotated_img.save(annotated_path)
        print(f"Saved annotated image to: {annotated_path}")

        # --- C. Crop Objects (New Logic) ---
        crops = ImageUtils.crop_objects(image, boxes)
        crop_dir = os.path.join(args.output_dir, "crops")
        os.makedirs(crop_dir, exist_ok=True)
        
        for i, crop in enumerate(crops):
            crop_path = os.path.join(crop_dir, f"crop_{i}.png")
            crop.save(crop_path)
            
        print(f"Saved {len(crops)} cropped images to {crop_dir}")

        # --- D. Extract Objects with Mask ---
        cutout_dir = os.path.join(args.output_dir, "cutouts")
        os.makedirs(cutout_dir, exist_ok=True)
        
        for i, mask_np in enumerate(masks):
            # 1. Extract image AND the mapping info (bbox)
            cutout, crop_bbox = ImageUtils.extract_object_with_mask(image, mask_np)
            
            # Save cutout
            safe_obj_name = args.object_name.replace(" ", "_")
            cutout_path = os.path.join(cutout_dir, f"{safe_obj_name}_cutout_{i}.png")
            cutout.save(cutout_path)
            
            print(f"Saved cutout {i} to: {cutout_path}")
            
            if crop_bbox:
                # Example of how to use the mapping
                x_off, y_off, _, _ = crop_bbox
                print(f"  > Mapping Info (Offset): x={x_off}, y={y_off}")
                
                # Hypothetical check: Map the center of the cutout back to the original image
                center_crop = (cutout.width // 2, cutout.height // 2)
                center_orig = ImageUtils.map_point_to_original(center_crop, crop_bbox)
                print(f"  > Center of cutout {center_crop} maps to {center_orig} in original image")
        
    else:
        print("No objects detected.")

if __name__ == "__main__":
    main()