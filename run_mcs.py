import argparse
import os
import sys
import json
import csv
import re
import numpy as np
import cv2
import torch
from tqdm import tqdm
from PIL import Image

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from src.stage1_segmentation.config import SegmentationConfig
from src.stage1_segmentation.processor import MirrorSegmentationProcessor
from src.stage1_segmentation.utils import ImageUtils
from src.stage2_feature_extraction.detectors.dinov2 import DINOv2Detector
from src.stage3_feature_matching.matchers.mutual_nn import MutualNNMatcher
from src.stage4_analysis.scorers.factory import get_scorer
from src.stage4_analysis.vanishing_point import VanishingPointAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Mirror Symmetry Evaluation")
    
    parser.add_argument("--image_folder", type=str, required=True, 
                        help="Path to folder containing input images (format: image_XXX.jpg).")
    parser.add_argument("--names_file", type=str, required=True, 
                        help="Path to text file containing object names (one per line).")
    parser.add_argument("--output_json", type=str, default="evaluation_report.json", 
                        help="Path to save the final JSON report.")
    parser.add_argument("--output_mask_dir", type=str, default="saved_masks", 
                    help="Folder to save the extracted segmentation masks.")
    parser.add_argument("--output_viz_dir", type=str, default="visualization_output", 
                        help="Folder to save visualization images (if enabled).")
    parser.add_argument("--visualize", action='store_true', 
                        help="If set, saves images with matching lines and centroid drawn.")
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'], 
                        help='DINOv2 model size.')
    parser.add_argument('--top_k', type=int, default=10, 
                        help='Number of strongest matches for each object to use for scoring.')
    parser.add_argument('--sim_threshold', type=float, default=0.5, 
                    help='Similarity threshold (pi) for mutual NN matcher.')
    
    return parser.parse_args()

def load_data(image_folder, names_file):
    """
    Loads image paths and matches them to lines in the text file based on index.
    Expects filenames like 'image_005.jpg'. Automatically handles 0-based and 1-based indexing.
    """
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    if not os.path.exists(names_file):
        print(f"Error: Names file not found: {names_file}")
        return []

    with open(names_file, 'r') as f:
        all_lines = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(all_lines)} lines from names file.")

    pattern = re.compile(r"image_(\d+)")
    
    # Scan directory and collect valid files with their parsed index
    files = sorted(os.listdir(image_folder))
    parsed_files = []
    
    for f in files:
        name, ext = os.path.splitext(f)
        if ext.lower() not in valid_exts:
            continue
            
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            parsed_files.append((f, idx))
            
    if not parsed_files:
        print(f"No valid images found in {image_folder}.")
        return []

    # Dynamically determine the base offset (0-based or 1-based)
    base_index = min(idx for _, idx in parsed_files)
    
    dataset = []
    for f, idx in parsed_files:
        # Subtract the base_index so the first image always maps to line 0
        adjusted_idx = idx - base_index
        
        if adjusted_idx < len(all_lines):
            full_path = os.path.join(image_folder, f)
            raw_objects = all_lines[adjusted_idx].split(',')
            object_list = [obj.strip() for obj in raw_objects if obj.strip()]
            dataset.append((full_path, object_list))
        else:
            print(f"[Warning] Adjusted image index {adjusted_idx} (original {idx}) exceeds lines in names file. Skipping {f}.")
    
    print(f"Found {len(dataset)} valid image-name pairs (Detected base index: {base_index}).")
    return dataset

def process_single_image(
    image_path, object_list, 
    seg_processor, feature_detector, feature_matcher, scorer, 
    top_k, visualize=False, output_viz_dir=None, output_mask_dir=None
):
    """
    Runs pipeline for one image. Returns metrics dict.
    Optionally saves visualization.
    """
    try:
        # Load image once
        pil_image = seg_processor.load_image(image_path)
        image_shape = (pil_image.size[1], pil_image.size[0]) # (H, W) for bounding
        
        if visualize:
             vis_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        all_lines_global = []
        accumulated_errors = []
        
        base_filename = os.path.splitext(os.path.basename(image_path))[0]

        for object_name in object_list:
            # --- Stage 1: Segmentation ---
            seg_results = seg_processor.process_image(pil_image, object_name)
            masks = seg_results.get("masks", [])
            scores = seg_results.get("scores", [])

            if len(masks) < 2:
                accumulated_errors.append(f"[{object_name}: < 2 masks]")
                continue

            # Select top 2 masks
            scored_masks = list(zip(scores, masks))
            scored_masks.sort(key=lambda x: x[0], reverse=True)
            top_2_masks = scored_masks[:2]

            if output_mask_dir:
                os.makedirs(output_mask_dir, exist_ok=True)
                for i, (score, mask) in enumerate(top_2_masks):
                    # Convert mask to numpy array
                    mask_np = np.array(mask)
                    
                    # Segmentation masks are often boolean (True/False) or 0/1. 
                    # We must scale them to 0-255 so they save properly as images.
                    if mask_np.dtype == bool or mask_np.max() <= 1:
                        mask_np = (mask_np * 255).astype(np.uint8)
                    else:
                        mask_np = mask_np.astype(np.uint8)
                    
                    # Create Image and save
                    mask_img = Image.fromarray(mask_np)
                    
                    # Naming convention: image_005_cat_mask_0.png
                    mask_filename = f"{base_filename}_{object_name}_mask_{i}.png"
                    mask_save_path = os.path.join(output_mask_dir, mask_filename)
                    mask_img.save(mask_save_path)

            # Extract Cutouts
            cutouts = []
            bboxes = []
            for _, mask in top_2_masks:
                cutout, bbox = ImageUtils.extract_object_with_mask(pil_image, mask)
                if bbox:
                    cutouts.append(cutout)
                    bboxes.append(bbox)

            if len(cutouts) != 2:
                accumulated_errors.append(f"[{object_name}: cutout failed]")
                continue

            # --- Stage 2: Feature Extraction (DINOv2) ---
            img1_np = np.array(cutouts[0])
            img2_np = np.array(cutouts[1])

            kp1_raw, desc1_raw = feature_detector.detect_and_compute(img1_np)
            kp2_raw, desc2_raw = feature_detector.detect_and_compute(img2_np)

            kp1, desc1 = ImageUtils.filter_features_by_mask(kp1_raw, desc1_raw, cutouts[0])
            kp2, desc2 = ImageUtils.filter_features_by_mask(kp2_raw, desc2_raw, cutouts[1])

            if len(kp1) == 0 or len(kp2) == 0:
                accumulated_errors.append(f"[{object_name}: no features]")
                continue

            # --- Stage 3: Matching ---
            raw_matches = feature_matcher.match(desc1, desc2)
            
            if len(raw_matches) == 0:
                accumulated_errors.append(f"[{object_name}: no matches]")
                continue

            # --- Stage 4: Local to Global Mapping ---
            raw_matches.sort(key=lambda x: x.distance)
            matches_to_use = raw_matches[:top_k] # Top K per object
            
            for m in matches_to_use:
                pt1_local = kp1[m.queryIdx].pt
                pt2_local = kp2[m.trainIdx].pt
                
                # Map to Global Coordinates
                pt1_global = np.array([pt1_local[0] + bboxes[0][0], pt1_local[1] + bboxes[0][1]])
                pt2_global = np.array([pt2_local[0] + bboxes[1][0], pt2_local[1] + bboxes[1][1]])
                
                # Append to the aggregate list for the whole image
                all_lines_global.append((pt1_global, pt2_global))

        if len(all_lines_global) == 0:
            error_msg = " | ".join(accumulated_errors) if accumulated_errors else "No valid objects found"
            return {'error': f'All objects failed: {error_msg}', 'final_score': 0.0}

        # Calculate one single score using all aggregated lines
        metrics = scorer.calculate_score(all_lines_global)

        # Calculate intersections here so they are always available for the JSON report
        intersections = VanishingPointAnalyzer.find_pairwise_intersections(
            all_lines_global, image_shape=image_shape
        )
        
        # Convert NumPy arrays to Python lists for JSON serialization
        metrics['intersections'] = intersections.tolist() if len(intersections) > 0 else []
        if 'centroid' in metrics and isinstance(metrics['centroid'], np.ndarray):
            metrics['centroid'] = metrics['centroid'].tolist()

        # Add metadata
        metrics['num_matches'] = len(all_lines_global)
        metrics['error'] = " | ".join(accumulated_errors) if accumulated_errors else 'None'

        # --- Visualization Logic ---
        if visualize and output_viz_dir:
            # Draw Matching Lines (Yellow)
            for l in all_lines_global:
                pt1 = (int(l[0][0]), int(l[0][1]))
                pt2 = (int(l[1][0]), int(l[1][1]))
                cv2.line(vis_image, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis_image, pt1, 3, (0, 0, 255), -1) 
                cv2.circle(vis_image, pt2, 3, (0, 0, 255), -1)

            # Draw Intersections (Cyan Cluster) using the already computed list
            for p in metrics['intersections']:
                ix, iy = int(p[0]), int(p[1])
                cv2.circle(vis_image, (ix, iy), 2, (255, 255, 0), -1)

            # Draw Centroid (Green)
            if 'centroid' in metrics and metrics['centroid']:
                cx, cy = int(metrics['centroid'][0]), int(metrics['centroid'][1])
                h, w, _ = vis_image.shape
                if 0 <= cx < w and 0 <= cy < h:
                    cv2.circle(vis_image, (cx, cy), 8, (0, 255, 0), -1)
            
            # Save Image
            filename = os.path.basename(image_path)
            save_path = os.path.join(output_viz_dir, f"viz_{filename}")
            cv2.imwrite(save_path, vis_image)

        return metrics
    except Exception as e:
        print(str(e))
        return {'error': str(e), 'final_score': 0.0}

def main():
    args = parse_args()
    
    # Setup
    if not os.path.exists(args.image_folder):
        print(f"Error: Folder not found: {args.image_folder}")
        return
        
    dataset = load_data(args.image_folder, args.names_file)
    if not dataset:
        print("No valid data pairs found.")
        return
    
    # Create viz directory if needed
    if args.visualize:
        os.makedirs(args.output_viz_dir, exist_ok=True)
        print(f"Visualization enabled. Images will be saved to: {args.output_viz_dir}")

    # Initialize Models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}...")
    
    seg_config = SegmentationConfig(device=device)
    seg_processor = MirrorSegmentationProcessor(seg_config)
    feature_detector = DINOv2Detector(model_type=args.model_size, input_size=448, device=device)
    feature_matcher = MutualNNMatcher(similarity_threshold=args.sim_threshold, use_cuda=(device=='cuda'))
    
    # Use Centroid Scorer
    scorer = get_scorer('centroid')

    # Main Loop
    results = []
    total_score = 0.0
    
    print("Starting Batch Evaluation...")
    for image_path, object_list in tqdm(dataset, desc="Processing"):
        
        metrics = process_single_image(
            image_path, object_list,
            seg_processor, feature_detector, feature_matcher, scorer,
            args.top_k,
            visualize=args.visualize,
            output_viz_dir=args.output_viz_dir,
            output_mask_dir=args.output_mask_dir
        )
        
        score = metrics.get('final_score', 0.0)
        total_score += score
        
        # Building the structured dictionary for JSON
        results.append({
            'filename': os.path.basename(image_path),
            'object_list': object_list,
            'final_score': score,
            'sum_distance': metrics.get('sum_distance', 0),
            'mean_distance': metrics.get('mean_distance', 0),
            'num_matches': metrics.get('num_matches', 0),
            'centroid': metrics.get('centroid', []),
            'intersections': metrics.get('intersections', []),
            'error_status': metrics.get('error', 'None')
        })

    # Reporting
    avg_score = total_score / len(dataset) if dataset else 0.0
    
    print("\n" + "="*40)
    print(f"BATCH PROCESSING COMPLETE")
    print(f"Total Images: {len(dataset)}")
    print(f"Average Mirror Consistency Score: {avg_score:.5f}")
    print("="*40)

    # --- Save JSON Report ---
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"JSON Report saved to: {args.output_json}")
    except IOError as e:
        print(f"Error saving JSON: {e}")

if __name__ == "__main__":
    main()