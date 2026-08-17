import os
import re
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate CLIP Score using Transformers with Safetensors.")
    
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing generated images.")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Path to the text file containing the prompts.")
    parser.add_argument("--output_json", type=str, default="clip_scores.json",
                        help="Path to save the final JSON report.")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16",
                        help="HuggingFace model path for CLIP.")
    
    return parser.parse_args()

def load_data(image_folder, prompts_file):
    """Loads images and aligns them with prompts using dynamic index matching."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    
    if not os.path.exists(prompts_file):
        print(f"Error: Prompts file not found: {prompts_file}")
        return []

    with open(prompts_file, 'r', encoding='utf-8') as f:
        all_prompts = [line.strip() for line in f.readlines()]

    pattern = re.compile(r"image_(\d+)")
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

    base_index = min(idx for _, idx in parsed_files)
    
    dataset = []
    for f, idx in parsed_files:
        adjusted_idx = idx - base_index
        
        if adjusted_idx < len(all_prompts):
            full_path = os.path.join(image_folder, f)
            dataset.append({
                'image_path': full_path, 
                'prompt': all_prompts[adjusted_idx]
            })
            
    print(f"Found {len(dataset)} valid image-prompt pairs (Base index: {base_index}).")
    return dataset

def main():
    args = parse_args()
    
    dataset = load_data(args.image_folder, args.prompts_file)
    if not dataset:
        return

    # Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing HuggingFace CLIP Model on {device} (Forcing Safetensors)...")
    
    # Bypass TorchMetrics and initialize Transformers directly to force use_safetensors=True
    try:
        model = CLIPModel.from_pretrained(args.model_name, use_safetensors=True).to(device)
        processor = CLIPProcessor.from_pretrained(args.model_name, use_safetensors=True)
    except Exception as e:
        print(f"Failed to load the model: {e}")
        return

    results = []
    total_score = 0.0

    print("Calculating CLIP Scores...")
    with torch.no_grad():
        for item in tqdm(dataset, desc="Processing"):
            try:
                # Load PIL Image directly
                image = Image.open(item['image_path']).convert("RGB")
                prompt = item['prompt']
                
                # Prepare inputs for the model
                inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
                
                # Pass inputs through the model to get the projected embeddings directly
                outputs = model(**inputs)
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                
                # Normalize features (Crucial for correct Cosine Similarity)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                
                # Exact TorchMetrics CLIPScore calculation: max(100 * cos_sim, 0)
                cos_sim = (image_features * text_features).sum(dim=-1).squeeze()
                score_val = max(100.0 * float(cos_sim.cpu()), 0.0)
                score_val = round(score_val, 4)
                
                total_score += score_val
                
                results.append({
                    "filename": os.path.basename(item['image_path']),
                    "prompt": prompt,
                    "clip_score": score_val
                })
                
            except Exception as e:
                print(f"\nError processing {item['image_path']}: {e}")
                results.append({
                    "filename": os.path.basename(item['image_path']),
                    "prompt": item['prompt'],
                    "clip_score": 0.0,
                    "error": str(e)
                })

    # Compile Final JSON Structure
    overall_average = total_score / len(dataset) if dataset else 0.0
    
    final_output = {
        "overall_average_clip_score": round(overall_average, 4),
        "total_images_processed": len(dataset),
        "individual_scores": results
    }

    print("\n" + "="*40)
    print("CLIP SCORE EVALUATION COMPLETE")
    print(f"Overall Average CLIP Score: {overall_average:.4f}")
    print("="*40)

    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"Detailed results saved to: {args.output_json}")
    except IOError as e:
        print(f"Error saving JSON: {e}")

if __name__ == "__main__":
    main()