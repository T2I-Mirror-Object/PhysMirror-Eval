import argparse
import os
import textwrap
from PIL import Image, ImageDraw, ImageFont

def get_font(size):
    """Attempt to load a scalable font, fallback to default if not found."""
    try:
        # Standard Windows font
        return ImageFont.truetype("arial.ttf", size)
    except IOError:
        try:
            # Standard Linux/Mac font
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except IOError:
            print("Warning: Could not load truetype font. Using small default font.")
            return ImageFont.load_default()

def create_ordered_grids(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    if not os.path.exists(args.prompts_file):
        print(f"Error: Prompts file not found at {args.prompts_file}")
        return
        
    with open(args.prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
        
    font_title = get_font(60)
    font_label = get_font(60)
    
    order = ["SDXL", "FLUX.1-dev", "Ours"]

    for i, prompt in enumerate(prompts):
        # 1-indexed for Ours (Omini), 0-indexed for baselines (SDXL and FLUX.1-dev)
        omini_idx = i + 1
        baseline_idx = i
        
        paths = {
            "SDXL": os.path.join(args.sdxl_dir, f"image_{baseline_idx:03d}.png"),
            "FLUX.1-dev": os.path.join(args.flux_dir, f"image_{baseline_idx:03d}.png"),
            "Ours": os.path.join(args.omini_dir, f"image_{omini_idx:03d}.png")
        }
        
        # Verify files exist
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            print(f"Skipping index {i}: Missing files {missing}")
            continue
            
        # Load images in order
        images = [Image.open(paths[name]).convert("RGB") for name in order]
        
        # Assuming images are roughly the same size (e.g., 1024x1024 for FLUX/SDXL)
        img_w, img_h = images[0].size
        padding = 30
        
        # Calculate text dimensions
        wrapped_prompt = textwrap.fill(f"Prompt: {prompt}", width=120)
        
        # Use textbbox to get actual text height
        draw_temp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = draw_temp.multiline_textbbox((0, 0), wrapped_prompt, font=font_title)
        prompt_height = bbox[3] - bbox[1] + 40
        label_height = 80
        
        # Create canvas
        total_width = (img_w * 3) + (padding * 4)
        total_height = prompt_height + img_h + label_height + (padding * 2)
        
        canvas = Image.new("RGB", (total_width, total_height), "white")
        draw = ImageDraw.Draw(canvas)
        
        # Draw Prompt Text
        draw.multiline_text((padding, padding), wrapped_prompt, fill="black", font=font_title)
        
        # Paste Images and Draw Labels
        y_offset_img = prompt_height + padding
        y_offset_label = y_offset_img + img_h + 20
        
        for idx, img in enumerate(images):
            # Resize if slightly mismatched, to fit cleanly
            if img.size != (img_w, img_h):
                img = img.resize((img_w, img_h), Image.Resampling.LANCZOS)
                
            x_offset = padding + (idx * (img_w + padding))
            canvas.paste(img, (x_offset, y_offset_img))
            
            # Label is the method name (SDXL, FLUX.1-dev, Ours)
            label = order[idx]
            
            # Center the label under the image
            label_bbox = draw.textbbox((0, 0), label, font=font_label)
            label_w = label_bbox[2] - label_bbox[0]
            label_x = x_offset + (img_w - label_w) // 2
            
            draw.text((label_x, y_offset_label), label, fill="black", font=font_label)
            
        # Save combined image
        out_filename = f"combined_{i:03d}.png"
        canvas.save(os.path.join(args.output_dir, out_filename))
        print(f"Generated {out_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine 3 generation methods into a single ordered study image.")
    parser.add_argument("--sdxl_dir", type=str, default="study/sdxl_baseline", help="Path to sdxl_baseline")
    parser.add_argument("--flux_dir", type=str, default="study/flux_baseline", help="Path to flux_baseline")
    parser.add_argument("--omini_dir", type=str, default="study/images_flux_omini", help="Path to images_flux_omini")
    parser.add_argument("--prompts_file", type=str, default="data/prompts.txt", help="Path to prompts.txt")
    parser.add_argument("--output_dir", type=str, default="study/carousel_images", help="Output directory")
    
    args = parser.parse_args()
    create_ordered_grids(args)
