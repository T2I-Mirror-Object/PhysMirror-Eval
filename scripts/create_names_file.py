import os
import re

def create_names_file(prompts_path, coco_path, output_path):
    # 1. Load valid COCO objects into a set for fast lookup
    if not os.path.exists(coco_path):
        print(f"Error: COCO objects file not found at {coco_path}")
        return

    with open(coco_path, 'r') as f:
        valid_coco_objects = {line.strip().lower() for line in f.readlines()}

    # 2. Process the prompts
    if not os.path.exists(prompts_path):
        print(f"Error: Prompts file not found at {prompts_path}")
        return

    with open(prompts_path, 'r') as f:
        prompts = f.readlines()

    extracted_lines = []
    
    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue

        # Isolate the objects portion of the string
        delimiter = " in front of the mirror"
        if delimiter in prompt:
            objects_substring = prompt.split(delimiter)[0]
        else:
            print(f"[Warning] Delimiter not found in prompt: {prompt}")
            objects_substring = prompt # Fallback

        # Split into individual item strings (e.g., "a cat", "a pair of skis")
        raw_items = objects_substring.split(',')
        
        cleaned_objects = []
        for item in raw_items:
            item = item.strip()
            
            # Updated Regex: Match "a pair of", "a", or "an" at the start of the string
            # It checks for the longest match first to ensure "a pair of" is fully removed
            obj_name = re.sub(r'^(a\s+pair\s+of|a|an)\s+', '', item, flags=re.IGNORECASE).strip()
            
            # Validate against the COCO list
            if obj_name.lower() in valid_coco_objects:
                cleaned_objects.append(obj_name)
            else:
                print(f"[Warning] Extracted '{obj_name}' but it is not in the COCO list.")

        # Join with comma and add to our final list
        extracted_lines.append(",".join(cleaned_objects))

    # 3. Write to the output file
    with open(output_path, 'w') as f:
        for line in extracted_lines:
            f.write(line + "\n")

    print(f"Success: Wrote {len(extracted_lines)} lines to {output_path}")

if __name__ == "__main__":
    PROMPTS_FILE = "data/3000_one_object_prompts.txt"
    COCO_FILE = "data/300_objects.txt"
    OUTPUT_FILE = "data/names_file_3000.txt"
    
    create_names_file(PROMPTS_FILE, COCO_FILE, OUTPUT_FILE)