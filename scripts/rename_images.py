import os
import re
import argparse

def pad_image_filenames(directory, prefix, target_digits):
    """
    Scans a directory for files matching a specific prefix and zero-pads 
    the numeric portion of the filename to a specified length.
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # Regex pattern to match: {prefix}{any digits}{.extension}
    # Example: "image_" + "12" + ".jpg"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)(\.[^.]+)$")
    
    renamed_count = 0
    skipped_count = 0
    
    print(f"Scanning directory: {directory}")
    
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        
        if match:
            current_num_str = match.group(1)
            extension = match.group(2)
            
            # Convert to int to strip existing zeros, then pad to target length
            new_num_str = str(int(current_num_str)).zfill(target_digits)
            new_filename = f"{prefix}{new_num_str}{extension}"
            
            # Only rename if the filename actually needs to change
            if filename != new_filename:
                old_path = os.path.join(directory, filename)
                new_path = os.path.join(directory, new_filename)
                
                # Safety check: avoid overwriting existing files
                if os.path.exists(new_path):
                    print(f"Warning: Cannot rename '{filename}' because '{new_filename}' already exists.")
                    skipped_count += 1
                    continue
                    
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                renamed_count += 1
                
    print(f"\nProcess complete. Renamed {renamed_count} files (Skipped {skipped_count} collisions).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standardize image filename numbering by padding with zeros."
    )
    
    # Define CLI arguments
    parser.add_argument(
        "-d", "--dir", 
        required=True, 
        help="Path to the directory containing the images."
    )
    parser.add_argument(
        "-p", "--prefix", 
        default="image_", 
        help="The text prefix before the numbers (default: 'image_')."
    )
    parser.add_argument(
        "-n", "--digits", 
        type=int, 
        default=4, 
        help="The target number of digits for the numeric part (default: 4)."
    )
    
    args = parser.parse_args()
    
    pad_image_filenames(args.dir, args.prefix, args.digits)