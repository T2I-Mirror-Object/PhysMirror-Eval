import os
import re
import argparse

def find_missing_images(directory, prefix, start_num, end_num, digits):
    """
    Scans a directory for images with a specific prefix and numbering,
    then identifies any missing sequence numbers in the specified range.
    """
    if not os.path.isdir(directory):
        print(f"Error: The directory '{directory}' does not exist.")
        return

    # 1. Generate the set of all expected numbers (e.g., 1 through 3000)
    expected_numbers = set(range(start_num, end_num + 1))
    found_numbers = set()

    # 2. Regex to match the prefix, capture the digits, and ignore the extension
    # Example: matches "image_0042.jpg" and extracts "0042"
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.[^.]+$")

    print(f"Scanning directory: '{directory}'...")
    
    # 3. Scan the directory once and collect all valid numbers
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Convert extracted string (e.g., "0042") to an integer (42)
            num = int(match.group(1))
            found_numbers.add(num)

    # 4. Calculate missing numbers using set logic
    missing_numbers = sorted(expected_numbers - found_numbers)

    # 5. Output the results
    print("-" * 40)
    if not missing_numbers:
        print(f"✅ Success: No missing images found between {start_num} and {end_num}.")
    else:
        print(f"❌ Found {len(missing_numbers)} missing images in the sequence:")
        for num in missing_numbers:
            # Reconstruct the expected filename for easy reading
            expected_filename = f"{prefix}{str(num).zfill(digits)}.*"
            print(f"   -> Missing: {expected_filename}")
            
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check a directory for missing sequentially numbered images."
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
        "-s", "--start", 
        type=int, 
        default=1, 
        help="The starting number of your sequence (default: 1)."
    )
    parser.add_argument(
        "-e", "--end", 
        type=int, 
        default=3000, 
        help="The ending number of your sequence (default: 3000)."
    )
    parser.add_argument(
        "-n", "--digits", 
        type=int, 
        default=4, 
        help="The number of digits in the filename for display purposes (default: 4)."
    )
    
    args = parser.parse_args()
    
    find_missing_images(args.dir, args.prefix, args.start, args.end, args.digits)