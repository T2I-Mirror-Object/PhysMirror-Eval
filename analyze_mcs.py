import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MCS JSON report and generate contour diagrams.")
    
    parser.add_argument("--input_json", type=str, required=True,
                        help="Path to the input evaluation_report.json file.")
    parser.add_argument("--contour_dir", type=str, default="contour_diagrams",
                        help="Directory to save the generated contour diagrams.")
    parser.add_argument("--output_scores", type=str, default="average_scores.json",
                        help="Path to save the summary of average scores.")
    
    return parser.parse_args()

def calculate_average_score(data_subset, name):
    """Calculates and prints the average score for a slice of the dataset."""
    if not data_subset:
        print(f"[{name}] No data found.")
        return 0.0
        
    scores = [item['final_score'] for item in data_subset if 'final_score' in item]
    if not scores:
        return 0.0
        
    avg_score = sum(scores) / len(scores)
    
    print(f"[{name}] Images: {len(scores)} | Average MCS: {avg_score:.5f}")
    return avg_score

def generate_contour_diagrams_and_fix_scores(data, output_dir):
    """Generates contour plots and fixes erroneous scores in place."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nGenerating contour diagrams in: {output_dir}")
    
    fixed_count = 0
    
    for item in data:
        filename = item.get('filename', 'unknown.jpg')
        intersections = np.array(item.get('intersections', []))
        centroid = item.get('centroid', [])
        
        # 1. Identify and fix the logic flaw immediately
        if len(intersections) < 3:
            print(f"Fixing {filename}: Not enough intersections ({len(intersections)}). Score set to 0.")
            item['final_score'] = 0.0
            item['error_status'] = "not enough intersection points"
            fixed_count += 1
            continue
            
        score = item.get('final_score', 0.0)
        
        x = intersections[:, 0]
        y = intersections[:, 1]
        
        plt.figure(figsize=(8, 8))
        
        try:
            sns.kdeplot(
                x=x, y=y, 
                cmap="mako",
                fill=True, 
                thresh=0.05,
                levels=10,
                alpha=0.8
            )
        except Exception as e:
            print(f"Could not compute KDE for {filename}: {e}")
            plt.close()
            continue

        plt.scatter(
            x, y, 
            color='white', s=15, edgecolor='black', 
            alpha=0.6, label='Intersections'
        )
        
        if centroid:
            plt.scatter(
                centroid[0], centroid[1], 
                color='red', marker='*', s=250, edgecolor='white', 
                label='Centroid (Effective VP)'
            )
            
        plt.title(f"Intersection Density: {filename}\nMCS: {score:.4f}", fontsize=14)
        plt.xlabel("X Coordinate (Pixels)")
        plt.ylabel("Y Coordinate (Pixels)")
        
        plt.gca().invert_yaxis()
        plt.legend(loc="upper right")
        plt.grid(True, linestyle='--', alpha=0.3)
        
        out_path = os.path.join(output_dir, f"contour_{os.path.splitext(filename)[0]}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"\nFixed {fixed_count} items with insufficient intersection points.")
    return data

def main():
    args = parse_args()
    
    if not os.path.exists(args.input_json):
        print(f"Error: Could not find {args.input_json}")
        return

    # Load data
    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sort data by filename
    data.sort(key=lambda x: x.get('filename', ''))

    print("="*40)
    print("MIRROR CONSISTENCY SCORE (MCS) ANALYSIS")
    print("="*40)

    # STEP 1: Generate diagrams and fix the scores IN PLACE
    data = generate_contour_diagrams_and_fix_scores(data, args.contour_dir)

    # STEP 2: Overwrite the input JSON file with the corrected data
    print(f"\nOverwriting original JSON file to save corrected scores...")
    try:
        with open(args.input_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully updated {args.input_json}")
    except IOError as e:
        print(f"Error saving updated JSON: {e}")

    # STEP 3: Slice the dataset exactly as requested
    subset_1_obj = data[0:160]    # Indices 0 to 159
    subset_2_obj = data[160:260]  # Indices 160 to 259
    subset_3_obj = data[260:360]  # Indices 260 to 359

    # STEP 4: Calculate averages using the newly corrected data
    print("\nCalculating Average Scores:")
    summary_scores = {
        "1_Object_Average_MCS": calculate_average_score(subset_1_obj, "1-Object Prompts"),
        "2_Object_Average_MCS": calculate_average_score(subset_2_obj, "2-Object Prompts"),
        "3_Object_Average_MCS": calculate_average_score(subset_3_obj, "3-Object Prompts"),
        "Overall_Average_MCS": calculate_average_score(data, "Overall Dataset")
    }

    # Save the summary dictionary
    try:
        with open(args.output_scores, 'w', encoding='utf-8') as f:
            json.dump(summary_scores, f, indent=4)
        print(f"\nAverage scores successfully saved to: {args.output_scores}")
    except IOError as e:
        print(f"Error saving scores JSON: {e}")

    print(f"\nAll operations complete. Check the '{args.contour_dir}' folder for your diagrams.")

if __name__ == "__main__":
    main()