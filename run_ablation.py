import argparse
import subprocess
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run MCS Ablation Study Grid Search")
    
    parser.add_argument("--main_script", type=str, default="run_mcs.py",
                        help="Path to your main evaluation script.")
    parser.add_argument("--image_folder", type=str, required=True,
                        help="Path to folder containing input images.")
    parser.add_argument("--names_file", type=str, required=True,
                        help="Path to text file containing object names.")
    parser.add_argument("--output_dir", type=str, default="ablation_results",
                        help="Directory to save the resulting JSON files.")
    
    # Use nargs='+' to accept a list of values from the command line
    parser.add_argument("--k_values", type=int, nargs='+', default=[5, 10, 15, 20],
                        help="List of Top-K values to test.")
    parser.add_argument("--pi_values", type=float, nargs='+', default=[0.4, 0.5, 0.6, 0.7],
                        help="List of similarity threshold (pi) values to test.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Starting ablation runs. Output directory: {args.output_dir}")
    print(f"Testing K values: {args.k_values}")
    print(f"Testing Pi values: {args.pi_values}")

    for k in args.k_values:
        for pi in args.pi_values:
            print(f"\n--- Running K={k}, Pi={pi} ---")
            output_json = os.path.join(args.output_dir, f"mcs_results_K{k}_pi{pi}.json")
            
            if os.path.exists(output_json):
                print(f"Result already exists at {output_json}. Skipping.")
                continue

            cmd = [
                "python", args.main_script,
                "--image_folder", args.image_folder,
                "--names_file", args.names_file,
                "--top_k", str(k),
                "--sim_threshold", str(pi),
                "--output_json", output_json
            ]
            
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error occurred while running K={k}, Pi={pi}: {e}")

    print("\nAll ablation runs complete!")

if __name__ == "__main__":
    main()