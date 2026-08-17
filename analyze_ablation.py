import argparse
import json
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze MCS Ablation Results")
    
    parser.add_argument("--results_dir", type=str, default="ablation_results",
                        help="Directory containing the JSON result files.")
    parser.add_argument("--baseline_k", type=int, default=10,
                        help="The baseline K value to compare against.")
    parser.add_argument("--baseline_pi", type=float, default=0.5,
                        help="The baseline pi value to compare against.")
    parser.add_argument("--k_values", type=int, nargs='+', default=[5, 10, 15, 20],
                        help="List of Top-K values to analyze.")
    parser.add_argument("--pi_values", type=float, nargs='+', default=[0.4, 0.5, 0.6, 0.7],
                        help="List of pi values to analyze.")
    parser.add_argument("--output_plot", type=str, default="ablation_study_results.png",
                        help="Path to save the generated plot.")
    
    return parser.parse_args()

def load_scores(filepath, filename_order=None):
    """Loads scores from JSON and optionally enforces a specific filename order."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found {filepath}")
        return None
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    score_dict = {item['filename']: item['final_score'] for item in data}
    
    if filename_order:
        # Return a list of scores matching the exact order of the baseline
        return [score_dict.get(f, 0.0) for f in filename_order]
    else:
        # Return dict and the order of filenames
        filenames = list(score_dict.keys())
        return [score_dict[f] for f in filenames], filenames

def main():
    args = parse_args()
    
    # 1. Load the baseline
    baseline_file = os.path.join(args.results_dir, f"mcs_results_K{args.baseline_k}_pi{args.baseline_pi}.json")
    print(f"Loading baseline from: {baseline_file}")
    
    baseline_result = load_scores(baseline_file)
    if not baseline_result:
        print("Baseline file is missing. Cannot proceed.")
        return
        
    baseline_array, filenames_order = baseline_result

    # 2. Analyze Varying K (pi fixed at baseline)
    k_correlations = []
    valid_k_values = []
    
    for k in args.k_values:
        test_file = os.path.join(args.results_dir, f"mcs_results_K{k}_pi{args.baseline_pi}.json")
        test_array = load_scores(test_file, filenames_order)
        
        if test_array:
            corr, _ = spearmanr(baseline_array, test_array)
            k_correlations.append(corr)
            valid_k_values.append(k)

    # 3. Analyze Varying Pi (K fixed at baseline)
    pi_correlations = []
    valid_pi_values = []
    
    for pi in args.pi_values:
        test_file = os.path.join(args.results_dir, f"mcs_results_K{args.baseline_k}_pi{pi}.json")
        test_array = load_scores(test_file, filenames_order)
        
        if test_array:
            corr, _ = spearmanr(baseline_array, test_array)
            pi_correlations.append(corr)
            valid_pi_values.append(pi)

    # 4. Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot K
    ax1.plot(valid_k_values, k_correlations, marker='o', linestyle='-', color='b')
    ax1.set_title("Robustness to Top-K pairs")
    ax1.set_xlabel("Number of Pairs (K)")
    ax1.set_ylabel("Spearman Rank Correlation")
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(True)
    ax1.axvline(x=args.baseline_k, color='r', linestyle='--', label=f'Baseline (K={args.baseline_k})')
    ax1.legend()

    # Plot Pi
    ax2.plot(valid_pi_values, pi_correlations, marker='s', linestyle='-', color='g')
    ax2.set_title("Robustness to Similarity Threshold (π)")
    ax2.set_xlabel("Similarity Threshold (π)")
    ax2.set_ylabel("Spearman Rank Correlation")
    ax2.set_ylim([0.0, 1.05])
    ax2.grid(True)
    ax2.axvline(x=args.baseline_pi, color='r', linestyle='--', label=f'Baseline (π={args.baseline_pi})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=300)
    print(f"\nAnalysis complete! Saved plot to {args.output_plot}")

if __name__ == "__main__":
    main()