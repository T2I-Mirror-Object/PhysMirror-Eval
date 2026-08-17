import argparse
import csv
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import pyiqa
except ImportError:
    raise ImportError("Please install pyiqa: pip install pyiqa")

AVAILABLE_METRICS = ["clipiqa", "maniqa", "musiq"]

HARDCODED_DIRS = [
    "flux_baseline",
    "sdxl_baseline",
    "images_flux_omini",
    "images_seg2any",
    "images_flux_synmirror_lora_depth",
]

SCRIPT_DIR = Path(__file__).parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute no-reference IQA scores (CLIP-IQA, MANIQA, MUSIQ) "
                    "for images in model subdirectories."
    )
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=AVAILABLE_METRICS,
        choices=AVAILABLE_METRICS,
        help=f"IQA metrics to compute. Default: all ({', '.join(AVAILABLE_METRICS)})."
    )
    parser.add_argument(
        "--output_csv", type=str, default="iqa_results.csv",
        help="Path to save per-image results CSV. Default: iqa_results.csv"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--model_dirs", type=str, nargs="+",
        default=HARDCODED_DIRS,
        help="Subdirectory names to evaluate (relative to script location). "
             f"Default: {HARDCODED_DIRS}"
    )
    return parser.parse_args()


def get_image_files(directory: Path):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted(f.name for f in directory.iterdir()
                  if f.suffix.lower() in valid_ext)


def load_done_models(out_path: Path) -> set:
    """Return set of model names already fully saved in the output CSV."""
    if not out_path.exists():
        return set()
    try:
        df = pd.read_csv(out_path)
        done = set(df["model"].unique())
        print(f"Resuming: found {len(done)} already-finished model(s) in {out_path}: {done}\n")
        return done
    except Exception as e:
        print(f"Warning: could not read existing CSV ({e}), starting fresh.\n")
        return set()


def append_rows_to_csv(out_path: Path, rows: list, metrics: list):
    """Append a list of rows to CSV (write header only if file doesn't exist yet)."""
    fieldnames = ["model", "image"] + metrics
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_dirs = [SCRIPT_DIR / name for name in args.model_dirs]

    print(f"Script directory : {SCRIPT_DIR}")
    print(f"Models to evaluate ({len(model_dirs)}): {[d.name for d in model_dirs]}")
    print(f"Metrics          : {args.metrics}")
    print(f"Device           : {args.device}")
    print(f"Output CSV       : {out_path}\n")

    # Resume: skip already-finished models
    done_models = load_done_models(out_path)

    # Load IQA metric models once
    print("Loading IQA models...")
    metric_fns = {}
    for metric_name in args.metrics:
        print(f"  [{metric_name}] loading...")
        metric_fns[metric_name] = pyiqa.create_metric(metric_name, device=args.device)
    print()

    error_log = []

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping.")
            continue

        model_name = model_dir.name

        if model_name in done_models:
            print(f"Skipping [{model_name}] — already done.")
            continue

        image_files = get_image_files(model_dir)
        if not image_files:
            print(f"No images found in {model_dir}, skipping.")
            continue

        print(f"Processing [{model_name}] — {len(image_files)} images")

        model_rows = []
        model_stats = {m: [] for m in args.metrics}

        for fname in tqdm(image_files, desc=model_name, ncols=80):
            img_path = str(model_dir / fname)
            row = {"model": model_name, "image": fname}
            had_error = False

            for metric_name, metric_fn in metric_fns.items():
                try:
                    with torch.no_grad():
                        score = metric_fn(img_path)
                    score_val = float(score.item() if hasattr(score, "item") else score)
                    row[metric_name] = round(score_val, 6)
                    model_stats[metric_name].append(score_val)
                except Exception as e:
                    tqdm.write(f"  [SKIP] {fname} | {metric_name} | {type(e).__name__}: {e}")
                    row[metric_name] = None
                    had_error = True

            if had_error:
                error_log.append(f"{model_name}/{fname}")

            model_rows.append(row)

        # Save this model's rows to CSV immediately
        append_rows_to_csv(out_path, model_rows, args.metrics)
        print(f"  -> Saved {len(model_rows)} rows for [{model_name}] to {out_path}")

    # Build and print summary from full CSV
    if out_path.exists():
        df = pd.read_csv(out_path)
        print("\n" + "=" * 65)
        print("SUMMARY — Average No-Reference IQA Scores per Model")
        print("=" * 65)

        summary_rows = []
        order = [d.name for d in model_dirs]
        for model_name in order:
            subset = df[df["model"] == model_name]
            if subset.empty:
                continue
            row = {"Model": model_name}
            for metric in args.metrics:
                if metric in subset.columns:
                    scores = subset[metric].dropna().tolist()
                    row[metric.upper()] = f"{sum(scores) / len(scores):.4f}" if scores else "N/A"
                else:
                    row[metric.upper()] = "N/A"
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        print(summary_df.to_string(index=False))

        summary_path = out_path.with_name(out_path.stem + "_summary" + out_path.suffix)
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")

    if error_log:
        print(f"\n[WARNING] {len(error_log)} image(s) were skipped due to errors:")
        for p in error_log:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
