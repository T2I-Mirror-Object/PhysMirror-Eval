import argparse
import csv
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as T

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    raise ImportError("Please install torchmetrics[image]: pip install torchmetrics[image]")

HARDCODED_DIRS = [
    "flux_baseline",
    "sdxl_baseline",
    "images_flux_omini",
    "images_seg2any",
    "images_flux_synmirror_lora_depth",
]

GROUND_TRUTH_DIR = "ground_truth"
SCRIPT_DIR = Path(__file__).parent

RESIZE_TRANSFORM = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Lambda(lambda x: (x * 255).to(torch.uint8)),
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute FID scores between model output directories and a ground truth directory."
    )
    parser.add_argument(
        "--model_dirs", type=str, nargs="+", default=HARDCODED_DIRS,
        help=f"Subdirectory names to evaluate (relative to script location). Default: {HARDCODED_DIRS}"
    )
    parser.add_argument(
        "--ground_truth_dir", type=str, default=GROUND_TRUTH_DIR,
        help=f"Ground truth directory name (relative to script location). Default: {GROUND_TRUTH_DIR}"
    )
    parser.add_argument(
        "--output_csv", type=str, default="fid_results.csv",
        help="Path to save FID results CSV. Default: fid_results.csv"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (cuda or cpu)."
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size for loading images. Default: 32"
    )
    return parser.parse_args()


def get_image_files(directory: Path):
    valid_ext = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return sorted(f for f in directory.iterdir() if f.suffix.lower() in valid_ext)


def load_images_as_batches(image_files: list, batch_size: int, device: str):
    """Yield batches of uint8 tensors (B, 3, 299, 299) from a list of image paths."""
    batch = []
    for img_path in image_files:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = RESIZE_TRANSFORM(img)  # (3, 299, 299) uint8
            batch.append(tensor)
        except Exception as e:
            tqdm.write(f"  [SKIP] {img_path.name}: {e}")
            continue

        if len(batch) == batch_size:
            yield torch.stack(batch).to(device)
            batch = []

    if batch:
        yield torch.stack(batch).to(device)


def compute_fid(gt_files: list, model_files: list, device: str, batch_size: int) -> float:
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    fid.reset()

    with torch.no_grad():
        for batch in load_images_as_batches(gt_files, batch_size, device):
            fid.update(batch, real=True)

        for batch in load_images_as_batches(model_files, batch_size, device):
            fid.update(batch, real=False)

    return float(fid.compute().item())


def main():
    args = parse_args()

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_dir = SCRIPT_DIR / args.ground_truth_dir
    model_dirs = [SCRIPT_DIR / name for name in args.model_dirs]

    print(f"Script directory   : {SCRIPT_DIR}")
    print(f"Ground truth dir   : {gt_dir}")
    print(f"Models to evaluate : {[d.name for d in model_dirs]}")
    print(f"Device             : {args.device}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Output CSV         : {out_path}\n")

    if not gt_dir.exists():
        print(f"Error: ground truth directory not found: {gt_dir}")
        return

    gt_files = get_image_files(gt_dir)
    if not gt_files:
        print(f"Error: no images found in ground truth directory: {gt_dir}")
        return

    print(f"Ground truth images: {len(gt_files)}\n")

    results = []

    for model_dir in model_dirs:
        if not model_dir.exists():
            print(f"Warning: {model_dir} does not exist, skipping.")
            continue

        model_name = model_dir.name
        model_files = get_image_files(model_dir)

        if not model_files:
            print(f"Warning: no images found in {model_dir}, skipping.")
            continue

        print(f"Computing FID for [{model_name}] — {len(model_files)} images vs {len(gt_files)} ground truth images...")

        try:
            fid_score = compute_fid(gt_files, model_files, args.device, args.batch_size)
            print(f"  FID = {fid_score:.4f}")
            results.append({"model": model_name, "fid_score": round(fid_score, 4)})
        except Exception as e:
            print(f"  [ERROR] {model_name}: {type(e).__name__}: {e}")
            results.append({"model": model_name, "fid_score": None})

    # Write CSV
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "fid_score"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {out_path}")

    # Print summary table
    if results:
        print("\n" + "=" * 50)
        print("SUMMARY — FID Scores (lower = more similar to GT)")
        print("=" * 50)
        df = pd.DataFrame(results)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
