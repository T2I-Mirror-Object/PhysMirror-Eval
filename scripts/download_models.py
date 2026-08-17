# download_models.py
import os
from huggingface_hub import snapshot_download

# Create checkpoints directory
save_dir = "checkpoints/grounding-dino"
os.makedirs(save_dir, exist_ok=True)

print(f"Downloading GroundingDINO to {save_dir}...")
snapshot_download(
    repo_id="IDEA-Research/grounding-dino-base",
    local_dir=save_dir,
    local_dir_use_symlinks=False  # Important: Actual files, not symlinks
)
print("Download complete.")