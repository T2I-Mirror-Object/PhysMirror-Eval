from dataclasses import dataclass
import os

@dataclass
class SegmentationConfig:
    """Configuration for the LangSAM model and segmentation process."""
    model_type: str = "sam2.1_hiera_large"       # SAM model type
    box_threshold: float = 0.3      # Threshold for object detection
    text_threshold: float = 0.25    # Threshold for text alignment
    device: str = "cuda"            # 'cuda' or 'cpu'
    gdino_path: str = os.path.join("checkpoints", "grounding-dino")