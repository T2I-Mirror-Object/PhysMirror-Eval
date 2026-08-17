import logging
from typing import Tuple, List, Dict, Any
from PIL import Image
import numpy as np
import torch
from lang_sam import LangSAM

from src.stage1_segmentation.config import SegmentationConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MirrorSegmentationProcessor:
    """
    Handles the segmentation of an object and its mirror reflection
    using the Lang-Segment-Anything model.
    """

    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.model = self._load_model()

    def _load_model(self) -> LangSAM:
        """Initializes the LangSAM model."""
        logger.info(f"Loading LangSAM model ({self.config.model_type})...")
        try:
            # Checkpoint handling is delegated to the library defaults
            model = LangSAM(
                sam_type=self.config.model_type,
                gdino_model_ckpt_path=self.config.gdino_path,
                gdino_processor_ckpt_path=self.config.gdino_path
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def load_image(self, image_path: str) -> Image.Image:
        """Loads and converts image to RGB."""
        try:
            image_pil = Image.open(image_path).convert("RGB")
            return image_pil
        except Exception as e:
            logger.error(f"Could not load image at {image_path}: {e}")
            raise

    def process_image(self, image_pil: Image.Image, object_name: str) -> Dict[str, np.ndarray]:
        """
        Segment the image based on the object name.
        
        Args:
            image_pil: The input image.
            object_name: The name of the object (e.g., "coffee mug").
            
        Returns:
            result_dict: Dictionary containing 'masks', 'boxes', 'scores'.
                         All values are np.ndarray.
        """
        # Prompt includes both the object and its reflection
        text_prompt = f"{object_name}. reflection of {object_name} in the mirror."
        
        logger.info(f"Predicting with prompt: '{text_prompt}'")
        
        # Run inference in batch mode (required by the library)
        results = self.model.predict(
            [image_pil], 
            [text_prompt],
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold
        )
        
        # Extract the result for the first (and only) image
        # Result is a dict: {'boxes': ..., 'scores': ..., 'masks': ..., 'mask_scores': ...}
        result = results[0]
        
        # Check if masks were found. The library returns empty lists/arrays if nothing is found.
        # We need to safely handle cases where 'masks' might be empty or None depending on exact library behavior
        masks = result.get("masks", [])
        
        if len(masks) == 0:
            logger.warning("No masks detected.")
        else:
            logger.info(f"Detected {len(masks)} instances.")

        return result