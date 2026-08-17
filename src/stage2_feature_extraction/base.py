# stage2_feature_extraction/base.py
from abc import ABC, abstractmethod
from typing import Tuple, List, Any
import numpy as np

class FeatureDetector(ABC):
    """
    Abstract base class for feature detection algorithms.
    Unifies Classical (SIFT) and Deep (DINOv2) pipelines.
    """

    @abstractmethod
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[Any], np.ndarray]:
        """
        Args:
            image: Input image (H, W, 3) as numpy array (BGR or RGB).

        Returns:
            keypoints: List of cv2.KeyPoint objects (or equivalent objects with .pt attributes).
            descriptors: Numpy array of shape (N, D) where N is num_keypoints, D is dimension.
        """
        pass