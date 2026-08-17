# stage3_feature_matching/base.py
from abc import ABC, abstractmethod
from typing import List
import numpy as np
import cv2

class FeatureMatcher(ABC):
    """
    Abstract base class for feature matching algorithms.
    """

    @abstractmethod
    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """
        Computes matches between two sets of descriptors.

        Args:
            descriptors1: (N, D) array from Source image.
            descriptors2: (M, D) array from Target image.

        Returns:
            List of cv2.DMatch objects containing queryIdx, trainIdx, and distance.
        """
        pass