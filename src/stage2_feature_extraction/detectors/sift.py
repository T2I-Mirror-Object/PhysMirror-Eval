# stage2_feature_extraction/detectors/sift.py
import cv2
import numpy as np
from typing import Tuple, List

from ..base import FeatureDetector

class SIFTFeatureDetector(FeatureDetector):
    """
    Standard OpenCV SIFT Wrapper.
    """
    def __init__(self, n_features: int = 0):
        self.sift = cv2.SIFT_create(nfeatures=n_features)

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        # SIFT works best on Grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        kps, des = self.sift.detectAndCompute(gray, None)
        
        # Safety check for empty images
        if des is None:
            des = np.array([])
            
        return list(kps), des