# stage3_feature_matching/matchers/knn.py
import cv2
import numpy as np
from typing import List

from ..base import FeatureMatcher

class KNNFeatureMatcher(FeatureMatcher):
    """
    Standard Wrapper for OpenCV KNN Matching with Lowe's Ratio Test.
    """

    def __init__(self, norm_type=cv2.NORM_L2, ratio_threshold: float = 0.75):
        self.ratio_threshold = ratio_threshold
        # Create BFMatcher (Brute Force)
        self.matcher = cv2.BFMatcher(normType=norm_type, crossCheck=False)

    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[cv2.DMatch]:
        if descriptors1 is None or descriptors2 is None:
            return []
        if len(descriptors1) < 2 or len(descriptors2) < 2:
            return []

        # k=2 for Ratio Test
        raw_matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for m, n in raw_matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
                
        return good_matches