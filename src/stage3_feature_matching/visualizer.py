import cv2
import numpy as np
from typing import List

class MatchVisualizer:
    @staticmethod
    def draw_matches(
        img1: np.ndarray, kp1: List[cv2.KeyPoint],
        img2: np.ndarray, kp2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        matches_thickness: int = 1
    ) -> np.ndarray:
        """
        Draws lines connecting the matched keypoints between two images.
        """
        # Ensure images are compatible for drawing (convert grayscale to BGR if needed)
        def _ensure_bgr(img):
            if len(img.shape) == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img

        vis_img1 = _ensure_bgr(img1)
        vis_img2 = _ensure_bgr(img2)

        # Draw matches
        # flags=2 (DRAW_RICH_KEYPOINTS) is not used here to keep it cleaner
        output_img = cv2.drawMatches(
            vis_img1, kp1,
            vis_img2, kp2,
            matches,
            None, # outImg
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0), # Green lines
            singlePointColor=None
        )
        
        return output_img