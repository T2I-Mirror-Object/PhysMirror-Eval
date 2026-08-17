import cv2
import numpy as np
from typing import List

class KeypointVisualizer:
    @staticmethod
    def draw_keypoints(image: np.ndarray, keypoints: List[cv2.KeyPoint], color=(0, 255, 0)) -> np.ndarray:
        """
        Draws keypoints on the image.
        """
        # Convert to BGR for OpenCV drawing if it's grayscale
        if len(image.shape) == 2:
            out_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4: # RGBA
            out_img = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            out_img = image.copy()

        # cv2.drawKeypoints can handle drawing, but doing it manually allows more control if needed
        # Flag DRAW_RICH_KEYPOINTS shows the size and orientation of the keypoints
        out_img = cv2.drawKeypoints(out_img, keypoints, out_img, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        return out_img