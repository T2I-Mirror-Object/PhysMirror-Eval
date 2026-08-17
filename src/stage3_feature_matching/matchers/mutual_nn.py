# stage3_feature_matching/matchers/mutual_nn.py
import numpy as np
import cv2
import torch
from typing import List

from ..base import FeatureMatcher

class MutualNNMatcher(FeatureMatcher):
    """
    Implements Mutual Nearest Neighbor (Best Buddies) matching.
    Ideal for semantic descriptors (DINOv2) where 1-to-1 bijective mapping is required.
    """

    def __init__(self, similarity_threshold: float = 0.5, use_cuda: bool = True):
        """
        Args:
            similarity_threshold (float): Minimum cosine similarity (0 to 1) to keep a match.
            use_cuda (bool): Whether to use GPU for matrix multiplication.
        """
        self.threshold = similarity_threshold
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def match(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """
        Computes bidirectional matches and returns only those that are mutual nearest neighbors.
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []

        # 1. Convert to PyTorch for fast Matrix Multiplication
        # Ensure input is float32 to avoid type mismatch during matrix multiplication if mixed
        desc1 = torch.from_numpy(descriptors1).float().to(self.device)
        desc2 = torch.from_numpy(descriptors2).float().to(self.device)

        # 2. Compute Cosine Similarity Matrix
        # Note: We assume descriptors are already L2 Normalized from Stage 2.
        # Matrix: (N, M)
        sim_matrix = torch.matmul(desc1, desc2.T)

        # 3. Find Best Matches in Both Directions
        # val_12: Best score for each row (Source -> Target)
        # idx_12: Index of best match in Target for each Source item
        val_12, idx_12 = torch.max(sim_matrix, dim=1)
        
        # val_21: Best score for each col (Target -> Source)
        # idx_21: Index of best match in Source for each Target item
        val_21, idx_21 = torch.max(sim_matrix, dim=0)

        # 4. Filter for Mutual Consistency (The "Handshake")
        # CPU conversion for indexing loop
        idx_12_np = idx_12.cpu().numpy()
        idx_21_np = idx_21.cpu().numpy()
        scores_np = val_12.float().cpu().numpy()

        matches = []
        num_source_points = descriptors1.shape[0]

        for i in range(num_source_points):
            # Who is the best partner for 'i'?
            j = idx_12_np[i]

            # Does 'j' also think 'i' is the best partner?
            if idx_21_np[j] == i:
                score = scores_np[i]
                
                # Check Threshold
                if score > self.threshold:
                    # Convert Similarity to Distance (1 - Sim) for DMatch compatibility
                    dist = 1.0 - score
                    
                    # Create standard OpenCV DMatch object
                    # queryIdx = index in descriptors1
                    # trainIdx = index in descriptors2
                    dmatch = cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=dist)
                    matches.append(dmatch)

        return matches