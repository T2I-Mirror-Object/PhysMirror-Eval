import numpy as np
import math
from typing import List, Tuple, Dict, Any

from .base import MirrorScorer
from ..vanishing_point import VanishingPointAnalyzer

class CentroidDistanceScorer(MirrorScorer):
    """
    Cluster Tightness Scorer (Mean Distance Normalized).
    
    1. Finds intersections of ALL provided lines.
    2. Computes the Centroid (Mean) of these intersections.
    3. Score = exp(-MeanDistance / Scale)
    """
    
    def __init__(self, normalization_scale: float = 150.0):
        self.normalization_scale = normalization_scale

    def calculate_score(self, lines: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        
        # Validation
        if len(lines) < 3:
            return {
                'final_score': 0.0,
                'sum_distance': float('inf'),
                'mean_distance': float('inf'),
                'centroid': np.array([0.0, 0.0]),
                'scorer_type': 'CentroidDistance'
            }

        # Find Intersections of ALL lines
        intersections = VanishingPointAnalyzer.find_pairwise_intersections(lines)
        
        if len(intersections) == 0:
             return {
                'final_score': 0.0,
                'sum_distance': float('inf'),
                'scorer_type': 'CentroidDistance'
            }

        # Calculate Centroid (The effective Vanishing Point)
        centroid = np.mean(intersections, axis=0)

        # Calculate Distances (Tightness)
        diffs = intersections - centroid
        distances = np.linalg.norm(diffs, axis=1)
        
        sum_dist = float(np.sum(distances))
        mean_dist = float(np.mean(distances))

        # Normalize Score (0.0 to 1.0) using MEAN distance
        final_score = math.exp(-mean_dist / self.normalization_scale)

        return {
            'final_score': final_score,
            'sum_distance': sum_dist,
            'mean_distance': mean_dist,
            'centroid': centroid,
            'num_intersections': len(intersections),
            'scorer_type': 'CentroidDistance'
        }