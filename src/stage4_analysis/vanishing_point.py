import numpy as np
from typing import List, Tuple

class VanishingPointAnalyzer:
    """
    Core Geometric Analysis: Line equations and Intersections.
    """

    @staticmethod
    def compute_line_equation(p1: np.ndarray, p2: np.ndarray) -> Tuple[float, float, float]:
        """Returns normalized (a, b, c) for line ax + by + c = 0."""
        x1, y1 = p1
        x2, y2 = p2
        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        norm = np.sqrt(a*a + b*b) + 1e-8
        return a/norm, b/norm, c/norm

    @staticmethod
    def find_pairwise_intersections(
        lines: List[Tuple[np.ndarray, np.ndarray]], 
        image_shape: Tuple[int, int] = None,
        margin_factor: float = 1.0
    ) -> np.ndarray:
        """
        Computes intersection points for every pair of lines using Cramer's Rule.
        """
        intersections = []
        num_lines = len(lines)
        eqs = [VanishingPointAnalyzer.compute_line_equation(l[0], l[1]) for l in lines]
        
        # Define bounds
        if image_shape:
            h, w = image_shape[:2]
            min_x, max_x = -w * margin_factor, w * (1 + margin_factor)
            min_y, max_y = -h * margin_factor, h * (1 + margin_factor)

        for i in range(num_lines):
            for j in range(i + 1, num_lines):
                a1, b1, c1 = eqs[i]
                a2, b2, c2 = eqs[j]
                
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-5: continue # Parallel
                
                x = (b1 * c2 - b2 * c1) / det
                y = (c1 * a2 - c2 * a1) / det
                
                if image_shape:
                    if not (min_x < x < max_x and min_y < y < max_y):
                        continue
                        
                intersections.append([x, y])
                
        return np.array(intersections)