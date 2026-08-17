from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Any

class MirrorScorer(ABC):
    """
    Abstract Interface for Mirror Symmetry Evaluation Strategies.
    """
    
    @abstractmethod
    def calculate_score(self, lines: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        """
        Computes a score describing how well the lines support the mirror hypothesis.

        Args:
            lines: List of line segments (start_point, end_point).

        Returns:
            Dict containing 'final_score' (0.0-1.0) and other debug metrics.
        """
        pass