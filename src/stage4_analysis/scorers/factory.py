from .base import MirrorScorer
from .centroid_distance import CentroidDistanceScorer

def get_scorer(scorer_name: str, **kwargs) -> MirrorScorer:
    """Factory method."""
    if scorer_name == 'centroid':
        scale = kwargs.get('normalization_scale', 5000.0)
        return CentroidDistanceScorer(normalization_scale=scale)
    else:
        # Default fallback or error
        print(f"Warning: Unknown scorer '{scorer_name}'. Defaulting to Centroid.")
        return CentroidDistanceScorer()