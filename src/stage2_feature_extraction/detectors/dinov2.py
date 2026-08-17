# stage2_feature_extraction/detectors/dinov2.py
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from typing import Tuple, List, Literal

from ..base import FeatureDetector

class DINOv2Detector(FeatureDetector):
    """
    Adapts the DINOv2 Dense Feature Extractor to the standard FeatureDetector interface.
    
    Treats every patch in the Vision Transformer grid as a 'Keypoint'.
    """

    # Model registry
    MODELS = {
        'small': 'dinov2_vits14',
        'base':  'dinov2_vitb14',
        'large': 'dinov2_vitl14',
    }

    def __init__(self, model_type: Literal['small', 'base', 'large'] = 'base', 
                 input_size: int = 448, 
                 device: str = 'cuda'):
        """
        Args:
            model_type: Size of the model backbone.
            input_size: Resolution to resize input images to (must be multiple of 14).
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.input_size = input_size
        self.patch_size = 14  # Fixed DINOv2 patch size
        
        print(f"[INFO] Initializing DINOv2 ({model_type}) on {self.device}...")
        
        # Load Backbone
        model_name = self.MODELS[model_type]
        # Bypass GitHub API fork-validation to avoid 504 timeouts on Kaggle
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(self.device)
        self.model.eval()

        # Standard Normalization
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Extracts dense features and wraps them as KeyPoints.
        """
        # Robust Image Loading (Fixing the 4-channel crash)
        if isinstance(image, np.ndarray):
            # Convert Numpy -> PIL
            # We assume input is BGR if it came from OpenCV, or RGB/RGBA if from PIL
            if image.shape[2] == 3:
                # Likely BGR from OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # Likely RGBA (this was causing your crash)
                # We drop the alpha channel simply by converting to RGB
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            pil_img = Image.fromarray(image)
        else:
            # If it's already a PIL image
            pil_img = image

        # FORCE RGB: This ensures we always have 3 channels (dropping Alpha if present)
        pil_img = pil_img.convert("RGB")
        
        original_size = pil_img.size # (W, H)
        
        # Now this will always produce a (3, H, W) tensor
        img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            features_dict = self.model.forward_features(img_tensor)
            # (1, N_patches, Dim)
            raw_tokens = features_dict['x_norm_patchtokens']
            
            # L2 Normalize (Crucial for cosine similarity matching later)
            raw_tokens = F.normalize(raw_tokens, dim=-1, p=2)

        # Convert to Keypoints & Descriptors
        # Flattened tokens are our descriptors
        descriptors = raw_tokens.squeeze(0).cpu().numpy() # (N, Dim)
        
        # Generate Grid Keypoints
        keypoints = self._generate_grid_keypoints(
            num_patches=descriptors.shape[0], 
            original_size=original_size
        )

        return keypoints, descriptors

    def _generate_grid_keypoints(self, num_patches: int, original_size: Tuple[int, int]) -> List[cv2.KeyPoint]:
        """
        Map the implicit 1D patch sequence back to 2D image coordinates.
        """
        grid_dim = int(np.sqrt(num_patches)) # e.g., 32 for 448px input
        
        # Calculate scaling factor from Model Input (448) back to Original Image
        # Note: We resized the image to square (input_size, input_size)
        # So we need to map the square grid back to potentially rectangular original image
        orig_w, orig_h = original_size
        
        step_x = orig_w / grid_dim
        step_y = orig_h / grid_dim
        
        keypoints = []
        
        for i in range(num_patches):
            # Row-major order
            row = i // grid_dim
            col = i % grid_dim
            
            # Center of the patch
            x = (col + 0.5) * step_x
            y = (row + 0.5) * step_y
            
            kpt = cv2.KeyPoint(x=float(x), y=float(y), size=float(step_x))
            keypoints.append(kpt)
            
        return keypoints