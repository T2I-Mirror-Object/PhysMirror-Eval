import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Any
import cv2

class ImageUtils:
    """
    Utility class for image manipulation: drawing boxes, annotations, and cropping.
    """

    @staticmethod
    def draw_boxes(image: Image.Image, boxes: np.ndarray, scores: Optional[np.ndarray] = None) -> Image.Image:
        """
        Draws bounding boxes and optional scores on the image.
        
        Args:
            image: Original PIL Image.
            boxes: Numpy array of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
            scores: Optional numpy array of shape (N,) containing confidence scores.
            
        Returns:
            Annotated PIL Image.
        """
        # Create a copy to avoid modifying the original image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(annotated_image)
        
        # Load a default font (optional, falls back to default if unavailable)
        try:
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        if len(boxes) == 0:
            return annotated_image

        for i, box in enumerate(boxes):
            # Box is expected to be [x_min, y_min, x_max, y_max]
            # Ensure coordinates are within image bounds and integers
            x0, y0, x1, y1 = box.astype(int)
            
            # Draw rectangle
            # Outline color: Red (255, 0, 0) with width 3
            draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
            
            # Draw label if scores are provided
            if scores is not None and i < len(scores):
                score = scores[i]
                text = f"{score:.2f}"
                
                # Draw text background for readability
                text_bbox = draw.textbbox((x0, y0), text, font=font)
                draw.rectangle(text_bbox, fill="red")
                draw.text((x0, y0), text, fill="white", font=font)

        return annotated_image

    @staticmethod
    def crop_objects(image: Image.Image, boxes: np.ndarray) -> List[Image.Image]:
        """
        Crops the image based on the provided bounding boxes.
        
        Args:
            image: Original PIL Image.
            boxes: Numpy array of shape (N, 4) in [x_min, y_min, x_max, y_max] format.
            
        Returns:
            List of PIL Images (crops).
        """
        crops = []
        if len(boxes) == 0:
            return crops

        width, height = image.size

        for box in boxes:
            # Cast to int
            x0, y0, x1, y1 = box.astype(int)
            
            # Clamp coordinates to image boundaries to prevent errors
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(width, x1)
            y1 = min(height, y1)
            
            # Perform crop
            crop = image.crop((x0, y0, x1, y1))
            crops.append(crop)
            
        return crops

    @staticmethod
    def extract_object_with_mask(
        image: Image.Image, 
        mask: np.ndarray
    ) -> Tuple[Image.Image, Optional[Tuple[int, int, int, int]]]:
        """
        Applies mask and crops to content. 
        explicitly blacks out the background to prevent "ghost" pixels.
        """
        # 1. Resize mask if needed
        mask_h, mask_w = mask.shape[-2:]
        img_w, img_h = image.size
        if (mask_h, mask_w) != (img_h, img_w):
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize((img_w, img_h), resample=Image.NEAREST)
            mask = np.array(mask_pil)

        # 2. Prepare Mask
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        mask = np.squeeze(mask)
        
        # Convert image to Numpy to manipulate pixels
        image_np = np.array(image.convert("RGB"))
        
        # Create a boolean mask (True where mask > 0)
        binary_mask = mask > 0
        
        # Black out the background (where mask is 0)
        # We use broadcasting: image_np is (H,W,3), binary_mask is (H,W)
        # We multiply by the mask (0 or 1) to zero out background pixels
        image_np = image_np * binary_mask[:, :, np.newaxis]
        
        # Convert back to PIL
        image_cleaned = Image.fromarray(image_np.astype(np.uint8))

        # 3. Create RGBA
        image_rgba = image_cleaned.convert("RGBA")
        mask_image = Image.fromarray(mask, mode='L')
        image_rgba.putalpha(mask_image)

        # 4. Get Bounding Box and Crop
        bbox = mask_image.getbbox()
        
        if bbox:
            return image_rgba.crop(bbox), bbox
        else:
            return image_rgba, None

    @staticmethod
    def map_point_to_original(point_in_crop: Tuple[int, int], crop_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Converts a point (x, y) from the cropped image coordinate system 
        to the original image coordinate system.
        """
        x_crop, y_crop = point_in_crop
        x_offset, y_offset, _, _ = crop_bbox
        return (x_crop + x_offset, y_crop + y_offset)

    @staticmethod
    def map_point_to_crop(point_in_original: Tuple[int, int], crop_bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Converts a point (x, y) from the original image coordinate system 
        to the cropped image coordinate system.
        """
        x_orig, y_orig = point_in_original
        x_offset, y_offset, _, _ = crop_bbox
        return (x_orig - x_offset, y_orig - y_offset)

    @staticmethod
    def filter_features_by_mask(keypoints: List[Any], descriptors: np.ndarray, pil_cutout: Image.Image) -> Tuple[List[Any], np.ndarray]:
        """
        Filters keypoints and descriptors to keep only those that fall on the foreground (Alpha > 128).
        
        Args:
            keypoints: List of cv2.KeyPoint objects.
            descriptors: Numpy array of descriptors (N, D).
            pil_cutout: PIL Image (RGBA) containing the object and alpha mask.
            
        Returns:
            Tuple(List[KeyPoint], np.ndarray): Filtered keypoints and descriptors.
        """
        # Convert PIL -> Numpy
        img_np = np.array(pil_cutout)
        
        # Validation: We need an Alpha channel (4 channels)
        if img_np.ndim != 3 or img_np.shape[2] != 4:
            # If no alpha channel exists, we cannot filter. Return everything.
            return keypoints, descriptors

        mask = img_np[:, :, 3] # Extract Alpha channel
        height, width = mask.shape

        valid_kps = []
        valid_descs = []

        for i, kp in enumerate(keypoints):
            x, y = int(kp.pt[0]), int(kp.pt[1])
            
            # 1. Check Bounds
            if 0 <= x < width and 0 <= y < height:
                # 2. Check Mask Value (Threshold 128 ensures we are on the object, not the edge feathering)
                if mask[y, x] > 128:
                    valid_kps.append(kp)
                    valid_descs.append(descriptors[i])

        # Return aligned data
        if len(valid_kps) == 0:
            return [], np.array([])
            
        return valid_kps, np.array(valid_descs)