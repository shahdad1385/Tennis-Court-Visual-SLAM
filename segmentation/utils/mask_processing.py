import cv2
import numpy as np
from typing import Tuple


class MaskProcessor:
    """
    Post-processing utilities for segmentation masks.
    Includes color-based masking and morphological operations.
    """

    # Class IDs
    CLASS_BACKGROUND = 0
    CLASS_PITCH_LINES = 1
    CLASS_POLES = 2

    # Colors for visualization (BGR)
    COLOR_PITCH_LINES = (0, 255, 0)  # Green
    COLOR_POLES = (0, 0, 255)        # Red

    @staticmethod
    def color_mask(mask: np.ndarray, class_id: int) -> np.ndarray:
        """
        Extract binary mask for a specific class ID.

        Args:
            mask: Raw segmentation mask (H, W).
            class_id: Class index to extract.

        Returns:
            Binary mask (H, W) with values 0 or 255.
        """
        return ((mask == class_id) * 255).astype(np.uint8)

    @staticmethod
    def morphological_cleanup(binary_mask: np.ndarray, 
                              kernel_size: int = 5, 
                              iterations: int = 2) -> np.ndarray:
        """
        Apply dilation and erosion to clean up edge noise.

        Args:
            binary_mask: Input binary mask (0 or 255).
            kernel_size: Size of the structuring element.
            iterations: Number of times to apply each operation.

        Returns:
            Cleaned binary mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Erosion to remove small noise
        cleaned = cv2.erode(binary_mask, kernel, iterations=iterations)
        # Dilation to restore shape
        cleaned = cv2.dilate(cleaned, kernel, iterations=iterations)
        
        return cleaned

    @staticmethod
    def create_overlay(frame: np.ndarray, mask: np.ndarray, class_id: int, alpha: float = 0.4) -> np.ndarray:
        """
        Create a colored overlay of the mask on the original frame.

        Args:
            frame: Original BGR image.
            mask: Segmentation mask.
            class_id: Class ID to colorize.
            alpha: Transparency factor.

        Returns:
            Overlay image.
        """
        overlay = frame.copy()
        binary = MaskProcessor.color_mask(mask, class_id)
        
        if class_id == MaskProcessor.CLASS_PITCH_LINES:
            color = MaskProcessor.COLOR_PITCH_LINES
        elif class_id == MaskProcessor.CLASS_POLES:
            color = MaskProcessor.COLOR_POLES
        else:
            color = (128, 128, 128)

        overlay[binary == 255] = color
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
