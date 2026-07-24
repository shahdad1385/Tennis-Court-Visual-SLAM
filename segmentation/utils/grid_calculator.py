import cv2
import numpy as np
from typing import List, Tuple, Optional


class GridCalculator:
    """
    Calculates grid distances from the robot base using segmented line masks
    and a homography matrix.
    """

    def __init__(self, homography_matrix: Optional[np.ndarray] = None, 
                 ground_roi: Tuple[float, float, float, float] = (-2.0, 2.0, 0.0, 10.0)):
        """
        Args:
            homography_matrix: 3x3 matrix mapping image to bird's-eye view.
            ground_roi: (x_min, x_max, y_min, y_max) in meters.
        """
        self.H = homography_matrix
        self.roi = ground_roi

    def set_homography(self, H: np.ndarray):
        self.H = H

    def extract_line_contours(self, binary_mask: np.ndarray, min_area: int = 100) -> List[np.ndarray]:
        """
        Extract contours from a binary mask of pitch lines.

        Args:
            binary_mask: Binary mask (0 or 255) where 255 represents pitch lines.
            min_area: Minimum contour area to keep.

        Returns:
            List of contours (each is a (N, 1, 2) array of pixel coords).
        """
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in contours if cv2.contourArea(c) > min_area]

    def mask_to_birdseye(self, binary_mask: np.ndarray, output_size: Tuple[int, int] = (400, 1000)) -> np.ndarray:
        """
        Warp a segmentation mask to bird's-eye view using the stored homography.

        Args:
            binary_mask: Input mask (H, W).
            output_size: (width, height) of the output bird's-eye image.

        Returns:
            Warped bird's-eye mask.
        """
        if self.H is None:
            raise ValueError("Homography matrix not set. Call set_homography() first.")
        
        return cv2.warpPerspective(binary_mask, self.H, output_size)

    def estimate_distances_from_mask(self, binary_mask: np.ndarray, 
                                     output_size: Tuple[int, int] = (400, 1000)) -> dict:
        """
        Estimate grid distances from a binary mask.

        Args:
            binary_mask: Binary mask (H, W) where 255 represents pitch lines.
            output_size: (width, height) for bird's-eye transformation.

        Returns:
            Dict containing:
                - "warped_mask": The bird's-eye view of the mask.
                - "line_contours": List of contours in the warped view.
                - "metric_lines": List of ((mx1, my1), (mx2, my2)) metric coordinates.
        """
        if self.H is None:
            return {"warped_mask": None, "line_contours": [], "metric_lines": []}

        # Warp the mask to bird's-eye view
        warped_mask = self.mask_to_birdseye(binary_mask, output_size)
        
        # Clean up the warped mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        warped_mask = cv2.dilate(warped_mask, kernel, iterations=1)

        # Extract contours
        contours = self.extract_line_contours(warped_mask)
        
        # Convert pixel coordinates to metric
        x_min, x_max, y_min, y_max = self.roi
        w, h = output_size
        metric_lines = []

        for contour in contours:
            for i in range(len(contour) - 1):
                x1, y1 = contour[i][0]
                x2, y2 = contour[i+1][0]
                
                # Linear mapping from pixel to metric
                m_x1 = (x1 / w) * (x_max - x_min) + x_min
                m_y1 = (y1 / h) * (y_max - y_min) + y_min
                m_x2 = (x2 / w) * (x_max - x_min) + x_min
                m_y2 = (y2 / h) * (y_max - y_min) + y_min
                
                metric_lines.append(((float(m_x1), float(m_y1)), (float(m_x2), float(m_y2))))

        return {
            "warped_mask": warped_mask,
            "line_contours": contours,
            "metric_lines": metric_lines
        }
