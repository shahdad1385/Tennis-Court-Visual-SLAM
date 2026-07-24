import numpy as np
from typing import Dict, Optional, Tuple
from ..models.seg_detector import SegmentationModel
from ..utils.mask_processing import MaskProcessor
from ..utils.grid_calculator import GridCalculator


class SegmentationPipeline:
    """
    End-to-end segmentation pipeline for pitch lines and poles.
    """

    def __init__(self, model: SegmentationModel, grid_calculator: Optional[GridCalculator] = None):
        """
        Args:
            model: Initialized SegmentationModel.
            grid_calculator: Optional GridCalculator for distance estimation.
        """
        self.model = model
        self.grid_calc = grid_calculator
        self.mask_proc = MaskProcessor()

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single camera frame.

        Args:
            frame: BGR image.

        Returns:
            Dict with:
                - "mask": Raw segmentation mask.
                - "lines_mask": Processed binary mask for pitch lines.
                - "poles_mask": Processed binary mask for poles.
                - "grid_result": Output from GridCalculator (if available).
                - "overlays": Dictionary of visualization overlays.
        """
        # 1. Model Inference
        raw_mask = self.model.predict(frame)

        # 2. Post-processing for Pitch Lines (Class 1)
        lines_mask_raw = self.mask_proc.color_mask(raw_mask, MaskProcessor.CLASS_PITCH_LINES)
        lines_mask = self.mask_proc.morphological_cleanup(lines_mask_raw, kernel_size=3, iterations=1)

        # 3. Post-processing for Poles (Class 2)
        poles_mask_raw = self.mask_proc.color_mask(raw_mask, MaskProcessor.CLASS_POLES)
        poles_mask = self.mask_proc.morphological_cleanup(poles_mask_raw, kernel_size=5, iterations=2)

        # 4. Grid Distance Estimation
        grid_result = None
        if self.grid_calc is not None:
            grid_result = self.grid_calc.estimate_distances_from_mask(lines_mask)

        # 5. Visualization
        overlays = {
            "lines": self.mask_proc.create_overlay(frame, lines_mask, MaskProcessor.CLASS_PITCH_LINES),
            "poles": self.mask_proc.create_overlay(frame, poles_mask, MaskProcessor.CLASS_POLES),
        }
        
        # Combined overlay
        combined = frame.copy()
        combined = self.mask_proc.create_overlay(combined, lines_mask, MaskProcessor.CLASS_PITCH_LINES, alpha=0.3)
        combined = self.mask_proc.create_overlay(combined, poles_mask, MaskProcessor.CLASS_POLES, alpha=0.5)
        overlays["combined"] = combined

        return {
            "mask": raw_mask,
            "lines_mask": lines_mask,
            "poles_mask": poles_mask,
            "grid_result": grid_result,
            "overlays": overlays
        }
