import numpy as np
from typing import Optional


class SegmentationModel:
    """
    Lightweight inference class for TensorRT or ONNX segmentation models.
    """

    def __init__(self, model_path: str, input_shape: tuple = (1, 3, 320, 320), backend: str = "onnx"):
        """
        Args:
            model_path: Path to .onnx or .trt model file.
            input_shape: Model input dimensions (N, C, H, W).
            backend: "onnx" for ONNX Runtime or "tensorrt" for TensorRT.
        """
        self.input_shape = input_shape
        self.backend = backend
        self.session = None
        self.input_name = None

        if backend == "onnx":
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        elif backend == "tensorrt":
            # Placeholder for TensorRT initialization using pycuda/tensorrt
            # import tensorrt as trt
            raise NotImplementedError("TensorRT backend is a placeholder. Use ONNX for now.")
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def predict(self, frame: np.ndarray) -> np.ndarray:
        """
        Run inference on a single frame.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            np.ndarray: Segmentation mask (H, W) with class IDs.
        """
        original_h, original_w = frame.shape[:2]
        
        # Preprocess: Resize, BGR to RGB, Normalize, Transpose to NCHW
        resized = cv2.resize(frame, (self.input_shape[3], self.input_shape[2]))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.float32) / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)

        # Run inference
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Post-process: Argmax to get class IDs
        # Assuming output shape is (1, num_classes, H, W) or (1, H, W)
        logits = outputs[0]
        if len(logits.shape) == 4:
            mask = np.argmax(logits, axis=1)[0]
        else:
            mask = logits[0]

        # Resize mask back to original dimensions
        mask = cv2.resize(mask.astype(np.uint8), (original_w, original_h), interpolation=cv2.INTER_NEAREST)
        
        return mask
