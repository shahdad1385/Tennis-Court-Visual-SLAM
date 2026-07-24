import cv2
import numpy as np

class IPMProcessor:
    def __init__(self, height=0.6, tilt_deg=15, k_matrix=None, ground_roi=None, output_size=(400, 1000)):
        """
        Initialize Inverse Perspective Mapping processor.
        
        Args:
            height (float): Camera height above ground in meters.
            tilt_deg (float): Camera tilt angle in degrees (0 = horizontal).
            k_matrix (np.array): 3x3 Intrinsic matrix. Defaults to example values.
            ground_roi (tuple): (x_min, x_max, y_min, y_max) in meters.
            output_size (tuple): (width, height) of the bird's-eye output image.
        """
        self.height = height
        self.tilt_rad = np.deg2rad(tilt_deg)
        self.K = k_matrix if k_matrix is not None else np.array([[500, 0, 320],
                                                                  [0, 500, 240],
                                                                  [0,   0,   1]])
        
        # Default ground ROI: 4m wide (X: -2 to 2), 10m long (Y: 0 to 10)
        self.rox = ground_roi if ground_roi else (-2.0, 2.0, 0.0, 10.0)
        self.output_size = output_size # (width, height)
        
        # Pre-compute homography matrix
        self.H = self._compute_homography()

    def _compute_extrinsic(self):
        """
        Compute rotation and translation matrices for camera to world transformation.
        World Frame: X (lateral), Y (forward), Z (up).
        Camera Frame: X (right), Y (down), Z (optical axis).
        """
        # Translation from world to camera
        # Camera is at (0, 0, height) in world
        t = np.array([[0], [0], [self.height]])
        
        # Rotation matrix (Pitch around X-axis)
        # Note: Rotating the world coordinates into camera frame.
        # World Z (up) needs to align with Camera -Y (up in camera) if looking straight down? 
        # Let's use standard extrinsic definition: 
        # R transforms world point to camera point: Pc = R * (Pw - t)
        
        # If tilt_deg is 0 (horizontal), Camera Z aligns with World Y (forward).
        # If tilt_deg is 90 (down), Camera Z aligns with World -Z (down).
        
        # We want rotation around X-axis.
        # At 0 deg: Yc = -Zw, Zc = Yw (looking forward)
        # At 90 deg: Yc = -Yw, Zc = -Zw (looking down)
        
        # Let's construct R such that:
        # Zc = sin(theta) * Xw + cos(theta) * Yw ? No.
        
        # Let's standardise:
        # World: X=Right, Y=Forward, Z=Up
        # Camera: X=Right, Y=Down, Z=Forward
        # 
        # Pitch rotation:
        #   R = [1      0           0        ]
        #       [0  cos(theta) -sin(theta) ]
        #       [0  sin(theta)  cos(theta) ]
        #
        # But we need to handle the fact that Camera Y is Down.
        # So we need an extra rotation of 180 around X?
        
        # Let's simplify by defining R directly for the tilt.
        # We want to project points on the ground (Z=0) to the image.
        
        # Rotation around X axis (pitch)
        c = np.cos(self.tilt_rad)
        s = np.sin(self.tilt_rad)
        
        # This R rotates a point from World to Camera frame assuming:
        # World: X_lateral, Y_forward, Z_up
        # Camera: X_right, Y_down, Z_forward
        
        # When tilt=0 (horizontal), Camera Z aligns with World Y.
        # When tilt=90 (down), Camera Z aligns with World -Z.
        
        R = np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]
        ])
        
        return R, t

    def _compute_homography(self):
        """
        Compute Homography matrix H that maps a point in the bird's-eye view
        to a point in the original image.
        
        We define 4 points in the Metric Ground Plane (Z=0)
        and their corresponding points in the Image Plane.
        """
        R, t = self._compute_extrinsic()
        
        x_min, x_max, y_min, y_max = self.rox
        w, h = self.output_size
        
        # 4 corners of the Ground ROI in World Coordinates (X, Y, 0)
        # Order: Top-Left (closest, left), Top-Right, Bottom-Left, Bottom-Right
        # In our metric frame: Y is forward. 
        # Top of birdseye image usually corresponds to Y=0 (closest to camera).
        # Bottom of birdseye image corresponds to Y=max (farthest).
        
        # Metric points (X, Y, Z=0)
        metric_pts = np.array([
            [x_min, y_min, 0], # Top-Left
            [x_max, y_min, 0], # Top-Right
            [x_min, y_max, 0], # Bottom-Left
            [x_max, y_max, 0]  # Bottom-Right
        ], dtype=np.float32).T # 3x4
        
        # Project to image plane: p_img = K * (R * (P_world - t))
        # Note: P_world is 3x4, t is 3x1
        # P_cam = R @ (metric_pts - t)
        p_cam = R @ (metric_pts - t)
        
        # Normalize by Z (depth)
        # p_img_homo = K @ p_cam
        p_img_homo = self.K @ p_cam
        
        # Convert to 2D pixel coordinates (u, v)
        img_pts = (p_img_homo[:2] / p_img_homo[2]).T # 4x2
        
        # Destination points in Bird's-Eye Image
        # These map to the corners of the output_size (w, h)
        birdseye_pts = np.array([
            [0, 0],       # Top-Left
            [w, 0],       # Top-Right
            [0, h],       # Bottom-Left
            [w, h]        # Bottom-Right
        ], dtype=np.float32)
        
        # Compute Perspective Transform Matrix
        # birdseye_pts = H @ img_pts
        H, status = cv2.findHomography(img_pts, birdseye_pts, cv2.RANSAC, 5.0)
        return H

    def process_frame(self, frame):
        """
        Process a single frame.
        Returns:
            birdseye_img: The warped bird's-eye view.
        """
        # Warp the image
        birdseye_img = cv2.warpPerspective(frame, self.H, self.output_size)
        return birdseye_img

    def detect_grid_lines(self, birdseye_img):
        """
        Detect lines in the bird's-eye view and return their metric coordinates.
        """
        gray = cv2.cvtColor(birdseye_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return birdseye_img.copy(), []
            
        x_min, x_max, y_min, y_max = self.rox
        w, h = self.output_size
        
        # Flatten lines to (N, 4) to handle both OpenCV 4.x and 5.x output formats
        lines = lines.reshape(-1, 4)
        
        # Draw lines on a copy for visualization
        vis_img = birdseye_img.copy()
        
        # For visualization of lines
        for x1, y1, x2, y2 in lines:
            cv2.line(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # For demonstration, map endpoints to metric coordinates
        metric_lines = []
        for x1, y1, x2, y2 in lines:
            # Pixel to Metric conversion
            m_x1 = (x1 / w) * (x_max - x_min) + x_min
            m_y1 = (y1 / h) * (y_max - y_min) + y_min
            m_x2 = (x2 / w) * (x_max - x_min) + x_min
            m_y2 = (y2 / h) * (y_max - y_min) + y_min
            
            metric_lines.append(((m_x1, m_y1), (m_x2, m_y2)))
            
            # Annotate end points on the image
            cv2.putText(vis_img, f"({m_x1:.1f},{m_y1:.1f})", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return vis_img, metric_lines

def main():
    # Initialize processor
    processor = IPMProcessor(height=0.6, tilt_deg=15)
    
    # Example usage with a video stream (or image)
    # Using 0 for default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize for consistent processing if needed
        frame = cv2.resize(frame, (640, 480))
        
        # Generate Bird's Eye View
        birdseye = processor.process_frame(frame)
        
        # Detect lines and get metrics
        result_img, metrics = processor.detect_grid_lines(birdseye)
        
        # Display results
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Birds Eye View + Lines', result_img)
        
        if metrics:
            print(f"Detected {len(metrics)} lines.")
            # print("Metric coordinates (meters):", metrics) # Uncomment to see data
        
        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
