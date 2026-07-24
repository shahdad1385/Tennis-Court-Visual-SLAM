import cv2
import numpy as np
from ipm_ground_grid import IPMProcessor

def create_test_grid_image(width=640, height=480):
    """Create a synthetic image with a perspective grid for testing."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (200, 200, 200) # Gray background
    
    # Draw perspective lines
    # Vanishing point roughly at center-top
    vp_x, vp_y = width // 2, 50
    
    # Vertical-ish lines
    for i in range(-5, 6):
        x_bottom = width // 2 + i * 60
        cv2.line(img, (vp_x, vp_y), (x_bottom, height), (0, 0, 0), 2)
        
    # Horizontal-ish lines (getting closer together at the top)
    for j in range(1, 10):
        y = int(vp_y + (height - vp_y) * (j / 10)**1.5)
        cv2.line(img, (0, y), (width, y), (0, 0, 0), 2)
        
    return img

def test_ipm():
    processor = IPMProcessor(height=0.6, tilt_deg=15)
    
    # Create synthetic input
    test_img = create_test_grid_image()
    cv2.imwrite("test_input_grid.png", test_img)
    print("Created synthetic test image: test_input_grid.png")
    
    # Process
    birdseye = processor.process_frame(test_img)
    cv2.imwrite("test_birdseye.png", birdseye)
    print("Generated Bird's Eye View: test_birdseye.png")
    
    # Detect lines
    result_img, metrics = processor.detect_grid_lines(birdseye)
    cv2.imwrite("test_result.png", result_img)
    print(f"Detected {len(metrics)} line segments.")
    if metrics:
        print("Sample metric coordinates (meters):")
        for line in metrics[:3]:
            print(f"  {line}")

if __name__ == "__main__":
    test_ipm()
