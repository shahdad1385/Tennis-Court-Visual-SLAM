import numpy as np

# Total number of keypoints
NUM_KEYPOINTS = 8

# Keypoint names in order (matches YOLO label index)
KEYPOINT_NAMES = [
    "left_pole_base",      # 0
    "left_pole_top",       # 1
    "right_pole_base",     # 2
    "right_pole_top",      # 3
    "service_line_center", # 4
    "service_line_left",   # 5
    "baseline_center",     # 6
    "service_line_right",  # 7
]

# Index mappings for convenience
KEYPOINT_INDICES = {name: i for i, name in enumerate(KEYPOINT_NAMES)}

# Grouped indices
POLE_INDICES = [0, 1, 2, 3]   # Net pole keypoints
GRID_INDICES = [4, 5, 6, 7]   # Grid intersection keypoints

# 3D world coordinates (meters) — origin at center of net on ground
# Coordinate system: X = lateral (right+), Y = forward (away+), Z = up+
OBJ_POINTS_POLES = np.array([
    [-5.485,  0.000, 0.00],  # Left Pole Base
    [-5.485,  0.000, 1.07],  # Left Pole Top
    [ 5.485,  0.000, 0.00],  # Right Pole Base
    [ 5.485,  0.000, 1.07],  # Right Pole Top
], dtype=np.float64)

OBJ_POINTS_GRID = np.array([
    [ 0.000,  6.400, 0.00],  # Service line center
    [-5.485,  6.400, 0.00],  # Service line left
    [ 0.000, 11.885, 0.00],  # Baseline center
    [ 5.485,  6.400, 0.00],  # Service line right
], dtype=np.float64)

# Keypoint visibility constants (YOLO format)
VIS_NOT_VISIBLE = 0
VIS_OCCLUDED = 1
VIS_FULLY_VISIBLE = 2

# YOLO flip indices for horizontal augmentation
# Maps original index → flipped index
FLIP_IDX = [2, 3, 0, 1, 4, 5, 7, 6]

# Skeleton connections for visualization (pairs of keypoint indices)
SKELETON = [
    (0, 1),  # Left pole base → top
    (2, 3),  # Right pole base → top
    (0, 2),  # Pole base line (net)
    (4, 5),  # Service line center → left
    (4, 7),  # Service line center → right
    (5, 6),  # Service line left → baseline center
]

# Colors for each keypoint (BGR)
KP_COLORS = [
    (0, 0, 255),    # Red — left pole base
    (0, 0, 200),    # Dark red — left pole top
    (255, 0, 0),    # Blue — right pole base
    (200, 0, 0),    # Dark blue — right pole top
    (0, 255, 0),    # Green — service line center
    (0, 200, 0),    # Dark green — service line left
    (0, 255, 255),  # Yellow — baseline center
    (0, 200, 200),  # Dark yellow — service line right
]

SKELETON_COLOR = (255, 255, 255)  # White
