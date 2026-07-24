"""
Generate a minimal synthetic dataset for testing the YOLOv8-Pose pipeline.
Creates random bounding boxes with keypoint positions in YOLO format.
"""
import os
import numpy as np
from pathlib import Path


def create_sample_dataset(output_dir: str, num_images: int = 50):
    """Create a minimal sample dataset with synthetic annotations."""
    output_dir = Path(output_dir)

    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    count = 0
    for i in range(num_images):
        split = "train" if i < int(num_images * 0.8) else "val"

        # Create a blank placeholder image (in real use, use actual court images)
        img_path = output_dir / split / "images" / f"court_{i:04d}.txt"
        lbl_path = output_dir / split / "labels" / f"court_{i:04d}.txt"

        # Generate random keypoints in normalized [0, 1] coordinates
        # Simulating a camera looking at the court
        kpts = np.random.rand(8, 3).astype(np.float64)
        kpts[:, 0] = np.clip(kpts[:, 0], 0.1, 0.9)  # x range
        kpts[:, 1] = np.clip(kpts[:, 1] * 0.6 + 0.2, 0.1, 0.9)  # y range
        kpts[:, 2] = np.random.choice([1.0, 2.0], size=8)  # visibility

        # Compute bounding box from keypoints (with padding)
        x_min, y_min = kpts[:, 0].min(), kpts[:, 1].min()
        x_max, y_max = kpts[:, 0].max(), kpts[:, 1].max()
        pad = 0.05
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) + 2 * pad
        h = (y_max - y_min) + 2 * pad
        w = max(w, 0.1)
        h = max(h, 0.1)

        # YOLO format: class_id cx cy w h kp1_x kp1_y kp1_v ... kp8_x kp8_y kp8_v
        kpt_str = " ".join(f"{kpts[j, 0]:.6f} {kpts[j, 1]:.6f} {int(kpts[j, 2])}" for j in range(8))
        line = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpt_str}\n"

        with open(lbl_path, "w") as f:
            f.write(line)

        # Write placeholder image path (empty file for structure)
        with open(img_path, "w") as f:
            f.write("placeholder")

        count += 1

    print(f"Created {count} sample annotations in {output_dir}")
    print(f"  Train: {int(num_images * 0.8)}, Val: {num_images - int(num_images * 0.8)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate sample keypoint dataset")
    parser.add_argument("--output", type=str, default="keypoint_detection/dataset",
                        help="Output directory for dataset")
    parser.add_argument("--num", type=int, default=50,
                        help="Number of images to generate")
    args = parser.parse_args()
    create_sample_dataset(args.output, args.num)
