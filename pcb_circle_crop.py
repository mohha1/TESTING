#!/usr/bin/env python3
"""
Dead-simple PCB alignment: find center, keep circular region only.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def find_ring_center(image_path):
    """Find center of copper ring using color thresholding."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load: {image_path}")
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([30, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 60, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        raise ValueError(f"No copper found in: {image_path}")
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return img, (cx, cy)


def align_pcb(image_path, circle_radius=230, output_path=None):
    """
    Find center of copper ring, keep only circular region.
    Everything outside the circle becomes black.
    """
    img, (cx, cy) = find_ring_center(image_path)
    h, w = img.shape[:2]
    
    # Create circular mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (cx, cy), circle_radius, 255, -1)
    
    # Apply mask: keep only inside circle
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    result = cv2.bitwise_and(img, mask_3ch)
    
    # Crop to bounding box of circle (optional, for smaller files)
    x1 = max(0, cx - circle_radius)
    y1 = max(0, cy - circle_radius)
    x2 = min(w, cx + circle_radius)
    y2 = min(h, cy + circle_radius)
    cropped = result[y1:y2, x1:x2]
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
    
    return cropped


def batch_align(input_dir, output_dir, circle_radius=230):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in input_path.iterdir() 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    print(f"Processing {len(image_files)} images, circle radius: {circle_radius}")
    
    for i, img_file in enumerate(image_files, 1):
        try:
            out_file = output_path / f"aligned_{img_file.name}"
            align_pcb(img_file, circle_radius, out_file)
            print(f"  [{i}/{len(image_files)}] OK {img_file.name}")
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] FAIL {img_file.name}: {e}")
    
    print(f"Done. Output in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Simple PCB circular crop')
    parser.add_argument('--input_dir', '-i', required=True, help='Input folder')
    parser.add_argument('--output_dir', '-o', default='aligned', help='Output folder')
    parser.add_argument('--radius', '-r', type=int, default=230, help='Circle radius (default: 230)')
    
    args = parser.parse_args()
    batch_align(args.input_dir, args.output_dir, args.radius)


if __name__ == "__main__":
    main()