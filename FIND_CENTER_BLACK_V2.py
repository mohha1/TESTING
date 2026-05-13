#!/usr/bin/env python3
"""
Dead-simple PCB alignment: find center via black hole inside copper ring, keep circular region only.
"""
import cv2
import numpy as np
import argparse
from pathlib import Path

def find_ring_center(image_path):
    """Find center by locating the black hole inside the copper ring."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect copper region first
    mask1 = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([30, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 60, 80]), np.array([180, 255, 255]))
    copper_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    copper_mask = cv2.morphologyEx(copper_mask, cv2.MORPH_CLOSE, kernel)

    # Find the largest copper contour
    contours, _ = cv2.findContours(copper_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"No copper found in: {image_path}")
    largest_copper = max(contours, key=cv2.contourArea)

    # Create a filled mask of the copper area to use as search zone
    copper_filled = np.zeros_like(copper_mask)
    cv2.drawContours(copper_filled, [largest_copper], -1, 255, -1)

    # Look for dark pixels ONLY inside the copper region
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    hole_mask = cv2.bitwise_and(dark_mask, copper_filled)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel2)
    hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel2)

    hole_contours, _ = cv2.findContours(hole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not hole_contours:
        raise ValueError(f"No black hole found inside copper in: {image_path}")

    largest_hole = max(hole_contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_hole)
    cx, cy = int(cx), int(cy)

    print(f"  Hole center: ({cx}, {cy}), hole radius: {int(radius)}px")

    return img, (cx, cy)

def align_pcb(image_path, circle_radius=230, output_path=None):
    """
    Find center of copper ring via black hole, keep only circular region.
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

    # Crop to bounding box of circle
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