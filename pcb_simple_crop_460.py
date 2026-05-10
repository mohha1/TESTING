#!/usr/bin/env python3
"""
Dead-simple PCB alignment: find center, crop square.
No transforms, no masks, no Hough circles.
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def find_ring_center(image_path):
    """
    Find the center of the big copper ring using simple color thresholding.
    Returns (cx, cy) or raises error.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot load: {image_path}")
    
    # Simple HSV threshold for copper (salmon/pink)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 60, 80]), np.array([30, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 60, 80]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find center of mass of the mask (the big copper blob)
    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        raise ValueError(f"No copper found in: {image_path}")
    
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    
    return img, (cx, cy)


def align_pcb(image_path, crop_size=460, output_path=None):
    """
    Find center of copper ring, crop a square around it.
    No scaling, no rotation, no warping.
    """
    img, (cx, cy) = find_ring_center(image_path)
    h, w = img.shape[:2]
    
    # Calculate crop bounds
    half = crop_size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    
    # If crop goes outside image, shift center
    if x2 - x1 < crop_size:
        x1 = max(0, min(cx - half, w - crop_size))
        x2 = x1 + crop_size
    if y2 - y1 < crop_size:
        y1 = max(0, min(cy - half, h - crop_size))
        y2 = y1 + crop_size
    
    cropped = img[y1:y2, x1:x2]
    
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cropped)
    
    return cropped


def batch_align(input_dir, output_dir, crop_size=460):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted([f for f in input_path.iterdir() 
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
    
    print(f"Processing {len(image_files)} images, crop size: {crop_size}x{crop_size}")
    
    for i, img_file in enumerate(image_files, 1):
        try:
            out_file = output_path / f"aligned_{img_file.name}"
            align_pcb(img_file, crop_size, out_file)
            print(f"  [{i}/{len(image_files)}] OK {img_file.name}")
        except Exception as e:
            print(f"  [{i}/{len(image_files)}] FAIL {img_file.name}: {e}")
    
    print(f"Done. Output in: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Simple PCB center crop')
    parser.add_argument('--input_dir', '-i', required=True, help='Input folder')
    parser.add_argument('--output_dir', '-o', default='aligned', help='Output folder')
    parser.add_argument('--crop_size', '-c', type=int, default=460, help='Crop size (default: 460)')
    
    args = parser.parse_args()
    batch_align(args.input_dir, args.output_dir, args.crop_size)


if __name__ == "__main__":
    main()