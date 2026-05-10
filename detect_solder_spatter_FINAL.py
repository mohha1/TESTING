"""
Solder Spatter Detector  v5  — DEFINITIVE
==========================================
Detection logic (from real image data):
  - Geometry  : 440x440, centre (220,220), ring r=130-210px  [hardcoded, always correct]
  - Ring gray : 120-128, std ~1.5, clean max = 156
  - Spatter   : white pixels 200+  (clearly visible, well separated from ring)

v5 key fix:
  Previous versions used blob area filtering which destroyed tiny dispersed spatter.
  v5 uses TWO complementary detectors — whichever fires first = DEFECTIVE:

  DETECTOR A — Pixel count:
    Count every ring pixel above SPATTER_THRESHOLD (160).
    Even 5 isolated bright pixels = suspicious. Threshold: MIN_BRIGHT_PIXELS.
    Catches fine dispersed spatter (many tiny specks).

  DETECTOR B — Blob detector (no morphological open — raw contours):
    Any connected group of 2+ bright pixels = a blob.
    Catches larger individual spatter drops.

  Both detectors annotate the output image independently.

Requirements:
    pip install opencv-python numpy

Usage:
    python detect_solder_spatter.py
"""

import cv2
import numpy as np
import csv
import shutil
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

INPUT_FOLDER  = r"C:\Users\TESTER\Desktop\PROJECTEOPENVINO\ALLINGIMAGES\ALIGNED"
OUTPUT_FOLDER = r"C:\Users\TESTER\Desktop\PROJECTEOPENVINO\ALLINGIMAGES\RESULTS"

# ── Geometry (hardcoded from diagnostic — never changes) ──────
IMG_SIZE = 440
CENTRE   = (220, 220)
INNER_R  = 125          # inner edge of red ring
OUTER_R  = 218          # outer edge of red ring  ← increased (was 210)
ERODE_PX = 2            # shrink mask inward to avoid circle-edge noise ← reduced (was 4)

# ── DETECTOR A: raw bright pixel count ───────────────────────
# Pixels in ring above this = spatter candidate
SPATTER_THRESHOLD  = 150   # lowered for more sensitivity (was 160, ring max = 156)
# Minimum number of bright pixels to flag as DEFECTIVE
# Clean images: 0 pixels above 155. Set to 2 for maximum sensitivity.
MIN_BRIGHT_PIXELS  = 2

# ── DETECTOR B: blob detector (no open/erode on spatter mask) ─
MIN_BLOB_AREA = 2     # 2+ connected pixels = a blob
MAX_BLOB_AREA = 5000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ══════════════════════════════════════════════════════════════

def make_ring_mask(h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, CENTRE, OUTER_R, 255, -1)
    cv2.circle(mask, CENTRE, INNER_R, 0,   -1)
    if ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (ERODE_PX*2+1, ERODE_PX*2+1))
        mask = cv2.erode(mask, k, iterations=1)
    return mask

# Pre-built mask for standard image size
_RING_MASK_CACHE = {}

def get_ring_mask(h, w):
    if (h, w) not in _RING_MASK_CACHE:
        _RING_MASK_CACHE[(h, w)] = make_ring_mask(h, w)
    return _RING_MASK_CACHE[(h, w)]


def detect_spatter(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return "ERROR", 0, 0, 0, None, "Could not read image"

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ring_mask = get_ring_mask(h, w)

    # ── Ring stats (logging only) ─────────────────────────────────────────────
    ring_pixels  = gray[ring_mask == 255]
    ring_median  = float(np.median(ring_pixels))
    ring_max     = int(ring_pixels.max())
    ring_std     = float(np.std(ring_pixels))

    # ── Bright pixel mask (NO morphological ops — don't destroy tiny spatter) ──
    _, bright_mask = cv2.threshold(gray, SPATTER_THRESHOLD, 255, cv2.THRESH_BINARY)
    spatter_mask   = cv2.bitwise_and(bright_mask, ring_mask)

    # ── DETECTOR A: total bright pixel count ─────────────────────────────────
    bright_pixel_count = int(np.count_nonzero(spatter_mask))
    detector_a_fired   = bright_pixel_count >= MIN_BRIGHT_PIXELS

    # ── DETECTOR B: connected blob count ─────────────────────────────────────
    contours, _ = cv2.findContours(spatter_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    valid_blobs      = [c for c in contours
                        if MIN_BLOB_AREA <= cv2.contourArea(c) <= MAX_BLOB_AREA]
    blob_count       = len(valid_blobs)
    total_blob_area  = int(sum(cv2.contourArea(c) for c in valid_blobs))
    detector_b_fired = blob_count > 0

    # ── Final decision ────────────────────────────────────────────────────────
    is_defective = detector_a_fired or detector_b_fired
    status       = "DEFECTIVE" if is_defective else "OK"

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = img.copy()

    # Ring boundary
    cv2.circle(annotated, CENTRE, OUTER_R, (0, 220, 0), 2)
    cv2.circle(annotated, CENTRE, INNER_R, (0, 220, 0), 2)

    # Highlight ALL bright pixels (including isolated ones) in red
    # Use a dilated version for visibility of tiny specks
    spatter_visible = spatter_mask.copy()
    k_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spatter_visible = cv2.dilate(spatter_visible, k_vis, iterations=1)
    annotated[spatter_visible == 255] = (0, 0, 255)  # paint red directly

    # Draw bounding boxes around blobs (Detector B hits)
    for cnt in valid_blobs:
        x, y, bw, bh = cv2.boundingRect(cnt)
        p = 4
        cv2.rectangle(annotated,
                      (max(0, x-p),    max(0, y-p)),
                      (min(w, x+bw+p), min(h, y+bh+p)),
                      (0, 100, 255), 1)

    # Status label
    color = (0, 0, 255) if is_defective else (0, 200, 0)
    trigger = []
    if detector_a_fired: trigger.append(f"A:pixels={bright_pixel_count}")
    if detector_b_fired: trigger.append(f"B:blobs={blob_count}")
    trigger_str = " ".join(trigger) if trigger else "none"

    cv2.putText(annotated, f"{status} | {trigger_str}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2, cv2.LINE_AA)
    cv2.putText(annotated,
                f"thresh={SPATTER_THRESHOLD}  median={ring_median:.0f}  "
                f"max={ring_max}  bright_px={bright_pixel_count}",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    msg = (f"bright_px={bright_pixel_count}  blobs={blob_count}  "
           f"blob_area={total_blob_area}px²  "
           f"ring(med={ring_median:.0f} max={ring_max} std={ring_std:.1f})  "
           f"thresh={SPATTER_THRESHOLD}  trigger=[{trigger_str}]")

    return status, bright_pixel_count, blob_count, total_blob_area, annotated, msg


def main():
    input_path  = Path(INPUT_FOLDER)
    output_path = Path(OUTPUT_FOLDER)

    if not input_path.exists():
        print(f"[ERROR] Input folder not found: {input_path}")
        return

    defective_dir = output_path / "DEFECTIVE"
    ok_dir        = output_path / "OK"
    for d in [defective_dir, ok_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_files = sorted([f for f in input_path.iterdir()
                          if f.suffix.lower() in IMAGE_EXTENSIONS])
    total = len(image_files)

    if total == 0:
        print(f"[WARNING] No images found in {input_path}")
        return

    print(f"\n{'─'*75}")
    print(f"  Solder Spatter Detector  v5  (dual detector — definitive)")
    print(f"  Input     : {input_path}")
    print(f"  Output    : {output_path}")
    print(f"  Images    : {total}")
    print(f"  Ring      : centre={CENTRE}  r={INNER_R}–{OUTER_R}px")
    print(f"  Threshold : {SPATTER_THRESHOLD}  (clean ring max=156, spatter=200+)")
    print(f"  Det.A fires if bright_pixels >= {MIN_BRIGHT_PIXELS}")
    print(f"  Det.B fires if any blob >= {MIN_BLOB_AREA}px²")
    print(f"{'─'*75}\n")

    csv_path = output_path / "report.csv"
    defective_count = ok_count = error_count = 0

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "status", "bright_pixels",
                         "blob_count", "blob_area_px2", "detail"])

        for i, img_file in enumerate(image_files, 1):
            status, bpx, blobs, area, annotated, msg = detect_spatter(img_file)
            print(f"[{i:>5}/{total}]  {status:<10}  {img_file.name:<42}  {msg}")

            dest_dir = (defective_dir if status == "DEFECTIVE"
                        else output_path if status == "ERROR"
                        else ok_dir)
            out_name = img_file.stem + "_result" + img_file.suffix

            if annotated is not None:
                # Side-by-side: annotated (left) | original (right)
                original = cv2.imread(str(img_file))
                side_by_side = np.hstack([annotated, original])
                ext = img_file.suffix.lower()
                if ext in (".jpg", ".jpeg"):
                    cv2.imwrite(str(dest_dir / out_name), side_by_side,
                                [cv2.IMWRITE_JPEG_QUALITY, 97])
                elif ext == ".png":
                    cv2.imwrite(str(dest_dir / out_name), side_by_side,
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    cv2.imwrite(str(dest_dir / out_name), side_by_side)
            else:
                shutil.copy2(img_file, dest_dir / out_name)

            writer.writerow([img_file.name, status, bpx, blobs, area, msg])

            if   status == "DEFECTIVE": defective_count += 1
            elif status == "OK":        ok_count        += 1
            else:                       error_count     += 1

    print(f"\n{'─'*75}")
    print(f"  DONE  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total     : {total}")
    print(f"  OK        : {ok_count}")
    print(f"  DEFECTIVE : {defective_count}")
    print(f"  ERROR     : {error_count}")
    print(f"  Report    : {csv_path}")
    print(f"{'─'*75}")
    print(f"\n  If you see false OKs: lower SPATTER_THRESHOLD (try 155)")
    print(f"  If you see false DEFECTIVEs: raise MIN_BRIGHT_PIXELS (try 5 or 8)\n")


if __name__ == "__main__":
    main()
