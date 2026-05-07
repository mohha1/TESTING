# test_padim.py
# Run with:
# & "C:\Program Files\Python313\python.exe" "C:\Users\TESTER\Desktop\Nueva carpeta (4)\test_padim.py"

from pathlib import Path
from openvino import Core
import numpy as np
import cv2
import json

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR   = Path(r"C:\Users\TESTER\Desktop\Nueva carpeta (4)\MODEL")
IMAGES_DIR  = Path(r"C:\Users\TESTER\Desktop\Nueva carpeta (4)\IMAGES TO INFER")
OUTPUT_DIR  = Path(r"C:\Users\TESTER\Desktop\Nueva carpeta (4)\IMAGES SENT")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load model ─────────────────────────────────────────────────────────────────
xml_file = list(MODEL_DIR.glob("*.xml"))
if not xml_file:
    raise FileNotFoundError("No .xml model file found in MODEL folder.")

xml_path  = xml_file[0]
meta_path = MODEL_DIR / "metadata.json"

print(f"Loading model: {xml_path.name}")

core     = Core()
model    = core.read_model(str(xml_path))
compiled = core.compile_model(model, "CPU")

# ── Print outputs (for verification) ──────────────────────────────────────────
print("\nMODEL OUTPUTS:")
for i, out in enumerate(compiled.outputs):
    print(f"  Output {i}: {out.any_name}  shape={out.partial_shape}")

# ── Get outputs by name ───────────────────────────────────────────────────────
score_output = compiled.output("pred_score")
map_output   = compiled.output("anomaly_map")
mask_output  = compiled.output("pred_mask")

# ── Load threshold from metadata ───────────────────────────────────────────────
threshold = 0.5

if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)

    threshold = meta.get(
        "image_threshold",
        meta.get(
            "pixel_threshold",
            meta.get("threshold", 0.5)
        )
    )

    print(f"Threshold from metadata: {threshold:.4f}")

else:
    print(f"⚠️ metadata.json not found, using default threshold: {threshold}")

# ── Input size ─────────────────────────────────────────────────────────────────
H, W = 256, 256

input_layer = compiled.input(0)

# ── Collect images ─────────────────────────────────────────────────────────────
images = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))

if not images:
    raise FileNotFoundError("No images found in IMAGES TO INFER folder.")

print(f"\nRunning inference on {len(images)} images...\n")

ok_list = []
ng_list = []

# ── Run inference ──────────────────────────────────────────────────────────────
for img_path in sorted(images):

    # ── Load image ─────────────────────────────────────────────────────────────
    img_bgr = cv2.imread(str(img_path))

    if img_bgr is None:
        print(f"⚠️ Could not read image: {img_path.name}")
        continue

    orig_h, orig_w = img_bgr.shape[:2]

    # ── Preprocess ─────────────────────────────────────────────────────────────
    img_resized = cv2.resize(img_bgr, (W, H))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    img_norm = img_rgb.astype(np.float32) / 255.0

    img_chw = np.transpose(img_norm, (2, 0, 1))

    img_batch = np.expand_dims(img_chw, axis=0)

    # ── Inference ──────────────────────────────────────────────────────────────
    results = compiled({input_layer: img_batch})

    # ── Outputs ────────────────────────────────────────────────────────────────
    score = float(results[score_output].squeeze())

    anomaly_map = results[map_output].squeeze()

    pred_mask = results[mask_output].squeeze().astype(np.uint8)

    # ── Normalize anomaly map to 0-255 ────────────────────────────────────────
    map_min = anomaly_map.min()
    map_max = anomaly_map.max()

    if map_max > map_min:
        heatmap = (
            (anomaly_map - map_min)
            / (map_max - map_min)
            * 255
        ).astype(np.uint8)
    else:
        heatmap = np.zeros((H, W), dtype=np.uint8)

    # ── Resize outputs back to original image size ────────────────────────────
    heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h))

    mask_resized = cv2.resize(
        pred_mask * 255,
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST
    )

    # ── Visualization ─────────────────────────────────────────────────────────
    heatmap_bgr = cv2.cvtColor(heatmap_resized, cv2.COLOR_GRAY2BGR)

    heatmap_color = cv2.applyColorMap(
        heatmap_resized,
        cv2.COLORMAP_JET
    )

    mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    # ── Combined output image ─────────────────────────────────────────────────
    combined = np.hstack([
        img_bgr,
        heatmap_bgr,
        heatmap_color,
        mask_bgr
    ])

    # ── Label ─────────────────────────────────────────────────────────────────
    label = "NG" if score >= threshold else "OK"

    color = (0, 0, 255) if label == "NG" else (0, 200, 0)

    cv2.putText(
        combined,
        f"{label}  score={score:.4f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    out_name = f"{label}_{img_path.stem}_heatmap.png"

    cv2.imwrite(str(OUTPUT_DIR / out_name), combined)

    print(
        f"  {'❌ NG' if label == 'NG' else '✅ OK'}  "
        f"score={score:.4f}  "
        f"{img_path.name}"
    )

    if label == "NG":
        ng_list.append(img_path.name)
    else:
        ok_list.append(img_path.name)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"RESULTS:  {len(ok_list)} OK  |  {len(ng_list)} NG")
print("=" * 60)

print(f"\n✅ OK ({len(ok_list)}):")
for name in ok_list:
    print(f"   {name}")

print(f"\n❌ NG ({len(ng_list)}):")
for name in ng_list:
    print(f"   {name}")

print(f"\n🖼️ Heatmap images saved to:\n   {OUTPUT_DIR}")