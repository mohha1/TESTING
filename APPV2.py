import json
import math
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk


C = {
    "bg_dark": "#1A1A1A",
    "bg_panel": "#222222",
    "bg_card": "#2A2A2A",
    "bg_hover": "#333333",
    "bg_input": "#1A1A1A",
    "denso_red": "#E30613",
    "denso_red_dk": "#B8000F",
    "ok_green": "#2ECC71",
    "warning": "#F39C12",
    "white": "#FFFFFF",
    "text_primary": "#F0F0F0",
    "text_secondary": "#A0A0A0",
    "text_dim": "#606060",
    "border": "#3A3A3A",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ReferenceData:
    image_name: str
    image_size: Tuple[int, int]
    points: List[Tuple[float, float]]
    template_radius: int = 42


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def cv_read(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def cv_write(path: Path, image: np.ndarray) -> None:
    ext = path.suffix or ".png"
    ok, data = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    data.tofile(str(path))


def order_corner_points(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    pts = np.array(points, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(d)]  # top-right
    ordered[3] = pts[np.argmax(d)]  # bottom-left
    return [(float(x), float(y)) for x, y in ordered]


def clamp_roi(cx: int, cy: int, radius: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(width, cx + radius + 1)
    y2 = min(height, cy + radius + 1)
    return x1, y1, x2, y2


def nearest_component_centroid(mask: np.ndarray, local_point: Tuple[int, int]) -> Optional[Tuple[float, float, float]]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return None

    lx, ly = local_point
    best = None
    best_score = float("inf")
    h, w = mask.shape[:2]
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < 20 or area > (w * h * 0.45):
            continue

        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        ww = stats[label, cv2.CC_STAT_WIDTH]
        hh = stats[label, cv2.CC_STAT_HEIGHT]
        aspect = max(ww, hh) / max(1, min(ww, hh))
        if aspect > 2.4:
            continue

        cx, cy = centroids[label]
        contains_click = labels[min(max(ly, 0), h - 1), min(max(lx, 0), w - 1)] == label
        distance = math.hypot(cx - lx, cy - ly)
        score = distance - (1000 if contains_click else 0)
        if score < best_score:
            best_score = score
            best = (float(cx), float(cy), math.sqrt(area / math.pi))
    return best


def refine_hole_center(image: np.ndarray, point: Tuple[float, float], radius: int = 54) -> Tuple[float, float]:
    h, w = image.shape[:2]
    cx, cy = int(round(point[0])), int(round(point[1]))
    x1, y1, x2, y2 = clamp_roi(cx, cy, radius, w, h)
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return point

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    lx, ly = cx - x1, cy - y1
    click_v = int(hsv[min(max(ly, 0), hsv.shape[0] - 1), min(max(lx, 0), hsv.shape[1] - 1), 2])

    light_mask = ((hsv[:, :, 2] > 165) & (hsv[:, :, 1] < 95)).astype(np.uint8)
    dark_mask = (hsv[:, :, 2] < 80).astype(np.uint8)

    _, otsu_light = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_dark = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks = [light_mask, dark_mask, otsu_light, otsu_dark] if click_v >= 120 else [dark_mask, light_mask, otsu_dark, otsu_light]

    kernel = np.ones((3, 3), np.uint8)
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        found = nearest_component_centroid(mask, (lx, ly))
        if found:
            fx, fy, _ = found
            return (x1 + fx, y1 + fy)

    return point


def crop_template(image: np.ndarray, point: Tuple[float, float], radius: int) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = clamp_roi(int(round(point[0])), int(round(point[1])), radius, w, h)
    return image[y1:y2, x1:x2].copy()


def locate_by_template(
    image: np.ndarray,
    template: np.ndarray,
    expected: Tuple[float, float],
    search_radius: int,
) -> Tuple[Tuple[float, float], float]:
    h, w = image.shape[:2]
    th, tw = template.shape[:2]
    ex, ey = int(round(expected[0])), int(round(expected[1]))
    x1, y1, x2, y2 = clamp_roi(ex, ey, search_radius, w, h)
    search = image[y1:y2, x1:x2]

    if search.shape[0] < th or search.shape[1] < tw:
        search = image
        x1, y1 = 0, 0

    search_gray = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    found = (x1 + max_loc[0] + tw / 2.0, y1 + max_loc[1] + th / 2.0)
    return refine_hole_center(image, found), float(max_val)


def make_board_alpha_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)

    green_dominant = (green > red + 10) & (green > blue + 8)
    green_board = ((hue >= 36) & (hue <= 98) & (sat > 45) & (val > 28) & green_dominant).astype(np.uint8) * 255
    saturated = ((sat > 55) & (val > 35)).astype(np.uint8) * 255
    non_black = (gray > 16).astype(np.uint8) * 255
    fallback = cv2.bitwise_and(saturated, non_black)

    mask = green_board if cv2.countNonZero(green_board) > cv2.countNonZero(fallback) * 0.35 else fallback

    h, w = mask.shape
    close_size = max(15, (min(h, w) // 45) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask.astype(np.float32) / 255.0

    largest = max(contours, key=cv2.contourArea)
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=1)

    flood = clean.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    clean = cv2.bitwise_or(clean, cv2.bitwise_not(flood))

    repair_size = max(15, (min(h, w) // 45) | 1)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (repair_size, repair_size))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, repair_kernel, iterations=2)

    background_like = ((clean > 0) & (sat < 90) & (val > 80) & ~green_dominant).astype(np.uint8)
    bg_count, bg_labels, _, _ = cv2.connectedComponentsWithStats(background_like, 8)
    if bg_count > 1:
        border_labels = set(np.unique(bg_labels[0, :]))
        border_labels.update(np.unique(bg_labels[-1, :]))
        border_labels.update(np.unique(bg_labels[:, 0]))
        border_labels.update(np.unique(bg_labels[:, -1]))
        border_labels.discard(0)
        for label in border_labels:
            clean[bg_labels == label] = 0

    edge_trim = max(1, min(h, w) // 260)
    if edge_trim > 0:
        trim_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_trim * 2 + 1, edge_trim * 2 + 1))
        clean = cv2.erode(clean, trim_kernel, iterations=1)

    feather = max(5, (min(h, w) // 95) | 1)
    alpha = cv2.GaussianBlur(clean, (feather, feather), 0).astype(np.float32) / 255.0
    return np.clip(alpha, 0.0, 1.0)


def align_and_extract(
    image: np.ndarray,
    source_points: Sequence[Tuple[float, float]],
    target_points: Sequence[Tuple[float, float]],
    output_size: Tuple[int, int],
) -> np.ndarray:
    src = np.array(source_points, dtype=np.float32)
    dst = np.array(target_points, dtype=np.float32)
    width, height = output_size
    transform, _ = cv2.findHomography(src, dst, method=0)
    if transform is None:
        transform = cv2.getPerspectiveTransform(src, dst)

    warped = cv2.warpPerspective(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    alpha = make_board_alpha_mask(warped)
    return np.clip(warped.astype(np.float32) * alpha[:, :, None], 0, 255).astype(np.uint8)


class PCBAlignerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DENSO PCB Hole Aligner")
        self.root.geometry("1180x760")
        self.root.configure(bg=C["bg_dark"])

        self.input_folder: Optional[Path] = None
        self.output_folder: Optional[Path] = None
        self.images: List[Path] = []
        self.current_image: Optional[np.ndarray] = None
        self.current_photo: Optional[ImageTk.PhotoImage] = None
        self.reference: Optional[ReferenceData] = None
        self.clicked_points: List[Tuple[float, float]] = []
        self.templates: List[np.ndarray] = []
        self.preview_scale = 1.0
        self.preview_offset = (0, 0)
        self.processing = False

        self.build_ui()

    def build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background=C["bg_card"], foreground=C["text_primary"], bordercolor=C["border"], padding=8)
        style.map("TButton", background=[("active", C["bg_hover"])])
        style.configure("Accent.TButton", background=C["denso_red"], foreground=C["white"], bordercolor=C["denso_red_dk"])
        style.map("Accent.TButton", background=[("active", C["denso_red_dk"])])
        style.configure("TProgressbar", troughcolor=C["bg_input"], background=C["denso_red"], bordercolor=C["border"])

        shell = tk.Frame(self.root, bg=C["bg_dark"])
        shell.pack(fill=tk.BOTH, expand=True)

        sidebar = tk.Frame(shell, bg=C["bg_panel"], width=320, padx=16, pady=16)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        title = tk.Label(sidebar, text="PCB Hole Aligner", bg=C["bg_panel"], fg=C["white"], font=("Segoe UI", 18, "bold"))
        title.pack(anchor="w", pady=(0, 18))

        self.input_label = self.info_label(sidebar, "Input folder", "No folder selected")
        ttk.Button(sidebar, text="Choose input folder", command=self.choose_input).pack(fill=tk.X, pady=(0, 12))

        self.output_label = self.info_label(sidebar, "Output folder", "Auto: aligned_output")
        ttk.Button(sidebar, text="Choose output folder", command=self.choose_output).pack(fill=tk.X, pady=(0, 18))

        self.status_label = tk.Label(
            sidebar,
            text="Select a folder, then click the 4 reference screw holes on the first image.",
            bg=C["bg_card"],
            fg=C["text_secondary"],
            justify=tk.LEFT,
            wraplength=268,
            padx=12,
            pady=12,
        )
        self.status_label.pack(fill=tk.X, pady=(0, 14))

        ttk.Button(sidebar, text="Reset reference points", command=self.reset_points).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(sidebar, text="Save reference", command=self.save_reference).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(sidebar, text="Load reference", command=self.load_reference).pack(fill=tk.X, pady=(0, 18))
        ttk.Button(sidebar, text="Apply on all", style="Accent.TButton", command=self.apply_all).pack(fill=tk.X, pady=(0, 14))

        self.progress = ttk.Progressbar(sidebar, mode="determinate")
        self.progress.pack(fill=tk.X)

        self.log = tk.Text(sidebar, bg=C["bg_input"], fg=C["text_secondary"], insertbackground=C["white"], relief=tk.FLAT, height=11)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(14, 0))

        content = tk.Frame(shell, bg=C["bg_dark"], padx=14, pady=14)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(content, bg=C["bg_input"], highlightthickness=1, highlightbackground=C["border"], cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda _: self.render_image())

    def info_label(self, parent: tk.Widget, label: str, value: str) -> tk.Label:
        tk.Label(parent, text=label, bg=C["bg_panel"], fg=C["text_dim"], font=("Segoe UI", 9, "bold")).pack(anchor="w")
        widget = tk.Label(parent, text=value, bg=C["bg_panel"], fg=C["text_primary"], anchor="w", wraplength=280)
        widget.pack(fill=tk.X, pady=(2, 10))
        return widget

    def log_line(self, text: str) -> None:
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.root.update_idletasks()

    def ui(self, callback, *args, **kwargs) -> None:
        self.root.after(0, lambda: callback(*args, **kwargs))

    def set_status(self, text: str, color: str = C["text_secondary"]) -> None:
        self.status_label.configure(text=text, fg=color)
        self.root.update_idletasks()

    def choose_input(self) -> None:
        selected = filedialog.askdirectory(title="Select folder with PCB images")
        if not selected:
            return
        self.input_folder = Path(selected)
        self.output_folder = self.output_folder or (self.input_folder / "aligned_output")
        self.input_label.configure(text=str(self.input_folder))
        self.output_label.configure(text=str(self.output_folder))
        self.images = list_images(self.input_folder)
        self.reset_points(clear_reference=True)
        if not self.images:
            self.set_status("No supported images found in the selected folder.", C["warning"])
            return
        self.current_image = cv_read(self.images[0])
        self.render_image()
        self.set_status(f"Loaded {self.images[0].name}. Click the 4 screw holes.", C["text_secondary"])
        self.log_line(f"Loaded {len(self.images)} image(s).")

    def choose_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if not selected:
            return
        self.output_folder = Path(selected)
        self.output_label.configure(text=str(self.output_folder))

    def reset_points(self, clear_reference: bool = True) -> None:
        self.clicked_points = []
        self.templates = []
        if clear_reference:
            self.reference = None
        self.render_image()
        self.set_status("Reference points cleared. Click the 4 screw holes.", C["text_secondary"])

    def image_to_canvas(self, point: Tuple[float, float]) -> Tuple[float, float]:
        ox, oy = self.preview_offset
        return ox + point[0] * self.preview_scale, oy + point[1] * self.preview_scale

    def canvas_to_image(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self.current_image is None:
            return None
        ox, oy = self.preview_offset
        ix = (x - ox) / self.preview_scale
        iy = (y - oy) / self.preview_scale
        h, w = self.current_image.shape[:2]
        if ix < 0 or iy < 0 or ix >= w or iy >= h:
            return None
        return ix, iy

    def render_image(self) -> None:
        self.canvas.delete("all")
        if self.current_image is None:
            self.canvas.create_text(30, 30, anchor="nw", text="Choose an input folder to begin", fill=C["text_dim"], font=("Segoe UI", 14))
            return

        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        h, w = self.current_image.shape[:2]
        self.preview_scale = min(canvas_w / w, canvas_h / h, 1.0)
        draw_w, draw_h = int(w * self.preview_scale), int(h * self.preview_scale)
        self.preview_offset = ((canvas_w - draw_w) // 2, (canvas_h - draw_h) // 2)

        rgb = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((draw_w, draw_h), Image.Resampling.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(pil)
        self.canvas.create_image(*self.preview_offset, anchor="nw", image=self.current_photo)

        points = self.reference.points if self.reference else self.clicked_points
        for idx, point in enumerate(points, start=1):
            cx, cy = self.image_to_canvas(point)
            r = 7
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline=C["denso_red"], width=2)
            self.canvas.create_text(cx + 12, cy - 12, text=str(idx), fill=C["white"], font=("Segoe UI", 11, "bold"), anchor="w")

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.current_image is None or self.processing:
            return
        if self.reference is not None:
            self.set_status("Reference already set. Use Reset reference points to click again.", C["warning"])
            return

        point = self.canvas_to_image(event.x, event.y)
        if point is None:
            return
        refined = refine_hole_center(self.current_image, point)
        self.clicked_points.append(refined)
        self.log_line(f"Hole {len(self.clicked_points)}: x={refined[0]:.1f}, y={refined[1]:.1f}")

        if len(self.clicked_points) == 4:
            ordered = order_corner_points(self.clicked_points)
            h, w = self.current_image.shape[:2]
            self.reference = ReferenceData(self.images[0].name if self.images else "", (w, h), ordered)
            self.templates = [crop_template(self.current_image, p, self.reference.template_radius) for p in ordered]
            self.set_status("Reference ready. Press Apply on all to process the folder.", C["ok_green"])
        else:
            self.set_status(f"Clicked {len(self.clicked_points)}/4 screw holes.", C["text_secondary"])
        self.render_image()

    def reference_path(self) -> Optional[Path]:
        if not self.input_folder:
            return None
        return self.input_folder / "pcb_reference.json"

    def save_reference(self) -> None:
        if not self.reference:
            self.set_status("Click 4 screw holes before saving a reference.", C["warning"])
            return
        path = self.reference_path()
        if path is None:
            return
        self.write_reference_file(path)
        self.log_line(f"Saved reference: {path.name}")

    def write_reference_file(self, path: Path) -> None:
        if not self.reference:
            return
        path.write_text(json.dumps(asdict(self.reference), indent=2), encoding="utf-8")

    def load_reference(self) -> None:
        if not self.input_folder or not self.images:
            self.set_status("Choose an input folder before loading a reference.", C["warning"])
            return
        path = self.reference_path()
        if path is None or not path.exists():
            self.set_status("No pcb_reference.json found in the input folder.", C["warning"])
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        self.reference = ReferenceData(
            image_name=data["image_name"],
            image_size=tuple(data["image_size"]),
            points=[tuple(p) for p in data["points"]],
            template_radius=int(data.get("template_radius", 42)),
        )
        ref_image = cv_read(self.input_folder / self.reference.image_name) if (self.input_folder / self.reference.image_name).exists() else self.current_image
        self.templates = [crop_template(ref_image, p, self.reference.template_radius) for p in self.reference.points]
        self.set_status("Reference loaded. Press Apply on all to process the folder.", C["ok_green"])
        self.render_image()

    def ensure_ready(self) -> bool:
        if not self.input_folder or not self.images:
            self.set_status("Choose an input folder with images first.", C["warning"])
            return False
        if not self.output_folder:
            self.output_folder = self.input_folder / "aligned_output"
            self.output_label.configure(text=str(self.output_folder))
        if not self.reference or len(self.templates) != 4:
            self.set_status("Click the 4 screw holes on the first image first.", C["warning"])
            return False
        return True

    def apply_all(self) -> None:
        if self.processing or not self.ensure_ready():
            return
        thread = threading.Thread(target=self.process_folder, daemon=True)
        thread.start()

    def process_folder(self) -> None:
        assert self.output_folder is not None
        assert self.reference is not None
        self.processing = True
        try:
            self.output_folder.mkdir(parents=True, exist_ok=True)
            target_points = self.reference.points
            output_size = self.reference.image_size
            self.ui(self.progress.configure, maximum=len(self.images), value=0)
            self.ui(self.set_status, "Processing folder...", C["text_secondary"])

            search_radius = max(90, int(max(output_size) * 0.16))
            for index, path in enumerate(self.images, start=1):
                image = cv_read(path)
                sx = image.shape[1] / output_size[0]
                sy = image.shape[0] / output_size[1]
                expected_points = [(p[0] * sx, p[1] * sy) for p in target_points]
                source_points = []
                confidences = []
                for template, expected in zip(self.templates, expected_points):
                    point, confidence = locate_by_template(image, template, expected, search_radius)
                    source_points.append(point)
                    confidences.append(confidence)

                aligned = align_and_extract(image, order_corner_points(source_points), target_points, output_size)
                output_path = self.output_folder / f"{path.stem}_aligned.png"
                cv_write(output_path, aligned)
                self.ui(self.progress.configure, value=index)
                self.ui(self.log_line, f"{path.name} -> {output_path.name}  confidence min={min(confidences):.2f}")

            ref_path = self.reference_path()
            if ref_path is not None:
                self.write_reference_file(ref_path)
                self.ui(self.log_line, f"Saved reference: {ref_path.name}")
            self.ui(self.set_status, f"Done. Saved {len(self.images)} aligned image(s).", C["ok_green"])
        except Exception as exc:
            self.ui(self.set_status, f"Error: {exc}", C["warning"])
            self.ui(messagebox.showerror, "Processing error", str(exc))
        finally:
            self.processing = False


def main() -> None:
    root = tk.Tk()
    PCBAlignerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
