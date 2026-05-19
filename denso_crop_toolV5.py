"""
DENSO Image Crop Tool - Enhanced GUI v4 (Responsive Layout Fix)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import json

# ═══════════════════════════════════════════════════════════════════════════════
# DENSO CORPORATE PALETTE
# ═══════════════════════════════════════════════════════════════════════════════
C = {
    "bg_dark":       "#1A1A1A",
    "bg_panel":      "#222222",
    "bg_card":       "#2A2A2A",
    "bg_hover":      "#333333",
    "bg_input":      "#1A1A1A",
    "denso_red":     "#E30613",
    "denso_red_dk":  "#B8000F",
    "denso_red_lt":  "#FF4D5A",
    "ok_green":      "#2ECC71",
    "ok_green_dk":   "#27AE60",
    "warning":       "#F39C12",
    "warning_dk":    "#D68910",
    "white":         "#FFFFFF",
    "text_primary":  "#F0F0F0",
    "text_secondary":"#A0A0A0",
    "text_dim":      "#606060",
    "border":        "#3A3A3A",
    "border_light":  "#4A4A4A",
    "accent_blue":   "#3498DB",
}

FONT_MONO   = ("Consolas", 9)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MEDIUM = ("Segoe UI", 10)
FONT_LARGE  = ("Segoe UI", 11, "bold")
FONT_TITLE  = ("Segoe UI", 14, "bold")
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ═══════════════════════════════════════════════════════════════════════════════
# ROI CLASS
# ═══════════════════════════════════════════════════════════════════════════════
class ROI:
    def __init__(self, name="ROI", x1=0, y1=0, x2=100, y2=100):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_dict(self):
        return {"name": self.name, "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["x1"], d["y1"], d["x2"], d["y2"])

    def normalize(self):
        self.x1, self.x2 = min(self.x1, self.x2), max(self.x1, self.x2)
        self.y1, self.y2 = min(self.y1, self.y2), max(self.y1, self.y2)

    @property
    def width(self):
        return abs(self.x2 - self.x1)

    @property
    def height(self):
        return abs(self.y2 - self.y1)

    @property
    def area(self):
        return self.width * self.height

# ═══════════════════════════════════════════════════════════════════════════════
# CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════
class DensoButton(tk.Canvas):
    """Custom flat button with hover effects and rounded corners."""
    def __init__(self, master, text, command=None, bg_color=None, fg_color=None,
                 width=120, height=32, icon_text=None, **kwargs):
        self.bg_color = bg_color or C["denso_red"]
        self.fg_color = fg_color or C["white"]
        self.hover_color = self._darken(self.bg_color, 0.85)
        self.command = command
        self.btn_text = text
        self.icon_text = icon_text

        super().__init__(master, width=width, height=height, bg=C["bg_panel"],
                         highlightthickness=0, cursor="hand2", **kwargs)

        self.radius = 6
        self._draw(self.bg_color)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        self.bind("<ButtonRelease-1>", self._on_release)

    def _darken(self, hex_color, factor):
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _draw(self, color):
        self.delete("all")
        w, h = int(self.cget("width")), int(self.cget("height"))
        r = self.radius
        self.create_oval(0, 0, r*2, r*2, fill=color, outline=color)
        self.create_oval(w-r*2, 0, w, r*2, fill=color, outline=color)
        self.create_oval(0, h-r*2, r*2, h, fill=color, outline=color)
        self.create_oval(w-r*2, h-r*2, w, h, fill=color, outline=color)
        self.create_rectangle(r, 0, w-r, h, fill=color, outline=color)
        self.create_rectangle(0, r, w, h-r, fill=color, outline=color)

        text = self.btn_text
        if self.icon_text:
            text = f"{self.icon_text}  {text}"
        self.create_text(w//2, h//2, text=text, fill=self.fg_color,
                         font=FONT_SMALL, anchor="center")

    def _on_enter(self, e):
        self._draw(self.hover_color)

    def _on_leave(self, e):
        self._draw(self.bg_color)

    def _on_click(self, e):
        self._draw(self._darken(self.bg_color, 0.7))

    def _on_release(self, e):
        self._draw(self.hover_color)
        if self.command:
            self.command()

class DensoIconButton(tk.Label):
    """Small icon button for list actions."""
    def __init__(self, master, text, command=None, color=C["denso_red"], **kwargs):
        self.command = command
        super().__init__(master, text=text, bg=C["bg_card"], fg=color,
                         font=("Segoe UI", 10), cursor="hand2", **kwargs)
        self.bind("<Enter>", lambda e: self.config(bg=C["bg_hover"]))
        self.bind("<Leave>", lambda e: self.config(bg=C["bg_card"]))
        self.bind("<Button-1>", lambda e: self.command() if self.command else None)

class DensoEntry(tk.Entry):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=C["bg_input"], fg=C["text_primary"],
                         insertbackground=C["white"], font=FONT_MONO,
                         relief="flat", bd=0, highlightthickness=1,
                         highlightcolor=C["denso_red"], highlightbackground=C["border"],
                         **kwargs)

class DensoLabel(tk.Label):
    def __init__(self, master, text, color=None, font=None, **kwargs):
        super().__init__(master, text=text, bg=C["bg_panel"],
                         fg=color or C["text_primary"], font=font or FONT_SMALL, **kwargs)

class DensoFrame(tk.Frame):
    def __init__(self, master, bg=None, **kwargs):
        super().__init__(master, bg=bg or C["bg_panel"], **kwargs)

class Card(tk.Frame):
    """Elevated card with subtle border."""
    def __init__(self, master, title=None, **kwargs):
        super().__init__(master, bg=C["bg_card"], bd=1, relief="solid",
                         highlightbackground=C["border"], highlightthickness=1, **kwargs)
        if title:
            header = tk.Frame(self, bg=C["bg_card"], height=30)
            header.pack(fill="x", padx=0, pady=0)
            header.grid_propagate(False)
            tk.Label(header, text=title, bg=C["bg_card"], fg=C["text_secondary"],
                     font=FONT_SMALL).pack(side="left", padx=10, pady=5)
            accent = tk.Frame(self, bg=C["denso_red"], height=2)
            accent.pack(fill="x", padx=0, pady=0)

class EditableCoord(tk.Frame):
    """A label that turns into an entry when clicked for editing coordinates."""
    def __init__(self, master, value, callback, bg_color, **kwargs):
        super().__init__(master, bg=bg_color, **kwargs)
        self.callback = callback
        self.value = value
        self.bg_color = bg_color

        self.label = tk.Label(self, text=str(value), bg=bg_color, fg=C["text_secondary"],
                              font=FONT_MONO, cursor="xterm", padx=2)
        self.label.pack()
        self.label.bind("<Button-1>", self._start_edit)
        self.label.bind("<Enter>", lambda e: self.label.config(fg=C["denso_red"]))
        self.label.bind("<Leave>", lambda e: self.label.config(fg=C["text_secondary"]))

        self.entry = None

    def _start_edit(self, event):
        self.label.pack_forget()
        self.entry = tk.Entry(self, bg=C["bg_input"], fg=C["text_primary"],
                              insertbackground=C["white"], font=FONT_MONO,
                              relief="flat", bd=1, highlightthickness=1,
                              highlightcolor=C["denso_red"], highlightbackground=C["border"],
                              width=6, justify="center")
        self.entry.insert(0, str(self.value))
        self.entry.pack()
        self.entry.focus_set()
        self.entry.select_range(0, tk.END)
        self.entry.bind("<Return>", self._confirm_edit)
        self.entry.bind("<FocusOut>", self._confirm_edit)
        self.entry.bind("<Escape>", self._cancel_edit)

    def _confirm_edit(self, event=None):
        if self.entry:
            try:
                new_val = int(self.entry.get())
                self.value = new_val
                self.callback(new_val)
            except ValueError:
                pass
            self.entry.destroy()
            self.entry = None
            self.label.config(text=str(self.value))
            self.label.pack()

    def _cancel_edit(self, event=None):
        if self.entry:
            self.entry.destroy()
            self.entry = None
            self.label.pack()

    def update_value(self, new_value):
        self.value = new_value
        self.label.config(text=str(new_value))

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
class DensoCropApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DENSO Image Crop Tool")
        self.root.configure(bg=C["bg_dark"])
        self.root.geometry("1500x950")
        self.root.minsize(1300, 750)

        self.input_folder = ""
        self.output_folder = ""
        self.image_files = []
        self.current_image_idx = 0
        self.current_image = None
        self.current_photo = None
        self.zoom_level = 1.0
        self.rois = []
        self.selected_roi_idx = None
        self.creating_roi = False
        self.create_start = None
        self.panning = False
        self.pan_start = None

        self._build_ui()
        self._apply_styles()
        self._load_placeholder()

    def _apply_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TScrollbar", background=C["bg_card"], troughcolor=C["bg_dark"],
                        bordercolor=C["border"], arrowcolor=C["text_secondary"], arrowsize=12)
        style.map("TScrollbar",
                  background=[("active", C["bg_hover"]), ("pressed", C["denso_red"])])
        style.configure("Horizontal.TScale", background=C["bg_panel"], troughcolor=C["bg_dark"])

    def _load_placeholder(self):
        self.canvas.delete("all")
        w, h = 800, 600
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.create_rectangle(0, 0, w, h, fill=C["bg_dark"], outline="")
        for i in range(0, w, 40):
            self.canvas.create_line(i, 0, i, h, fill=C["border"], width=1)
        for i in range(0, h, 40):
            self.canvas.create_line(0, i, w, i, fill=C["border"], width=1)
        cx, cy = w//2, h//2
        self.canvas.create_text(cx, cy-30, text="[ IMAGE PREVIEW ]", fill=C["text_dim"],
                                font=("Segoe UI", 16, "bold"))
        self.canvas.create_text(cx, cy+10, text="Select an input folder to load images",
                                fill=C["text_secondary"], font=FONT_MEDIUM)
        self.canvas.create_text(cx, cy+35, text="or drag to create a new ROI zone",
                                fill=C["text_dim"], font=FONT_SMALL)

    def _build_ui(self):
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # ═══ TOP HEADER BAR ═══
        header = tk.Frame(self.root, bg=C["bg_dark"], height=50)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.grid_propagate(False)
        header.grid_columnconfigure(1, weight=1)

        logo_frame = tk.Frame(header, bg=C["bg_dark"])
        logo_frame.grid(row=0, column=0, sticky="w", padx=15, pady=8)

        tk.Label(logo_frame, text="DENSO", bg=C["bg_dark"], fg=C["white"],
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=(0, 2))
        tk.Label(logo_frame, text="Image Crop Tool", bg=C["bg_dark"], fg=C["text_secondary"],
                 font=FONT_MEDIUM).pack(side="left", padx=(4, 0))

        self.header_status = tk.Label(header, text="Ready", bg=C["bg_dark"],
                                      fg=C["text_dim"], font=FONT_SMALL)
        self.header_status.grid(row=0, column=2, sticky="e", padx=15)

        accent = tk.Frame(self.root, bg=C["denso_red"], height=2)
        accent.grid(row=0, column=0, sticky="ew", padx=0, pady=(48, 0))

        # ═══ MAIN CONTENT ═══
        content = tk.Frame(self.root, bg=C["bg_dark"])
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(12, 10))
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)

        # ─── LEFT SIDEBAR (scrollable canvas) ───
        sidebar_container = tk.Frame(content, bg=C["bg_panel"], width=340)
        sidebar_container.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        sidebar_container.grid_propagate(False)
        sidebar_container.grid_rowconfigure(0, weight=1)
        sidebar_container.grid_columnconfigure(0, weight=1)

        sidebar_canvas = tk.Canvas(sidebar_container, bg=C["bg_panel"], highlightthickness=0)
        sidebar_canvas.grid(row=0, column=0, sticky="nsew")

        sidebar_scroll = ttk.Scrollbar(sidebar_container, orient="vertical", command=sidebar_canvas.yview)
        sidebar_scroll.grid(row=0, column=1, sticky="ns")
        sidebar_canvas.configure(yscrollcommand=sidebar_scroll.set)

        sidebar = tk.Frame(sidebar_canvas, bg=C["bg_panel"])
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw", width=320)
        sidebar.bind("<Configure>", lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all")))

        # Folder Card
        folder_card = Card(sidebar, title="FOLDERS")
        folder_card.pack(fill="x", padx=10, pady=(10, 5))

        inner = tk.Frame(folder_card, bg=C["bg_card"], padx=12, pady=10)
        inner.pack(fill="both", expand=True)

        DensoLabel(inner, "Input Folder", color=C["text_secondary"]).grid(row=0, column=0, sticky="w", pady=(0, 4))
        input_row = tk.Frame(inner, bg=C["bg_card"])
        input_row.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        input_row.grid_columnconfigure(0, weight=1)
        self.input_entry = DensoEntry(input_row, width=25)
        self.input_entry.grid(row=0, column=0, sticky="ew")
        DensoIconButton(input_row, "...", command=self._browse_input,
                        color=C["text_secondary"]).grid(row=0, column=1, padx=(4, 0))

        DensoLabel(inner, "Output Folder", color=C["text_secondary"]).grid(row=2, column=0, sticky="w", pady=(0, 4))
        output_row = tk.Frame(inner, bg=C["bg_card"])
        output_row.grid(row=3, column=0, sticky="ew")
        output_row.grid_columnconfigure(0, weight=1)
        self.output_entry = DensoEntry(output_row, width=25)
        self.output_entry.grid(row=0, column=0, sticky="ew")
        DensoIconButton(output_row, "...", command=self._browse_output,
                        color=C["text_secondary"]).grid(row=0, column=1, padx=(4, 0))

        # ROI Card
        roi_card = Card(sidebar, title="ROI ZONES")
        roi_card.pack(fill="x", padx=10, pady=5)

        roi_inner = tk.Frame(roi_card, bg=C["bg_card"], padx=10, pady=8)
        roi_inner.pack(fill="both", expand=True)

        roi_btns = tk.Frame(roi_inner, bg=C["bg_card"])
        roi_btns.pack(fill="x", pady=(0, 8))
        DensoButton(roi_btns, "Add ROI", command=self._add_roi,
                    bg_color=C["denso_red"], width=90, height=28, icon_text="+").pack(side="left", padx=(0, 4))
        DensoButton(roi_btns, "Remove", command=self._remove_roi,
                    bg_color=C["warning"], width=80, height=28, icon_text="-").pack(side="left", padx=4)

        self.roi_count_label = tk.Label(roi_btns, text="0 zones", bg=C["bg_card"],
                                        fg=C["text_dim"], font=FONT_SMALL)
        self.roi_count_label.pack(side="right")

        self.roi_canvas = tk.Canvas(roi_inner, bg=C["bg_card"], highlightthickness=0, height=180)
        self.roi_canvas.pack(fill="both", expand=True)

        roi_scroll = ttk.Scrollbar(roi_inner, orient="vertical", command=self.roi_canvas.yview)
        roi_scroll.pack(side="right", fill="y")
        self.roi_canvas.configure(yscrollcommand=roi_scroll.set)

        self.roi_list_frame = tk.Frame(self.roi_canvas, bg=C["bg_card"])
        self.roi_canvas.create_window((0, 0), window=self.roi_list_frame, anchor="nw", width=290)
        self.roi_list_frame.bind("<Configure>",
            lambda e: self.roi_canvas.configure(scrollregion=self.roi_canvas.bbox("all")))

        config_frame = tk.Frame(roi_inner, bg=C["bg_card"])
        config_frame.pack(fill="x", pady=(8, 0))
        DensoButton(config_frame, "Save", command=self._save_config,
                    bg_color=C["ok_green"], width=70, height=24, icon_text="").pack(side="left", padx=(0, 4))
        DensoButton(config_frame, "Load", command=self._load_config,
                    bg_color=C["bg_hover"], fg_color=C["text_primary"],
                    width=70, height=24, icon_text="").pack(side="left", padx=4)

        # Navigation Card
        nav_card = Card(sidebar, title="NAVIGATION")
        nav_card.pack(fill="x", padx=10, pady=5)

        nav_inner = tk.Frame(nav_card, bg=C["bg_card"], padx=12, pady=10)
        nav_inner.pack(fill="both", expand=True)

        counter_frame = tk.Frame(nav_inner, bg=C["bg_card"])
        counter_frame.pack(fill="x", pady=(0, 8))

        self.prev_btn = DensoButton(counter_frame, "Prev", command=self._prev_image,
                                     bg_color=C["bg_hover"], fg_color=C["text_primary"],
                                     width=60, height=26, icon_text="<")
        self.prev_btn.pack(side="left")

        self.img_counter = tk.Label(counter_frame, text="0 / 0", bg=C["bg_card"],
                                     fg=C["text_primary"], font=FONT_MONO)
        self.img_counter.pack(side="left", expand=True)

        self.next_btn = DensoButton(counter_frame, "Next", command=self._next_image,
                                     bg_color=C["bg_hover"], fg_color=C["text_primary"],
                                     width=60, height=26, icon_text=">")
        self.next_btn.pack(side="right")

        self.filename_label = tk.Label(nav_inner, text="No image loaded", bg=C["bg_card"],
                                        fg=C["text_dim"], font=FONT_SMALL, wraplength=280)
        self.filename_label.pack(fill="x", pady=(0, 8))

        zoom_frame = tk.Frame(nav_inner, bg=C["bg_card"])
        zoom_frame.pack(fill="x")
        tk.Label(zoom_frame, text="Zoom", bg=C["bg_card"], fg=C["text_secondary"],
                 font=FONT_SMALL).pack(side="left")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = tk.Scale(zoom_frame, from_=0.2, to=4.0, resolution=0.1,
                                    orient="horizontal", variable=self.zoom_var,
                                    bg=C["bg_card"], fg=C["text_primary"],
                                    highlightthickness=0, troughcolor=C["bg_dark"],
                                    activebackground=C["denso_red"], showvalue=False,
                                    length=180, command=self._on_zoom)
        self.zoom_scale.pack(side="left", padx=(10, 0))
        self.zoom_value_label = tk.Label(zoom_frame, text="100%", bg=C["bg_card"],
                                          fg=C["text_primary"], font=FONT_MONO, width=5)
        self.zoom_value_label.pack(side="left", padx=(5, 0))

        # Action Card - CROP BUTTONS (always visible, packed last)
        action_card = Card(sidebar, title="ACTIONS")
        action_card.pack(fill="x", padx=10, pady=5)

        action_inner = tk.Frame(action_card, bg=C["bg_card"], padx=12, pady=10)
        action_inner.pack(fill="both", expand=True)

        DensoButton(action_inner, "Crop All Images", command=self._crop_all,
                    bg_color=C["ok_green"], width=280, height=36, icon_text="").pack(fill="x", pady=(0, 6))
        DensoButton(action_inner, "Crop Current Only", command=self._crop_current,
                    bg_color=C["denso_red"], width=280, height=36, icon_text="").pack(fill="x", pady=(0, 6))

        self.stats_label = tk.Label(action_inner, text="0 images | 0 ROIs | 0 crops ready",
                                     bg=C["bg_card"], fg=C["text_dim"], font=FONT_SMALL)
        self.stats_label.pack(fill="x", pady=(4, 0))

        # ─── RIGHT PANEL (Image Viewer) ───
        viewer_frame = tk.Frame(content, bg=C["bg_dark"], bd=1, relief="solid",
                                highlightbackground=C["border"], highlightthickness=1)
        viewer_frame.grid(row=0, column=1, sticky="nsew")
        viewer_frame.grid_rowconfigure(0, weight=1)
        viewer_frame.grid_columnconfigure(0, weight=1)

        # Image canvas directly at top
        self.canvas_frame = tk.Frame(viewer_frame, bg=C["bg_dark"])
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.canvas_frame.grid_rowconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, bg=C["bg_dark"], highlightthickness=0,
                                cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        h_scroll = ttk.Scrollbar(self.canvas_frame, orient="horizontal", command=self.canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)

        # Bottom info bar
        info_bar = tk.Frame(viewer_frame, bg=C["bg_panel"], height=28)
        info_bar.grid(row=1, column=0, sticky="ew")
        info_bar.grid_propagate(False)

        self.canvas_info = tk.Label(info_bar, text="Left-click + drag: Create ROI  |  Click ROI: Select  |  Right-click: Pan  |  Scroll: Zoom",
                                     bg=C["bg_panel"], fg=C["text_dim"], font=("Segoe UI", 8))
        self.canvas_info.pack(side="left", padx=10, pady=4)

        self.mouse_pos_label = tk.Label(info_bar, text="", bg=C["bg_panel"],
                                         fg=C["text_dim"], font=FONT_MONO)
        self.mouse_pos_label.pack(side="right", padx=10, pady=4)

        self.canvas.bind("<Motion>", self._on_mouse_move)

    # ═══════════════════════════════════════════════════════════════════════════
    # FOLDER OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _browse_input(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_folder = folder
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, folder)
            self._load_images()

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, folder)
            self._update_stats()

    def _load_images(self):
        self.image_files = []
        for f in sorted(os.listdir(self.input_folder)):
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                self.image_files.append(os.path.join(self.input_folder, f))

        if self.image_files:
            self.current_image_idx = 0
            self._load_current_image()
            self.header_status.config(text=f"Loaded {len(self.image_files)} images")
        else:
            self.header_status.config(text="No images found")
            self._load_placeholder()
        self._update_stats()

    def _load_current_image(self):
        if not self.image_files:
            return
        path = self.image_files[self.current_image_idx]
        try:
            self.current_image = Image.open(path)
            self.img_counter.config(text=f"{self.current_image_idx + 1} / {len(self.image_files)}")
            self.filename_label.config(text=os.path.basename(path), fg=C["text_primary"])
            self._update_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot load image:\n{str(e)}")

    def _update_stats(self):
        total_crops = len(self.image_files) * len(self.rois) if self.image_files else 0
        self.stats_label.config(
            text=f"{len(self.image_files)} images | {len(self.rois)} ROIs | {total_crops} crops ready")
        self.roi_count_label.config(text=f"{len(self.rois)} zone{'s' if len(self.rois) != 1 else ''}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CANVAS & RENDERING
    # ═══════════════════════════════════════════════════════════════════════════
    def _update_canvas(self):
        if self.current_image is None:
            return

        w, h = self.current_image.size
        zoom = self.zoom_level
        new_w, new_h = int(w * zoom), int(h * zoom)

        display_img = self.current_image.resize((new_w, new_h), Image.LANCZOS)
        self.current_photo = ImageTk.PhotoImage(display_img)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.current_photo, tags="bg_image")
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        # Subtle grid overlay
        grid_spacing = int(50 * zoom)
        if grid_spacing > 20:
            for x in range(0, new_w, grid_spacing):
                self.canvas.create_line(x, 0, x, new_h, fill=C["border"], width=1, tags="grid")
            for y in range(0, new_h, grid_spacing):
                self.canvas.create_line(0, y, new_w, y, fill=C["border"], width=1, tags="grid")
            self.canvas.tag_lower("grid")

        self._draw_rois()

    def _draw_rois(self):
        if self.current_image is None:
            return

        zoom = self.zoom_level
        img_w, img_h = self.current_image.size

        for i, roi in enumerate(self.rois):
            x1 = max(0, min(roi.x1, img_w)) * zoom
            y1 = max(0, min(roi.y1, img_h)) * zoom
            x2 = max(0, min(roi.x2, img_w)) * zoom
            y2 = max(0, min(roi.y2, img_h)) * zoom

            is_selected = (i == self.selected_roi_idx)
            color = C["denso_red"] if is_selected else C["warning"]
            width = 3 if is_selected else 2

            if is_selected:
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=C["denso_red"],
                                             stipple="gray50", outline="", tags=f"roi_fill_{i}")

            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width, tags=f"roi_{i}")

            label_text = f" {roi.name} "
            label_y = y1 - 18 if y1 > 25 else y2 + 18
            text_w = len(label_text) * 6
            self.canvas.create_rectangle(x1, label_y-8, x1+text_w, label_y+8,
                                         fill=color, outline="", tags=f"roi_label_bg_{i}")
            self.canvas.create_text(x1+2, label_y, text=label_text, anchor="w",
                                     fill=C["white"], font=("Segoe UI", 8, "bold"),
                                     tags=f"roi_label_{i}")

            # Show crop size: "240x210 -> 735x610  |  495x400"
            dim_text = f"{roi.width}x{roi.height}"
            dim_y = y2 + 14 if not is_selected or y1 <= 25 else y1 - 28
            self.canvas.create_text(x1, dim_y, text=dim_text, anchor="nw",
                                     fill=color, font=FONT_MONO, tags=f"roi_dim_{i}")

            if is_selected:
                handle_size = 5
                handles = [(x1, y1, "nw"), (x2, y1, "ne"), (x1, y2, "sw"), (x2, y2, "se")]
                for j, (hx, hy, corner) in enumerate(handles):
                    self.canvas.create_rectangle(hx-handle_size, hy-handle_size,
                                                  hx+handle_size, hy+handle_size,
                                                  fill=C["white"], outline=color, width=2,
                                                  tags=f"roi_handle_{i}_{j}")

        for i in range(len(self.rois)):
            for tag in [f"roi_{i}", f"roi_fill_{i}", f"roi_label_{i}", f"roi_label_bg_{i}"]:
                self.canvas.tag_bind(tag, "<Button-1>", lambda e, idx=i: self._select_roi(idx))

    # ═══════════════════════════════════════════════════════════════════════════
    # ROI OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _add_roi(self):
        if self.current_image is None:
            messagebox.showwarning("No Images", "Please load images first.")
            return
        w, h = self.current_image.size
        margin_x = w // 8
        margin_y = h // 8
        roi = ROI(f"ROI_{len(self.rois) + 1}", margin_x, margin_y, w - margin_x, h - margin_y)
        self.rois.append(roi)
        self.selected_roi_idx = len(self.rois) - 1
        self._update_roi_list()
        self._update_canvas()
        self._update_stats()

    def _remove_roi(self):
        if self.selected_roi_idx is not None and 0 <= self.selected_roi_idx < len(self.rois):
            self.rois.pop(self.selected_roi_idx)
            self.selected_roi_idx = None
            self._update_roi_list()
            self._update_canvas()
            self._update_stats()

    def _select_roi(self, idx):
        self.selected_roi_idx = idx
        self._update_canvas()
        self._update_roi_list()

    def _update_roi_list(self):
        for widget in self.roi_list_frame.winfo_children():
            widget.destroy()

        if not self.rois:
            empty = tk.Label(self.roi_list_frame, text="No ROIs defined\nClick 'Add ROI' to create one",
                              bg=C["bg_card"], fg=C["text_dim"], font=FONT_SMALL,
                              justify="center", pady=20)
            empty.pack(fill="both", expand=True)
        else:
            for i, roi in enumerate(self.rois):
                is_sel = (i == self.selected_roi_idx)
                bg = C["bg_hover"] if is_sel else C["bg_card"]

                row = tk.Frame(self.roi_list_frame, bg=bg, padx=8, pady=6)
                row.pack(fill="x", pady=1)
                row.bind("<Button-1>", lambda e, idx=i: self._select_roi(idx))

                strip = tk.Frame(row, bg=C["denso_red"] if is_sel else C["warning"], width=3)
                strip.pack(side="left", fill="y", padx=(0, 8))
                strip.pack_propagate(False)

                info = tk.Frame(row, bg=bg)
                info.pack(side="left", fill="both", expand=True)

                name_row = tk.Frame(info, bg=bg)
                name_row.pack(fill="x")

                tk.Label(name_row, text=roi.name, bg=bg, fg=C["text_primary"],
                         font=("Segoe UI", 9, "bold")).pack(side="left")

                badge = tk.Label(name_row, text=f"{roi.width}x{roi.height}", bg=C["bg_panel"],
                                  fg=C["text_secondary"], font=("Segoe UI", 7), padx=4)
                badge.pack(side="left", padx=(6, 0))

                # Editable coordinates row
                coord_frame = tk.Frame(info, bg=bg)
                coord_frame.pack(fill="x", pady=(2, 0))

                tk.Label(coord_frame, text="(", bg=bg, fg=C["text_dim"], font=FONT_MONO).pack(side="left")

                # Editable x1
                x1_edit = EditableCoord(coord_frame, roi.x1,
                    lambda v, idx=i, attr="x1": self._update_roi_coord(idx, attr, v), bg)
                x1_edit.pack(side="left")

                tk.Label(coord_frame, text=",", bg=bg, fg=C["text_dim"], font=FONT_MONO).pack(side="left")

                # Editable y1
                y1_edit = EditableCoord(coord_frame, roi.y1,
                    lambda v, idx=i, attr="y1": self._update_roi_coord(idx, attr, v), bg)
                y1_edit.pack(side="left")

                tk.Label(coord_frame, text=") -> (", bg=bg, fg=C["text_dim"], font=FONT_MONO).pack(side="left")

                # Editable x2
                x2_edit = EditableCoord(coord_frame, roi.x2,
                    lambda v, idx=i, attr="x2": self._update_roi_coord(idx, attr, v), bg)
                x2_edit.pack(side="left")

                tk.Label(coord_frame, text=",", bg=bg, fg=C["text_dim"], font=FONT_MONO).pack(side="left")

                # Editable y2
                y2_edit = EditableCoord(coord_frame, roi.y2,
                    lambda v, idx=i, attr="y2": self._update_roi_coord(idx, attr, v), bg)
                y2_edit.pack(side="left")

                tk.Label(coord_frame, text=")", bg=bg, fg=C["text_dim"], font=FONT_MONO).pack(side="left")

                # Crop size display
                size_text = f"  |  Crop: {roi.width}x{roi.height}"
                tk.Label(coord_frame, text=size_text, bg=bg, fg=C["ok_green"], font=FONT_MONO).pack(side="left")

                actions = tk.Frame(row, bg=bg)
                actions.pack(side="right", padx=(4, 0))

                DensoIconButton(actions, "\u270E", command=lambda idx=i: self._rename_roi(idx),
                                color=C["accent_blue"]).pack(side="left", padx=2)
                DensoIconButton(actions, "\u2715", command=lambda idx=i: self._delete_roi(idx),
                                color=C["denso_red"]).pack(side="left", padx=2)

        self.roi_list_frame.update_idletasks()
        self.roi_canvas.configure(scrollregion=self.roi_canvas.bbox("all"))

    def _update_roi_coord(self, idx, attr, value):
        """Update a specific coordinate of an ROI and refresh."""
        roi = self.rois[idx]
        setattr(roi, attr, value)
        self._update_canvas()
        self._update_roi_list()

    def _rename_roi(self, idx):
        new_name = simpledialog.askstring("Rename ROI", "Enter new name:",
                                          initialvalue=self.rois[idx].name)
        if new_name and new_name.strip():
            self.rois[idx].name = new_name.strip()
            self._update_roi_list()
            self._update_canvas()

    def _delete_roi(self, idx):
        self.rois.pop(idx)
        if self.selected_roi_idx == idx:
            self.selected_roi_idx = None
        elif self.selected_roi_idx is not None and self.selected_roi_idx > idx:
            self.selected_roi_idx -= 1
        self._update_roi_list()
        self._update_canvas()
        self._update_stats()

    # ═══════════════════════════════════════════════════════════════════════════
    # CANVAS INTERACTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _on_canvas_click(self, event):
        items = self.canvas.find_overlapping(event.x-5, event.y-5, event.x+5, event.y+5)
        for item in items:
            tags = self.canvas.gettags(item)
            for tag in tags:
                if tag.startswith("roi_") and not tag.startswith("roi_label_") and not tag.startswith("roi_handle_") and not tag.startswith("roi_dim_"):
                    return

        self.creating_roi = True
        self.create_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))

    def _on_canvas_drag(self, event):
        if self.creating_roi and self.create_start:
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            x1, y1 = self.create_start
            self.canvas.delete("temp_roi")
            self.canvas.create_rectangle(x1, y1, x, y, outline=C["ok_green"],
                                         width=2, dash=(6, 4), tags="temp_roi")
            zoom = self.zoom_level
            dx = abs(int((x - x1) / zoom))
            dy = abs(int((y - y1) / zoom))
            mid_x = (x1 + x) / 2
            mid_y = (y1 + y) / 2
            self.canvas.create_text(mid_x, mid_y - 15, text=f"{dx} x {dy}",
                                     fill=C["ok_green"], font=FONT_MONO, tags="temp_roi")

    def _on_canvas_release(self, event):
        if self.creating_roi and self.create_start:
            x1, y1 = self.create_start
            x2, y2 = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            self.canvas.delete("temp_roi")

            zoom = self.zoom_level
            img_x1 = int(min(x1, x2) / zoom)
            img_y1 = int(min(y1, y2) / zoom)
            img_x2 = int(max(x1, x2) / zoom)
            img_y2 = int(max(y1, y2) / zoom)

            if abs(img_x2 - img_x1) > 10 and abs(img_y2 - img_y1) > 10:
                roi = ROI(f"ROI_{len(self.rois) + 1}", img_x1, img_y1, img_x2, img_y2)
                self.rois.append(roi)
                self.selected_roi_idx = len(self.rois) - 1
                self._update_roi_list()
                self._update_canvas()
                self._update_stats()

            self.creating_roi = False
            self.create_start = None

    def _on_right_click(self, event):
        self.panning = True
        self.pan_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def _on_mousewheel(self, event):
        delta = 0.1 if event.delta > 0 else -0.1
        new_zoom = max(0.2, min(4.0, self.zoom_level + delta))
        self._set_zoom(new_zoom)

    def _on_shift_mousewheel(self, event):
        self.canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_mouse_move(self, event):
        if self.panning and self.pan_start:
            dx = self.pan_start[0] - event.x
            dy = self.pan_start[1] - event.y
            self.canvas.xview_scroll(dx, "units")
            self.canvas.yview_scroll(dy, "units")
            self.pan_start = (event.x, event.y)

        if self.current_image:
            zoom = self.zoom_level
            cx = int(self.canvas.canvasx(event.x) / zoom)
            cy = int(self.canvas.canvasy(event.y) / zoom)
            w, h = self.current_image.size
            if 0 <= cx < w and 0 <= cy < h:
                self.mouse_pos_label.config(text=f"{cx}, {cy}")
            else:
                self.mouse_pos_label.config(text="")

    def _on_zoom(self, value):
        self._set_zoom(float(value))

    def _set_zoom(self, value):
        self.zoom_level = round(value, 1)
        self.zoom_var.set(self.zoom_level)
        self.zoom_value_label.config(text=f"{int(self.zoom_level*100)}%")
        self._update_canvas()

    # ═══════════════════════════════════════════════════════════════════════════
    # NAVIGATION
    # ═══════════════════════════════════════════════════════════════════════════
    def _prev_image(self):
        if self.image_files and self.current_image_idx > 0:
            self.current_image_idx -= 1
            self._load_current_image()

    def _next_image(self):
        if self.image_files and self.current_image_idx < len(self.image_files) - 1:
            self.current_image_idx += 1
            self._load_current_image()

    # ═══════════════════════════════════════════════════════════════════════════
    # CROP OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    def _crop_all(self):
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return
        if not self.rois:
            messagebox.showwarning("No ROIs", "Please define at least one ROI.")
            return
        if not self.output_folder:
            messagebox.showwarning("No Output", "Please select an output folder.")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        total = len(self.image_files) * len(self.rois)
        count = 0
        errors = []

        progress = tk.Toplevel(self.root)
        progress.title("Cropping...")
        progress.configure(bg=C["bg_panel"])
        progress.geometry("400x150")
        progress.transient(self.root)
        progress.grab_set()

        tk.Label(progress, text="Processing images...", bg=C["bg_panel"],
                 fg=C["text_primary"], font=FONT_MEDIUM).pack(pady=(15, 10))

        progress_bar = ttk.Progressbar(progress, length=350, mode="determinate", maximum=total)
        progress_bar.pack(pady=5)

        progress_label = tk.Label(progress, text=f"0 / {total}", bg=C["bg_panel"],
                                   fg=C["text_secondary"], font=FONT_MONO)
        progress_label.pack(pady=5)

        self.root.update()

        for img_path in self.image_files:
            try:
                img = Image.open(img_path)
                base_name = os.path.splitext(os.path.basename(img_path))[0]

                for roi in self.rois:
                    roi.normalize()
                    x1 = max(0, min(roi.x1, img.width))
                    y1 = max(0, min(roi.y1, img.height))
                    x2 = max(0, min(roi.x2, img.width))
                    y2 = max(0, min(roi.y2, img.height))

                    if x2 > x1 and y2 > y1:
                        cropped = img.crop((x1, y1, x2, y2))
                        out_name = f"{base_name}_{roi.name}.png"
                        out_path = os.path.join(self.output_folder, out_name)
                        cropped.save(out_path, "PNG")
                        count += 1

                        progress_bar["value"] = count
                        progress_label.config(text=f"{count} / {total}")
                        self.header_status.config(text=f"Cropping... {count}/{total}")
                        self.root.update()
            except Exception as e:
                errors.append(f"{os.path.basename(img_path)}: {str(e)}")

        progress.destroy()
        self.header_status.config(text=f"Done! Cropped {count} images")

        if errors:
            messagebox.showwarning("Completed with Errors",
                                   f"Cropped {count}/{total} images.\n\nErrors:\n" + "\n".join(errors[:5]))
        else:
            messagebox.showinfo("Success", f"Cropping complete!\n{count} images saved to:\n{self.output_folder}")

    def _crop_current(self):
        if not self.image_files:
            messagebox.showwarning("No Images", "Please load images first.")
            return
        if not self.rois:
            messagebox.showwarning("No ROIs", "Please define at least one ROI.")
            return
        if not self.output_folder:
            messagebox.showwarning("No Output", "Please select an output folder.")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        img_path = self.image_files[self.current_image_idx]

        try:
            img = Image.open(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            count = 0

            for roi in self.rois:
                roi.normalize()
                x1 = max(0, min(roi.x1, img.width))
                y1 = max(0, min(roi.y1, img.height))
                x2 = max(0, min(roi.x2, img.width))
                y2 = max(0, min(roi.y2, img.height))

                if x2 > x1 and y2 > y1:
                    cropped = img.crop((x1, y1, x2, y2))
                    out_name = f"{base_name}_{roi.name}.png"
                    out_path = os.path.join(self.output_folder, out_name)
                    cropped.save(out_path, "PNG")
                    count += 1

            self.header_status.config(text=f"Cropped {count} regions")
            messagebox.showinfo("Success", f"Cropped {count} regions saved to:\n{self.output_folder}")
        except Exception as e:
            messagebox.showerror("Error", f"Error processing image:\n{str(e)}")

    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIG SAVE/LOAD
    # ═══════════════════════════════════════════════════════════════════════════
    def _save_config(self):
        if not self.rois:
            messagebox.showwarning("No ROIs", "No ROIs to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Save ROI Configuration")
        if path:
            config = {"rois": [roi.to_dict() for roi in self.rois]}
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
            self.header_status.config(text=f"Config saved: {os.path.basename(path)}")

    def _load_config(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load ROI Configuration")
        if path:
            try:
                with open(path, 'r') as f:
                    config = json.load(f)
                self.rois = [ROI.from_dict(d) for d in config.get("rois", [])]
                self.selected_roi_idx = None
                self._update_roi_list()
                self._update_canvas()
                self._update_stats()
                self.header_status.config(text=f"Config loaded: {os.path.basename(path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config:\n{str(e)}")

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = DensoCropApp(root)
    root.mainloop()
