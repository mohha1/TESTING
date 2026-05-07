from pathlib import Path
from openvino import Core

# ── Model path ────────────────────────────────────────────────────────────────
MODEL_DIR = Path(r"C:\Users\TESTER\Desktop\Nueva carpeta (4)\MODEL")

xml_file = list(MODEL_DIR.glob("*.xml"))
if not xml_file:
    raise FileNotFoundError("No .xml file found.")

xml_path = xml_file[0]

# ── Load model ────────────────────────────────────────────────────────────────
core = Core()
model = core.read_model(str(xml_path))
compiled = core.compile_model(model, "CPU")

# ── Print inputs ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL INPUTS")
print("="*60)

for i, inp in enumerate(compiled.inputs):
    print(f"\nInput {i}")
    print(f"Name  : {inp.any_name}")
    print(f"Shape : {inp.partial_shape}")
    print(f"Type  : {inp.get_element_type()}")

# ── Print outputs ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODEL OUTPUTS")
print("="*60)

for i, out in enumerate(compiled.outputs):
    print(f"\nOutput {i}")
    print(f"Name  : {out.any_name}")
    print(f"Shape : {out.partial_shape}")
    print(f"Type  : {out.get_element_type()}")

print("\n" + "="*60)
print(f"Total outputs: {len(compiled.outputs)}")
print("="*60)