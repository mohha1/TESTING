import openvino as ov
from pathlib import Path

model_path = "path/to/your/model.xml"

core = ov.Core()
model = core.read_model(str(model_path))
compiled = core.compile_model(model, "CPU")

print("=== INPUTS ===")
for inp in compiled.inputs:
    print(f"  Name: {inp.get_any_name()}")
    print(f"  Shape: {inp.shape}")
    print(f"  Type: {inp.element_type}")
    
print("\n=== OUTPUTS ===")
for i, out in enumerate(compiled.outputs):
    print(f"  [{i}] Name: {out.get_any_name()}")
    print(f"      Shape: {out.shape}")
    print(f"      Type: {out.element_type}")
