import nbformat as nbf

nb = nbf.v4.new_notebook()

# Add test cells
nb.cells = [
    nbf.v4.new_markdown_cell("# Binder Test Notebook"),
    
    nbf.v4.new_code_cell("""
import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
"""),
    
    nbf.v4.new_code_cell("""
# Test imports
try:
    import numpy as np
    print(f"✅ numpy {np.__version__}")
except Exception as e:
    print(f"❌ numpy: {e}")

try:
    import matplotlib.pyplot as plt
    print(f"✅ matplotlib")
except Exception as e:
    print(f"❌ matplotlib: {e}")

try:
    import ipywidgets
    print(f"✅ ipywidgets {ipywidgets.__version__}")
except Exception as e:
    print(f"❌ ipywidgets: {e}")

try:
    import ipyvolume as ipv
    print(f"✅ ipyvolume {ipv.__version__}")
except Exception as e:
    print(f"❌ ipyvolume: {e}")

try:
    import nibabel as nib
    print(f"✅ nibabel {nib.__version__}")
except Exception as e:
    print(f"❌ nibabel: {e}")

try:
    import neuropythy as ny
    print(f"✅ neuropythy {ny.__version__}")
except Exception as e:
    print(f"❌ neuropythy: {e}")
"""),
    
    nbf.v4.new_code_cell("""
# Test simple widget
from ipywidgets import IntSlider, Output
import ipywidgets as widgets
from IPython.display import display

slider = IntSlider(value=5, min=0, max=10, description='Test:')
output = Output()

def on_value_change(change):
    with output:
        print(f"Value: {change['new']}")

slider.observe(on_value_change, names='value')
display(slider, output)
""")
]

with open('test_imports.ipynb', 'w') as f:
    nbf.write(nb, f)

print("✅ Created test_imports.ipynb")
