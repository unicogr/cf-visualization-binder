import json

# Read the notebook
with open('surfCFviz.ipynb', 'r') as f:
    nb = json.load(f)

# Find the imports cell and make it safer
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and i == 1:  # First code cell with imports
        # Wrap imports in try-except
        cell['source'] = [
            "# Safe imports with error handling\n",
            "import sys\n",
            "import warnings\n",
            "warnings.filterwarnings('ignore')\n",
            "\n",
            "# Core imports\n",
            "try:\n",
            "    import os\n",
            "    import glob\n",
            "    import yaml\n",
            "    from pathlib import Path\n",
            "    import pickle\n",
            "    import numpy as np\n",
            "    import nibabel as nib\n",
            "    print('✅ Core imports successful')\n",
            "except Exception as e:\n",
            "    print(f'❌ Core imports failed: {e}')\n",
            "    raise\n",
            "\n",
            "# Neuroimaging imports\n",
            "try:\n",
            "    import neuropythy as ny\n",
            "    from neuropythy.geometry import Mesh, Tesselation\n",
            "    print('✅ Neuropythy loaded')\n",
            "except Exception as e:\n",
            "    print(f'❌ Neuropythy failed: {e}')\n",
            "    print('Installing neuropythy...')\n",
            "    import subprocess\n",
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'neuropythy'])\n",
            "    import neuropythy as ny\n",
            "    from neuropythy.geometry import Mesh, Tesselation\n",
            "\n",
            "# Visualization imports\n",
            "try:\n",
            "    import pandas as pd\n",
            "    import ipyvolume as ipv\n",
            "    import matplotlib.pyplot as plt\n",
            "    from matplotlib.patches import Patch, Wedge\n",
            "    import matplotlib.colors as mcolors\n",
            "    from matplotlib import cm\n",
            "    from mpl_toolkits.axes_grid1.inset_locator import inset_axes\n",
            "    print('✅ Visualization imports successful')\n",
            "except Exception as e:\n",
            "    print(f'❌ Visualization imports failed: {e}')\n",
            "    raise\n",
            "\n",
            "# Additional scientific imports\n",
            "try:\n",
            "    from nilearn.surface import vol_to_surf\n",
            "    from nilearn import surface, plotting, signal\n",
            "    from scipy.spatial.distance import pdist, squareform\n",
            "    from scipy.spatial import cKDTree\n",
            "    from scipy.stats import pearsonr\n",
            "    print('✅ Scientific imports successful')\n",
            "except Exception as e:\n",
            "    print(f'⚠️ Some scientific imports failed: {e}')\n",
            "\n",
            "# Widget imports\n",
            "try:\n",
            "    from ipywidgets import FloatText, HBox, VBox, Textarea, Output, Dropdown, FloatSlider, interactive_output\n",
            "    from traitlets import link\n",
            "    from IPython.display import display\n",
            "    print('✅ Widget imports successful')\n",
            "except Exception as e:\n",
            "    print(f'❌ Widget imports failed: {e}')\n",
            "    raise\n",
            "\n",
            "# Optional imports\n",
            "try:\n",
            "    from brainspace.gradient import GradientMaps\n",
            "    print('✅ BrainSpace loaded')\n",
            "except:\n",
            "    print('⚠️ BrainSpace not available (optional)')\n",
            "\n",
            "import math\n",
            "import gc\n",
            "\n",
            "print('\\n✅ All critical imports completed!')\n"
        ]
        break

# Save the modified notebook
with open('surfCFviz.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✅ Notebook updated with safe imports")
