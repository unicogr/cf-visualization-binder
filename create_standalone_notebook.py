# This script creates a standalone version with inline color palettes
import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Add markdown cell with title
nb.cells.append(nbf.v4.new_markdown_cell("# Whole-brain Connective Field mapping"))

# Add imports cell
imports = '''import os
import glob
import yaml
from pathlib import Path
import pickle
import numpy as np
import nibabel as nib
import neuropythy as ny
from neuropythy.geometry import Mesh, Tesselation
import pandas as pd
import ipyvolume as ipv
from nilearn.surface import vol_to_surf
import matplotlib.pyplot as plt
from nilearn import surface, plotting
from matplotlib.patches import Patch, Wedge
import matplotlib.colors as mcolors
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import cKDTree
from ipywidgets import FloatText, HBox, VBox, Textarea, Output, Dropdown, FloatSlider, interactive_output
from traitlets import link
import math
import gc
from nilearn import signal
from scipy.stats import pearsonr'''

nb.cells.append(nbf.v4.new_code_cell(imports))

# Add color palette cell (inline)
color_palettes = '''# Inline color palettes (replacing cfmap.color_palettes)
def get_color_palettes():
    # Eccentricity colors (example - adjust to match your actual colors)
    eccen_hex = ['#FF0000', '#FF4500', '#FF8C00', '#FFD700', '#FFFF00', 
                 '#ADFF2F', '#00FF00', '#00FA9A', '#00CED1', '#1E90FF', 
                 '#0000FF', '#8A2BE2', '#9400D3', '#FF1493', '#FF69B4']
    
    # Polar colors (20-color wheel)
    polar_hex = ['#FF0000', '#FF4000', '#FF8000', '#FFBF00', '#FFFF00',
                 '#BFFF00', '#80FF00', '#40FF00', '#00FF00', '#00FF40',
                 '#00FF80', '#00FFBF', '#00FFFF', '#00BFFF', '#0080FF',
                 '#0040FF', '#0000FF', '#4000FF', '#8000FF', '#BF00FF']
    
    # Create matplotlib colormaps
    from matplotlib.colors import ListedColormap
    eccen_cmap = ListedColormap(eccen_hex)
    polar_cmap = ListedColormap(polar_hex)
    
    return {
        'eccentricity': {
            'hex': eccen_hex,
            'matplotlib_cmap': eccen_cmap
        },
        'polar': {
            'hex': polar_hex,
            'matplotlib_cmap': polar_cmap
        }
    }

color_palettes = get_color_palettes()
eccen_colors = color_palettes["eccentricity"]
polar_colors = color_palettes["polar"]'''

nb.cells.append(nbf.v4.new_code_cell(color_palettes))

# Save notebook
with open('surfCFviz_binder.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Created surfCFviz_binder.ipynb")
