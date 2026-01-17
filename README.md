# Connective Field Visualization

Interactive visualization of whole-brain connective field mapping results.

## Launch Interactive App

Click the button below to launch the interactive visualization:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/YOUR_USERNAME/cf-visualization-binder/main?urlpath=voila%2Frender%2FsurfCFviz.ipynb)

## Features

- Interactive 3D brain surface visualization
- Multiple CF parameters: eccentricity, polar angle, CF size, R²
- Adjustable R² thresholds
- Custom and HSV colormaps for polar angle
- Real-time parameter updates

## Local Usage

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run with Voilà: `voila surfCFviz.ipynb`

## Data Requirements

This notebook requires:
- FreeSurfer subject files (`.surf`, `.curv`)
- CF model results (`.npz` files)

