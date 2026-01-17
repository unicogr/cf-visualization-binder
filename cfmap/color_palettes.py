import matplotlib.colors as mcolors



def get_color_palettes():
    """
    Returns color palettes for eccentricity and polar angle mappings.
    
    Returns:
        dict: Dictionary containing eccentricity and polar palettes in various formats
    """
    
    # Eccentricity palette RGB values (0-255)
    ecc_rgb_values = [
        [255, 40, 0],   # Red
        [255, 130, 0],  # Orange-red
        [255, 210, 0],  # Orange-yellow
        [255, 255, 0],  # Yellow
        [115, 255, 0],  # Yellow-green
        [31, 255, 0],   # Green
        [0, 255, 207],  # Turquoise
        [0, 231, 255],  # Cyan
        [20, 140, 255], # Light blue
        [40, 60, 255]   # Blue
    ]
    
    # Polar palette RGB values (0-255)
    polar_rgb_values = [
        [106, 189, 69],   # Color1
        [203, 219, 42],   # Color2
        [254, 205, 8],    # Color3
        [242, 104, 34],   # Color4
        [237, 32, 36],    # Color5
        [237, 32, 36],    # Color6
        [242, 104, 34],   # Color7
        [254, 205, 8],    # Color8
        [203, 219, 42],   # Color9
        [106, 189, 69],   # Color10
        [106, 189, 69],   # Color11
        [110, 205, 221],  # Color12
        [50, 178, 219],   # Color13
        [62, 105, 179],   # Color14
        [57, 84, 165],    # Color15
        [57, 84, 165],    # Color16
        [62, 105, 179],   # Color17
        [50, 178, 219],   # Color18
        [110, 205, 221],  # Color19
        [106, 189, 69]    # Color20
    ]
    
    def create_palette_formats(rgb_values):
        """Helper function to create different palette formats."""
        # Normalize to 0-1 range for matplotlib
        norm_values = [[r/255, g/255, b/255] for r, g, b in rgb_values]
        
        # Create hex values for other libraries
        hex_values = [mcolors.rgb2hex(rgb) for rgb in norm_values]
        
        # Create named colors with format for easy access
        named_colors = {f"color{i+1}": hex_values[i] for i in range(len(hex_values))}
        
        return {
            "rgb_0_255": rgb_values,         # Original RGB (0-255)
            "rgb_0_1": norm_values,          # Normalized RGB (0-1)
            "hex": hex_values,               # Hex color codes
            "named": named_colors,           # Named colors
            "matplotlib_cmap": mcolors.LinearSegmentedColormap.from_list("custom_cmap", norm_values)  # Matplotlib colormap
        }
    
    # Create palettes
    eccentricity_palette = create_palette_formats(ecc_rgb_values)
    polar_palette = create_palette_formats(polar_rgb_values)
    
    return {
        "eccentricity": eccentricity_palette,
        "polar": polar_palette
    }