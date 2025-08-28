from PIL import Image
from torchvision.transforms.functional import to_tensor

# Determine correct resampling filter based on Pillow version
try:
    RESAMP_BILINEAR = Image.Resampling.BILINEAR  # Pillow >= 10
except AttributeError:
    RESAMP_BILINEAR = Image.BILINEAR             # Pillow < 10

MODEL_H = 32  # Target height for CRNN input

def preprocess(png_path):
    """
    Loads and resizes a CAPTCHA image to a fixed height, preserving aspect ratio.
    Converts to grayscale and returns a normalized tensor.

    Args:
        png_path (str): Path to the input PNG image.

    Returns:
        torch.Tensor: Image tensor of shape (1, H, W), normalized to [0,1]
    """
    
    img = Image.open(png_path).convert('L')  # Convert to grayscale

    w, h = img.size

    new_w = int(w * (MODEL_H / h)) # Compute new width to preserve aspect ratio given fixed MODEL_H

    img = img.resize((new_w, MODEL_H), RESAMP_BILINEAR) # Resize image to (new_w, MODEL_H) using bilinear interpolation

    return to_tensor(img)  # Returns (1, H, W) tensor in [0,1]
