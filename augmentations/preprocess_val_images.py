import os
from PIL import Image
from tqdm import tqdm  # For progress bar

# Adjust the import if augmentation.py is in the same directory
from augmentation import estimate_bg_color, remove_background as remove_bg, make_glyphs_black

# Define relative paths based on current script location (Captcha-Cracker/augmentations)
input_dir = os.path.join("..", "data", "part2", "val", "images")
output_dir = os.path.join("..", "data", "part2", "val", "images_preprocessed")
os.makedirs(output_dir, exist_ok=True)

# Valid image file extensions
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")

# Basic preprocessing: background removal and glyph blackening
def preprocess_image(img: Image.Image) -> Image.Image:
    # Estimate background color and remove it
    bg_color = estimate_bg_color(img)
    img = remove_bg(img, bg_color)

    # Convert all non-white pixels to black
    img = make_glyphs_black(img)

    return img

# Iterate through all valid images in the input directory
for filename in tqdm(os.listdir(input_dir), desc="Processing images"):
    if not filename.lower().endswith(valid_extensions):
        continue  # Skip non-image files

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    try:
        img = Image.open(input_path)
        processed_img = preprocess_image(img)
        processed_img.save(output_path)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
