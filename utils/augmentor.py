import os
import sys
from PIL import Image
from tqdm import tqdm

# Add the augmentations module to the path to allow import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "augmentations")))
import augmentation

def apply_augmentation_to_folder(input_dir, output_dir, params):
    """
    Applies augmentation to all PNG images in a folder.
    Skips already-processed images individually (file-by-file).
    Detects and handles corrupted images gracefully.
    
    Args:
        input_dir (str): Path to folder with original images
        output_dir (str): Path where augmented images will be saved
        params (dict): Dictionary of kwargs to pass to augment_image
    """
    os.makedirs(output_dir, exist_ok=True)

    img_files = [f for f in os.listdir(input_dir) if f.endswith(".png")]
    print(f" Augmenting {len(img_files)} images from {input_dir} → {output_dir}")

    skipped = 0
    processed = 0
    failed = 0
    corrupted_input = 0

    for fname in tqdm(img_files):
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        # Check if output file exists and is valid
        if os.path.exists(out_path):
            try:
                with Image.open(out_path) as im:
                    im.verify()  # Try to verify output image
                skipped += 1
                continue
            except Exception:
                print(f" Corrupted output file detected, removing: {out_path}")
                os.remove(out_path)

        # Check input image validity before proceeding
        try:
            with Image.open(in_path) as im:
                im.verify()
        except Exception as e:
            print(f" Corrupted input image skipped: {fname} ({e})")
            corrupted_input += 1
            continue

        try:
            img = Image.open(in_path).convert("RGB")
            aug_img = augmentation.augment_image(img, **params)
            aug_img.save(out_path)
            processed += 1
        except Exception as e:
            print(f" Failed to augment {fname}: {e}")
            failed += 1

    print(f"\n Augmentation completed:")
    print(f" {processed} images processed")
    print(f" {skipped} images already done")
    print(f" {corrupted_input} input images corrupted and skipped")
    print(f" {failed} failed during augmentation")
