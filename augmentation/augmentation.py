from typing import Optional, Tuple
import random

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

def estimate_bg_color(
        img: Image.Image
    ) -> Tuple[int, int, int]:
    """
    Estimate a plausible background color from image borders.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image (RGB recommended).

    Returns
    -------
    tuple[int, int, int]
        Estimated background color as (R, G, B).
    """
    arr = np.asarray(img.convert("RGB"))
    top = arr[0, :, :]
    bottom = arr[-1, :, :]
    left = arr[:, 0, :]
    right = arr[:, -1, :]
    border = np.concatenate([top, bottom, left, right], axis=0)
    rgb = np.median(border, axis=0).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def rotate_image(
    img: Image.Image,
    angle_range: Optional[Tuple[float, float]],
    fill: Tuple[int, int, int],
) -> Image.Image:
    """
    Rotate the image by a random angle within `angle_range`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    angle_range : tuple[float, float] or None
        (min_deg, max_deg). If None, rotation is skipped.
    fill : tuple[int, int, int]
        Fill color for exposed areas after rotation. (Calculated from image borders)

    Returns
    -------
    PIL.Image.Image
        Rotated image, cropped back to original size.
    """
    if not angle_range:
        return img

    w, h = img.size
    margin = int(0.1 * max(w, h))
    padded = TF.pad(img, [margin, margin, margin, margin], fill=fill)

    angle = random.uniform(angle_range[0], angle_range[1])
    rotated = TF.rotate(
        padded,
        angle=angle,
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=fill,
    )
    return TF.center_crop(rotated, [h, w])


def shear_image(
    img: Image.Image,
    shear_range: Optional[Tuple[float, float]],
    fill: Tuple[int, int, int],
) -> Image.Image:
    """
    Apply a horizontal shear with angle sampled from `shear_range`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    shear_range : tuple[float, float] or None
        (min_deg, max_deg). If None, shear is skipped.
    fill : tuple[int, int, int]
        Fill color for exposed areas.

    Returns
    -------
    PIL.Image.Image
        Sheared image.
    """
    if not shear_range:
        return img

    w, h = img.size
    margin = int(0.1 * max(w, h))
    padded = TF.pad(img, [margin, margin, margin, margin], fill=fill)

    shear = random.uniform(shear_range[0], shear_range[1])
    out = TF.affine(
        padded,
        angle=0.0,
        translate=(0, 0),
        scale=1.0,
        shear=[shear, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=fill,
    )
    return TF.center_crop(out, [h, w])


def resize_image(
    img: Image.Image,
    target_size: Optional[Tuple[int, int]],
) -> Image.Image:
    """
    Resize the image to a fixed width and height.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    target_size : tuple[int, int] or None
        (target_width, target_height). If None, scaling is skipped.
    Returns
    -------
    PIL.Image.Image
        Resized image with exact target dimensions.
    """
    if not target_size:
        return img

    target_w, target_h = target_size
    return img.resize((target_w, target_h), Image.BILINEAR)


def grayscale_image(img: Image.Image, propability: float) -> Image.Image:
    """
    Convert to 3-channel grayscale with probability `propability`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    p_gray : float
        Probability in [0, 1]. 0 → never, 1 → always.

    Returns
    -------
    PIL.Image.Image
        Possibly grayscaled image.
    """
    if random.random() >= propability:
        return img
    return TF.rgb_to_grayscale(img, num_output_channels=3)


def adjust_brightness(
    img: Image.Image,
    brightness_range: Optional[Tuple[float, float]],
) -> Image.Image:
    """
    Adjust image brightness.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    brightness_range : tuple[float, float] or None
        Multiplicative factor range (e.g., (0.85, 1.15)).
        If None, brightness is unchanged.

    Returns
    -------
    PIL.Image.Image
        Image with brightness adjustment.
    """
    if not brightness_range:
        return img
    b = random.uniform(brightness_range[0], brightness_range[1])
    return TF.adjust_brightness(img, b)


def adjust_contrast(
    img: Image.Image,
    contrast_range: Optional[Tuple[float, float]],
) -> Image.Image:
    """
    Adjust image contrast.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    contrast_range : tuple[float, float] or None
        Multiplicative factor range (e.g., (0.85, 1.15)).
        If None, contrast is unchanged.

    Returns
    -------
    PIL.Image.Image
        Image with contrast adjustment.
    """
    if not contrast_range:
        return img
    c = random.uniform(contrast_range[0], contrast_range[1])
    return TF.adjust_contrast(img, c)


def blur_image(
    img: Image.Image,
    p_blur: float,
    blur_radius: Optional[Tuple[float, float]],
) -> Image.Image:
    """
    Apply Gaussian blur with probability `p_blur`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    p_blur : float
        Probability in [0, 1].
    blur_radius : tuple[float, float] or None
        (min_radius, max_radius). If None, blur is skipped.

    Returns
    -------
    PIL.Image.Image
        Possibly blurred image.
    """
    if (random.random() >= p_blur) or (not blur_radius):
        return img
    radius = random.uniform(blur_radius[0], blur_radius[1])
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def add_noise(img: Image.Image, noise_std: Optional[Tuple[float, float]]) -> Image.Image:
    """
    Add zero-mean Gaussian noise on [0, 1] with std sampled from `noise_std`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    noise_std : tuple[float, float] or None
        Standard deviation range. If None or <= 0, noise is skipped.

    Returns
    -------
    PIL.Image.Image
        Noisy image.
    """
    if not noise_std:
        return img
    std = random.uniform(noise_std[0], noise_std[1])
    if std <= 0:
        return img
    t = TF.to_tensor(img)
    t = torch.clamp(t + torch.randn_like(t) * std, 0.0, 1.0)
    return TF.to_pil_image(t)


def add_lines(
    img: Image.Image,
    propability: float,
    line_count: Optional[Tuple[int, int]],
    line_thickness: Optional[Tuple[int, int]],
) -> Image.Image:
    """
    Draw a few thin distractor lines with probability `propability`.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    propability : float
        Probability in [0, 1].
    line_count : tuple[int, int] or None
        Inclusive range for number of lines. If None, skip drawing.
    line_thickness : tuple[int, int] or None
        Inclusive range for line thickness in pixels. If None, uses 1.

    Returns
    -------
    PIL.Image.Image
        Image with optional distractor lines.
    """
    if (random.random() >= propability) or (not line_count):
        return img

    out = img.copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    n = random.randint(line_count[0], line_count[1])
    thickness = 1
    if line_thickness:
        thickness = random.randint(line_thickness[0], line_thickness[1])

    # Choose a dark-ish line color relative to the estimated background.
    bg = estimate_bg_color(out)
    line_color = tuple(max(0, c - random.randint(40, 80)) for c in bg)

    for _ in range(n):
        x0, y0 = random.randint(0, w - 1), random.randint(0, h - 1)
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        draw.line([(x0, y0), (x1, y1)], fill=line_color, width=thickness)
    return out


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def augment_image(
    img: Image.Image,
    angle_range: Optional[Tuple[float, float]] = (-15, 15),
    shear_range: Optional[Tuple[float, float]] = (-10, 10),
    target_size: Optional[Tuple[int, int]] = (640, 160),
    grayscale_probability: float = 0.2,
    brightness_range: Optional[Tuple[float, float]] = (0.8, 1.2),
    contrast_range: Optional[Tuple[float, float]] = (0.8, 1.2),
    blur_probability: float = 0.3,
    blur_radius: Optional[Tuple[float, float]] = (0.5, 2.0),
    noise_std: Optional[Tuple[float, float]] = (0.01, 0.05),
    lines_probability: float = 0.1,
    line_count: Optional[Tuple[int, int]] = (1, 3),
    line_thickness: Optional[Tuple[int, int]] = (1, 4),
) -> Image.Image:
    """
    Apply comprehensive augmentation pipeline to an image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image.
    angle_range : tuple[float, float] or None
        Rotation angle range in degrees.
    shear_range : tuple[float, float] or None
        Shear angle range in degrees.
    target_size : tuple[int, int] or None
        Target (width, height) for resizing.
    grayscale_probability : float
        Probability of converting to grayscale.
    brightness_range : tuple[float, float] or None
        Brightness adjustment factor range.
    contrast_range : tuple[float, float] or None
        Contrast adjustment factor range.
    blur_probability : float
        Probability of applying blur.
    blur_radius : tuple[float, float] or None
        Gaussian blur radius range.
    noise_std : tuple[float, float] or None
        Noise standard deviation range.
    lines_probability : float
        Probability of adding distractor lines.
    line_count : tuple[int, int] or None
        Number of lines to add.
    line_thickness : tuple[int, int] or None
        Line thickness range.

    Returns
    -------
    PIL.Image.Image
        Augmented image.
    """
    # Estimate background color for fill operations
    background_color = estimate_bg_color(img)

    # Apply geometric transformations
    img = rotate_image(img, angle_range, background_color)
    img = shear_image(img, shear_range, background_color)
    img = resize_image(img, target_size)
    
    # Apply color transformations
    img = grayscale_image(img, grayscale_probability)
    img = adjust_brightness(img, brightness_range)
    img = adjust_contrast(img, contrast_range)
    
    # Apply noise and blur
    img = blur_image(img, blur_probability, blur_radius)
    img = add_noise(img, noise_std)
    
    # Add distractor elements
    img = add_lines(img, lines_probability, line_count, line_thickness)
    
    return img
