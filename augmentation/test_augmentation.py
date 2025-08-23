from typing import Dict, List, Tuple, Optional
import math
import random
from PIL import Image
import matplotlib.pyplot as plt

from augmentation import augment_image


def _disabled_kwargs() -> Dict:
    """
    Build a kwargs dict that disables all augmentations.

    Returns
    -------
    dict
        Keyword arguments for augment_image with all effects disabled/neutral.
    """
    return dict(
        angle_range=None,             # disable rotation
        shear_range=None,             # disable shear
        target_size=None,             # no resizing
        grayscale_probability=0.0,    # never grayscale
        brightness_range=None,        # no brightness jitter
        contrast_range=None,          # no contrast jitter
        blur_probability=0.0,         # never blur
        blur_radius=None,             # (kept None when blur disabled)
        noise_std=None,               # no noise
        lines_probability=0.0,        # never draw lines
        line_count=None,
        line_thickness=None,
    )


def _scenarios() -> List[Tuple[str, Dict]]:
    """
    Define one-augmentation-only scenarios.

    Returns
    -------
    list[tuple[str, dict]]
        List of (title, kwargs) for each subplot test.
    """
    s: List[Tuple[str, Dict]] = []

    # Baseline
    s.append(("Original", _disabled_kwargs()))

    # Geometry
    s.append(("Rotation +15°", {**_disabled_kwargs(), "angle_range": (15.0, 15.0)}))
    s.append(("Rotation -15°", {**_disabled_kwargs(), "angle_range": (-15.0, -15.0)}))
    s.append(("Shear +10°", {**_disabled_kwargs(), "shear_range": (20.0, 20.0)}))
    s.append(("Shear -10°", {**_disabled_kwargs(), "shear_range": (-20.0, -20.0)}))
    s.append(("Resize 640×160 (same of course)", {**_disabled_kwargs(), "target_size": (640, 160)}))

    # Photometric
    s.append(("Grayscale)", {**_disabled_kwargs(), "grayscale_probability": 1.0}))
    s.append(("Brightness ×0.7", {**_disabled_kwargs(), "brightness_range": (0.7, 0.7)}))
    s.append(("Brightness ×1.5", {**_disabled_kwargs(), "brightness_range": (1.5, 1.5)}))
    s.append(("Contrast ×0.7", {**_disabled_kwargs(), "contrast_range": (0.7, 0.7)}))
    s.append(("Contrast ×1.5", {**_disabled_kwargs(), "contrast_range": (1.5, 1.5)}))

    # Blur / Noise / Lines
    s.append((
        "Blur (r=2.0)",
        {**_disabled_kwargs(), "blur_probability": 1.0, "blur_radius": (2.0, 2.0)}
    ))
    s.append((
        "Noise (std=0.2)",
        {**_disabled_kwargs(), "noise_std": (0.2, 0.2)}
    ))
    s.append((
        "Noise (std=0.1)",
        {**_disabled_kwargs(), "noise_std": (0.1, 0.1)}
    ))
    s.append((
        "Lines (5 @ 2px)",
        {
            **_disabled_kwargs(),
            "lines_probability": 1.0,
            "line_count": (5, 5),
            "line_thickness": (2, 2),
        }
    ))

    return s


def plot_augmentations_grid(
    image_path: str,
    save_path: Optional[str] = None,
    seed: Optional[int] = 123,
    max_cols: int = 4,
    dpi: int = 150,
) -> None:
    """
    Render a grid where each subplot shows exactly one augmentation enabled.

    Parameters
    ----------
    image_path : str
        Path to the test image (PNG/JPG).
    save_path : str or None, default None
        If set, saves the figure to this path (e.g., "aug_grid.png").
    seed : int or None, default 123
        Global RNG seed for reproducible visual tests.
    max_cols : int, default 4
        Maximum number of columns in the subplot grid.
    dpi : int, default 150
        Figure DPI for saving/showing.
    """
    if seed is not None:
        random.seed(seed)

    img = Image.open(image_path).convert("RGB")
    scenarios = _scenarios()

    n = len(scenarios)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), dpi=dpi)
    # Normalize axes to a 2D list
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                title, kwargs = scenarios[idx]
                # Call your augmentation with only one effect enabled
                aug = augment_image(img.copy(), **kwargs)
                ax.imshow(aug)
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                idx += 1
            else:
                ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # Example usage: update the path to your local test image.
    plot_augmentations_grid(image_path="test.png",
                            save_path="aug_grid.png")
