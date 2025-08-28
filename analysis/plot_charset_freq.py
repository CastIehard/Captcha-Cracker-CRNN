import os
import json
import matplotlib.pyplot as plt
from collections import Counter


def run_plot_charset_freq(label_path="data/part2/train/labels.json", save_dir="outputs/plots/"):
    """
    Analyze and plot character frequency distribution from labels.json.

    Args:
        label_path (str): Path to labels.json (Detectron-style).
        save_dir (str): Output directory to save the frequency plot.
    """

    # Check if label file exists
    if not os.path.exists(label_path):
        print(f" Label file not found: {label_path}")
        return

    # Load labels
    with open(label_path, "r") as f:
        labels = json.load(f)

    # Count character occurrences
    counter = Counter()
    for entry in labels:
        captcha = entry.get("captcha_string", "")
        counter.update(captcha)

    # Count character occurrences using Counter
    if not counter:
        print(" No characters found in labels.")
        return

    # Sort characters by frequency
    chars, freqs = zip(*sorted(counter.items(), key=lambda x: -x[1]))

    # Plot
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.bar(chars, freqs, color="salmon", edgecolor="black")
    plt.title("Character Frequency in Training Labels")
    plt.xlabel("Character")
    plt.ylabel("Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "charset_frequency.png")
    plt.savefig(save_path)
    plt.close()

    print(f" Saved character frequency plot to: {save_path}")


if __name__ == "__main__":
    run_plot_charset_freq()
