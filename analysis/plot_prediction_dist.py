import os
import json
import matplotlib.pyplot as plt
from collections import Counter


def run_plot_prediction_dist(pred_json_dir="outputs/predictions/", save_dir="outputs/plots/"):
    """
    Analyze and plot predicted character frequency from all prediction JSON files in a directory.

    Args:
        pred_json_dir (str): Directory containing prediction JSON files.
        save_dir (str): Directory to save the frequency plot.
    """

    if not os.path.isdir(pred_json_dir):
        print(f" Prediction directory not found: {pred_json_dir}")
        return

    pred_counter = Counter()

    # Aggregate predictions across all JSON files
    for fname in os.listdir(pred_json_dir):
        if not fname.endswith(".json"):
            continue

        json_path = os.path.join(pred_json_dir, fname)
        try:
            with open(json_path, "r") as f:
                predictions = json.load(f)
                for entry in predictions:
                    pred_text = entry.get("pred", "")
                    pred_counter.update(pred_text)
        except Exception as e:
            print(f" Failed to load {fname}: {e}")

    # If no predictions found
    if not pred_counter:
        print(" No predicted characters found in any prediction files.")
        return

    # Sort by frequency
    chars, freqs = zip(*sorted(pred_counter.items(), key=lambda x: -x[1]))

    # Plot
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 5))
    plt.bar(chars, freqs, color="skyblue", edgecolor="black")
    plt.title("Predicted Character Frequency")
    plt.xlabel("Character")
    plt.ylabel("Count")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "predicted_charset_frequency.png")
    plt.savefig(save_path)
    plt.close()

    print(f" Saved predicted character distribution plot to: {save_path}")


if __name__ == "__main__":
    run_plot_prediction_dist()
