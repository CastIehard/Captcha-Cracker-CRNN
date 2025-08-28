import os
import json
import matplotlib.pyplot as plt


def run_plot_curves(history_dir="outputs/tuning_all/logs/", output_dir="outputs/plots/curves/"):
    """
    Plot training loss and validation LER curves for all history files.

    Args:
        history_dir (str): Directory containing JSON history files per trial.
        output_dir (str): Directory to save generated plots.
    """

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over each history file in the directory
    for fname in os.listdir(history_dir):
        if not fname.endswith(".json"):
            continue # Skip non-JSON files

        trial_name = os.path.splitext(fname)[0]
        json_path = os.path.join(history_dir, fname)

        # Load history from JSON
        with open(json_path, "r") as f:
            history = json.load(f)

        train_loss = history.get("train_loss", [])
        val_ler = history.get("val_ler", [])
        epochs = list(range(1, len(train_loss) + 1)) # X-axis for plotting

        # Plot Training Loss Curve
        plt.figure()
        plt.plot(epochs, train_loss, marker='o', label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{trial_name} - Train Loss")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{trial_name}_loss.png"))
        plt.close()

        # Plot Validation LER Curve
        plt.figure()
        plt.plot(epochs, val_ler, marker='o', color='red', label="Validation LER")
        plt.xlabel("Epoch")
        plt.ylabel("LER")
        plt.title(f"{trial_name} - Validation LER")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{trial_name}_ler.png"))
        plt.close()

        print(f" Plots saved for: {trial_name}")


if __name__ == "__main__":
    run_plot_curves()
