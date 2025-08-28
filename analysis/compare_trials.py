import os
import json
import matplotlib.pyplot as plt

def run_comparison(history_dir, output_dir):
    """
    Compare validation LER across all trials and plot as a bar chart.

    Args:
        history_dir (str): Directory containing JSON history files.
        output_dir (str): Directory to save the comparison plot.
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    trial_names = []
    final_lers = []

    # Iterate through all JSON history files
    for fname in os.listdir(history_dir):
        if not fname.endswith(".json"):
            continue

        trial_name = os.path.splitext(fname)[0]
        json_path = os.path.join(history_dir, fname)

        with open(json_path, "r") as f:
            history = json.load(f)

        val_ler = history.get("val_ler", [])
        if val_ler:
            trial_names.append(trial_name)
            final_lers.append(val_ler[-1])  # Last epoch's LER

    if not trial_names:
        print(" No trials found to compare.")
        return

    # Sort trials by final LER
    sorted_trials = sorted(zip(trial_names, final_lers), key=lambda x: x[1])
    trial_names, final_lers = zip(*sorted_trials)

    # Plot bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(trial_names, final_lers, color='skyblue')
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Final Validation LER")
    plt.title("Comparison of Final LER Across Trials")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "compare_ler_across_trials.png")
    plt.savefig(plot_path)
    plt.close()

    print(f" Saved comparison plot: {plot_path}")
