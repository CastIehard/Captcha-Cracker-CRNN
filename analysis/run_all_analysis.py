import sys
import os

def main():
    """
    Entry point for running selected analysis scripts.
    """

    args = sys.argv[1:]

    # Default path
    output_base_dir = "outputs/tuning_small/"

    # Extract --output_base_dir if provided
    if "--output_base_dir" in args:
        idx = args.index("--output_base_dir")
        output_base_dir = args[idx + 1]
        del args[idx:idx + 2]  # Remove from args list so only tasks remain

    # Now args only contains task names
    tasks = args

    # Dynamically construct paths
    history_dir = os.path.join(output_base_dir, "logs")
    analysis_dir = os.path.join(output_base_dir, "analysis")
    pred_json_dir = os.path.join(output_base_dir, "predictions")

    if not tasks:
        print(" No analysis tasks specified. Provide script names like:")
        print("   python run_all_analysis.py plot_curves compare_trials")
        return

    print(f" Running analysis scripts: {tasks}")

    if "plot_curves" in tasks:
        from plot_curves import run_plot_curves
        run_plot_curves(history_dir=history_dir, output_dir=analysis_dir)

    if "compare_trials" in tasks:
        from compare_trials import run_comparison
        run_comparison(history_dir=history_dir, output_dir=analysis_dir)

    if "plot_charset_freq" in tasks:
        from plot_charset_freq import run_plot_charset_freq
        run_plot_charset_freq(save_dir=analysis_dir)

    if "plot_prediction_dist" in tasks:
        from plot_prediction_dist import run_plot_prediction_dist
        run_plot_prediction_dist(pred_json_dir=pred_json_dir, save_dir=analysis_dir)


if __name__ == "__main__":
    main()
