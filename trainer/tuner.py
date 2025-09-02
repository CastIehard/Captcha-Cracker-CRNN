import os
import yaml
import shutil
import torch
import optuna
import subprocess
from optuna.trial import TrialState
from torch.utils.data import DataLoader, Subset

from models.crnn import CRNN
from trainer.train import train
from trainer.evaluator import evaluate_ler, make_part2_predictions_json
from trainer.dataloader import CaptchaDataset, collate
from utils.seed import set_seed
from utils.parser import load_charset_from_labels
from utils.logger import Logger
from utils.augmentor import apply_augmentation_to_folder


def build_optimizer(name, model_params, lr):
    # Choose optimizer based on name
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def get_latest_epoch(ckpt_dir):
    # Returns the highest checkpoint epoch number from the checkpoint directory
    if not os.path.exists(ckpt_dir):
        return 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith("epoch_") and f.endswith(".pth")]
    if not files:
        return 0
    return max(int(f.replace("epoch_", "").replace(".pth", "")) for f in files)


def run_from_config(config_path):
    # Load config YAML
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Set up output directory paths
    output_base = cfg.get("output_base_dir", "outputs/tuning_optuna")
    cfg["checkpoint_dir"] = os.path.join(output_base, "checkpoints")
    cfg["final_model_dir"] = os.path.join(output_base, "models")
    cfg["history_dir"] = os.path.join(output_base, "logs")
    cfg["log_file_path"] = os.path.join(output_base, "training_log.txt")
    cfg["analysis_dir"] = os.path.join(output_base, "analysis")
    cfg["prediction_dir"] = os.path.join(output_base, "predictions")

    # Create required folders if they don't exist
    for d in [cfg["checkpoint_dir"], cfg["final_model_dir"], cfg["history_dir"],
              cfg["analysis_dir"], cfg["prediction_dir"]]:
        os.makedirs(d, exist_ok=True)

    logger = Logger(cfg["log_file_path"])
    logger.section(f"Optuna Tuning Started from {config_path}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load charset from label file
    char2idx, idx2char, charset = load_charset_from_labels(cfg["label_path"])

    scenarios = cfg.get("augmentation_scenarios", [{"name": "default", "params": None}])
    best_model_path = None
    best_ler = float("inf")

    for scenario in scenarios:
        aug_name = scenario["name"]
        aug_params = scenario["params"]
        print(f"\nStarting augmentation scenario: {aug_name}")
        logger.section(f"Scenario: {aug_name}")

        # Augmentation
        if aug_params:
            aug_image_dir = os.path.join(cfg["train_root"], f"images_{aug_name}")
            os.makedirs(aug_image_dir, exist_ok=True)
            apply_augmentation_to_folder(cfg["original_image_dir"], aug_image_dir, aug_params)
        else:
            aug_image_dir = cfg["original_image_dir"]

        aug_subdir = os.path.basename(aug_image_dir)
        train_ds = CaptchaDataset(cfg["train_root"], char2idx, image_subdir=aug_subdir)
        val_ds = CaptchaDataset(cfg["val_root"], char2idx)

        # Optional: restrict dataset size for debugging
        #train_ds = Subset(train_ds, list(range(min(len(train_ds), 1000))))
        #val_ds = Subset(val_ds, list(range(min(len(val_ds), 1000))))

        # Set up or resume Optuna study
        study = optuna.create_study(
            direction="minimize",
            study_name=f"study_{aug_name}",
            storage="sqlite:///optuna_study.db",
            load_if_exists=True
        )

        # Manually assign consistent trial numbers
        trial_id_map = {}
        existing_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        for i, trial in enumerate(existing_trials):
            trial_id_map[trial._trial_id] = i

        current_trial_index = len(trial_id_map)

        def objective(trial):
            nonlocal current_trial_index

            # Ensure consistent trial folder names by manual tracking
            trial_id = trial._trial_id
            if trial_id not in trial_id_map:
                trial_id_map[trial_id] = current_trial_index
                current_trial_index += 1
            trial_number = trial_id_map[trial_id]
            trial_name = f"{aug_name}_trial_{trial_number + 1}"

            # hyperparameters
            batch_size = trial.suggest_categorical("batch_size", cfg["batch_size"])
            lr = trial.suggest_float("lr", float(cfg["learning_rate"][0]), float(cfg["learning_rate"][1]), log=True)
            epochs = trial.suggest_categorical("epochs", cfg["epochs"])
            opt_name = trial.suggest_categorical("optimizer", cfg["optimizer"])

            logger.section(f"Trial {trial_number + 1}: {trial_name}")
            logger.log(f"batch_size={batch_size}, lr={lr}, epochs={epochs}, optimizer={opt_name}")

            # Build dataloaders
            train_loader = DataLoader(
                train_ds, 
                batch_size=batch_size, 
                shuffle=True, 
                collate_fn=collate,
                num_workers=10,
                )
            val_loader = DataLoader(
                val_ds, 
                batch_size=batch_size, 
                shuffle=False, 
                collate_fn=collate,
                num_workers=10
                )

            model = CRNN(num_classes=1 + len(charset)).to(device)
            optimizer = build_optimizer(opt_name, model.parameters(), lr)

            model_dir = os.path.join(cfg["final_model_dir"], trial_name)
            ckpt_dir = os.path.join(cfg["checkpoint_dir"], trial_name)
            hist_path = os.path.join(cfg["history_dir"], f"{trial_name}.json")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            model_path = os.path.join(model_dir, "final_model.pth")
            start_epoch = get_latest_epoch(ckpt_dir)

            # Train the model
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=epochs,
                device=device,
                idx2char=idx2char,
                blank=cfg["ctc_blank_index"],
                save_path=model_path,
                checkpoint_dir=ckpt_dir,
                history_path=hist_path,
                start_epoch=start_epoch
            )

            # Evaluate validation LER
            val_ler = evaluate_ler(model, val_loader, device, idx2char, blank=cfg["ctc_blank_index"])
            logger.log(f"Val LER: {val_ler:.4f}")
            trial.set_user_attr("model_path", model_path)
            trial.set_user_attr("val_ler", val_ler)

            return val_ler

        # Resume only remaining trials
        remaining_trials = cfg.get("num_trials", 10) - len(existing_trials)
        if remaining_trials > 0:
            print(f" Resuming study: {study.study_name} — {len(existing_trials)} completed, {remaining_trials} remaining")
            study.optimize(objective, n_trials=remaining_trials)
        else:
            print(f" Study already completed: {len(existing_trials)}/{cfg.get('num_trials', 10)} trials.")

        # Record best model if LER improved
        best_trial = study.best_trial
        trial_model_path = best_trial.user_attrs["model_path"]
        trial_ler = best_trial.user_attrs["val_ler"]
        if trial_ler < best_ler:
            best_ler = trial_ler
            best_model_path = trial_model_path

    # Run visualization/analysis scripts if specified
    visualizations = cfg.get("analysis_scripts", [])
    if visualizations:
        logger.section("Running Analysis Scripts")
        try:
            subprocess.run(
                ["python", "analysis/run_all_analysis.py", "--output_base_dir", output_base] + visualizations,
                check=True
            )
            logger.log("Analysis scripts completed.")
        except subprocess.CalledProcessError as e:
            logger.log(f"Analysis failed: {e}")

    # Generate predictions using the best model
    print("\nGenerating predictions on test set...")
    os.makedirs(cfg["prediction_dir"], exist_ok=True)
    pred_json_path = os.path.join(
        cfg["prediction_dir"],
        f"{os.path.basename(config_path).split('.')[0]}_predictions.json"
    )

    final_model = CRNN(num_classes=1 + len(charset)).to(device)
    checkpoint = torch.load(best_model_path, map_location=device)
    final_model.load_state_dict(checkpoint["model_state"])
    final_model.eval()

    make_part2_predictions_json(
        model=final_model,
        test_root=cfg["test_root"],
        out_json_path=pred_json_path,
        device=device,
        idx2char=idx2char,
        blank=cfg["ctc_blank_index"]
    )

    print(f"Predictions saved to {pred_json_path}")
