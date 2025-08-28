import os
import time
import yaml
import torch
import optuna
import subprocess
from torch.utils.data import DataLoader

from models.crnn import CRNN
from trainer.train import train
from trainer.evaluator import evaluate_ler, make_part2_predictions_json
from trainer.dataloader import CaptchaDataset, collate
from utils.seed import set_seed
from utils.parser import load_charset_from_labels
from utils.checkpoint import save_checkpoint
from utils.augmentor import apply_augmentation_to_folder
from utils.logger import Logger


def build_optimizer(name, model_params, lr):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(model_params, lr=lr)
    elif name == "adamw":
        return torch.optim.AdamW(model_params, lr=lr)
    elif name == "sgd":
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def run_from_config(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    output_base = cfg.get("output_base_dir", "outputs/tuning_optuna")

    # Setup output directories
    cfg["checkpoint_dir"] = os.path.join(output_base, "checkpoints")
    cfg["final_model_dir"] = os.path.join(output_base, "models")
    cfg["history_dir"] = os.path.join(output_base, "logs")
    cfg["log_file_path"] = os.path.join(output_base, "training_log.txt")
    cfg["analysis_dir"] = os.path.join(output_base, "analysis")
    cfg["prediction_dir"] = os.path.join(output_base, "predictions")

    for d in [cfg["checkpoint_dir"], cfg["final_model_dir"], cfg["history_dir"],
              cfg["analysis_dir"], cfg["prediction_dir"]]:
        os.makedirs(d, exist_ok=True)

    logger = Logger(cfg["log_file_path"])
    logger.section(f"Optuna Tuning Started from {config_path}")

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char2idx, idx2char, charset = load_charset_from_labels(cfg["label_path"])

    scenarios = cfg.get("augmentation_scenarios", [{"name": "default", "params": None}])
    best_model_path = None
    best_ler = float("inf")

    for scenario in scenarios:
        aug_name = scenario["name"]
        aug_params = scenario["params"]
        print(f"\nStarting augmentation scenario: {aug_name}")
        logger.section(f"Scenario: {aug_name}")

        # Apply augmentation if needed
        if aug_params:
            aug_image_dir = os.path.join(cfg["train_root"], f"images_{aug_name}")
            os.makedirs(aug_image_dir, exist_ok=True)
            apply_augmentation_to_folder(cfg["original_image_dir"], aug_image_dir, aug_params)
        else:
            aug_image_dir = cfg["original_image_dir"]

        aug_subdir = os.path.basename(aug_image_dir)
        train_ds = CaptchaDataset(cfg["train_root"], char2idx, image_subdir=aug_subdir)
        val_ds = CaptchaDataset(cfg["val_root"], char2idx)

        def objective(trial):
            batch_size = trial.suggest_categorical("batch_size", cfg["batch_size"])
            lr_range = [float(x) for x in cfg["learning_rate"]]
            lr = trial.suggest_float("lr", min(lr_range), max(lr_range), log=True)
            epochs = trial.suggest_categorical("epochs", cfg["epochs"])
            opt_name = trial.suggest_categorical("optimizer", cfg["optimizer"])

            trial_name = f"{aug_name}_trial_{trial.number + 1}"
            logger.section(f"Trial {trial.number + 1}: {trial_name}")
            logger.log(f"batch_size={batch_size}, lr={lr}, epochs={epochs}, optimizer={opt_name}")

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

            model = CRNN(num_classes=1 + len(charset)).to(device)
            optimizer = build_optimizer(opt_name, model.parameters(), lr)

            model_dir = os.path.join(cfg["final_model_dir"], trial_name)
            ckpt_dir = os.path.join(cfg["checkpoint_dir"], trial_name)
            hist_path = os.path.join(cfg["history_dir"], f"{trial_name}.json")
            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            model_path = os.path.join(model_dir, "final_model.pth")

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
                start_epoch=0
            )

            val_ler = evaluate_ler(model, val_loader, device, idx2char, blank=cfg["ctc_blank_index"])
            logger.log(f"Val LER: {val_ler:.4f}")

            # Save model path in user attributes
            trial.set_user_attr("model_path", model_path)
            trial.set_user_attr("val_ler", val_ler)

            return val_ler

        # Run tuning
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=cfg.get("num_trials", 10))

        best_trial = study.best_trial
        trial_model_path = best_trial.user_attrs["model_path"]
        trial_ler = best_trial.user_attrs["val_ler"]

        if trial_ler < best_ler:
            best_ler = trial_ler
            best_model_path = trial_model_path

    # Run analysis
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

    # Predict test set
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
