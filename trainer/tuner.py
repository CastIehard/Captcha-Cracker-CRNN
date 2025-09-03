import os
import yaml
import torch
import subprocess
import random
import json
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


def sample_random_params(cfg, existing_params=None):
    """Sample random parameters from config ranges, avoiding duplicates"""
    if existing_params is None:
        existing_params = set()
    
    max_attempts = 1000
    for _ in range(max_attempts):
        # Sample parameters
        batch_size = random.choice(cfg["batch_size"])
        
        # Handle learning rate range or list
        if len(cfg["learning_rate"]) == 2 and isinstance(cfg["learning_rate"][0], (int, float)):
            # Range: sample log-uniform between min and max
            lr_min, lr_max = cfg["learning_rate"]
            lr = random.uniform(lr_min, lr_max)
        else:
            # List: sample from choices
            lr = random.choice(cfg["learning_rate"])
            
        epochs = random.choice(cfg["epochs"])
        optimizer = random.choice(cfg["optimizer"])
        
        # Create parameter tuple for duplicate checking
        param_tuple = (batch_size, lr, epochs, optimizer)
        
        if param_tuple not in existing_params:
            return {
                "batch_size": batch_size,
                "learning_rate": lr,
                "epochs": epochs,
                "optimizer": optimizer
            }, param_tuple
    
    # If we can't find unique params after many attempts, just return a random one
    return {
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": epochs,
        "optimizer": optimizer
    }, param_tuple


def load_existing_trials(trial_log_path):
    """Load existing trial parameters from log file"""
    existing_params = set()
    if os.path.exists(trial_log_path):
        try:
            with open(trial_log_path, 'r') as f:
                trials = json.load(f)
                for trial in trials:
                    params = trial['params']
                    param_tuple = (
                        params['batch_size'],
                        params['learning_rate'],
                        params['epochs'],
                        params['optimizer']
                    )
                    existing_params.add(param_tuple)
        except (json.JSONDecodeError, KeyError):
            pass
    return existing_params


def save_trial_result(trial_log_path, trial_num, params, val_ler, model_path):
    """Save trial result to log file"""
    trial_data = {
        "trial": trial_num,
        "params": params,
        "val_ler": val_ler,
        "model_path": model_path
    }
    
    # Load existing trials
    trials = []
    if os.path.exists(trial_log_path):
        try:
            with open(trial_log_path, 'r') as f:
                trials = json.load(f)
        except json.JSONDecodeError:
            trials = []
    
    # Add new trial
    trials.append(trial_data)
    
    # Save back
    with open(trial_log_path, 'w') as f:
        json.dump(trials, f, indent=2)


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
    output_base = cfg.get("output_base_dir", "outputs/training")
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
    logger.section(f"Training Started from {config_path}")

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

        # Set up trial tracking
        trial_log_path = os.path.join(cfg["history_dir"], f"{aug_name}_trials.json")
        existing_params = load_existing_trials(trial_log_path)
        num_trials = cfg.get("num_trials", 10)
        
        print(f"Running {num_trials} trials for scenario: {aug_name}")
        print(f"Found {len(existing_params)} existing trials")
        
        # Run trials
        for trial_num in range(1, num_trials + 1):
            print(f"\n--- Trial {trial_num}/{num_trials} ---")
            
            # Sample random parameters
            params, param_tuple = sample_random_params(cfg, existing_params)
            existing_params.add(param_tuple)
            
            batch_size = params["batch_size"]
            lr = params["learning_rate"]
            epochs = params["epochs"]
            opt_name = params["optimizer"]
            
            trial_name = f"{aug_name}_trial_{trial_num}"
            
            logger.section(f"Trial {trial_num}: {trial_name}")
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
            
            # Save trial result
            save_trial_result(trial_log_path, trial_num, params, val_ler, model_path)
            
            # Update best model if this trial performed better
            if val_ler < best_ler:
                best_ler = val_ler
                best_model_path = model_path
                print(f"New best LER: {val_ler:.4f}")

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
    logger.log(f"Training completed. Best LER: {best_ler:.4f}")
    logger.log(f"Best model saved at: {best_model_path}")
    logger.log(f"Predictions saved to: {pred_json_path}")
