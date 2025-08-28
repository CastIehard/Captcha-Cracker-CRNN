import os
import time
import yaml
import itertools
import torch
import subprocess
import numpy as np
from torch.utils.data import DataLoader

from utils.parser import load_charset_from_labels
from trainer.dataloader import CaptchaDataset, collate
from models.crnn import CRNN
from trainer.train import train
from trainer.evaluator import evaluate_ler, make_part2_predictions_json
from utils.seed import set_seed
from utils.checkpoint import save_checkpoint
from utils.augmentor import apply_augmentation_to_folder
from utils.logger import Logger



def get_param_grid(cfg):
    """
    Generate all combinations of hyperparameters for grid search.

    Args:
        cfg (dict): Configuration dictionary loaded from YAML.

    Returns:
        List of tuples: Each tuple contains one combination of (batch_size, learning_rate, epochs, optimizer).
    """

    return list(itertools.product(
        cfg['batch_size'],
        cfg['learning_rate'],
        cfg['epochs'],
        cfg.get('optimizer', ['adam']) # Default optimizer is Adam if not specified
    ))


def build_optimizer(name, model_params, lr):
    """
    Create an optimizer instance based on the name.

    Args:
        name (str): Optimizer name (adam, adamw, sgd).
        model_params (iterable): Model parameters to optimize.
        lr (float): Learning rate.

    Returns:
        torch.optim.Optimizer: Initialized optimizer.

    Raises:
        ValueError: If the optimizer name is not supported.
    """
    
    lr = float(lr)
    name = name.lower()
    if name == 'adam':
        return torch.optim.Adam(model_params, lr=lr)
    elif name == 'adamw':
        return torch.optim.AdamW(model_params, lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(model_params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def run_from_config(config_path):
    """
    Run the full training and evaluation pipeline using a configuration file.

    Args:
        config_path (str): Path to YAML configuration file.
    """

    # Start measuring total runtime
    overall_start = time.time()

    # Load configuration
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load base output directory
    output_base = cfg.get("output_base_dir", "outputs/tuning_small")

    # Automatically derive all related directories
    cfg["checkpoint_dir"]   = os.path.join(output_base, "checkpoints")
    cfg["final_model_dir"]  = os.path.join(output_base, "models")
    cfg["history_dir"]      = os.path.join(output_base, "logs")
    cfg["log_file_path"]    = os.path.join(output_base, "training_log.txt")
    cfg["analysis_dir"]     = os.path.join(output_base, "analysis")

    # Get log file path from config
    log_file_path = cfg.get("log_file_path", "training_log.txt")
    logger = Logger(log_file_path) # Initialize logger

    # Ensure the directory for the log file exists
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Log the start of the training run
    logger.section(f"New Run from Config: {config_path}")

    # Set random seed
    set_seed(cfg['seed'])

    # Load character set from label file
    char2idx, idx2char, charset = load_charset_from_labels(cfg['label_path'])

    # Generate all hyperparameter combinations
    param_grid = get_param_grid(cfg)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # Initialize best validation LER and model path
    best_ler = float("inf")
    best_model_path = None

    # Get augmentation scenarios from config, or use default
    scenarios = cfg.get("augmentation_scenarios", [{"name": "default", "params": None}])

    for scenario in scenarios:
        aug_name = scenario['name']
        aug_params = scenario['params']
        print(f"\n Starting augmentation scenario: {aug_name}")

        # Apply data augmentation if parameters are provided
        if aug_params is not None:
            aug_image_dir = os.path.join(cfg['train_root'], f"images_{aug_name}") # change the dir name if needed
            os.makedirs(aug_image_dir, exist_ok=True)
            apply_augmentation_to_folder(
                input_dir=cfg['original_image_dir'],
                output_dir=aug_image_dir,
                params=aug_params
            )
        else:
            aug_image_dir = cfg['original_image_dir']

        # Extract image subfolder name for dataset initialization
        aug_subdir = os.path.basename(aug_image_dir)

        # Create full training and validation datasets
        train_ds = CaptchaDataset(cfg['train_root'], char2idx, image_subdir=aug_subdir)
        val_ds = CaptchaDataset(cfg['val_root'], char2idx)

        # Loop through each hyperparameter combination (grid search)
        for i, (batch_size, lr, epochs, opt_name) in enumerate(param_grid):
            trial_start = time.time()  # Start trial timer
            trial_name = f"{aug_name}_trial_{i+1}" # change the dir name if needed

            print(f"\n Trial {i+1}/{len(param_grid)} under {aug_name}:\n"
                  f"   batch_size={batch_size}, lr={lr}, epochs={epochs}, optimizer={opt_name}")
            
            logger.section(f"Trial {i+1}/{len(param_grid)}: {trial_name}") # Log trial header
            logger.log(f"batch_size={batch_size}, lr={lr}, epochs={epochs}, optimizer={opt_name}") # Log hyperparameters

            # Create DataLoaders for training and validation
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, pin_memory=(device == 'cuda'))
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, pin_memory=(device == 'cuda'))

            # Initialize model and optimizer
            model = CRNN(num_classes=1 + len(charset)).to(device)
            optimizer = build_optimizer(opt_name, model.parameters(), lr)

            # Setup paths for saving model, checkpoints, and training history
            model_dir = os.path.join(cfg['final_model_dir'], trial_name)
            ckpt_dir = os.path.join(cfg['checkpoint_dir'], trial_name)
            hist_path = os.path.join(cfg['history_dir'], f"{trial_name}.json")

            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

            model_path = os.path.join(model_dir, "final_model.pth")

            # Start training loop
            train(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                num_epochs=epochs,
                device=device,
                idx2char=idx2char,
                blank=cfg['ctc_blank_index'],
                save_path=model_path,
                checkpoint_dir=ckpt_dir,
                history_path=hist_path,
                start_epoch=0
            )

            # Evaluate model on validation set using LER
            val_ler = evaluate_ler(model, val_loader, device, idx2char, blank=cfg['ctc_blank_index'])
            print(f"{trial_name} - Val LER: {val_ler:.4f}")
            logger.log(f"Val LER: {val_ler:.4f}") # Log validation LER

            trial_end = time.time()
            trial_minutes = (trial_end - trial_start) / 60
            print(f" Trial {i+1} duration: {trial_minutes:.2f} minutes")

            logger.log(f"Trial duration: {trial_minutes:.2f} minutes") # Log trial duration

            # Save model checkpoint if this is the best so far
            if val_ler < best_ler:
                best_ler = val_ler
                best_model_path = model_path

                # Copy final_model.pth to best_model.pth (without shutil)
                src = model_path
                dst = os.path.join(model_dir, "best_model.pth")
                with open(src, "rb") as f_src, open(dst, "wb") as f_dst:
                    f_dst.write(f_src.read())

                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epochs,
                    path=os.path.join(ckpt_dir, "checkpoint_best.pth"),
                    extra_info={"val_ler": val_ler}
                )

    print(f"\n Best model overall: {best_model_path} (val_LER = {best_ler:.4f})")
    logger.section("Training Complete") # Log end of tuning
    logger.log(f"Best model overall: {best_model_path} (val_LER = {best_ler:.4f})") # Log best model info

    # End of entire tuning process
    overall_end = time.time()
    total_minutes = (overall_end - overall_start) / 60

    print(f"\n Total tuning time: {total_minutes:.2f} minutes")
    logger.log(f"Total tuning time: {total_minutes:.2f} minutes") # Log total runtime


    # Run analysis scripts if specified in config
    analysis_dir = cfg["analysis_dir"]
    os.makedirs(analysis_dir, exist_ok=True)

    visualizations = cfg.get("analysis_scripts", [])
    if visualizations:
        logger.section("Running Analysis Scripts")
        logger.log(f"Executing: {visualizations}")
        try:
            subprocess.run(
                ["python", "analysis/run_all_analysis.py", "--output_base_dir", output_base] + visualizations,
                check=True
            )
            logger.log(" All analysis scripts ran successfully.")
        except subprocess.CalledProcessError as e:
            logger.log(f" Analysis script failed: {e}")


    # Run predictions on test set (optional)
    print("\nGenerating predictions on test set...")

    # Define prediction directory under output_base
    prediction_dir = os.path.join(cfg["output_base_dir"], "predictions")
    os.makedirs(prediction_dir, exist_ok=True)

    # Set prediction output path
    pred_json_path = os.path.join(prediction_dir, f"{os.path.basename(config_path).split('.')[0]}_predictions.json")

    # Reload best model
    best_model = CRNN(num_classes=1 + len(charset)).to(device)
    checkpoint = torch.load(best_model_path, map_location=device)
    best_model.load_state_dict(checkpoint["model_state"])
    best_model.eval()

    make_part2_predictions_json(
        model=best_model,
        test_root=cfg["test_root"],
        out_json_path=pred_json_path,
        device=device,
        idx2char=idx2char,
        blank=cfg["ctc_blank_index"]
    )

    print(f" Predictions saved to {pred_json_path}")