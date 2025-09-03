## Project Overview

This project focuses on developing and evaluating a robust CAPTCHA recognition model using CRNN (Convolutional Recurrent Neural Network) and CTC (Connectionist Temporal Classification) loss. The main objective is to evaluate different data augmentation strategies and hyperparameter configurations to improve the model's robustness and generalization.

### Technologies Used
- **PyTorch** for model implementation and training
- **CTC Loss** for sequence-level supervision without character alignment
- **Data Augmentation** with custom pipelines for realistic CAPTCHA distortions
- **Grid Search** for hyperparameter tuning across multiple training scenarios

### Key Features
- Modular augmentation system with full control over geometric, color, blur, noise, and distractor-line parameters
- YAML-based configuration for tuning multiple parameters at once
- Automated logging and checkpointing per trial
- Built-in analysis suite for visualizing learning curves, error samples, confidence scores, and character distribution

## Directory Structure

```
captcha-cracker/
├── configs/ # YAML configuration files for tuning and experiments
│ ├── tuning_S.yaml # small range tuning config
│ ├── tuning_M.yaml # medium range tuning config
│ ├── tuning_L.yaml # large range tuning config
│ ├── crnn_best_parameters.yaml # final settings for base model
│ └── crnn_best_parameters_augmentation_part3.yaml # final settings for augmented model
│
├── data/ # Dataset root directory
│ └── part2/ # Provided dataset (organized into train/val/test)
│
├── models/ # Model definitions
│ └── crnn.py # CRNN architecture
│
├── trainer/ # Training and evaluation logic
│ ├── dataloader.py # CaptchaDataset and collate function
│ ├── train.py # Training loop with checkpoints
│ ├── evaluator.py # Evaluation metrics and prediction utilities
│ └── tuner.py # Hyperparameter tuning orchestrator
│
├── utils/ # Utility modules
│ ├── augmentor.py # Data augmentation controller
│ ├── parser.py # Charset parsing and text-to-target utilities
│ ├── checkpoint.py # Save/load model checkpoints
│ ├── seed.py # Random seed control
│ └── logger.py # Simple logging utility
│
├── analysis/ # Post-analysis and visualization scripts
│ ├── run_all_analysis.py # Runner for executing multiple analysis tasks
│ ├── plot_curves.py
│ ├── compare_trials.py
│ ├── plot_charset_freq.py
│ └── plot_prediction_dist.py
│
├── outputs/ # Results from experiments (auto-generated)
│ └── tuning/ # Logs, checkpoints, Visualization, and models for tuning runs
│
├── main.py # Entry point to launch training/tuning
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

## Training and Tuning

### Running an Experiment
You can launch training or hyperparameter tuning with:

```bash
python main.py --config configs/tuning_M.yaml
```

- Replace `tuning_M.yaml` with `tuning_S.yaml` or `tuning_L.yaml` depending on the scale of your experiment.  
- The results (checkpoints, logs, final models) will be saved under `outputs/`.

---

### How Tuning Works
- The core training loop is implemented in **`trainer/train.py`**.
- Hyperparameter tuning is orchestrated by **`trainer/tuner.py`**, which performs **grid search**:
  - All combinations of specified parameters (`batch_size`, `learning_rate`, `epochs`, `optimizer`) are iterated automatically.
  - Each trial is run independently and logs its own history, checkpoints, and final model.
  - The best model is determined by **Validation LER (Levenshtein Error Rate)**.

---

### Configurable Parameters
All experiments are configured via **YAML files** under `configs/`.

Key parameters include:

- **Dataset paths**
  - `train_root`, `val_root`, `test_root`
  - `label_path` → JSON file with captcha labels
  - `original_image_dir` → un-augmented training images

- **Tuning settings**
  - `batch_size` → list of batch sizes to try (e.g. `[4, 8, 16]`)
  - `learning_rate` → learning rates in scientific notation (e.g. `[1e-3, 5e-4, 1e-4]`)
  - `epochs` → training epochs per trial
  - `optimizer` → choice of optimizer (`adam`, `adamw`, `sgd`)
  - `seed` → random seed for reproducibility
  - `ctc_blank_index` → blank token index for CTC loss

- **Augmentation scenarios**
  - Multiple named augmentation configurations can be defined.
  - Parameters: `angle_range`, `shear_range`, `brightness_range`, `contrast_range`, `noise_std`, `blur_probability`, `blur_radius`, `lines_probability`, `line_count`, `line_thickness`.

- **Output directories**
  - `checkpoint_dir`, `final_model_dir`, `history_dir`
  - `log_file_path`

- **Post-analysis**
  - `analysis_scripts` → list of visualization/analysis tasks to run automatically after tuning.

---

With this design, you can easily add new configs or augmentation scenarios without modifying the training code.




# Post Analysis

After training and hyperparameter tuning, this project provides **automatic post-analysis scripts** to help evaluate and visualize the results.  
These scripts run either manually (via `python analysis/run_all_analysis.py ...`) or automatically if specified in the config file.

## Automatic Execution
- The config file (`configs/*.yaml`) can specify which analysis scripts to run under the `analysis_scripts` section.
- After tuning, `tuner.py` will call these scripts automatically if they are listed.

## Available Analysis Scripts
All analysis utilities are stored in the `analysis/` folder. Each script serves a specific purpose:

- **plot_curves.py**: Plots training loss and validation LER curves for each trial.
- **compare_trials.py**: Compares final LER across all trials in a bar chart.
- **plot_charset_freq.py**: Analyzes and plots character frequency in training labels.
- **plot_prediction_dist.py**: Analyzes distribution of predicted characters across the test set.

These tools help debug model behavior, compare augmentation effects, and ensure the model generalizes well.


# Output Structure

All experiment results are saved under the **`outputs/`** directory.  
This folder is automatically created during training and organized into subfolders for clarity.

## Directory Layout

```
outputs/
├── tuning_L/                         # ← output_base_dir (e.g., tuning_L, tuning_small)
│   ├── checkpoints/                  # Checkpoints saved for each trial (per epoch)
│   │   ├── aug_type_A_trial_1/
│   │   │   ├── epoch_1.pth
│   │   │   ├── epoch_2.pth
│   │   │   └── checkpoint_best.pth
│   │   └── ...
│   │
│   ├── models/                       # Final and best models for each trial
│   │   ├── aug_type_A_trial_1/
│   │   │   ├── final_model.pth
│   │   │   └── best_model.pth
│   │   └── ...
│   │
│   ├── logs/                         # Training history logs (JSON) per trial
│   │   ├── aug_type_A_trial_1.json
│   │   └── aug_type_B_trial_2.json
│   │
│   ├── predictions/                  # Predictions on the test set (JSON)
│   │   ├── tuning_L_predictions.json
│   │   └── ...
│   │
│   ├── analysis/                     # All visualization outputs collected here
│   │   ├── curves/                   # Training curves (Loss and LER)
│   │   │   ├── aug_type_A_trial_1_loss.png
│   │   │   ├── aug_type_A_trial_1_ler.png
│   │   │   └── ...
│   │   │
│   │   ├── comparison/              # Bar charts comparing trials (e.g., LER)
│   │   │   └── final_ler_comparison.png
│   │   │
│   │   │
│   │   └──  charset/                 # Frequency distribution of predicted characters
│   │   │   └── predicted_charset_frequency.png
│   │
│   └── training_log.txt             # Global log file with trial-wise info

```

## Contents Explained

- **checkpoints/** → Per-epoch saved states for resuming training or debugging.
- **models/** → Final trained model and best-performing model per trial.
- **logs/** → Detailed training histories (`train_loss`, `val_ler`) in JSON format.
- **training_log.txt** → Human-readable log with trial hyperparameters and validation metrics.
- **predictions/** → Saved prediction results in JSON format for test evaluation.
- **plots/** → Generated analysis figures.

---


# Google Colab (Optional)

This project can also be run in **Google Colab** by using `main.ipynb`, making it easy to experiment without setting up a local environment.  
Two approaches are supported:

---

## 1. Manual Upload (ZIP files)
You can manually upload both the dataset and project folder as zip archives.

### Steps:
1. Upload your dataset (`part2.zip`) which must contain:
   ```
   part2/
   ├── train/
   ├── val/
   └── test/
   ```
2. Upload the project code (`captcha-cracker.zip`) containing:
   ```
   captcha-cracker/
   ├── main.py
   ├── configs/
   ├── trainer/
   └── ...
   ```
3. Use the provided Colab setup script to extract the files:
   ```python
   import os, zipfile

   # Extract dataset
   with zipfile.ZipFile("part2.zip", 'r') as zip_ref:
       zip_ref.extractall("data")

   # Extract project
   with zipfile.ZipFile("captcha-cracker.zip", 'r') as zip_ref:
       zip_ref.extractall("captcha-cracker")

   %cd captcha-cracker
   ```

4. Install dependencies and run training:
   ```bash
   !pip install -r requirements.txt
   !python main.py --config configs/tuning_M.yaml
   ```

---

## 2. GitHub Clone
Instead of uploading the zipped project folder, you can directly clone the repository:

```bash
!git clone https://github.com/captcha-cracker.git
%cd captcha-cracker
!pip install -r requirements.txt
!python main.py --config configs/tuning_M.yaml
```

---


