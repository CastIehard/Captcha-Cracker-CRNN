import os
import json
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from trainer.evaluator import evaluate_ler
from utils.checkpoint import save_checkpoint


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    device,
    crit=None,
    idx2char=None,
    blank=0,
    save_path="outputs/checkpoints/crnn_captcha.pth",
    checkpoint_dir="outputs/checkpoints/",
    history_path="outputs/logs/train_history.json",
    start_epoch=0,
):
    """
    Core training loop with checkpoint and history logging.

    Args:
        model: CRNN model
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        optimizer: torch optimizer
        num_epochs: total number of epochs
        device: 'cuda' or 'cpu'
        crit: loss function (optional)
        idx2char: mapping for evaluation
        blank: CTC blank index
        save_path: path to final model
        checkpoint_dir: directory for saving intermediate checkpoints
        history_path: path to save training history JSON
        start_epoch: start index for resuming training
    """

    # Use default CTC Loss if not specified
    if crit is None:
        crit = nn.CTCLoss(blank=blank, zero_infinity=True)

    # Move model to target device
    model.to(device)

    # Ensure output directories exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize training history
    history = {
        "train_loss": [],
        "val_ler": []
    }

    # Load existing history if resuming
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                history = json.load(f)
        except Exception:
            print(f" Failed to load existing history from {history_path}, starting fresh.")

    # Load previous checkpoint if resuming
    if start_epoch > 0:
        prev_ckpt_path = os.path.join(checkpoint_dir, f"epoch_{start_epoch}.pth")
        if os.path.exists(prev_ckpt_path):
            print(f"Loading previous checkpoint from {prev_ckpt_path} to resume from epoch {start_epoch+1}")
            checkpoint = torch.load(prev_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        else:
            print(f" Warning: Checkpoint for epoch {start_epoch} not found. Starting fresh.")

    # Start training from specified epoch
    for epoch in range(start_epoch, num_epochs):
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")

        # If checkpoint exists, skip training and just record from it
        if os.path.exists(ckpt_path):
            print(f" Skipping epoch {epoch+1}, loading existing checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Use logged metrics if available
            train_loss = checkpoint.get("train_loss", 0.0)
            val_ler = checkpoint.get("val_ler", 1.0)
            history["train_loss"].append(train_loss)
            history["val_ler"].append(val_ler)
            continue

        model.train()
        total_loss, steps = 0.0, 0

        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for X, targets, in_lens, tar_lens in pbar:
            X = X.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            in_lens = in_lens.to(device, non_blocking=True)
            tar_lens = tar_lens.to(device, non_blocking=True)

            logp = model(X)
            loss = crit(logp, targets, in_lens, tar_lens)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss)
            steps += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        val_ler = evaluate_ler(model, val_loader, device, idx2char, blank)
        avg_loss = total_loss / max(steps, 1)
        history["train_loss"].append(avg_loss)
        history["val_ler"].append(val_ler)

        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f}  val_LER: {val_ler:.4f}")

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_ler": val_ler,
            "train_loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved at: {ckpt_path}")

    # Save final model checkpoint
    final_ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": num_epochs,
        "val_ler": history["val_ler"][-1] if history["val_ler"] else None,
    }
    torch.save(final_ckpt, save_path)
    print(f" Final model saved to {save_path}")

    # Save training history
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f" Training history saved to {history_path}")
