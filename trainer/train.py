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

    # Start training over epochs
    for epoch in range(start_epoch, num_epochs):
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
        # Skip epoch if checkpoint already exists
        if os.path.exists(ckpt_path):
            print(f" Skipping epoch {epoch+1}, loading existing checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device) 
            model.load_state_dict(checkpoint["model_state"]) 
            optimizer.load_state_dict(checkpoint["optimizer_state"])

            # Optional: If you saved these in the checkpoint
            train_loss = checkpoint.get("train_loss", 0.0)
            val_ler = checkpoint.get("val_ler", 1.0)

            history["train_loss"].append(train_loss)
            history["val_ler"].append(val_ler)

            continue

        model.train()
        total_loss, steps = 0.0, 0

        # Initialize progress bar
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for X, targets, in_lens, tar_lens in pbar:
            # Move batch data to device
            X = X.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            in_lens = in_lens.to(device, non_blocking=True)
            tar_lens = tar_lens.to(device, non_blocking=True)

            # Forward pass
            logp = model(X)  # Output: (T, B, C)

            # Compute loss
            loss = crit(logp, targets, in_lens, tar_lens)

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Clip gradients for stability
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # Update weights
            optimizer.step()
            
            # Accumulate loss for reporting
            total_loss += float(loss)
            steps += 1

            # Update progress bar
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Validation phase using Levenshtein Error Rate (LER) 
        val_ler = evaluate_ler(
            model=model,
            dataloader=val_loader,
            device=device,
            idx2char=idx2char,
            blank=blank
        )

        # Average training loss for this epoch
        avg_loss = total_loss / max(steps, 1)
        history["train_loss"].append(avg_loss)
        history["val_ler"].append(val_ler)

        print(f"Epoch {epoch+1}/{num_epochs} - train_loss: {avg_loss:.4f}  val_LER: {val_ler:.4f}")

        # Save checkpoint for this epoch
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_ler": val_ler,
            "train_loss": avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"Checkpoint saved at: {ckpt_path}")

    # Save final model checkpoint after all epoch
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
