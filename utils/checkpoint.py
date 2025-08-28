import torch
import os

def save_checkpoint(model, optimizer, epoch, path, extra_info=None):
    """
    Saves training checkpoint including model, optimizer, and metadata.

    Args:
        model (nn.Module): The model to save
        optimizer (Optimizer): The optimizer to save
        epoch (int): Current epoch number
        path (str): Path to save checkpoint (e.g., 'checkpoints/ckpt.pth')
        extra_info (dict): Optional metadata (e.g., val_acc)
    """

    # Ensure the directory for the checkpoint exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create dictionary containing model/optimizer states and epoch
    checkpoint = {
        'model_state': model.state_dict(), # All learnable parameters
        'optimizer_state': optimizer.state_dict(), # Optimizer internal state
        'epoch': epoch, # Current epoch number
    }

    # Add extra information if provided
    if extra_info:
        checkpoint.update(extra_info)

    # Save checkpoint to file
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at: {path}")


def load_checkpoint(path, model, optimizer=None, device='cpu'):
    """
    Loads checkpoint and restores model (+ optionally optimizer) state.

    Args:
        path (str): Path to checkpoint
        model (nn.Module): Model object to load into
        optimizer (Optimizer, optional): Optimizer to restore
        device (str): Device to map checkpoint (e.g., 'cuda' or 'cpu')

    Returns:
        int: The epoch to resume from (typically epoch + 1)
    """

    # Check if checkpoint exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f" Checkpoint not found: {path}")

    # Load checkpoint file (map to given device)
    checkpoint = torch.load(path, map_location=device)

    # Restore model weights
    model.load_state_dict(checkpoint['model_state'])

    # Restore optimizer state if provided
    if optimizer and 'optimizer_state' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state'])

    # Resume from next epoch after the checkpointed one
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Checkpoint loaded from: {path} (resuming from epoch {start_epoch})")

    return start_epoch
