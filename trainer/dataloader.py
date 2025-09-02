import os
import json
import torch
from torch.utils.data import Dataset
from utils.preprocess import preprocess
from utils.parser import text_to_targets

class CaptchaDataset(Dataset):
    """
    Custom PyTorch Dataset for loading CAPTCHA images and their labels.

    Args:
        root (str): Root directory containing labels.json and images folder.
        char2idx (dict): Dictionary mapping characters to label indices.
        image_subdir (str): Subdirectory name where images are stored (default: 'images').
    """

    def __init__(self, root, char2idx, image_subdir='images'):
        self.root = root
        self.char2idx = char2idx
        self.image_subdir = image_subdir # Allows for augmented image subfolders

        # Load metadata from labels.json
        label_path = os.path.join(root, 'labels.json')
        with open(label_path, 'r') as f:
            self.meta = json.load(f)

    def __len__(self):
        """Return the total number of CAPTCHA samples."""
        return len(self.meta)

    def __getitem__(self, idx):
        """
        Load a single image and its label sequence.

        Args:
            idx (int): Index of the sample.

        Returns:
            x (Tensor): Preprocessed image tensor of shape (C, H, W).
            y (Tensor): Target label indices as a 1D tensor.
        """
        # Load entry from metadata
        entry = self.meta[idx]
        image_id = entry['image_id']

        # Build full image path
        img_path = os.path.join(self.root, self.image_subdir, f"{image_id}.png")

        # Load and preprocess the image
        x = preprocess(img_path)

        # Convert text label to numerical target tensor
        y = text_to_targets(entry['captcha_string'], self.char2idx)

        return x, y


def collate(batch, stride=32):
    """
    Custom collate function for DataLoader to handle variable-width CAPTCHA images.

    Args:
        batch (list): List of (image_tensor, label_tensor) tuples.
        stride (int): Downsampling stride of the network (for calculating input lengths).

    Returns:
        padded_images (Tensor): Batch of padded image tensors (B, C, H, maxW).
        targets (Tensor): Concatenated target sequences for all samples (1D tensor).
        input_lengths (Tensor): Sequence lengths (after CNN downsampling) for each image.
        target_lengths (Tensor): Lengths of each label sequence.
    """
    
    # Unzip images and labels
    xs, ys = zip(*batch)

    H = xs[0].shape[1]

    # Get original widths of all images
    widths = [x.shape[2] for x in xs]
    maxW = max(widths) # Find the maximum width in the batch

    padded = []
    input_lengths = []
    for x, w in zip(xs, widths):
        padW = maxW - w # Calculate how much padding is needed to right-align
        if padW:
            # Pad width dimension (right side)
            x = torch.nn.functional.pad(x, (0, padW, 0, 0))
        padded.append(x)

        # Input length after stride-based downsampling
        input_lengths.append(w // stride)

    # Concatenate all target sequences into one 1D tensor
    targets = torch.cat(ys)

    # Store the original lengths of each label sequence
    target_lengths = torch.tensor([len(y) for y in ys])

    # Stack padded images into a batch tensor
    return torch.stack(padded), targets, torch.tensor(input_lengths), target_lengths
