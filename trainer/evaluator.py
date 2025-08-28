import os
import json
import glob
import torch
import Levenshtein
from PIL import Image
from tqdm import tqdm
from utils.preprocess import preprocess
from utils.decoder import decode_greedy

def evaluate_seq_acc(model, dataloader, device, idx2char, blank=0):
    """
    Evaluates sequence-level accuracy on a validation set.

    Accuracy is computed by comparing predicted sequences
    to ground truth strings, and counting exact matches.

    Args:
        model (nn.Module): Trained CRNN model
        dataloader (DataLoader): Validation data loader
        device (torch.device): Device to run inference on
        idx2char (dict): Index-to-character mapping
        blank (int): Index of CTC blank token

    Returns:
        float: Sequence-level accuracy (0.0 to 1.0)
    """
    model.eval()
    n_total, n_correct = 0, 0

    with torch.no_grad():
        for X, targets, input_lens, target_lens in dataloader:
            X = X.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            logp = model(X)  # Shape: (T, B, C)

            # Greedy decode predictions to strings
            preds = decode_greedy(logp, idx2char, blank=blank)

            # Convert flat target tensor to string list
            gts = []
            offset = 0
            for L in target_lens.tolist():
                label_seq = targets[offset:offset+L].tolist()
                gt_str = ''.join(idx2char[i] for i in label_seq)
                gts.append(gt_str)
                offset += L

            # Compare predictions with ground truths
            for pred, gt in zip(preds, gts):
                n_correct += int(pred == gt)
                n_total += 1

    return n_correct / max(n_total, 1)


def predict_image_string(model, img_path, device, idx2char, blank=0):
    """
    Runs CRNN model on a single image and returns predicted string.

    Args:
        model (nn.Module): Trained CRNN model
        img_path (str): Path to input image
        device (torch.device): Device to run on
        idx2char (dict): Index-to-character mapping
        blank (int): Index of CTC blank

    Returns:
        str: Decoded CAPTCHA string
    """
    model.eval()

    # Preprocess image and add batch dimension
    x = preprocess(img_path).unsqueeze(0).to(device)  # (1,1,H,W)

    with torch.no_grad():
        logp = model(x) # Shape: (T, 1, C)

    # Decode to string
    return decode_greedy(logp, idx2char, blank)[0]


def make_part2_predictions_json(model, test_root, out_json_path, device, idx2char, blank=0):
    """
    Generates predictions for all test images and saves in Detectron-style JSON format.

    Args:
        model (nn.Module): Trained CRNN model
        test_root (str): Path to test set root (must contain 'images' subdir)
        out_json_path (str): Output path for saving JSON
        device (torch.device): CUDA or CPU
        idx2char (dict): Index-to-character mapping
        blank (int): Index for CTC blank
    """

    img_dir = os.path.join(test_root, 'images')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.png')))

    results = []
    for p in img_paths:
        fname = os.path.basename(p)
        image_id = os.path.splitext(fname)[0]

        # Get image dimensions
        with Image.open(p) as im:
            w, h = im.size

        # Predict string from image
        pred_str = predict_image_string(model, p, device, idx2char, blank)

        # Append result in Detectron format
        results.append({
            "height": int(h),
            "width": int(w),
            "image_id": image_id,
            "captcha_string": pred_str,
            "annotations": []  # Not used for CRNN output
        })

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)

    # Save predictions to JSON
    with open(out_json_path, 'w') as f:
        json.dump(results, f, indent=2)


def _edit_distance(a: str, b: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.

    Args:
        a (str): First string
        b (str): Second string

    Returns:
        int: Edit distance between a and b
    """

    la, lb = len(a), len(b)
    # Initialize DP table
    dp = [[0]*(lb+1) for _ in range(la+1)]

    for i in range(la+1): 
        dp[i][0] = i
    for j in range(lb+1): 
        dp[0][j] = j

    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1,       # delete
                           dp[i][j-1] + 1,       # insert
                           dp[i-1][j-1] + cost)  # replace

    return dp[la][lb]

@torch.no_grad()
def evaluate_ler(model, dataloader, device, idx2char, blank=0):
    """
    Computes the average Levenshtein Error Rate (LER) across a dataset.

    LER = edit_distance(pred, gt) / max(1, len(gt))

    Args:
        model (nn.Module): Trained CRNN model
        dataloader (DataLoader): Dataset loader
        device (torch.device): CUDA or CPU
        idx2char (dict): Index-to-character mapping
        blank (int): CTC blank token index

    Returns:
        float: Average LER
    """

    model.eval()
    total_err = 0.0
    n = 0
    for X, targets, in_lens, tar_lens in dataloader:
        X = X.to(device, non_blocking=True)
        logp = model(X)  # Shape: (T, B, C)
        preds = decode_greedy(logp, idx2char, blank=blank)  # Decoded strings

        # Unflatten targets into list of strings
        offs = 0
        gts = []
        for L in tar_lens.tolist():
            seq = targets[offs:offs+L].tolist()
            gts.append(''.join(idx2char[i] for i in seq))
            offs += L

        # Compute normalized edit distance per sample
        for p, g in zip(preds, gts):
            ed = _edit_distance(p, g)
            total_err += ed / max(1, len(g))
            n += 1

    return total_err / max(1, n)


def make_predictions_json(model, dataloader, device, idx2char, blank, output_path):
    """
    Predicts captcha strings for the test dataset and saves to JSON.

    Args:
        model: Trained CRNN model
        dataloader: DataLoader for test set
        device: CPU or CUDA
        idx2char: index-to-char mapping
        blank: CTC blank index
        output_path: where to save predictions.json
    """

    model.eval()
    model.to(device)

    preds = []

    with torch.no_grad():
        for images, _, _, _ in tqdm(dataloader, desc="Predicting on test set"):
            images = images.to(device)

            # Forward pass
            log_probs = model(images)

            # Greedy decode
            decoded = decode_greedy(log_probs, idx2char, blank)

            # get corresponding image_id from dataset
            for i, pred in enumerate(decoded):
                img_name = dataloader.dataset[i]['image_id']  # assumes CaptchaDataset returns dict
                preds.append({
                    "image_id": img_name,
                    "captcha_string": pred
                })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(preds, f, indent=2)

    print(f" Saved test predictions to: {output_path}")
