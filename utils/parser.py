import json
import torch

def load_charset_from_labels(label_path):
    """
    Loads the charset from a Detectron-style labels.json file and returns
    character-index mappings for CTC training/decoding.

    Args:
        label_path (str): Path to labels.json

    Returns:
        tuple:
            - char2idx (dict): mapping from character to index (1-based, 0 is CTC blank)
            - idx2char (dict): mapping from index to character
            - charset (list): sorted list of unique characters
    """

    # Open and parse the labels.json file
    with open(label_path, 'r') as f:
        ann = json.load(f)

    # Extract all unique characters from every captcha_string in the dataset
    # Example: [{"captcha_string": "AB12"}, {"captcha_string": "C3D4"}] → {'A','B','1','2','C','3','D','4'}
    charset = sorted({c for e in ann for c in e['captcha_string']})

    # Map each character to an index (starting at 1, since 0 is reserved for CTC blank token)
    char2idx = {c: i + 1 for i, c in enumerate(charset)}

    # Reverse mapping: index → character
    idx2char = {i + 1: c for i, c in enumerate(charset)}

    return char2idx, idx2char, charset


def text_to_targets(s, char2idx):
    """
    Converts a captcha string into a tensor of integer class indices.

    Args:
        s (str): Captcha string (e.g., "AB12")
        char2idx (dict): Character-to-index mapping

    Returns:
        torch.Tensor: 1D tensor of integer indices
    """
    
    # Convert each character in the string to its corresponding index
    # Example: "AB1" → [char2idx['A'], char2idx['B'], char2idx['1']]
    return torch.tensor([char2idx[c] for c in s], dtype=torch.long)
