import torch

def decode_greedy(log_probs, idx2char, blank=0):
    """
    Performs greedy decoding from CTC log probabilities.

    Args:
        log_probs (Tensor): Log probabilities from CRNN (T, B, C)
            - T = sequence length (time steps)
            - B = batch size
            - C = number of classes (including blank)
        idx2char (dict): Index-to-character mapping
        blank (int): Index of CTC blank token

    Returns:
        List[str]: Decoded sequences for each sample in the batch
    """

    # Get the most likely class at each time step
    path = log_probs.argmax(dim=-1)  # (T, B)

    T, B = path.shape
    sequences = []

    # Process each sequence in the batch independently
    for b in range(B):
        prev = blank # Track the previous symbol (initialize as blank)
        chars = [] # Store decoded characters for this sequence

        for t in range(T):
            p = int(path[t, b]) # Predicted class index at time t for sample b

            # Apply CTC decoding rules (Ignore blanks & Collapse consecutive repeated characters)
            if p != blank and p != prev:
                chars.append(idx2char[p])

            prev = p # Update previous symbol

        # Join characters into final string for this sample
        sequences.append(''.join(chars))

    return sequences
