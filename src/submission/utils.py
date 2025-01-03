"""
Utility Functions:
These utility functions provide essential support for model training, evaluation, and sampling.
Modifications to these functions are not recommended, but you may add your own utilities as needed.
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed):
    """
    Sets the seed for reproducibility across random, NumPy, and PyTorch libraries.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    """
    Truncates logits to only retain the top-k elements, setting the rest to negative infinity.

    Args:
        logits (torch.Tensor): The logits tensor.
        k (int): The number of top elements to retain.

    Returns:
        torch.Tensor: Logits tensor with only the top-k elements retained.
    """
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    Generates a sequence of predictions using a given model.

    Args:
        model (torch.nn.Module): The model used for sampling.
        x (torch.Tensor): Conditioning sequence of indices (shape: (batch_size, sequence_length)).
        steps (int): Number of steps to sample.
        temperature (float): Temperature to scale logits; higher values increase randomness.
        sample (bool): If True, samples from the probability distribution. If False, selects the most likely prediction.
        top_k (int, optional): If provided, restricts sampling to the top-k predictions.

    Returns:
        torch.Tensor: The generated sequence (shape: (batch_size, sequence_length + steps)).
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        # Restrict context to the model's block size
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]

        # Forward pass through the model
        logits, _ = model(x_cond)

        # Extract logits for the last position and scale by temperature
        logits = logits[:, -1, :] / temperature

        # Optionally truncate logits to top-k values
        if top_k is not None:
            logits = top_k_logits(logits, top_k)

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Sample or choose the most likely token
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)

        # Append the predicted token to the sequence
        x = torch.cat((x, ix), dim=1)

    return x


def evaluate_places(filepath, predicted_places):
    """
    Evaluates the accuracy of predicted birthplaces against ground truth data.

    Args:
        filepath (str): Path to the file containing ground truth data (tab-separated: name, birthplace).
        predicted_places (list of str): List of predicted birthplaces.

    Returns:
        tuple: (total, correct) as floats, where `total` is the number of samples
               and `correct` is the count of correct predictions.
    """
    with open(filepath, encoding="utf-8") as fin:
        lines = [x.strip().split("\t") for x in fin]

        # Handle cases where ground truth is unavailable
        if len(lines[0]) == 1:
            print(
                "!!! No ground truth is provided; this will be done on the autograder, returning (0, 0)"
            )
            return (0, 0)

        # Extract true birthplaces from the file
        true_places = [x[1] for x in lines]
        total = len(true_places)

        # Ensure predictions match the ground truth size
        assert total == len(predicted_places)

        # Count correct predictions
        correct = len(
            [1 for true, pred in zip(true_places, predicted_places) if true == pred]
        )

        return (float(total), float(correct))
