import random
import torch
from torch.utils.data import Dataset
import argparse

"""
The input-output pairs (x, y) in the NameDataset are structured as follows:

- `x`: Encodes the input question and answer concatenated with masking and padding.
- `y`: Encodes the target sequence, which predicts the answer position.

Padding characters (PAD_CHAR) in `y` prevent the model from learning to predict the question text.

The NameDataset leverages the `pretraining_dataset` to ensure consistent vocabulary and configurations.
"""


class NameDataset(Dataset):
    def __init__(self, data, pretraining_dataset):
        self.MASK_CHAR = pretraining_dataset.MASK_CHAR  # Masking character
        self.PAD_CHAR = pretraining_dataset.PAD_CHAR  # Padding character
        self.itos = pretraining_dataset.itos  # Index-to-string mapping
        self.stoi = pretraining_dataset.stoi  # String-to-index mapping
        self.block_size = pretraining_dataset.block_size  # Maximum input size
        self.data = list(
            data.encode("utf-8").decode("ascii", errors="ignore").split("\n")
        )

    def __len__(self):
        """Returns the total number of data entries in the dataset."""
        return len(self.data) - 1

    def __getitem__(self, idx):
        """Generates a single data example (x, y)."""
        inp, oup = self.data[idx].split("\t")
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR * (self.block_size - len(x))
        y = self.PAD_CHAR * (len(inp) - 1) + x[len(inp) :]
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y


"""
Implements a dataset for a span corruption objective.

Vocabulary Specification:
- `self.stoi`: Maps characters to integer indices.
- `self.itos`: Maps indices back to characters.

Masking Specification:
- Truncates documents randomly within defined limits.
- Reorganizes the input as:
  [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] MASK_CHAR [pads].
- Ensures consistent input-output string lengths via padding.

This implementation supports training tasks requiring character-level predictions.
"""


class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = "\u2047"  # Masking character
        self.PAD_CHAR = "\u25a1"  # Padding character

        chars = list(sorted(set(data)))
        assert self.MASK_CHAR not in chars, "Mask character should not be in dataset."
        assert self.PAD_CHAR not in chars, "Padding character should not be in dataset."
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        data_size, vocab_size = len(data), len(chars)
        print(f"Dataset contains {data_size} characters, {vocab_size} unique.")

        self.block_size = block_size
        self.max_context_size = int(block_size * 3 / 4)
        self.data = data.split("\n")

    def __len__(self):
        """Returns the total number of documents in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Generates a single data example (x, y) based on span corruption logic."""
        document = self.data[idx]
        min_length, max_length = 4, min(int(self.block_size * 3 / 4), len(document))
        truncated_document = document[: random.randint(min_length, max_length)]

        truncated_length = len(truncated_document)
        avg_masked_length = max(1, truncated_length // 4)
        masked_length = random.randint(
            1, min(truncated_length - 2, 2 * avg_masked_length)
        )

        start_idx = random.randint(0, truncated_length - masked_length)
        prefix = truncated_document[:start_idx]
        masked_content = truncated_document[start_idx : start_idx + masked_length]
        suffix = truncated_document[start_idx + masked_length :]

        masked_string = (
            prefix
            + self.MASK_CHAR
            + suffix
            + self.MASK_CHAR
            + masked_content
            + self.MASK_CHAR
        )

        if len(masked_string) >= self.block_size + 1:
            masked_string = masked_string[: self.block_size + 1]
        else:
            num_pads = self.block_size + 1 - len(masked_string)
            masked_string += self.PAD_CHAR * num_pads

        input_string = masked_string[:-1]
        output_string = masked_string[1:]
        x = torch.tensor([self.stoi.get(c, 0) for c in input_string], dtype=torch.long)
        y = torch.tensor([self.stoi.get(c, 0) for c in output_string], dtype=torch.long)

        return x, y


"""
Main script for dataset debugging and visualization.
"""
if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument(
        "dataset_type",
        help="Specify the dataset type to sample from.",
        choices=["namedata", "charcorruption"],
    )
    args = argp.parse_args()

    if args.dataset_type == "namedata":
        corruption_dataset = CharCorruptionDataset(
            open("./../data/wiki.txt", encoding="utf-8").read(), 128
        )
        name_dataset = NameDataset(
            open("./../data/birth_places_train.tsv", encoding="utf-8").read(),
            corruption_dataset,
        )
        for _, example in zip(range(4), name_dataset):
            x, y = example
            print("x:", "".join([name_dataset.itos[int(c)] for c in x]))
            print("y:", "".join([name_dataset.itos[int(c)] for c in y]))
    elif args.dataset_type == "charcorruption":
        corruption_dataset = CharCorruptionDataset(
            open("./../data/wiki.txt", encoding="utf-8").read(), 128
        )
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print("x:", "".join([corruption_dataset.itos[int(c)] for c in x]))
            print("y:", "".join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
