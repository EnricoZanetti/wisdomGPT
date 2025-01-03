#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage:
    run.py --function=<function> --variant=<attention-model> --pretrain_corpus_path=<file> [--writing_params_path=<file>] [--finetune_corpus_path=<file>] [--reading_params_path=<file>] [--eval_corpus_path=<file>] [--outputs_path=<file>] [options]

Options:
    -h --help                               Show this help message.
    --compile                               Compile the model for optimization.
    --no-compile                            Skip model compilation.
    --backend=<str>                         Backend for model compilation [default: inductor] {inductor, aot_eager, cudagraphs}.
    --function=<function>                   Specify the operation: 'pretrain', 'finetune', or 'evaluate'.
    --variant=<attention-model>             Specify the model variant: 'vanilla' or 'perceiver'.
    --pretrain_corpus_path=<file>           Path to the pretraining corpus.
    --writing_params_path=<file>            Path to save model parameters after training.
    --reading_params_path=<file>            Path to load model parameters for fine-tuning or evaluation.
    --finetune_corpus_path=<file>           Path to the fine-tuning corpus.
    --eval_corpus_path=<file>               Path to the evaluation corpus.
    --outputs_path=<file>                   Path to save predictions during evaluation.
    --tb_expt_name=<str>                    TensorBoard experiment name [default: run].
    --bottleneck_dim=<n>                    Bottleneck dimension for Perceiver models [default: 32].
    --pretrain_lr=<value>                   Learning rate for pretraining [default: 6e-3].
    --finetune_lr=<value>                   Learning rate for fine-tuning [default: 6e-4].
"""

from docopt import docopt
from datetime import datetime
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.nn import functional as F
import random
from torch.utils.tensorboard import SummaryWriter
import sys

# Import necessary modules and utilities
from core import (
    GPT,
    GPTConfig,
    CharCorruptionDataset,
    NameDataset,
    TrainerConfig,
    Trainer,
    evaluate_places,
    sample,
    initialize_vanilla_model,
    initialize_perceiver_model,
    finetune,
    pretrain,
    train,
)

random.seed(0)


def create_model(args, mconf):
    """
    Initialize the specified model variant.

    Args:
        args (dict): Command-line arguments parsed by docopt.
        mconf (GPTConfig): Configuration object for the model.

    Returns:
        torch.nn.Module: Initialized model.
    """
    if args["--variant"] == "vanilla":
        return initialize_vanilla_model(mconf)
    elif args["--variant"] == "perceiver":
        bottleneck_dim = int(args["--bottleneck_dim"])
        return initialize_perceiver_model(mconf, bottleneck_dim)
    else:
        raise ValueError("Invalid model variant specified in --variant.")


def evaluate(args, pretrain_dataset, device, model):
    """
    Evaluate the model on a specified corpus.

    Args:
        args (dict): Command-line arguments.
        pretrain_dataset (Dataset): Pretraining dataset to access vocabulary.
        device (torch.device): Device to run evaluation.
        model (torch.nn.Module): The trained model to evaluate.
    """
    assert (
        args["--outputs_path"] is not None
    ), "Outputs path must be specified for evaluation."
    assert (
        args["--reading_params_path"] is not None
    ), "Model parameters path must be specified for evaluation."
    assert (
        args["--eval_corpus_path"] is not None
    ), "Evaluation corpus path must be specified."

    model.load_state_dict(torch.load(args["--reading_params_path"], weights_only=True))
    correct, total = 0, 0

    with open(args["--outputs_path"], "w", encoding="utf-8") as fout:
        predictions = []
        for line in tqdm(open(args["--eval_corpus_path"], encoding="utf-8")):
            x = line.split("\t")[0] + "⁇"
            x = torch.tensor([pretrain_dataset.stoi[s] for s in x], dtype=torch.long)[
                None, ...
            ].to(device)
            pred = sample(model, x, 32, sample=False)[0]
            completion = "".join([pretrain_dataset.itos[int(i)] for i in pred])
            predicted_place = completion.split("⁇")[1]
            predictions.append(predicted_place)
            fout.write(predicted_place + "\n")

        total, correct = evaluate_places(args["--eval_corpus_path"], predictions)

    if total > 0:
        print(f"Correct: {correct} out of {total} ({correct/total*100:.2f}%)")
    else:
        print(f'Predictions written to {args["--outputs_path"]}; no targets provided.')


def setup_device():
    """
    Sets up the PyTorch device based on available hardware.

    Returns:
        torch.device: Device to use for training and evaluation.
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    """
    Main function to handle pretraining, fine-tuning, or evaluation based on user input.
    """
    args = docopt(__doc__)
    device = setup_device()

    # Keep block size constant
    block_size = 128
    pretrain_text = open(args["--pretrain_corpus_path"], encoding="utf-8").read()
    pretrain_dataset = CharCorruptionDataset(pretrain_text, block_size)

    # Configure model
    mconf = GPTConfig(
        pretrain_dataset.vocab_size,
        pretrain_dataset.block_size,
        n_layer=4,
        n_head=8,
        n_embd=256,
    )
    model = create_model(args, mconf)

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(
        log_dir=f'expt/{args["--function"]}/{args["--tb_expt_name"]}_{args["--variant"]}_'
        f'{int(args["--bottleneck_dim"])}_pt_lr_{float(args["--pretrain_lr"])}_'
        f'ft_lr_{float(args["--finetune_lr"])}_{datetime_str}'
    )

    if args["--compile"]:
        try:
            model = torch.compile(model, backend=args["--backend"])
            print("Model successfully compiled.")
        except Exception as e:
            print(f"Model compilation failed: {e}")

    model = model.to(device)

    if args["--function"] == "finetune":
        assert (
            args["--finetune_corpus_path"] is not None
        ), "Finetune corpus path must be specified."
        assert (
            args["--writing_params_path"] is not None
        ), "Writing parameters path must be specified."
        reading_params_path = args["--reading_params_path"]
        finetune_corpus_path = args["--finetune_corpus_path"]
        finetune_lr = float(args["--finetune_lr"])
        _, trainer = finetune(
            reading_params_path,
            finetune_corpus_path,
            pretrain_dataset,
            block_size,
            model,
            finetune_lr,
            writer,
        )
        train(model, args["--writing_params_path"], trainer)

    elif args["--function"] == "pretrain":
        assert (
            args["--writing_params_path"] is not None
        ), "Writing parameters path must be specified."
        pretrain_lr = float(args["--pretrain_lr"])
        _, trainer = pretrain(pretrain_dataset, block_size, model, pretrain_lr, writer)
        train(model, args["--writing_params_path"], trainer)

    elif args["--function"] == "evaluate":
        evaluate(args, pretrain_dataset, device, model)


if __name__ == "__main__":
    main()
