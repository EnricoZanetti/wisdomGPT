"""
Simple Training Loop:
This boilerplate code applies to any arbitrary neural network and is not specific to GPT models.

We recommend not modifying this file.
"""

import math
import logging
import sys
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    """
    Configuration class for the Trainer.

    Attributes:
        max_epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for the optimizer.
        betas (tuple): Betas for the Adam optimizer.
        grad_norm_clip (float): Maximum gradient norm for clipping.
        weight_decay (float): Weight decay for regularization.
        lr_decay (bool): Whether to use learning rate decay.
        warmup_tokens (float): Number of tokens for linear warmup.
        final_tokens (float): Tokens processed to reach 10% of the initial learning rate.
        ckpt_path (str): Path to save model checkpoints.
        num_workers (int): Number of workers for the DataLoader.
        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer for logging.
    """

    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 0
    writer = None

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    """
    Trainer class for managing the training and evaluation of a model.

    Attributes:
        model (torch.nn.Module): Model to be trained.
        train_dataset (Dataset): Training dataset.
        test_dataset (Dataset): Optional test dataset.
        config (TrainerConfig): Training configuration.
    """

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # Automatically use GPU if available
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
        elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device("mps")
            self.model = self.model.to(self.device)
        print(f"Using device: {self.device}", file=sys.stderr)

    def save_checkpoint(self):
        """
        Saves the model's state dictionary to the configured checkpoint path.
        """
        if self.config.ckpt_path is not None:
            ckpt_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            logger.info(f"Saving checkpoint to {self.config.ckpt_path}")
            torch.save(ckpt_model.state_dict(), self.config.ckpt_path)

    def train(self):
        """
        Manages the training and evaluation loop.
        Includes learning rate decay, gradient clipping, and checkpointing.
        """
        model, config = self.model, self.config

        # Define optimizer with separate parameter groups for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ]
        params_nodecay = [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ]
        optim_groups = [
            {"params": params_decay, "weight_decay": config.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(
            optim_groups, lr=config.learning_rate, betas=config.betas
        )
        step = 0

        def run_epoch(split):
            """
            Executes a single epoch of training or evaluation.

            Args:
                split (str): Specifies whether to 'train' or 'test'.
            """
            nonlocal step
            is_train = split == "train"
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(
                data, batch_size=config.batch_size, num_workers=config.num_workers
            )

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:
                # Move data to the appropriate device
                x, y = x.to(self.device), y.to(self.device)

                # Forward pass
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = loss.mean()  # Aggregate losses if using multiple GPUs
                    losses.append(loss.item())

                if is_train:
                    # Backward pass and optimization
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # Adjust learning rate
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # Count tokens processed in this step
                        if self.tokens < config.warmup_tokens:
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # Report progress
                    pbar.set_description(
                        f"Epoch {epoch + 1} Iter {it}: Train Loss {loss.item():.5f}. LR {lr:e}"
                    )
                    if config.writer:
                        config.writer.add_scalar("train/loss", loss.item(), step)
                        config.writer.add_scalar("train/lr", lr, step)

                step += 1

            if not is_train:
                logger.info(f"Test Loss: {np.mean(losses):.5f}")

        # Initialize token counter for learning rate decay
        self.tokens = 0

        # Training loop
        for epoch in range(config.max_epochs):
            run_epoch("train")
            if self.test_dataset is not None:
                run_epoch("test")
            self.save_checkpoint()
