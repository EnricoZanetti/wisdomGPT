from .model import GPT
from .dataset import NameDataset
from .trainer import Trainer, TrainerConfig

import torch
import random

random.seed(0)


def initialize_vanilla_model(mconf):
    """
    Initializes a standard GPT model using the provided configuration.

    Args:
        mconf: Model configuration object.

    Returns:
        A GPT model initialized with the given configuration.
    """
    attention_model = None

    attention_model = GPT(mconf)

    return attention_model


def initialize_perceiver_model(mconf, bottleneck_dim=32):
    """
    Initializes a Perceiver variant of the GPT model by adding a bottleneck
    dimension to the configuration.

    Args:
        mconf: Model configuration object.
        bottleneck_dim: Bottleneck dimension for the Perceiver variant.

    Returns:
        A Perceiver GPT model initialized with the modified configuration.
    """
    attention_model = None

    mconf.bottleneck_dim = bottleneck_dim
    attention_model = GPT(mconf)

    return attention_model


def finetune(
    reading_params_path,
    finetune_corpus_path,
    pretrain_dataset,
    block_size,
    model,
    finetune_lr=6e-4,
    writer=None,
):
    """
    Fine-tunes a GPT model on a specified corpus.

    Args:
        reading_params_path: Path to pretrained model parameters, or None for training from scratch.
        finetune_corpus_path: Path to the finetuning corpus.
        pretrain_dataset: Dataset used during pretraining, for configuration purposes.
        block_size: Maximum sequence length for the model.
        model: The GPT model to be fine-tuned.
        finetune_lr: Learning rate for fine-tuning.
        writer: Optional writer for logging.

    Returns:
        A tuple containing the TrainerConfig and Trainer object.
    """
    trainer_obj = None
    tconf = None

    max_epochs = 75 if reading_params_path is None else 10
    batch_size = 256
    learning_rate = finetune_lr
    lr_decay = True
    warmup_tokens = 512 * 20
    final_tokens = 200 * len(pretrain_dataset) * block_size
    num_workers = 4

    tconf = TrainerConfig(
        max_epochs=max_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lr_decay=lr_decay,
        warmup_tokens=warmup_tokens,
        final_tokens=final_tokens,
        num_workers=num_workers,
    )

    if reading_params_path is not None:
        pretrained_state_dict = torch.load(
            reading_params_path, map_location=torch.device("cpu")
        )
        model.load_state_dict(pretrained_state_dict)

    data = open(finetune_corpus_path, "r").read()
    train_dataset = NameDataset(data, pretrain_dataset)
    trainer_obj = Trainer(model, train_dataset, None, tconf)

    return tconf, trainer_obj


def pretrain(pretrain_dataset, block_size, model, pretrain_lr=6e-3, writer=None):
    """
    Pretrains a GPT model on a given dataset.

    Args:
        pretrain_dataset: Dataset for pretraining the model.
        block_size: Maximum sequence length for the model.
        model: The GPT model to be pretrained.
        pretrain_lr: Learning rate for pretraining.
        writer: Optional writer for logging.

    Returns:
        A tuple containing the TrainerConfig and Trainer object.
    """
    trainer_obj = None
    tconf = None

    tconf = TrainerConfig(
        max_epochs=650,
        batch_size=128,
        learning_rate=pretrain_lr,
        lr_decay=True,
        warmup_tokens=512 * 20,
        final_tokens=200 * len(pretrain_dataset) * block_size,
        num_workers=0,
    )

    trainer_obj = Trainer(model, pretrain_dataset, None, tconf)

    return tconf, trainer_obj


def train(model, writing_params_path, trainer_obj):
    """
    Trains the model using the Trainer object and saves the model parameters.

    Args:
        model: The GPT model to be trained.
        writing_params_path: Path to save the trained model parameters.
        trainer_obj: Trainer object that manages the training process.
    """
    trainer_obj.train()
    torch.save(model.state_dict(), writing_params_path)

    return
