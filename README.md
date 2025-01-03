# **WisdomGPT: A Framework for Knowledge-Enhanced Transformer Training**

**WisdomGPT** is a powerful framework designed to train, fine-tune, and evaluate **Transformer models** for knowledge-intensive tasks. It demonstrates how pretraining on a vast corpus of text containing world knowledge (e.g., Wikipedia) significantly enhances a model's ability to generalize beyond the training data. By fine-tuning the pretrained model on task-specific datasets, **WisdomGPT** enables models to perform well on knowledge-driven challenges.

## Key Concepts

1. **Knowledge-Intensive Tasks**:
   - Tasks that require external knowledge not explicitly present in the training data.

2. **Pretraining**:
   - Training a model on a large text corpus to acquire general linguistic and factual knowledge.

3. **Fine-Tuning**:
   - Adapting the pretrained model to specific tasks, allowing it to leverage learned knowledge effectively.

4. **Performance Improvement**:
   - Fine-tuned models outperform baseline models on held-out datasets by utilizing pretrained knowledge.

## Framework Overview

Built upon **Andrej Karpathy’s** [**minGPT**](https://github.com/karpathy/minGPT), **WisdomGPT** offers:

- A **simple and transparent** codebase for learning and experimenting with Transformer models.
- Seamless support for **pretraining**, **fine-tuning**, and **evaluation** workflows.
- Tools to run experiments locally or on Azure virtual machines with GPU acceleration.

## Installation

### **Using Conda**

Set up the environment for local development or Azure training:

```bash
conda env create -f environment.yml
conda activate wisdom-gpt
```

For GPU support, use the CUDA environment:

```bash
conda env create -f environment.yml
conda activate wisdom-gpt_cuda
```

## Usage

### Command-Line Interface

Run the `run.py` script to pretrain, fine-tune, or evaluate models:

```bash
python run.py --function=<function> --variant=<attention-model> --pretrain_corpus_path=<file> [options]
```

### Key Arguments

| Argument                   | Description                                                                 | Default Value   |
|----------------------------|-----------------------------------------------------------------------------|-----------------|
| `--function`               | Specify 'pretrain', 'finetune', or 'evaluate'.                              | None (required) |
| `--variant`                | Specify the model variant: 'vanilla' or 'perceiver'.                       | None (required) |
| `--pretrain_corpus_path`   | Path to the pretraining corpus.                                             | None (required) |
| `--writing_params_path`    | Path to save model parameters after training.                               | None            |
| `--reading_params_path`    | Path to load pretrained parameters for fine-tuning or evaluation.           | None            |
| `--finetune_corpus_path`   | Path to the fine-tuning corpus.                                             | None            |
| `--eval_corpus_path`       | Path to the evaluation corpus.                                              | None            |
| `--outputs_path`           | Path to save predictions during evaluation.                                 | None            |
| `--bottleneck_dim`         | Bottleneck dimension for Perceiver models.                                 | `32`            |
| `--pretrain_lr`            | Learning rate for pretraining.                                              | `6e-3`          |
| `--finetune_lr`            | Learning rate for fine-tuning.                                              | `6e-4`          |

### Example: Pretraining on Wikipedia

```bash
python run.py --function=pretrain --variant=vanilla --pretrain_corpus_path=data/wiki.txt --writing_params_path=model/pretrained_model.pth
```

### Example: Fine-Tuning for Knowledge-Intensive Tasks

```bash
python run.py --function=finetune --variant=perceiver --pretrain_corpus_path=data/wiki.txt --finetune_corpus_path=data/task_data.txt --reading_params_path=model/pretrained_model.pth --writing_params_path=model/finetuned_model.pth
```

### Example: Evaluation

```bash
python run.py --function=evaluate --variant=vanilla --reading_params_path=model/finetuned_model.pth --eval_corpus_path=data/eval_data.txt --outputs_path=results/predictions.txt
```

### Training Workflow

1. Upload the dataset and code to your VM.
2. Run the `run.py` script with the desired options.
3. Allocate approximately 5 hours for pretraining and fine-tuning.

## Outputs

1. **Model Parameters**:
   - Saved as `.pth` files after pretraining or fine-tuning.

2. **Evaluation Results**:
   - Predictions written to a specified `--outputs_path`.

## Acknowledgments

This project is inspired by:
- [Andrej Karpathy’s minGPT](https://github.com/karpathy/minGPT)
- The [Stanford XCS224N course](https://online.stanford.edu/courses/xcs224n-natural-language-processing-deep-learning).

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

- **Developer**: [Enrico Zanetti](https://www.linkedin.com/in/enrico-zanetti/)