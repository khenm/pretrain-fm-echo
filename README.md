# PyTorch Template Code

A clean, generic PyTorch template designed for computer vision tasks. This repository provides a robust foundation for deep learning projects with built-in support for distributed training, experiment tracking, and automated checkpoint management.

## Features

- **Configuration-Driven:** Manage experiments cleanly with YAML configurations.
- **Distributed Training:** Out-of-the-box support for Distributed Data Parallel (DDP).
- **Experiment Tracking:** Seamless integration with Weights & Biases (`wandb`).
- **Resilient Checkpointing:** Automated `TrainerState` management with auto-discovery and state resumption.
- **Modular Registry:** Dynamically build models, losses, dataloaders, and metrics via registries.

## Installation

This project targets Python 3.10+ and uses `uv` with `pyproject.toml` for modern, fast dependency management.

```bash
# Clone the repository
git clone https://github.com/username/pytorch-template-code.git
cd pytorch-template-code
```

## Project Structure

```text
pytorch-template-code/
├── configs/              # YAML configuration files
├── notebooks/            # Exploratory data analysis
├── scripts/              # Executable scripts 
│   └── train.py          # Main training entry point
└── src/                  # Core source code
    ├── datasets/         # Data loading and dataset definitions
    ├── losses/           # Custom loss functions
    ├── metrics/          # Evaluation metrics
    ├── models/           # Neural network architectures
    ├── registry.py       # Component builders and factory functions
    ├── runner.py         # Workspace setup and environment initialization
    ├── trainer.py        # Core training loop logic
    └── utils/            # Utilities (logging, DDP, state management)
├── pyproject.toml        # Project configuration and dependencies
└── uv.lock               # Lock file for dependencies
```

## Usage

### Training

Start a training session using the main training script. The default configuration is `configs/config.yaml`. We use `uv` to manage dependencies and run scripts. 

```bash
uv run scripts/train.py --config configs/config.yaml
```

**Resume Training:**
The runner supports intelligent auto-resumption from the latest checkpoint associated with the model in the vault.

```bash
uv run scripts/train.py --config configs/config.yaml --resume
```

**Distributed Training (DDP):**
To run distributed training across multiple GPUs, use `torchrun`:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py --config configs/config.yaml
```

## Configuration

The `config.yaml` controls the pipeline. Key sections:
- `model`: Architecture target and parameters.
- `data`: Dataset configurations.
- `training`: Hyperparameters, learning rates, epochs, and workspace paths (`runs/`, `checkpoints/`).
- `loss`: Active criteria registry keys.
- `wandb`: Experiment tracking toggles and project settings.

## FAQ

<details>
<summary><b>How to add a new model, dataset, or loss function?</b></summary>

This template uses a cleaner, modular registry system (`src/registry.py`). You do not need to modify the main training loop to add new components. Just define your class and use the corresponding decorator.

**1. Adding a New Model:**
```python
# src/models/my_model.py
from src.registry import register_model
import torch.nn as nn

@register_model("MyNewModel")
class MyNewModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Initialize layers...

    @classmethod
    def from_config(cls, cfg):
        # The runner calls this method to instantiate the model
        return cls(channels=cfg['model'].get('channels', 3))
```
*Tip: Ensure your new file is imported inside `src/models/__init__.py` so the decorator registers the class automatically.*

**2. Adding a New Dataset:**
```python
# src/datasets/my_dataset.py
from src.registry import register_dataset

@register_dataset("MY_CUSTOM_DATA")
class MyCustomDataset:
    @classmethod
    def get_dataloaders(cls, cfg):
        # Build and return standard PyTorch DataLoaders
        # return train_loader, val_loader
        pass
```

**3. Adding a New Loss Function:**
```python
# src/losses/my_loss.py
from src.registry import register_loss
import torch.nn as nn

@register_loss("MyCustomLoss")
class MyCustomLoss(nn.Module):
    def forward(self, pred, target):
        loss_val = ... 
        return loss_val
```

Once registered, you switch components entirely via `config.yaml`:
```yaml
model:
  name: "MyNewModel"
data:
  name: "MY_CUSTOM_DATA"
loss:
  types: ["MyCustomLoss"]
```
</details>

---

## 📄 Research Paper README Template

If you are adapting this repository for a research paper release, use the following template to ensure clarity, rigor, and reproducibility.


# [Paper Title: Subtitle]

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/1234.56789)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://your-project-page.github.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**[First Author]**, **[Second Author]**, **[Third Author]**

**[Conference/Journal Name Year]**

</div>

## Description

Official PyTorch implementation of **"[Paper Title]"**.

[Insert a clear, concise paragraph describing the problem, the proposed method, and the primary results. Mention any state-of-the-art achievements.]

<p align="center">
  <img src="docs/teaser.png" alt="Teaser" width="80%">
</p>

## Installation

This project uses `uv` for reproducible and fast dependency management.

```bash
# Clone the repository
git clone https://github.com/username/project-name.git
cd project-name

# Install dependencies
uv sync
```

## Data Preparation

[Provide explicitly detailed instructions for downloading and preparing the datasets used in the paper. Include a tree visualization of the expected directory structure.]

```text
data/
├── dataset_name/
│   ├── train/
│   ├── val/
│   └── test/
```

## Pre-trained Models

We provide pre-trained checkpoints for our models. You can download them from [Google Drive / Hugging Face](#) or use the provided script.

| Model Architecture | Params (M) | Metric 1 | Metric 2 | Weights |
|--------------------|------------|----------|----------|---------|
| Model-Small        | 10.5       | 85.0     | 42.1     | [Link](#) |
| Model-Large        | 85.2       | 88.5     | 55.3     | [Link](#) |

## Training

To reproduce the results reported in the paper, execute the training script with the corresponding configuration file.

```bash
# Standard training
python3 scripts/train.py --config configs/model_large.yaml

# Distributed Data Parallel (DDP) training on 4 GPUs
torchrun --nproc_per_node=4 scripts/train.py --config configs/model_large.yaml
```

*Note: Training configurations are located in `configs/`. You can adjust batch size and learning rate via command-line arguments if needed.*

## Evaluation

To evaluate a trained model or a pre-trained checkpoint on the test set:

```bash
python3 scripts/eval.py --config configs/model_large.yaml --resume path/to/checkpoint.ckpt
```

## Citation

If you use this code or our pre-trained models in your research, please cite our paper:

```bibtex
@inproceedings{author2026title,
  title     = {Paper Title: Subtitle},
  author    = {Author, First and Author, Second and Author, Third},
  booktitle = {Proceedings of the [Conference Name]},
  year      = {2026},
  pages     = {1--10}
}
```

## Acknowledgements

[Optionally, acknowledge any fundamental repositories or codebases that your code builds upon.]

