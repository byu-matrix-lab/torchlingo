# TorchLingo

<div align="center" markdown>

![TorchLingo Logo](assets/logo.svg){ width="200" }

**A beginner-friendly PyTorch library for learning Neural Machine Translation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/01-data-and-vocab.ipynb) -->

[Getting Started](getting-started/installation.md){ .md-button .md-button--primary }
[View Tutorials](tutorials/index.md){ .md-button }

</div>

---

## What is TorchLingo?

**TorchLingo** is a compact, educational library designed to teach you the fundamentals of **Neural Machine Translation (NMT)** using PyTorch. Whether you've never touched PyTorch before or you're looking to understand how Google Translate works under the hood, TorchLingo has you covered.

!!! tip "Perfect for Students"
    TorchLingo was built for university coursework. The code is intentionally simple, readable, and well-documented—no magic, no hidden complexity.

!!! note "Run in Google Colab"
    The easiest way to get started is with Google Colab—no installation required on your machine!
    
    ```python
    %pip install torchlingo
    ```
    
    Colab provides **free GPU access** which speeds up training significantly. Just go to **Runtime → Change runtime type → GPU**.

## ✨ Features

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Easy to Start**

    ---

    Get your first translation model running in under 5 minutes with our quickstart guide.

    [:octicons-arrow-right-24: Quick Start](getting-started/quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **Learn by Doing**

    ---

    Interactive Jupyter notebooks walk you through every concept step-by-step.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-code-tags:{ .lg .middle } **Readable Code**

    ---

    Every function is documented. Every tensor shape is explained. No black boxes.

    [:octicons-arrow-right-24: API Reference](reference/index.md)

-   :material-swap-horizontal:{ .lg .middle } **Two Architectures**

    ---

    Learn both classic **LSTM** and modern **Transformer** models side by side.

    [:octicons-arrow-right-24: Model Concepts](concepts/models.md)

</div>

## 🚀 Quick Example

Here's a taste of what working with TorchLingo looks like:

```python
from torchlingo.config import Config
from torchlingo.data_processing import NMTDataset, create_dataloaders
from torchlingo.models import SimpleTransformer

# 1. Create a configuration
config = Config(
    batch_size=32,
    learning_rate=1e-4,
    d_model=256,
)

# 2. Load your data
train_dataset = NMTDataset("data/train.tsv", config=config)

# 3. Create your model
model = SimpleTransformer(
    src_vocab_size=len(train_dataset.src_vocab),
    tgt_vocab_size=len(train_dataset.tgt_vocab),
    config=config,
)

# 4. Train and translate!
# (See our tutorials for the complete training loop)
```

## 🗺️ Where to Go Next

| If you want to...                          | Go here                                              |
| ------------------------------------------ | ---------------------------------------------------- |
| Install TorchLingo                         | [Installation Guide](getting-started/installation.md) |
| Build your first model in 5 minutes        | [Quick Start](getting-started/quickstart.md)          |
| Understand the theory behind NMT           | [What is NMT?](concepts/what-is-nmt.md)               |
| Follow step-by-step notebooks              | [Tutorials](tutorials/index.md)                       |
| Look up a specific function                | [API Reference](reference/index.md)                   |

## 📚 Philosophy

TorchLingo follows a few key principles:

1. **Explicit is better than implicit** — You can trace every tensor operation
2. **Notebooks are first-class** — Every concept is executable
3. **Start simple, add complexity later** — Basic tokenization → SentencePiece → Multilingual
4. **Docs that teach** — Every docstring explains *why*, not just *what*

Read more about our [design philosophy](concepts/what-is-nmt.md).

---

<div align="center" markdown>

**Ready to translate?** :fontawesome-solid-language:

[Get Started :material-arrow-right:](getting-started/installation.md){ .md-button .md-button--primary }

</div>
