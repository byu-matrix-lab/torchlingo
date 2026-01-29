# Installation

This guide will help you get TorchLingo installed and ready to use in just a few minutes.

## 🚀 Recommended: Google Colab (Easiest)

**The fastest way to get started!** Google Colab requires zero setup on your machine and provides free GPU access.

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/commanderjcc/torchlingo/blob/main/docs/docs/tutorials/01-data-and-vocab.ipynb) -->

### Step 1: Open a Colab Notebook

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click **File → New notebook** (or open one of our tutorial notebooks)

### Step 2: Enable GPU (Important for Training!)

1. Click **Runtime → Change runtime type**
2. Select **GPU** from the "Hardware accelerator" dropdown
3. Click **Save**

!!! tip "Free GPU Access"
    Colab provides free access to NVIDIA GPUs. Training is **10-50x faster** with GPU compared to CPU!

### Step 3: Install TorchLingo

In the first cell of your notebook, run:

```python
%pip install torchlingo
```

### Step 4: Verify Installation

```python
# Check that everything is working
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test TorchLingo
import torchlingo
from torchlingo.config import get_default_config
config = get_default_config()
print(f"\n✓ TorchLingo is ready! Default batch size: {config.batch_size}")
```

You should see output showing CUDA is available and a GPU name (like "Tesla T4").

---

## 💻 Local Installation

If you prefer to run locally or want to explore/modify the source code:

### Prerequisites

Before installing TorchLingo, make sure you have:

- **Python 3.10 or higher** - [Download Python](https://www.python.org/downloads/)
- **pip** (comes with Python)
- **Git** (optional, for cloning the repository)

!!! tip "Check your Python version"
    Open a terminal and run:
    ```bash
    python --version
    ```
    You should see `Python 3.10.x` or higher.

### Method 1: Install with pip (Simplest)

```bash
pip install torchlingo
```

### Method 2: Install from Source (For Development)

This is the best method if you want to explore the code, run tutorials, or modify things:

=== "macOS/Linux"

    ```bash
    # Clone the repository
    git clone https://github.com/commanderjcc/torchlingo.git
    cd torchlingo

    # Create a virtual environment
    python -m venv .venv
    source .venv/bin/activate

    # Install in editable mode
    pip install -e .
    ```

=== "Windows"

    ```powershell
    # Clone the repository
    git clone https://github.com/commanderjcc/torchlingo.git
    cd torchlingo

    # Create a virtual environment
    python -m venv .venv
    .venv\Scripts\activate

    # Install in editable mode
    pip install -e .
    ```

## Verify Installation

Let's make sure everything is working:

```python
# In a Python shell or script
import torchlingo
from torchlingo.config import get_default_config

config = get_default_config()
print(f"TorchLingo is ready! Default batch size: {config.batch_size}")
```

You should see output like:

```
TorchLingo is ready! Default batch size: 64
```

## Installing Dependencies

TorchLingo has a few key dependencies that are automatically installed:

| Package       | Purpose                                |
| ------------- | -------------------------------------- |
| `torch`       | Deep learning framework                |
| `pandas`      | Data loading and manipulation          |
| `sentencepiece` | Subword tokenization                  |
| `tensorboard` | Training visualization                 |
| `sacrebleu`   | Translation quality metrics            |

### Optional: Development Dependencies

If you want to run tests or contribute:

```bash
pip install -e ".[dev,test]"
```

### Optional: Documentation Dependencies

To build these docs locally:

```bash
pip install -e ".[docs]"
```

## GPU Support

TorchLingo works on CPU out of the box, but training is **much faster** with a GPU.

=== "NVIDIA GPU (CUDA)"

    PyTorch should automatically detect your NVIDIA GPU. Verify with:
    
    ```python
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    ```

=== "Apple Silicon (MPS)"

    On M1/M2/M3 Macs, PyTorch can use the Metal Performance Shaders backend:
    
    ```python
    import torch
    print(f"MPS available: {torch.backends.mps.is_available()}")
    ```

=== "CPU Only"

    No GPU? No problem! All tutorials work on CPU. Just expect longer training times for larger models.

## Common Issues

??? question "ImportError: No module named 'torchlingo'"
    Make sure you've activated your virtual environment:
    ```bash
    source .venv/bin/activate  # macOS/Linux
    .venv\Scripts\activate     # Windows
    ```

??? question "ModuleNotFoundError: No module named 'torch'"
    PyTorch didn't install correctly. Try:
    ```bash
    pip install torch --upgrade
    ```

??? question "Permission denied when installing"
    Don't use `sudo pip install`. Use a virtual environment instead (Method 1 above).

## Next Steps

Now that TorchLingo is installed, let's build something!

[Quick Start :material-arrow-right:](quickstart.md){ .md-button .md-button--primary }
