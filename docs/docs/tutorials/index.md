# Tutorials

Welcome to the TorchLingo tutorials! These interactive Jupyter notebooks will guide you through building a complete neural machine translation system.

## 🚀 Run in Google Colab (Recommended)

The easiest way to run these tutorials is in **Google Colab**—no installation required!

| Tutorial | Description | Open in Colab |
|----------|-------------|---------------|
| **1. Data and Vocabulary** | Load data, build vocabularies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/01-data-and-vocab.ipynb) |
| **2. Train a Tiny Model** | Build and train a Transformer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/02-train-tiny-model.ipynb) |
| **3. Inference and Beam Search** | Generate translations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/03-inference-and-beamsearch.ipynb) |

!!! tip "Enable GPU in Colab"
    For faster training, enable GPU: **Runtime → Change runtime type → GPU**
    
    Colab provides free access to NVIDIA GPUs!

## Learning Path

Follow these tutorials in order for the best learning experience:

<div class="grid cards" markdown>

-   :material-numeric-1-circle:{ .lg .middle } **Data and Vocabulary**

    ---

    Learn how to load parallel data, build vocabularies, and prepare your data for training.

    [:octicons-arrow-right-24: Start Tutorial](01-data-and-vocab.ipynb)

-   :material-numeric-2-circle:{ .lg .middle } **Train a Tiny Model**

    ---

    Build and train your first Transformer model on a small dataset—runs in seconds!

    [:octicons-arrow-right-24: Start Tutorial](02-train-tiny-model.ipynb)

-   :material-numeric-3-circle:{ .lg .middle } **Inference and Beam Search**

    ---

    Generate translations using greedy and beam search decoding strategies.

    [:octicons-arrow-right-24: Start Tutorial](03-inference-and-beamsearch.ipynb)

</div>

## Prerequisites

**For Google Colab (Recommended):**

- [x] A Google account
- [x] Basic Python knowledge

**For Local Setup:**

- [x] [TorchLingo installed](../getting-started/installation.md)
- [x] Basic Python knowledge
- [x] Jupyter Notebook installed (`pip install notebook`)

## Running the Tutorials

### Option 1: Google Colab (Recommended)

1. Click any "Open in Colab" badge above
2. Go to **Runtime → Change runtime type → GPU**
3. Run the first cell to install TorchLingo: `%pip install torchlingo`

### Option 2: Run Locally

```bash
# Navigate to the tutorials directory
cd docs/tutorials

# Start Jupyter
jupyter notebook
```

### Option 3: Read Online

You can read the tutorials directly in the documentation—the code cells and outputs are rendered for you.

## Tutorial Overview

### Tutorial 1: Data and Vocabulary

**Time**: ~15 minutes

You'll learn:

- Loading TSV, CSV, and other data formats
- Building source and target vocabularies
- Encoding and decoding text
- Creating PyTorch datasets

**Key classes covered**: `load_data()`, `SimpleVocab`, `NMTDataset`

### Tutorial 2: Train a Tiny Model

**Time**: ~20 minutes

You'll learn:

- Creating a Transformer model
- Setting up the training loop
- Teacher forcing explained
- Monitoring training progress
- Saving and loading checkpoints

**Key classes covered**: `SimpleTransformer`, `Config`, `collate_fn`

### Tutorial 3: Inference and Beam Search

**Time**: ~15 minutes

You'll learn:

- Greedy decoding
- Beam search decoding
- Comparing decoding strategies
- Evaluating with BLEU score

**Key concepts**: Inference modes, decoding algorithms, evaluation metrics

## Tips for Success

!!! tip "Run Every Cell"
    Execute cells in order—many depend on previous outputs.

!!! tip "Experiment!"
    Try changing hyperparameters, model sizes, and data. Breaking things is how you learn.

!!! tip "Check Tensor Shapes"
    When debugging, print tensor shapes liberally:
    ```python
    print(f"src shape: {src.shape}")
    ```

## What's Next?

After completing the tutorials:

- 📖 Dive deeper into [Concepts](../concepts/what-is-nmt.md)
- 🔍 Explore the [API Reference](../reference/index.md)
- 🚀 Try your own dataset

---

Ready to begin?

[Start Tutorial 1 :material-arrow-right:](01-data-and-vocab.ipynb){ .md-button .md-button--primary }
