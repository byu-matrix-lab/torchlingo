# Tutorials

Welcome to the TorchLingo tutorials! These interactive Jupyter notebooks will guide you through building a complete neural machine translation system.

## 🚀 Run in Google Colab (Recommended)

The easiest way to run these tutorials is in **Google Colab**—no installation required!

| Tutorial | Description | Open in Colab |
|----------|-------------|---------------|
| **1. Data and Vocabulary** | Load data, build vocabularies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/01-data-and-vocab.ipynb) |
| **2. Evaluating Translations** | Watch the default metric pick the worse system | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/02-evaluating-translations.ipynb) |
| **3. Train a Tiny Model** | Build and train a Transformer | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/03-train-tiny-model.ipynb) |
| **4. Attention and Alignment** | Measure what attention learns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/04-attention-and-alignment.ipynb) |
| **5. Attention on a Real Transformer** | Read a pretrained model's alignment on unseen text | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/05-transformer-attention.ipynb) |
| **6. Diagnosing Failures** | Break a model on purpose; watch each check fire | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/06-diagnosing-failures.ipynb) |
| **7. Translating Unseen Sentences** | A real model on text it never saw | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/07-real-translations.ipynb) |
| **8. Inference and Beam Search** | Generate translations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/byu-matrix-lab/torchlingo/blob/main/docs/docs/tutorials/08-inference-and-beamsearch.ipynb) |

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

-   :material-numeric-2-circle:{ .lg .middle } **Evaluating Translations**

    ---

    Score two systems three ways, and watch the default metric pick the one that changed the meaning.

    [:octicons-arrow-right-24: Start Tutorial](02-evaluating-translations.ipynb)

-   :material-numeric-3-circle:{ .lg .middle } **Train a Tiny Model**

    ---

    Build and train your first Transformer model on a small dataset—runs in seconds!

    [:octicons-arrow-right-24: Start Tutorial](03-train-tiny-model.ipynb)

-   :material-numeric-4-circle:{ .lg .middle } **Attention and Alignment**

    ---

    Train the same model with and without attention, then check whether it learned the *correct* alignment.

    [:octicons-arrow-right-24: Start Tutorial](04-attention-and-alignment.ipynb)

-   :material-numeric-5-circle:{ .lg .middle } **Attention on a Real Transformer**

    ---

    Read where a pretrained Transformer looked while it translated a sentence it never saw.

    [:octicons-arrow-right-24: Start Tutorial](05-transformer-attention.ipynb)

-   :material-numeric-6-circle:{ .lg .middle } **Diagnosing Failures**

    ---

    Break a model five different ways and watch a specific check catch each one.

    [:octicons-arrow-right-24: Start Tutorial](06-diagnosing-failures.ipynb)

-   :material-numeric-7-circle:{ .lg .middle } **Translating Unseen Sentences**

    ---

    Take a model trained on real data and read what it produces on sentences it never saw.

    [:octicons-arrow-right-24: Start Tutorial](07-real-translations.ipynb)

-   :material-numeric-8-circle:{ .lg .middle } **Inference and Beam Search**

    ---

    Generate translations using greedy and beam search decoding strategies.

    [:octicons-arrow-right-24: Start Tutorial](08-inference-and-beamsearch.ipynb)

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

### Tutorial 2: Evaluating Translations

Given two systems, which one ships? Fixed strings, so each example changes one thing.

You'll learn:

- Why BLEU, chrF and TER can rank the same two systems three different ways
- Why a corpus score and an average of sentence scores differ
- Why a score needs its signature, and how the wrong reference shape returns a plausible
  number and no error

**Key functions covered**: `compute_bleu`, `compute_chrf`, `compute_ter`

### Tutorial 3: Train a Tiny Model

**Time**: ~20 minutes

You'll learn:

- Creating a Transformer model
- Setting up the training loop
- Teacher forcing explained
- Monitoring training progress
- Saving and loading checkpoints

**Key classes covered**: `SimpleTransformer`, `Config`, `collate_fn`

### Tutorial 4: Attention and Alignment

**Time**: ~20 minutes

You'll learn:

- Why a fixed-size hidden state is a bottleneck — seen in the code, as a discarded variable
- Running the with-versus-without ablation on one flag
- Measuring whether attention learned the *correct* alignment, not just a lower loss
- Reading an alignment heatmap
- Why Transformer self-attention is the same operation

**Key classes covered**: `SimpleSeq2SeqLSTM(attention=True)`, `plot_attention`, `format_attention`

### Tutorial 5: Attention on a Real Transformer

Tutorial 4's inspection, on a pretrained Transformer reading a held-out sentence.

You'll learn:

- Reading cross-attention as an alignment map on real text, with subword pieces
- Why a model can find the right source words and still translate badly
- Spotting a dropped word as a column nothing attends to

**Key functions covered**: `attention_for_sequence`, `format_attention`, `greedy_decode`

### Tutorial 6: Diagnosing Failures

**Time**: ~20 minutes

Every other tutorial shows you something that works. This one breaks things on
purpose, so that a misbehaving model leaves you with a procedure instead of a
hunch.

You'll learn:

- The order to check things in, cheapest first — and why a *yes* at any level
  makes everything below it meaningless
- Catching a misaligned corpus before it costs you a training run
- Telling "needs more epochs" apart from "these parameters were never going to
  move", using one backward pass
- Spotting memorization from the *sign* of the train/validation gap
- Why a contaminated test set reports a number that is true of neither half

**Key concepts**: `ln(V)` as the "learned nothing" reference, gradient flow,
generalization gap, test contamination, `model.eval()`

**Key functions covered**: `diagnose_alignment`, `shuffle_target_side`, `compute_bleu`

### Tutorial 7: Translating Sentences It Has Never Seen

**Time**: ~15 minutes

Loads a model trained on 64,311 real English–Spanish pairs and points it at
talks it never saw. **The translations are not good** — seeing exactly *how*
they fall short is the point.

You'll learn:

- Why holding out whole *talks* matters, and why holding out random sentences
  flatters the score
- Reading a real model's output honestly instead of a memorized toy corpus
- What undertraining looks like from the outside

**Key concepts**: held-out evaluation, BLEU on real text, undertrained vs.
data-starved

### Tutorial 8: Inference and Beam Search

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
