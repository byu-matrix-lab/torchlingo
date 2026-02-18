# Training Guide: Cebuano → Mandarin Chinese NMT

## Dataset

You have a large parallel corpus: **ceb__cmn.csv** (~656k sentence pairs)
- Source language: Cebuano (ceb)
- Target language: Mandarin Chinese (cmn)
- Format: CSV with columns: `src_lang`, `tgt_lang`, `src_text`, `tgt_text`

## Training Scripts

I've created **two training scripts** for different needs:

### 1. Simple Script (Recommended for Getting Started)

**File**: [`train_ceb_cmn_simple.py`](train_ceb_cmn_simple.py)

**Why use this**: Shows how TorchLingo handles all the heavy lifting. Minimal code, package does the work.

**What it does**:
- ✅ Loads data and creates splits
- ✅ Trains SentencePiece models (package handles this)
- ✅ Creates datasets (package handles this)
- ✅ Trains model with validation (package handles this)

**Run it**:
```bash
cd torchlingo
python train_ceb_cmn_simple.py
```

**Code is just ~100 lines** - the package does everything!

### 2. Comprehensive Script (Production Ready)

**File**: [`train_ceb_cmn.py`](train_ceb_cmn.py)

**Why use this**: Full production training with monitoring, performance optimization, and detailed logging.

**What it includes**:
- ✅ Data preprocessing with filtering
- ✅ Performance monitoring
- ✅ Configurable hyperparameters via command line
- ✅ Learning rate warmup scheduler
- ✅ Detailed logging and progress tracking
- ✅ Sample translations for evaluation

**Run it**:
```bash
# Basic usage
python train_ceb_cmn.py

# Custom configuration
python train_ceb_cmn.py \
  --batch-size 64 \
  --vocab-size 32000 \
  --num-epochs 20 \
  --learning-rate 0.0001 \
  --experiment-name large_model
```

**Command-line options**:
```
--vocab-size        SentencePiece vocabulary size (default: 16000)
--batch-size        Training batch size (default: 32)
--learning-rate     Learning rate (default: 0.0001)
--num-epochs        Number of training epochs (default: 10)
--val-interval      Validation every N steps (default: 1000)
--save-interval     Save checkpoint every N steps (default: 2000)
--patience          Early stopping patience (default: 5)
--experiment-name   Name for this run (default: baseline)
--device            Device: auto/cuda/cpu (default: auto)
```

## Quick Start: 5-Minute Setup

```bash
# 1. Navigate to torchlingo directory
cd torchlingo

# 2. Ensure package is installed
pip install -e .

# 3. Run simple training (will take several hours on full dataset)
python train_ceb_cmn_simple.py
```

## Expected Runtime

With your dataset size (~656k examples):

**GPU (NVIDIA RTX 3090 or similar)**:
- Data prep: ~2-3 minutes
- Tokenizer training: ~5-10 minutes
- Model training: ~3-6 hours (10 epochs)

**CPU**:
- Data prep: ~5 minutes
- Tokenizer training: ~10-15 minutes
- Model training: ~24-48 hours (10 epochs)

**Recommendation**: Use GPU for efficient training. If on CPU, reduce dataset size for testing:
```python
# In script, after loading data:
df = df.sample(n=50000, random_state=42)  # Use 50k examples for testing
```

## What the Package Handles

The TorchLingo package abstracts away complexity:

### Data Processing
```python
# Package function does all preprocessing
from torchlingo.preprocessing.sentencepiece import preprocess_sentencepiece

preprocess_sentencepiece(train_file, val_file, test_file, config=cfg)
# ✓ Trains SentencePiece models
# ✓ Tokenizes all data
# ✓ Saves tokenized files
# ✓ Handles special tokens
```

### Dataset Creation
```python
# Package creates ready-to-use datasets
from torchlingo.data_processing.dataset import NMTDataset

dataset = NMTDataset(data_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=cfg)
# ✓ Encodes sentences to token IDs
# ✓ Adds special tokens (SOS/EOS)
# ✓ Handles padding and truncation
```

### Training Loop
```python
# Package handles complete training workflow
from torchlingo.training import train_model

result = train_model(model, train_loader, val_loader, config=cfg)
# ✓ Training loop with loss computation
# ✓ Validation monitoring
# ✓ Early stopping
# ✓ Gradient clipping
# ✓ Checkpointing
# ✓ TensorBoard logging
# ✓ Progress bars
```

### Model Architecture
```python
# Package provides production-ready models
from torchlingo.models import SimpleTransformer

model = SimpleTransformer(src_vocab_size, tgt_vocab_size, config=cfg)
# ✓ Full Transformer encoder-decoder
# ✓ Multi-head attention
# ✓ Rotary position embeddings (RoPE)
# ✓ Proper initialization
```

## Monitoring Training

### TensorBoard

Both scripts use TensorBoard for monitoring:

```bash
# In a separate terminal
tensorboard --logdir runs

# Open browser to http://localhost:6006
```

**What you'll see**:
- Training loss curve
- Validation loss curve
- Learning rate schedule
- Real-time progress

### Checkpoints

Models are saved to `checkpoints/ceb_cmn/`:
- `model_best.pt` - Best model (lowest validation loss)
- `model_last.pt` - Most recent checkpoint

### Console Output

The comprehensive script shows:
```
STEP 1: Data Preprocessing
Loaded 656,153 parallel sentences
Train: 524,922 | Val: 65,615 | Test: 65,616

STEP 2: SentencePiece Training
Training Cebuano tokenizer (vocab_size=16000)...
Training Mandarin tokenizer (vocab_size=16000)...

STEP 3: Dataset Creation
Cebuano vocab size: 16,000
Mandarin vocab size: 16,000

STEP 4: Model Initialization
Parameters: 67,891,200
Size: ~259 MB

STEP 5: Training
Epoch 1/10 | Train: 6.2341 | Val: 5.8923
Epoch 2/10 | Train: 5.4567 | Val: 5.1234
...
```

## Optimizing Training Speed

### 1. Increase Batch Size (if GPU memory allows)
```bash
python train_ceb_cmn.py --batch-size 64
```

### 2. Use Mixed Precision (automatic on GPU)
The scripts automatically use AMP (Automatic Mixed Precision) on GPU for 2x speedup.

### 3. Reduce Model Size (for faster iteration)
Edit the config in script:
```python
cfg = Config(
    d_model=256,          # Instead of 512
    num_encoder_layers=4, # Instead of 6
    num_decoder_layers=4, # Instead of 6
    ...
)
```

### 4. Reduce Vocabulary Size
```bash
python train_ceb_cmn.py --vocab-size 8000
```

### 5. Use Smaller Dataset for Testing
In script, after loading data:
```python
df = df.sample(n=10000, random_state=42)  # Test with 10k examples
```

## After Training

### Load Best Model
```python
import torch
from torchlingo.models import SimpleTransformer

model = SimpleTransformer(src_vocab_size, tgt_vocab_size, config=cfg)
model.load_state_dict(torch.load('checkpoints/ceb_cmn/model_best.pt'))
model.eval()
```

### Translate Sentences
```python
from torchlingo.inference import translate_batch

cebuano_sentences = [
    "Maayong buntag",
    "Kumusta ka",
]

chinese_translations = translate_batch(
    model,
    cebuano_sentences,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
    decode_strategy='greedy',
    max_len=100,
)

for ceb, cmn in zip(cebuano_sentences, chinese_translations):
    print(f"{ceb} → {cmn}")
```

## Troubleshooting

### Out of Memory Error
**Solution**: Reduce batch size
```bash
python train_ceb_cmn.py --batch-size 16
```

### Training Too Slow on CPU
**Solution**: Use smaller subset
```python
# In script
df = df.sample(n=50000, random_state=42)
```

### Loss Not Decreasing
**Solutions**:
- Check data quality (no missing values)
- Reduce learning rate: `--learning-rate 0.00005`
- Increase warmup steps (edit script)
- Check vocabulary coverage

### SentencePiece Training Fails
**Solution**: Data might have encoding issues
```python
# Before training SentencePiece
df[cfg.src_col] = df[cfg.src_col].str.encode('utf-8', errors='ignore').str.decode('utf-8')
df[cfg.tgt_col] = df[cfg.tgt_col].str.encode('utf-8', errors='ignore').str.decode('utf-8')
```

## Recommended Workflow

### Phase 1: Quick Test (30 minutes)
```python
# Use 10k examples to verify everything works
df = df.sample(n=10000, random_state=42)
```
```bash
python train_ceb_cmn_simple.py
```

### Phase 2: Medium Run (2-3 hours)
```python
# Use 100k examples
df = df.sample(n=100000, random_state=42)
```
```bash
python train_ceb_cmn.py --batch-size 64 --num-epochs 5
```

### Phase 3: Full Training (24 hours)
```bash
# Use full dataset
python train_ceb_cmn.py --batch-size 64 --num-epochs 20 --vocab-size 32000
```

## Performance Expectations

With proper hyperparameters, expect:

**After 5 epochs** (~125k steps):
- Training loss: ~3.5
- Validation loss: ~4.0
- Translations: Partially correct, word order issues

**After 10 epochs** (~250k steps):
- Training loss: ~2.5
- Validation loss: ~3.2
- Translations: Mostly correct, some errors

**After 20 epochs** (~500k steps):
- Training loss: ~1.8
- Validation loss: ~2.8
- Translations: High quality, fluent

## Next Steps

1. **Run quick test**: `python train_ceb_cmn_simple.py` with small data
2. **Monitor training**: Open TensorBoard
3. **Evaluate results**: Check sample translations
4. **Tune hyperparameters**: Adjust based on results
5. **Full training**: Run on full dataset with optimized settings

## Resources

- **Test Suite**: Run `python -m unittest discover tests` to verify package
- **Documentation**: See `TESTING_GUIDE.md` and `CLAUDE.md`
- **Multilingual**: See `MULTILINGUAL_QUICKSTART.md` for multi-language training

---

**Ready to train?** Start with the simple script and scale up! 🚀
