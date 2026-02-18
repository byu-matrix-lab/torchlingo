# Multilingual Training Analysis and Recommendations

## Current State

### ✅ What's Already Implemented

The codebase has **basic multilingual support** for bidirectional EN↔X translation:

1. **Language Tag Utilities** ([preprocessing/multilingual.py](src/torchlingo/preprocessing/multilingual.py))
   - `add_language_tags()`: Prepends language tags to source sentences
   - `preprocess_multilingual()`: Creates bidirectional training data (EN→X and X→EN)

2. **Config Support** ([config.py](src/torchlingo/config.py))
   - `lang_tag_en_to_x`: Default `"<2X>"` for EN→target language
   - `lang_tag_x_to_en`: Default `"<2E>"` for target→EN language

3. **Current Workflow**:
   ```python
   # Example: EN↔ES training
   df = pd.DataFrame({'src': ['hello'], 'tgt': ['hola']})

   # Adds "<2X> hello" → "hola"
   tagged_en_x = add_language_tags(df, '<2X>')

   # Creates reverse: "hola" → "<2E> hello"
   # Then combines and shuffles both directions
   preprocess_multilingual(train_file, val_file, test_file)
   ```

### ❌ What's Missing for True Multilingual Support

**Current limitation**: Only supports **two-language bidirectional** training (EN↔X). Does not support:

1. **Multiple Target Languages**: Cannot train EN→{ES, FR, DE, ...} in one model
2. **Many-to-Many**: Cannot train {EN, ES, FR}↔{EN, ES, FR} in one model
3. **Language Tag Vocabulary Management**: No automatic way to ensure language tags are in vocabulary
4. **Language-Specific Inference**: No built-in way to specify target language at inference time

## Recommendations for Comprehensive Multilingual Support

### Option 1: Minimal Changes (Quick Fix)

**Goal**: Support multiple target languages with minimal code changes

**Changes Needed**:

1. **Make language tags more flexible**:
   ```python
   # In config.py - add support for multiple language tags
   LANGUAGE_TAGS = {
       'es': '<2es>',
       'fr': '<2fr>',
       'de': '<2de>',
       'zh': '<2zh>',
   }
   ```

2. **Enhance `add_language_tags()` to accept language code**:
   ```python
   def add_language_tags_multi(
       df: pd.DataFrame,
       target_lang: str,  # e.g., 'es', 'fr', 'de'
       language_tags: Dict[str, str] = None,
       src_col: str = None,
       config: Config = None,
   ) -> pd.DataFrame:
       """Add language tag based on target language code."""
       cfg = config if config is not None else get_default_config()
       tags = language_tags or cfg.language_tags or LANGUAGE_TAGS

       if target_lang not in tags:
           raise ValueError(f"Unknown language: {target_lang}")

       tag = tags[target_lang]
       return add_language_tags(df, tag, src_col, config)
   ```

3. **Create multilingual dataset builder**:
   ```python
   def create_multilingual_dataset(
       data_sources: Dict[str, Path],  # {'es': path_to_en_es.tsv, 'fr': ...}
       config: Config = None,
   ) -> pd.DataFrame:
       """Combine multiple language pairs with appropriate tags."""
       combined = []
       for lang_code, data_file in data_sources.items():
           df = load_data(data_file)
           tagged = add_language_tags_multi(df, lang_code, config=config)
           combined.append(tagged)

       # Shuffle all languages together
       result = pd.concat(combined).sample(frac=1).reset_index(drop=True)
       return result
   ```

4. **Ensure language tags are in vocabulary**:
   ```python
   # For SimpleVocab: manually add to vocab
   vocab = SimpleVocab()
   vocab.build_vocab(sentences)
   # Add language tags to vocabulary
   for tag in ['<2es>', '<2fr>', '<2de>']:
       if tag not in vocab.token2idx:
           vocab.token2idx[tag] = len(vocab)
           vocab.idx2token[len(vocab)-1] = tag

   # For SentencePiece: include in training data or use user_defined_symbols
   ```

### Option 2: Comprehensive Implementation (Recommended)

**Goal**: Full-featured multilingual system with proper abstraction

**Architecture**:

```
MultilingualDataset
├── Language Registry (manages language codes and tags)
├── Tagged Vocabulary (handles language tags automatically)
└── Multilingual DataLoader (batches by language for efficiency)
```

**Key Components to Add**:

#### 1. Language Registry

```python
# src/torchlingo/preprocessing/language_registry.py

@dataclass
class LanguageInfo:
    code: str          # ISO code: 'es', 'fr', 'de'
    name: str          # Full name: 'Spanish', 'French', 'German'
    tag: str           # Model tag: '<2es>', '<2fr>', '<2de>'
    direction: str     # 'source', 'target', or 'both'

class LanguageRegistry:
    """Manages language codes, tags, and metadata for multilingual training."""

    def __init__(self):
        self.languages: Dict[str, LanguageInfo] = {}

    def register(self, code: str, name: str, tag: str = None, direction: str = 'both'):
        """Register a new language."""
        tag = tag or f"<2{code}>"
        self.languages[code] = LanguageInfo(code, name, tag, direction)

    def get_tag(self, code: str) -> str:
        """Get language tag for a given code."""
        return self.languages[code].tag

    def all_tags(self) -> List[str]:
        """Get all language tags."""
        return [info.tag for info in self.languages.values()]
```

#### 2. Enhanced Config

```python
# Add to config.py

class Config:
    # ... existing fields ...

    # New multilingual fields
    language_registry: Optional[LanguageRegistry] = None
    multilingual_mode: bool = False
    ensure_language_tags_in_vocab: bool = True
    language_tags: Dict[str, str] = field(default_factory=lambda: {
        'es': '<2es>', 'fr': '<2fr>', 'de': '<2de>',
        'zh': '<2zh>', 'ja': '<2ja>', 'ar': '<2ar>',
    })
```

#### 3. Multilingual Vocabulary Wrapper

```python
# src/torchlingo/data_processing/multilingual_vocab.py

class MultilingualVocab(BaseVocab):
    """Wrapper that ensures language tags are in vocabulary."""

    def __init__(
        self,
        base_vocab: BaseVocab,
        language_registry: LanguageRegistry,
        config: Config = None,
    ):
        self.base_vocab = base_vocab
        self.language_registry = language_registry
        self.language_tags = language_registry.all_tags()

        # Ensure language tags are registered
        if isinstance(base_vocab, SimpleVocab):
            self._add_language_tags_to_simple_vocab()

    def _add_language_tags_to_simple_vocab(self):
        """Add language tags to SimpleVocab."""
        for tag in self.language_tags:
            if tag not in self.base_vocab.token2idx:
                idx = len(self.base_vocab)
                self.base_vocab.token2idx[tag] = idx
                self.base_vocab.idx2token[idx] = tag

    def encode(self, sentence: str, add_special_tokens: bool = True) -> List[int]:
        """Encode with base vocab (language tags already in vocab)."""
        return self.base_vocab.encode(sentence, add_special_tokens)

    # ... delegate other methods to base_vocab ...
```

#### 4. Multilingual Dataset

```python
# src/torchlingo/data_processing/multilingual_dataset.py

class MultilingualNMTDataset(NMTDataset):
    """Dataset for multilingual NMT with automatic language tagging."""

    def __init__(
        self,
        data_sources: Dict[str, Path],  # {'es': path, 'fr': path, ...}
        language_registry: LanguageRegistry,
        src_vocab: BaseVocab,
        tgt_vocab: BaseVocab,
        config: Config = None,
    ):
        # Load and combine all language pairs with tags
        combined_data = self._combine_language_pairs(
            data_sources, language_registry
        )

        # Save to temp file and use parent NMTDataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            combined_data.to_csv(f.name, sep='\t', index=False)
            temp_file = Path(f.name)

        super().__init__(temp_file, src_vocab=src_vocab, tgt_vocab=tgt_vocab, config=config)

        # Clean up temp file
        temp_file.unlink()

    def _combine_language_pairs(
        self,
        data_sources: Dict[str, Path],
        language_registry: LanguageRegistry,
    ) -> pd.DataFrame:
        """Combine multiple language pairs with appropriate tags."""
        combined = []

        for lang_code, data_file in data_sources.items():
            df = load_data(data_file)
            tag = language_registry.get_tag(lang_code)

            # Add language tag to source column
            df_copy = df.copy()
            src_col = self.cfg.src_col
            df_copy[src_col] = f"{tag} " + df_copy[src_col].astype(str)

            combined.append(df_copy)

        # Shuffle all languages together
        result = pd.concat(combined).sample(frac=1).reset_index(drop=True)
        return result
```

#### 5. Multilingual Inference

```python
# Add to inference.py

def translate_multilingual(
    model: nn.Module,
    sentences: Sequence[str],
    target_language: str,  # 'es', 'fr', 'de', etc.
    language_registry: LanguageRegistry,
    src_vocab: BaseVocab,
    tgt_vocab: BaseVocab,
    decode_strategy: str = "greedy",
    config: Config = None,
) -> List[str]:
    """Translate sentences to specified target language."""

    # Get language tag
    lang_tag = language_registry.get_tag(target_language)

    # Prepend language tag to all sentences
    tagged_sentences = [f"{lang_tag} {sent}" for sent in sentences]

    # Use existing translate_batch
    return translate_batch(
        model,
        tagged_sentences,
        src_vocab,
        tgt_vocab,
        decode_strategy=decode_strategy,
        config=config,
    )
```

## Implementation Priority

### Phase 1: Quick Wins (1-2 hours)
1. ✅ Add `language_tags` dict to Config
2. ✅ Create `add_language_tags_multi()` helper
3. ✅ Create `create_multilingual_dataset()` utility
4. ✅ Add documentation and examples

### Phase 2: Vocabulary Integration (2-3 hours)
1. ⬜ Create `ensure_language_tags_in_vocab()` utility
2. ⬜ Add to `SimpleVocab.build_vocab()`
3. ⬜ Add `user_defined_symbols` support for SentencePiece training
4. ⬜ Write tests

### Phase 3: Full Framework (4-6 hours)
1. ⬜ Implement `LanguageRegistry`
2. ⬜ Implement `MultilingualVocab` wrapper
3. ⬜ Implement `MultilingualNMTDataset`
4. ⬜ Add `translate_multilingual()` to inference
5. ⬜ Comprehensive testing
6. ⬜ Example notebooks/scripts

## Example Usage (After Phase 3)

```python
from torchlingo.preprocessing import LanguageRegistry
from torchlingo.data_processing import MultilingualNMTDataset, MultilingualVocab
from torchlingo.inference import translate_multilingual

# Setup language registry
registry = LanguageRegistry()
registry.register('es', 'Spanish', '<2es>')
registry.register('fr', 'French', '<2fr>')
registry.register('de', 'German', '<2de>')

# Prepare data sources
data_sources = {
    'es': Path('data/en_es.tsv'),
    'fr': Path('data/en_fr.tsv'),
    'de': Path('data/en_de.tsv'),
}

# Build vocabulary with language tags
src_vocab = SimpleVocab()
src_vocab.build_vocab(all_source_sentences)
src_vocab = MultilingualVocab(src_vocab, registry)  # Adds language tags

# Create multilingual dataset
dataset = MultilingualNMTDataset(
    data_sources=data_sources,
    language_registry=registry,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
)

# Train model (same as before)
model = SimpleTransformer(...)
train_model(model, train_loader, ...)

# Inference with target language specification
translations_es = translate_multilingual(
    model,
    ["Hello world", "Good morning"],
    target_language='es',
    language_registry=registry,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
)

translations_fr = translate_multilingual(
    model,
    ["Hello world", "Good morning"],
    target_language='fr',
    language_registry=registry,
    src_vocab=src_vocab,
    tgt_vocab=tgt_vocab,
)
```

## Immediate Workaround (Current Codebase)

Until Phase 1 is implemented, you can do this manually:

```python
import pandas as pd
from torchlingo.preprocessing.multilingual import add_language_tags
from torchlingo.config import Config

# Load your language pair datasets
en_es = pd.read_csv('en_es.tsv', sep='\t')
en_fr = pd.read_csv('en_fr.tsv', sep='\t')
en_de = pd.read_csv('en_de.tsv', sep='\t')

# Manually tag each language
en_es_tagged = add_language_tags(en_es, '<2es>')
en_fr_tagged = add_language_tags(en_fr, '<2fr>')
en_de_tagged = add_language_tags(en_de, '<2de>')

# Combine and shuffle
multilingual_data = pd.concat([
    en_es_tagged,
    en_fr_tagged,
    en_de_tagged,
]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save combined dataset
multilingual_data.to_csv('multilingual_train.tsv', sep='\t', index=False)

# Build vocab on tagged data
src_vocab = SimpleVocab()
src_vocab.build_vocab(multilingual_data['src'].tolist())
# Language tags are now in vocab automatically!

# At inference time, manually prepend language tag
def translate_to_spanish(model, text):
    tagged = f"<2es> {text}"
    return translate_batch(model, [tagged], src_vocab, tgt_vocab)[0]
```

## Testing Requirements

For any multilingual implementation, tests should cover:

1. ✅ Language tag prepending
2. ✅ Multiple language pair combination
3. ✅ Language tags in vocabulary
4. ✅ Inference with different target languages
5. ✅ SentencePiece with language tags (user_defined_symbols)
6. ✅ Batch processing with mixed languages
7. ✅ End-to-end multilingual training

## References

- **mBART**: Uses `<2XX>` tokens for target language
- **M2M-100**: Many-to-many translation with language tokens
- **NLLB**: No Language Left Behind approach
- **Original Transformer Paper**: Mentions language embeddings

---

**Recommendation**: Start with **Phase 1** for immediate multilingual support, then implement **Phase 3** for a production-ready solution.
