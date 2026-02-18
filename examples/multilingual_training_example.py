"""Example: Training a multilingual NMT model.

This script demonstrates how to train a single model that can translate
from English to multiple target languages (Spanish, French, German) using
language tags.
"""

from pathlib import Path
import pandas as pd
import torch

from torchlingo.config import Config
from torchlingo.data_processing.dataset import NMTDataset
from torchlingo.data_processing.vocab import SimpleVocab
from torchlingo.data_processing.batching import collate_fn
from torchlingo.models import SimpleTransformer
from torchlingo.training import train_model
from torchlingo.inference import translate_batch
from torchlingo.preprocessing.multilingual_helpers import (
    create_multilingual_dataset,
    ensure_language_tags_in_vocab,
    save_multilingual_splits,
)


def create_sample_data(output_dir: Path):
    """Create sample parallel corpora for demonstration."""
    print("Creating sample parallel corpora...")

    # Sample EN-ES data
    en_es = pd.DataFrame({
        'src': [
            'hello world',
            'good morning',
            'how are you',
            'thank you',
            'goodbye',
        ] * 10,
        'tgt': [
            'hola mundo',
            'buenos días',
            'cómo estás',
            'gracias',
            'adiós',
        ] * 10,
    })

    # Sample EN-FR data
    en_fr = pd.DataFrame({
        'src': [
            'hello world',
            'good morning',
            'how are you',
            'thank you',
            'goodbye',
        ] * 10,
        'tgt': [
            'bonjour monde',
            'bonjour',
            'comment allez-vous',
            'merci',
            'au revoir',
        ] * 10,
    })

    # Sample EN-DE data
    en_de = pd.DataFrame({
        'src': [
            'hello world',
            'good morning',
            'how are you',
            'thank you',
            'goodbye',
        ] * 10,
        'tgt': [
            'hallo welt',
            'guten morgen',
            'wie geht es dir',
            'danke',
            'auf wiedersehen',
        ] * 10,
    })

    # Save files
    output_dir.mkdir(parents=True, exist_ok=True)
    en_es.to_csv(output_dir / 'en_es_train.tsv', sep='\t', index=False)
    en_fr.to_csv(output_dir / 'en_fr_train.tsv', sep='\t', index=False)
    en_de.to_csv(output_dir / 'en_de_train.tsv', sep='\t', index=False)

    print(f"Created sample data in {output_dir}/")


def main():
    """Main training workflow."""

    # Configuration
    data_dir = Path('data/multilingual_example')
    cfg = Config(
        batch_size=8,
        learning_rate=0.001,
        max_seq_length=50,
    )

    # Step 1: Create sample data
    create_sample_data(data_dir)

    # Step 2: Create multilingual training dataset
    print("\nCreating multilingual dataset...")
    data_sources = {
        'es': data_dir / 'en_es_train.tsv',
        'fr': data_dir / 'en_fr_train.tsv',
        'de': data_dir / 'en_de_train.tsv',
    }

    multilingual_df = create_multilingual_dataset(data_sources, config=cfg)

    # Save for later use
    multilingual_file = data_dir / 'multilingual_train.tsv'
    multilingual_df.to_csv(multilingual_file, sep='\t', index=False)
    print(f"Saved to {multilingual_file}")

    # Step 3: Build vocabularies
    print("\nBuilding vocabularies...")

    # Source vocabulary (includes language tags)
    src_vocab = SimpleVocab(min_freq=1)
    src_vocab.build_vocab(multilingual_df['src'].tolist())

    # Ensure language tags are in vocabulary (they should already be there)
    ensure_language_tags_in_vocab(src_vocab, ['es', 'fr', 'de'])

    # Target vocabulary (separate for each language, or combined)
    tgt_vocab = SimpleVocab(min_freq=1)
    tgt_vocab.build_vocab(multilingual_df['tgt'].tolist())

    print(f"Source vocab size: {len(src_vocab)}")
    print(f"Target vocab size: {len(tgt_vocab)}")

    # Verify language tags are present
    print(f"Language tags in vocab: <2es>, <2fr>, <2de> present")

    # Step 4: Create dataset and dataloader
    print("\nCreating dataset...")
    dataset = NMTDataset(
        multilingual_file,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        config=cfg,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Step 5: Create model
    print("\nInitializing model...")
    model = SimpleTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=128,
        n_heads=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_ff=512,
        dropout=0.1,
        config=cfg,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Step 6: Train model
    print("\nTraining multilingual model...")
    result = train_model(
        model,
        train_loader=train_loader,
        val_loader=None,
        num_epochs=10,
        gradient_clip=1.0,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        config=cfg,
    )

    print(f"\nTraining complete!")
    print(f"Final loss: {result.train_losses[-1]:.4f}")
    print(f"Loss improvement: {result.train_losses[0]:.4f} → {result.train_losses[-1]:.4f}")

    # Step 7: Test inference with different target languages
    print("\n" + "=" * 60)
    print("Testing multilingual inference")
    print("=" * 60)

    test_sentences = [
        "hello world",
        "good morning",
        "thank you",
    ]

    for target_lang, lang_name in [('es', 'Spanish'), ('fr', 'French'), ('de', 'German')]:
        print(f"\n{lang_name} translations:")

        # Prepend language tag
        tagged_sentences = [f"<2{target_lang}> {sent}" for sent in test_sentences]

        # Translate
        translations = translate_batch(
            model,
            tagged_sentences,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            decode_strategy='greedy',
            max_len=30,
            config=cfg,
        )

        # Display results
        for src, tgt in zip(test_sentences, translations):
            print(f"  {src:20} → {tgt}")

    print("\n" + "=" * 60)
    print("Multilingual training example complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
