import math
import types
import unittest
import tempfile
from pathlib import Path
from unittest import mock

import torch

from torchlingo.training import train_model, TrainResult
from torchlingo.inference import greedy_decode, beam_search_decode, translate_batch
from torchlingo.config import get_default_config
from torchlingo.data_processing.vocab import SimpleVocab, SentencePieceVocab


class DummyTransformer(torch.nn.Module):
    """Minimal Transformer-like model with encode/decode hooks for tests."""

    def __init__(self, vocab_size: int = 10, eos_idx: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
        self.dummy = torch.nn.Parameter(torch.zeros(1))  # ensures device discovery
        self.proj = torch.nn.Linear(4, vocab_size)

    def encode(self, src, src_key_padding_mask=None):
        batch, src_len = src.shape
        return torch.zeros(batch, src_len, 4, device=src.device)

    def decode(
        self,
        tgt,
        memory,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
        tgt_mask=None,
    ):
        batch, tgt_len = tgt.shape
        zeros = torch.zeros(batch, tgt_len, 4, device=tgt.device)
        logits = self.proj(zeros)
        logits = logits.clone()
        logits[:, -1, self.eos_idx] += 5.0  # make EOS best
        logits[:, -1, 4] += 4.0  # second-best to test ordering
        return logits

    def forward(self, src, tgt):
        memory = self.encode(src)
        return self.decode(tgt, memory)


class DummyLSTMModel(torch.nn.Module):
    """LSTM-based model exercising the LSTM decode path."""

    def __init__(self, vocab_size: int = 10, eos_idx: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.eos_idx = eos_idx
        self.src_embed = torch.nn.Embedding(vocab_size, 4, padding_idx=0)
        self.tgt_embed = torch.nn.Embedding(vocab_size, 4, padding_idx=0)
        self.encoder = torch.nn.LSTM(4, 4, batch_first=True)
        self.decoder = torch.nn.LSTM(4, 4, batch_first=True)
        self.output = torch.nn.Linear(4, vocab_size)
        torch.nn.init.zeros_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)
        with torch.no_grad():
            self.output.bias[self.eos_idx] = 5.0

    def forward(self, src, tgt):
        src_emb = self.src_embed(src)
        _, (h, c) = self.encoder(src_emb)
        tgt_emb = self.tgt_embed(tgt)
        dec_out, _ = self.decoder(tgt_emb, (h, c))
        return self.output(dec_out)


PARALLEL_SENTENCES: list[tuple[str, str]] = [
    (
        "the curious cat watches morning birds quietly",
        "el gato curioso observa aves matutinas en silencio",
    ),
    (
        "a small child paints bright dreams today",
        "un nino pequeno pinta suenos brillantes hoy",
    ),
    (
        "students gather early to discuss science",
        "estudiantes se reunen temprano para discutir ciencia",
    ),
    (
        "the old bridge stands over calm river",
        "el viejo puente se alza sobre rio tranquilo",
    ),
    (
        "morning coffee warms tired minds slowly",
        "el cafe matutino calienta mentes cansadas lentamente",
    ),
    (
        "the stormy night frightened the lonely traveler",
        "la noche tormentosa asusto al viajero solitario",
    ),
    (
        "friendly neighbors share fresh bread weekly",
        "vecinos amables comparten pan fresco cada semana",
    ),
    (
        "engineers design resilient systems for everyone",
        "ingenieros disenan sistemas resistentes para todos",
    ),
    (
        "musicians rehearse before the grand concert",
        "musicos ensayan antes del gran concierto",
    ),
    (
        "the library invites quiet reflection after work",
        "la biblioteca invita a una reflexion tranquila despues del trabajo",
    ),
    (
        "gardeners water new plants during sunrise",
        "jardineros riegan nuevas plantas durante el amanecer",
    ),
    (
        "the research team publishes transparent results frequently",
        "el equipo de investigacion publica resultados transparentes con frecuencia",
    ),
    (
        "campers celebrate under clear starlit skies",
        "campistas celebran bajo cielos claros y estrellados",
    ),
    (
        "teachers encourage curiosity in every lesson",
        "maestros fomentan la curiosidad en cada leccion",
    ),
    (
        "the rescue crew arrives within minutes always",
        "el equipo de rescate llega en pocos minutos siempre",
    ),
    (
        "writers capture memories in patient journals",
        "escritores capturan recuerdos en diarios pacientes",
    ),
    (
        "scientists analyze data to reduce uncertainty",
        "cientificos analizan datos para reducir incertidumbre",
    ),
    (
        "athletes train daily to achieve steady progress",
        "atletas se entrenan a diario para lograr progreso constante",
    ),
    (
        "the coastal wind carries salt and stories",
        "el viento costero lleva sal y historias",
    ),
    (
        "festival lights shimmer across the open plaza",
        "las luces del festival brillan a traves de la plaza abierta",
    ),
    (
        "volunteers organize supplies for displaced families",
        "voluntarios organizan suministros para familias desplazadas",
    ),
]


def _english_sentences() -> list[str]:
    return [src for src, _ in PARALLEL_SENTENCES]


def _spanish_sentences() -> list[str]:
    return [tgt for _, tgt in PARALLEL_SENTENCES]


def _build_simple_vocabs() -> tuple[SimpleVocab, SimpleVocab]:
    src_vocab = SimpleVocab(min_freq=1)
    tgt_vocab = SimpleVocab(min_freq=1)
    src_vocab.build_vocab(_english_sentences())
    tgt_vocab.build_vocab(_spanish_sentences())
    return src_vocab, tgt_vocab


def _load_sentencepiece_vocabs(
    config=None,
) -> tuple[SentencePieceVocab, SentencePieceVocab]:
    cfg = config if config is not None else get_default_config()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    src_model = data_dir / "sp_model_src.model"
    tgt_model = data_dir / "sp_model_tgt.model"
    if not src_model.exists() or not tgt_model.exists():
        raise unittest.SkipTest("SentencePiece model artifacts are missing in data/.")
    return (
        SentencePieceVocab(str(src_model), config=cfg),
        SentencePieceVocab(str(tgt_model), config=cfg),
    )


def _toy_loader(batch_size: int = 2):
    src = torch.tensor([[2, 5, 3], [2, 6, 3]], dtype=torch.long)
    tgt = torch.tensor([[2, 8, 3], [2, 9, 3]], dtype=torch.long)
    ds = torch.utils.data.TensorDataset(src, tgt)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


class TrainModelTests(unittest.TestCase):
    def test_train_model_without_validation_returns_losses_and_no_checkpoint(self):
        model = DummyTransformer()
        loader = _toy_loader()
        result = train_model(
            model, train_loader=loader, val_loader=None, num_epochs=2, save_dir=None
        )

        self.assertIsInstance(result, TrainResult)
        self.assertEqual(len(result.train_losses), 2)
        self.assertEqual(result.val_losses, [])
        self.assertIsNone(result.best_checkpoint)

    def test_train_model_with_validation_saves_best_checkpoint(self):
        model = DummyTransformer()
        train_loader = _toy_loader()
        val_loader = _toy_loader()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_model(
                model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=2,
                save_dir=Path(tmpdir),
                gradient_clip=1.0,
            )

            self.assertEqual(len(result.train_losses), 2)
            self.assertEqual(len(result.val_losses), 2)
            self.assertIsNotNone(result.best_checkpoint)
            self.assertTrue(result.best_checkpoint.exists())

    def test_train_model_warns_on_zero_step_epoch(self):
        """A loader that yields no batches should trigger a RuntimeWarning."""
        model = DummyTransformer()
        empty_ds = torch.utils.data.TensorDataset(
            torch.empty(0, 3, dtype=torch.long), torch.empty(0, 3, dtype=torch.long)
        )
        empty_loader = torch.utils.data.DataLoader(empty_ds, batch_size=2)
        with self.assertWarns(RuntimeWarning):
            result = train_model(
                model,
                train_loader=empty_loader,
                val_loader=None,
                num_epochs=1,
                save_dir=None,
            )
        self.assertEqual(len(result.train_losses), 1)

    def test_logged_train_loss_is_windowed_and_epoch_continuous(self):
        """train/loss should be the mean batch loss over the logging window,
        carried across epoch boundaries (not a per-epoch cumulative average
        that resets and produces a sawtooth drop each epoch)."""

        class _RecordingLoss(torch.nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.values = []

            def forward(self, logits, target):
                loss = self.base(logits, target)
                self.values.append(loss.item())
                return loss

        records: dict[str, list] = {}

        class _FakeWriter:
            def __init__(self, *a, **k):
                pass

            def add_scalar(self, tag, value, step):
                records.setdefault(tag, []).append((step, value))

            def close(self):
                pass

        # 3 single-sample batches per epoch, 2 epochs, log every 2 steps.
        src = torch.tensor([[2, 5, 3], [2, 6, 3], [2, 7, 3]], dtype=torch.long)
        tgt = torch.tensor([[2, 8, 3], [2, 9, 3], [2, 4, 3]], dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(src, tgt), batch_size=1, shuffle=False
        )

        cfg = get_default_config()
        cfg.use_tensorboard = True
        criterion = _RecordingLoss(torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_idx))

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.tensorboard_dir = Path(tmpdir)
            with mock.patch("torchlingo.training.SummaryWriter", _FakeWriter):
                train_model(
                    DummyTransformer(),
                    train_loader=loader,
                    val_loader=None,
                    num_epochs=2,
                    criterion=criterion,
                    config=cfg,
                    log_every=2,
                    save_dir=None,
                )

        losses = criterion.values  # per-step batch losses, in order
        logged = [v for _tag_step, v in records.get("train/loss", [])]

        # Logs fire at step 2 of each epoch. Window 1 = steps 1-2 of epoch 1.
        # Window 2 spans the epoch boundary: step 3 of epoch 1 plus steps 1-2 of
        # epoch 2 (3 batches) -> mean of losses[2], losses[3], losses[4]. The
        # old per-epoch average would instead reset and report mean(losses[3],
        # losses[4]); including losses[2] proves the window is epoch-continuous.
        self.assertEqual(len(logged), 2)
        self.assertAlmostEqual(logged[0], sum(losses[0:2]) / 2, places=5)
        self.assertAlmostEqual(logged[1], sum(losses[2:5]) / 3, places=5)

    def test_accumulation_rejects_invalid_steps(self):
        with self.assertRaises(ValueError):
            train_model(
                DummyTransformer(),
                train_loader=_toy_loader(),
                num_epochs=1,
                accumulation_steps=0,
            )

    def test_accumulation_optimizer_step_count_with_partial_window(self):
        """With 5 batches and accumulation_steps=2, the optimizer steps at
        batches 2, 4, and 5 (the final partial window is flushed) -> 3 steps."""
        model = DummyTransformer()
        src = torch.tensor([[2, 5, 3]] * 5, dtype=torch.long)
        tgt = torch.tensor([[2, 8, 3]] * 5, dtype=torch.long)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(src, tgt), batch_size=1, shuffle=False
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        calls = {"n": 0}
        real_step = opt.step

        def counting_step(self, *a, **k):
            calls["n"] += 1
            return real_step(*a, **k)

        # torch's LR schedulers wrap opt.step for call tracking and expect it
        # to be a bound method (they read step.__func__); a plain function
        # assigned directly would break that. types.MethodType keeps it bound.
        opt.step = types.MethodType(counting_step, opt)
        train_model(
            model,
            train_loader=loader,
            num_epochs=1,
            optimizer=opt,
            accumulation_steps=2,
            save_dir=None,
        )
        self.assertEqual(calls["n"], 3)

    def test_accumulation_matches_large_batch_update(self):
        """accumulation_steps=k over k equal micro-batches gives the same
        parameter update as a single batch of k x the size (no padding)."""
        from torchlingo.models import SimpleTransformer

        def make_model():
            torch.manual_seed(123)
            return SimpleTransformer(
                src_vocab_size=20,
                tgt_vocab_size=20,
                d_model=16,
                n_heads=2,
                num_encoder_layers=1,
                num_decoder_layers=1,
                d_ff=32,
                dropout=0.0,  # determinism
            )

        torch.manual_seed(0)
        # Fixed-length, pad-free samples so per-micro-batch token counts match.
        src = torch.randint(4, 20, (4, 5), dtype=torch.long)
        tgt = torch.randint(4, 20, (4, 6), dtype=torch.long)
        ds = torch.utils.data.TensorDataset(src, tgt)

        # A: two micro-batches of 2, accumulate 2 -> one optimizer step.
        model_a = make_model()
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
        train_model(
            model_a,
            torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False),
            num_epochs=1,
            optimizer=opt_a,
            accumulation_steps=2,
            save_dir=None,
        )

        # B: one batch of 4 -> one optimizer step.
        model_b = make_model()
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
        train_model(
            model_b,
            torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False),
            num_epochs=1,
            optimizer=opt_b,
            accumulation_steps=1,
            save_dir=None,
        )

        for (n_a, p_a), (_n_b, p_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            self.assertTrue(
                torch.allclose(p_a, p_b, atol=1e-6),
                msg=f"param {n_a} differs between accumulation and large-batch",
            )

    def test_nonfinite_gradient_skips_optimizer_step(self):
        """A non-finite gradient norm must skip the update (not poison weights)
        and surface a RuntimeWarning."""
        model = DummyTransformer()
        loader = _toy_loader()  # single batch
        opt = torch.optim.SGD(model.parameters(), lr=1.0)
        calls = {"n": 0}
        real_step = opt.step

        def counting_step(self, *a, **k):
            calls["n"] += 1
            return real_step(*a, **k)

        # torch's LR schedulers wrap opt.step for call tracking and expect it
        # to be a bound method (they read step.__func__); a plain function
        # assigned directly would break that. types.MethodType keeps it bound.
        opt.step = types.MethodType(counting_step, opt)
        # Force the clipped grad norm to look non-finite.
        with mock.patch(
            "torchlingo.training.clip_grad_norm_",
            return_value=torch.tensor(float("inf")),
        ):
            with self.assertWarns(RuntimeWarning):
                train_model(
                    model,
                    train_loader=loader,
                    num_epochs=1,
                    optimizer=opt,
                    gradient_clip=1.0,
                    save_dir=None,
                )
        self.assertEqual(calls["n"], 0)  # every step skipped

    def test_train_model_honors_gradient_clip(self):
        model = DummyTransformer()
        loader = _toy_loader()
        result = train_model(
            model,
            train_loader=loader,
            val_loader=None,
            num_epochs=1,
            gradient_clip=0.1,
            save_dir=None,
        )
        self.assertEqual(len(result.train_losses), 1)
        self.assertTrue(math.isfinite(result.train_losses[0]))


class GreedyDecodeTests(unittest.TestCase):
    def test_greedy_decode_transformer_path_returns_eos(self):
        cfg = get_default_config()
        model = DummyTransformer(eos_idx=cfg.eos_idx)
        src = torch.tensor([[cfg.sos_idx, 11, cfg.eos_idx]])

        decoded = greedy_decode(model, src, max_len=5, config=cfg)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0][0], cfg.sos_idx)
        self.assertIn(cfg.eos_idx, decoded[0])

    def test_greedy_decode_lstm_path_returns_eos(self):
        cfg = get_default_config()
        model = DummyLSTMModel(eos_idx=cfg.eos_idx)
        src = torch.tensor([[cfg.sos_idx, 8, cfg.eos_idx]])

        decoded = greedy_decode(model, src, max_len=5, config=cfg)
        self.assertEqual(decoded[0][0], cfg.sos_idx)
        self.assertIn(cfg.eos_idx, decoded[0])

    def test_greedy_decode_raises_on_missing_interfaces(self):
        class NoDecode(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))

        model = NoDecode()
        src = torch.tensor([[1, 2, 3]])
        with self.assertRaises(ValueError):
            greedy_decode(model, src)


class BeamSearchDecodeTests(unittest.TestCase):
    def test_beam_search_decode_returns_sequence_with_eos(self):
        cfg = get_default_config()
        model = DummyTransformer(eos_idx=cfg.eos_idx)
        src = torch.tensor([[cfg.sos_idx, 5, cfg.eos_idx]])

        tokens = beam_search_decode(model, src, beam_size=2, max_len=5, config=cfg)
        self.assertEqual(tokens[0], cfg.sos_idx)
        self.assertIn(cfg.eos_idx, tokens)

    def test_beam_search_decode_raises_on_batch_size_gt_one(self):
        cfg = get_default_config()
        model = DummyTransformer(eos_idx=cfg.eos_idx)
        src = torch.tensor(
            [
                [cfg.sos_idx, 5, cfg.eos_idx],
                [cfg.sos_idx, 6, cfg.eos_idx],
            ]
        )
        with self.assertRaises(ValueError):
            beam_search_decode(model, src, beam_size=2, max_len=5, config=cfg)

    def test_beam_search_decode_prefers_higher_log_prob_paths(self):
        cfg = get_default_config()

        class SkewedTransformer(DummyTransformer):
            def decode(
                self,
                tgt,
                memory,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None,
                tgt_mask=None,
            ):
                batch, tgt_len = tgt.shape
                logits = torch.full(
                    (batch, tgt_len, self.vocab_size), -10.0, device=tgt.device
                )
                logits[:, -1, 4] = 6.0
                logits[:, -1, self.eos_idx] = 5.0
                return logits

        model = SkewedTransformer(eos_idx=cfg.eos_idx)
        src = torch.tensor([[cfg.sos_idx, 5, cfg.eos_idx]])
        tokens = beam_search_decode(model, src, beam_size=2, max_len=4, config=cfg)
        self.assertIn(4, tokens)


class TranslateBatchTests(unittest.TestCase):
    def test_translate_batch_greedy_simple_vocab_handles_many_sentences(self):
        cfg = get_default_config()
        model = DummyTransformer(eos_idx=cfg.eos_idx)
        src_vocab, tgt_vocab = _build_simple_vocabs()
        english = _english_sentences()
        spanish = _spanish_sentences()
        target_ids = [
            tgt_vocab.encode(sentence, add_special_tokens=True) for sentence in spanish
        ]

        with mock.patch(
            "torchlingo.inference.greedy_decode", return_value=target_ids
        ) as mock_greedy:
            outputs = translate_batch(
                model,
                sentences=english,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy="greedy",
                max_len=32,
                config=cfg,
            )

        mock_greedy.assert_called_once()
        self.assertEqual(len(outputs), len(spanish))
        self.assertEqual(outputs, spanish)

    def test_translate_batch_beam_sentencepiece_handles_many_sentences(self):
        cfg = get_default_config()
        model = DummyTransformer(eos_idx=cfg.eos_idx)
        english = _english_sentences()
        spanish = _spanish_sentences()
        src_vocab, tgt_vocab = _load_sentencepiece_vocabs(config=cfg)
        beam_outputs = [
            tgt_vocab.encode(sentence, add_special_tokens=True) for sentence in spanish
        ]

        with mock.patch(
            "torchlingo.inference.beam_search_decode", side_effect=beam_outputs
        ) as mock_beam:
            outputs = translate_batch(
                model,
                sentences=english,
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                decode_strategy="beam",
                beam_size=4,
                max_len=32,
                config=cfg,
            )

        self.assertEqual(outputs, spanish)
        self.assertEqual(mock_beam.call_count, len(spanish))


if __name__ == "__main__":
    unittest.main()
