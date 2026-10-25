"""Meta's NLLB-200 as a translator, decoded the ways TorchLingo decodes.

An accessory, not part of TorchLingo: see :mod:`torchlingo_harness`.

NLLB-200 (No Language Left Behind) is one model that translates between any two of 200
languages. Two course notebooks use it as a point of comparison or a shortcut: A12 sets it
beside an LLM prompted with examples, and A13 may back-translate with it instead of training a
reverse model, where it covers the student's language. Its language list is FLORES-200's, with
codes such as ``eng_Latn`` and ``tel_Telu``.

:func:`load_nllb` returns a translator with the same four decoding strategies as
:func:`torchlingo.inference.translate_batch`, so a notebook can switch between its own model and
NLLB without changing how it asks for greedy, beam, sampled or noised-beam output. Pair it with
:func:`torchlingo.inference.translate_lines_to_file` for a decode that survives a dropped
session.

:func:`fine_tune` trains NLLB further on a student's pairs, both directions at once if the rows
say so, with :func:`coded_batch` building what it trains on: A14 fine-tunes NLLB that way, and
Tutorial 9 writes the same loop out step by step.

It needs Hugging Face ``transformers``, which Colab ships and TorchLingo does not require; it is
imported only when :func:`load_nllb` is called.

Example:
    Skipped under ``--doctest-modules``: it downloads a 2.4 GB model.

    >>> from torchlingo_harness.nllb import load_nllb                  # doctest: +SKIP
    >>> to_english = load_nllb("tel_Telu", "eng_Latn")                  # doctest: +SKIP
    >>> to_english(["మీరు ఎలా ఉన్నారు?"], decode_strategy="sample", seed=0)  # doctest: +SKIP
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from torchlingo.inference import DECODE_STRATEGIES, add_noise

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
FLORES_200_LIST = (
    "https://github.com/facebookresearch/flores/blob/main/flores200/README.md"
)


class NLLBTranslator:
    """A loaded NLLB model and tokenizer, translating from one language into another.

    Build one with :func:`load_nllb`. Calling it translates a list of sentences.

    Args:
        model: A Hugging Face sequence-to-sequence model with ``generate``.
        tokenizer: Its tokenizer, with ``src_lang`` already set.
        tgt_lang: The FLORES-200 code to translate into.
        device: Where the model is.
        name: The model's name, recorded by :meth:`settings`.
    """

    def __init__(
        self, model, tokenizer, tgt_lang: str, device: torch.device, name: str
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tgt_lang = tgt_lang
        self.device = device
        self.name = name
        self.target_id = tokenizer.convert_tokens_to_ids(tgt_lang)

    def settings(self, **decoding) -> dict:
        """What decides this translator's output, for a resumable job's record.

        Args:
            **decoding: The keyword arguments the job passes to the translator.

        Returns:
            The model, both languages and the decoding arguments, JSON-serializable.
        """
        return {
            "translator": self.name,
            "src_lang": self.tokenizer.src_lang,
            "tgt_lang": self.tgt_lang,
            **decoding,
        }

    def __call__(
        self,
        sentences: Sequence[str],
        decode_strategy: str = "greedy",
        beam_size: int = 5,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | Sequence[int] = 0,
        batch_size: int = 16,
        max_length: int = 200,
    ) -> list[str]:
        """Translate sentences, decoded as ``translate_batch`` decodes.

        Args:
            sentences: Sentences in the source language.
            decode_strategy: One of :data:`torchlingo.inference.DECODE_STRATEGIES`.
            beam_size: Beam width for "beam" and "beam+noise".
            temperature: For "sample".
            top_k: For "sample"; None samples from the full vocabulary. NLLB's vocabulary
                covers 200 languages and their scripts, and a draw from all of it at
                temperature 1 often lands in another script altogether; ``top_k=10``, which
                Edunov et al. (2018) also report, keeps the variety and loses the strays.
            seed: For "sample" and "beam+noise"; an int, or one seed per sentence.
            batch_size: Sentences per call to the model.
            max_length: The longest translation, in NLLB's tokens. Passed as ``max_length``,
                which NLLB's own generation settings already set, so as not to warn on every
                batch, as ``max_new_tokens`` would.

        Returns:
            One translation per sentence.

        Note:
            "beam+noise" applies :func:`torchlingo.inference.add_noise` with each sentence's
            own seed (``seed + i`` for sentence ``i``, given an int). "sample" cannot be seeded
            per sentence: Hugging Face draws from one random stream for a whole batch, so a
            sampled sentence depends on the batch it is in. It is seeded by its batch's first
            sentence, so the same call with the same ``batch_size`` gives the same output.

        Raises:
            ValueError: If ``decode_strategy`` is not one TorchLingo has.
        """
        if decode_strategy not in DECODE_STRATEGIES:
            raise ValueError(
                f"Unknown decode_strategy {decode_strategy!r}; choose one of "
                f"{', '.join(repr(s) for s in DECODE_STRATEGIES)}."
            )
        if isinstance(seed, int):
            seeds = [seed + i for i in range(len(sentences))]
        else:
            seeds = [int(s) for s in seed]
        out: list[str] = []
        for start in range(0, len(sentences), batch_size):
            batch = self.tokenizer(
                list(sentences[start : start + batch_size]),
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            options = {
                "forced_bos_token_id": self.target_id,
                "max_length": max_length,
            }
            if decode_strategy == "sample":
                torch.manual_seed(seeds[start])
                # top_k=0 turns off Hugging Face's default of sampling among the top 50 only.
                options.update(
                    do_sample=True, temperature=temperature, top_k=top_k or 0, top_p=1.0
                )
            else:
                beams = beam_size if decode_strategy in ("beam", "beam+noise") else 1
                options.update(do_sample=False, num_beams=beams)
            with torch.no_grad():
                ids = self.model.generate(**batch, **options)
            out += self.tokenizer.batch_decode(ids, skip_special_tokens=True)
        if decode_strategy == "beam+noise":
            return [add_noise(text, seed=s) for text, s in zip(out, seeds)]
        return out


def load_nllb(
    src_lang: str,
    tgt_lang: str,
    model_name: str = NLLB_MODEL,
    device: torch.device | None = None,
) -> NLLBTranslator:
    """Load NLLB-200 to translate from ``src_lang`` into ``tgt_lang``.

    Args:
        src_lang: FLORES-200 code of the source language, e.g. ``"tel_Telu"``.
        tgt_lang: FLORES-200 code of the target language, e.g. ``"eng_Latn"``.
        model_name: Which NLLB model. The default, the 600M-parameter distilled one, fits a
            Colab T4 beside a TorchLingo model.
        device: Where to run it. Defaults to CUDA if available, then Apple's GPU, then CPU.

    Returns:
        A translator: call it on a list of sentences.

    Raises:
        ValueError: If either code is not one of NLLB's languages.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    for code in (src_lang, tgt_lang):
        if tokenizer.convert_tokens_to_ids(code) == tokenizer.unk_token_id:
            raise ValueError(
                f"{code!r} is not one of NLLB's languages. Codes look like 'tel_Telu': "
                f"find yours in the FLORES-200 list, {FLORES_200_LIST}. If your language "
                "is not there, NLLB does not cover it."
            )
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, dtype=dtype)
    return NLLBTranslator(
        model.to(device).eval(), tokenizer, tgt_lang, device, model_name
    )


def coded_batch(tokenizer, frame, device: torch.device, max_pieces: int = 126) -> dict:
    """Tokenize pairs whose rows carry their own language codes, as NLLB trains on them.

    Each source becomes its code, its pieces and ``</s>``; each target the same with its own
    code. The codes come from the rows, not from the tokenizer's ``src_lang`` and
    ``tgt_lang``, so one batch can hold both directions of a bidirectional training set.

    Args:
        tokenizer: NLLB's tokenizer.
        frame: A DataFrame with columns ``src``, ``tgt``, ``src_code`` and ``tgt_code``.
        device: Where the tensors go.
        max_pieces: The longest sentence kept, in pieces, before the code and ``</s>``.

    Returns:
        ``input_ids``, ``attention_mask`` and ``labels``, padded; padding in the labels is
        ``-100``, which the model's loss skips.
    """

    def coded(texts, codes):
        return [
            [tokenizer.convert_tokens_to_ids(code)]
            + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))[:max_pieces]
            + [tokenizer.eos_token_id]
            for text, code in zip(texts, codes)
        ]

    def padded(rows, fill):
        width = max(map(len, rows))
        return torch.tensor(
            [r + [fill] * (width - len(r)) for r in rows], device=device
        )

    sources = coded(frame.src, frame.src_code)
    targets = coded(frame.tgt, frame.tgt_code)
    input_ids = padded(sources, tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
        "labels": padded(targets, -100),
    }


def fine_tune(
    model,
    tokenizer,
    train,
    val,
    checkpointer,
    *,
    steps: int,
    device: torch.device,
    batch_size: int = 16,
    accumulate: int = 2,
    learning_rate: float = 1e-4,
    warmup: int = 100,
    eval_every: int = 500,
    val_pairs: int = 500,
    seed: int = 479,
    space_needed_gb: float = 8.0,
):
    """Fine-tune NLLB on pairs with their language codes, resumably; keep the best checkpoint.

    The loop Tutorial 9 writes out, for notebooks whose lesson is elsewhere: Adafactor, whose
    small state keeps a checkpoint near the size of the weights (Adam's made each one about
    7 GB for NLLB's 600M parameters); gradient accumulation; bfloat16 arithmetic where the GPU
    has it; and a :class:`~torchlingo.training_checkpoint.TrainingCheckpointer`, so a dropped
    session resumes, and a finished run is not repeated. The data order is seeded per pass
    over the training set, so a resumed run sees the batches an uninterrupted one would.

    Args:
        model: NLLB, in float32, on ``device``. Left holding the best checkpoint's weights.
        tokenizer: NLLB's tokenizer.
        train: Training pairs, columns ``src``, ``tgt``, ``src_code`` and ``tgt_code``.
        val: Validation pairs, the same columns; the first ``val_pairs`` are used.
        checkpointer: Where the run is saved, and resumed from.
        steps: Optimizer steps in all, counting any already taken.
        device: Where the model is.
        batch_size: Pairs per forward pass.
        accumulate: Batches per optimizer step.
        learning_rate: Adafactor's, fixed after a linear warmup.
        warmup: Steps of warmup.
        eval_every: Steps between validation checks, each of which saves.
        val_pairs: How many validation pairs a check uses.
        seed: Seeds the model's dropout and the data order.
        space_needed_gb: Stop before training if the checkpoint folder has less free space,
            since a save that fails an hour in loses the hour. A third of it when resuming.

    Returns:
        The checkpointer's state: steps taken, validation losses, the best of them.

    Raises:
        SystemExit: If the checkpoint folder has too little free space.
    """
    import contextlib
    import shutil
    import time

    from transformers.optimization import Adafactor

    per_pass = max(1, len(train) // batch_size)
    orders: dict[int, object] = {}

    def micro_batch(k):
        n = k // per_pass
        if n not in orders:
            orders.clear()
            orders[n] = train.sample(frac=1, random_state=seed + n)
        start = (k % per_pass) * batch_size
        return coded_batch(tokenizer, orders[n][start : start + batch_size], device)

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def amp():
        if bf16:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def validation_loss():
        model.eval()
        held_out, total = val[:val_pairs], 0.0
        with torch.no_grad(), amp():
            for start in range(0, len(held_out), batch_size):
                chunk = held_out[start : start + batch_size]
                total += model(
                    **coded_batch(tokenizer, chunk, device)
                ).loss.item() * len(chunk)
        model.train()
        return total / len(held_out)

    torch.manual_seed(seed)
    optimizer = Adafactor(
        model.parameters(),
        lr=learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    resuming = checkpointer.has_checkpoint()
    needed = space_needed_gb / 3 if resuming else space_needed_gb
    folder = checkpointer.path_for("latest").parent
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free / 1e9
    if free < needed:
        raise SystemExit(
            f"{folder} has {free:.1f} GB free and the checkpoints need about {needed:.0f} GB. "
            "Free some space (Drive's trash counts until emptied), then run this again."
        )
    if resuming:
        checkpointer.load(model, optimizer, scheduler, map_location=device)
    state = checkpointer.state

    model.train()
    started, first_step, losses = time.monotonic(), state.global_step, []
    while state.global_step < steps:
        step_loss = 0.0
        for a in range(accumulate):
            with amp():
                batch = micro_batch(state.global_step * accumulate + a)
                loss = model(**batch).loss / accumulate
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        state.global_step += 1
        losses.append(step_loss)
        if state.global_step - first_step == 20:
            per_step = (time.monotonic() - started) / 20
            left = (steps - state.global_step) * per_step / 60
            print(
                f"{per_step:.2f} s a step: about {left:.0f} more minutes for {steps:,} steps."
            )
        if state.global_step % eval_every == 0 or state.global_step == steps:
            val_loss = validation_loss()
            best = val_loss < state.best_val_loss
            state.val_losses.append(val_loss)
            state.best_val_loss = min(state.best_val_loss, val_loss)
            print(
                f"step {state.global_step:6,}   training loss {sum(losses) / len(losses):.3f}"
                f"   validation loss {val_loss:.3f}{'   best so far' if best else ''}"
            )
            losses = []
            checkpointer.save(model, optimizer, scheduler, is_best=best)
        elif checkpointer.should_save():
            checkpointer.save(model, optimizer, scheduler)

    if checkpointer.has_checkpoint("best"):
        checkpointer.load(model, which="best", map_location=device, restore_rng=False)
    model.eval()
    return state


__all__ = [
    "FLORES_200_LIST",
    "NLLB_MODEL",
    "NLLBTranslator",
    "coded_batch",
    "fine_tune",
    "load_nllb",
]
