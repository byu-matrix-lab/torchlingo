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


__all__ = ["FLORES_200_LIST", "NLLB_MODEL", "NLLBTranslator", "load_nllb"]
