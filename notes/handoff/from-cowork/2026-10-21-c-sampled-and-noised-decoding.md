# Sampled and noised-beam decoding in TorchLingo, for A13's back-translation

*From the Cowork session, Wed Oct 21 2026, about 14:10. Eric asked for it in chat: "Let's add
sampled and noised output to torchlingo. Then it can be an option in A13." Context:
`Roadmap/Assignment Arc Plan_v2.md` (course folder), the A13 section.*

## Why

Edunov, Ott, Auli and Grangier (EMNLP 2018), "Understanding Back-Translation at Scale"
(arXiv 1808.09381): back-translations made by **sampling**, or by **beam search with noise added
to its output**, train a better forward model than beam or greedy back-translations, "in all but
resource poor settings". Many CS 479 languages are resource-poor, so the comparison is a real
question for a student, not a foregone conclusion: A13 can let a student generate the synthetic
sources one way or another and measure which helps their language.

## What to add (your design; this is what the assignment needs)

Today `translate_batch` and `translate_to_file` take `decode_strategy` "greedy" or "beam"
(`inference.py`, `beam_search_decode`, `config.beam_size`).

1. **`decode_strategy="sample"`**: ancestral sampling from the model's full distribution at each
   step, with a `temperature` (default 1.0) and, if you think it worth it, optional `top_k`
   (Edunov et al. also report restricted sampling, top-10). It needs a `seed`, so a run is
   reproducible, and the seed goes in `translate_to_file`'s fingerprint, as `beam_size` does now.
2. **`decode_strategy="beam+noise"`** (or `noise=True` with beam): beam search, then the paper's
   noise on each output: delete words with probability 0.1, replace words with a filler token
   with probability 0.1, and swap words no further than three positions apart (a random
   permutation within a window of 3). Seeded the same way. Noise applies to the synthetic
   *source* side, which is what back-translation produces.
3. **Batching and speed** comparable with greedy: 100K sentences of back-translation is the A13
   case. If sampling must run unbatched, then say so in the docstring and the notebook.
4. **Config and docs:** the new strategy names in `Config` validation, the docs page that lists
   decoding strategies, and tutorial 08 (inference and beam search) if a short section fits.
5. **Tests:** sampling with a fixed seed is deterministic; different seeds differ; temperature
   near 0 matches greedy; noise rates land near 0.1 on a long input; the fingerprint records the
   strategy and the seed.

## How A13 will use it (for the rewrite you are already doing)

- **Back-translation:** the student picks greedy, beam, sample or beam+noise for the synthetic
  sources, says why, and (optional, or required if the compute allows) compares two of them.
- **Scoring:** both systems, the baseline and the one with back-translated data, decoded on
  the A8/A9 test set with **greedy and with beam (k = 5)**, so the student separates what the
  data bought from what the decoder bought (Lecture 10: a comparison must pin decoding). The
  sacreBLEU paired bootstrap on the comparison.
- The A13 rewrite's other items are in the arc plan: the monolingual data named, NLLB allowed for
  back-translation, the Path A frame, rater agreement on the reverse model's English output, and
  the notebook submitted.

## Also from today, for A14 (Eric: "good idea")

A14 scores both directions with **one stated decoder** (greedy, or beam with its width), the same
for the bidirectional model and for the one-way models it is compared with.
