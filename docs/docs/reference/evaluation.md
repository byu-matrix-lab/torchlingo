# Evaluation

Translation-quality metrics, all wrapping [sacreBLEU](https://github.com/mjpost/sacrebleu).

For what these metrics *mean* and when each one misleads, read
[Evaluation](../concepts/evaluation.md) first. This page is the API.

| Function | Metric | Direction | Reach for it when |
| --- | --- | --- | --- |
| [`compute_bleu`](#torchlingo.evaluation.compute_bleu) | BLEU | higher is better | Reporting a headline number others will compare against |
| [`compute_chrf`](#torchlingo.evaluation.compute_chrf) | chrF | higher is better | Short text, or a morphologically rich target language |
| [`compute_ter`](#torchlingo.evaluation.compute_ter) | TER | **lower is better** | You care about edit effort — post-editing, for instance |

## Always report the signature

`compute_bleu` attaches a `.signature` to its result:

```python
result = compute_bleu(predictions, references)
print(result.score, result.signature)
# 35.36  nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
```

BLEU is sensitive enough to tokenization that the same translations can score
several points apart under different settings, so a bare number is not
comparable to anyone else's. The signature is what makes it reproducible.

## References: the shape that catches everyone

Both of these are accepted:

```python
compute_bleu(predictions, ["ref one", "ref two"])                    # one reference each
compute_bleu(predictions, [["ref 1a", "ref 1b"], ["ref 2a", "ref 2b"]])  # several each
```

Internally they are transposed into the *reference streams* sacreBLEU expects —
one list per reference slot, each running the length of the corpus — rather than
one list per sentence.

!!! warning "Why this is called out"
    Passing the per-sentence shape straight to sacreBLEU does not raise. It
    reads N sentences as N separate reference streams holding one sentence
    each, scores against that, and returns a plausible-looking number.

    `compute_chrf` and `compute_ter` did exactly that until 2026-09. On a
    three-sentence corpus chrF read 54.85 where the truth was 67.91, and TER
    read 50.00 where the truth was 25.00. Nothing caught it because nothing
    used them — neither function appeared in a test, a tutorial or a docs page.

    The three now share one helper, so they cannot disagree about what a
    reference list means, and `tests/test_metric_reference_shape.py` checks each
    against sacreBLEU called directly.

    If you call sacreBLEU yourself, transpose. If a corpus score looks oddly
    close to what you would get scoring each sentence alone, this is why.

## Mixing metrics

Nothing stops you scoring one set of translations three ways, and
[the concepts page](../concepts/evaluation.md) shows why you should. Two
cautions:

- **Direction differs.** TER is an error rate. A table with BLEU and TER columns
  needs to say so, or it will be misread.
- **Magnitudes are not comparable.** chrF routinely lands tens of points above
  BLEU on identical output. That is the metrics disagreeing about scale, not one
  of them being wrong.

## API Reference

::: torchlingo.evaluation
    options:
      show_source: true
      members:
        - compute_bleu
        - compute_chrf
        - compute_ter
        - evaluate_model
