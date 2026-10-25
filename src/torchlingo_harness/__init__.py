"""Accessories that ship with TorchLingo but are not part of it.

TorchLingo is the library CS 479 learns from: small, readable, and about the models students
build. Some course notebooks also need to call someone else's model, as a comparison or a
shortcut. That code lives here, outside the ``torchlingo`` package, so the library stays lean:
``import torchlingo`` never loads it, and nothing in ``torchlingo`` depends on it.

Modules:
    nllb: Meta's NLLB-200, decoded the ways TorchLingo decodes, for A12 and A13, and
        fine-tuned on a student's pairs, both directions at once, for A14.
"""
