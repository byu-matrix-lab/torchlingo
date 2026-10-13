## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Does the grader survive?** `grader.exe` is four unsigned PyInstaller binaries in a PhD
  student's OneDrive with no source, no repository, no license. Ammon has been asked for the
  source. Fallback: the checks are documented in Lecture 4 and belong in `torchlingo.diagnostics`.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The paper's specifics are verified before any number reaches a slide.
- **Learning Suite's assignment texts.** A13 and A14 still carry last year's OpenNMT directions;
  each is rewritten against its notebook. A7's due time differs between Learning Suite and the
  deck.
- **Where does A13's monolingual text come from?** A8 used all of each student's cleaned data, so
  the A13 notebook asks students for new sentences in their language. A course-provided source
  per language would remove that step.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and has no notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Each is
  now a Colab badge in the repository, but the Drive copies must stay alive while students are in
  them.
