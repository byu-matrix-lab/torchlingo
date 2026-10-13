## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Has Coulson's real-corpus Colab run of the A8 kickoff happened?** Repository task #152. It
  is the only end-to-end test before 8a; CI cannot mount Drive or draw a GPU.
- **Does the grader survive?** `grader.exe` is four unsigned PyInstaller binaries in a PhD
  student's OneDrive with no source, no repository, no license. Ammon has been asked for the
  source. Fallback: the checks are documented in Lecture 4 and belong in `torchlingo.diagnostics`.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The repository session will verify the paper's specifics before any
  number reaches a slide.
- **Learning Suite's assignment texts.** A9, A13 and A14 still carry last year's OpenNMT
  directions (A8's was replaced Sep 29; its step order needs the Sep 30 reorder); the current ones are in `torchlingo-private/notes/assignments/`. A7's due
  time differs
  between Learning Suite and the deck.
- **Lecture 9's load.** Its own subject plus an A8 debrief, diagnostics reading and a subword
  activity. The decoding block already moved out for this reason. Watch it.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and has no notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Now
  fixable as badges, but the Drive copies must stay alive while students are in them.
