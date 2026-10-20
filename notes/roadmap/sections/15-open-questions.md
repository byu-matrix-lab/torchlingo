## Open questions

- **Does "about 35 epochs" hold for a student with 30K pairs?** The 30-epoch convergence was
  measured at 100K pairs with a subword vocabulary. Revisit once the A5 audit says how many
  students are well below 100K; the audit itself still needs the submissions staged somewhere
  readable. The instruction's escape hatch is "keep going while validation loss falls".
- **Can students run the grader from source?** The source exists:
  `byu-matrix-lab/data-cleaning-pipeline-grader`, one `grader.py` run as
  `python3 grader.py source target output`, which would end the download-and-override-Gatekeeper
  step. Two things block pointing the Lecture 4 and 5 decks at it, both Eric's: the repository is
  private, so students cannot clone it, and it has no license. It was last pushed in September
  2024, so its checks need comparing with what Lectures 4 and 5 ask for first. Until then
  students use the four PyInstaller binaries.
- **An NLLB spotlight lecture** (Eric, Sep 27): what makes NLLB special beyond its data, namely
  the sparsely gated mixture-of-experts architecture with its regularisation against low-resource
  pairs overfitting, and the deliberate training curriculum, which has no coverage in the course.
  Home is Lecture 13 or 14. The paper's specifics are verified before any number reaches a slide.
- **Learning Suite's assignment texts.** A13 and A14 still carry last year's OpenNMT directions;
  each is rewritten against its notebook. A7's due time differs between Learning Suite and the
  deck.
- **Lecture 6 is overloaded** and nothing has been done about it.
- **Lecture 19 needs rescoping** around the material now in Lecture 5, and has no notebook.
- **What Lectures 15 to 23 and the final project need from the repository**, held until Lectures
  11 to 13 are done (Eric, Oct 19). None has a notebook, by design, but three may want code:
  Lecture 18's post-editing activity measures normalized edit distance (a small notebook, or
  TorchLingo's TER?); Lecture 19's IBM Model 1 worked by hand, set beside 8a's attention
  alignment, once the lecture is rescoped; and a "starting a project from your own A9 or A14
  model" page for the final projects, proposed Nov 16.
- **Do lectures after 12 have students prompt an LLM?** Eric, Oct 19: "almost certainly yes", to
  be confirmed against these scope notes. If a second notebook does, A12's prompting code becomes
  a TorchLingo helper; until then it stays in A12's notebook.
- **Deck links to personal Google Drive URLs.** Lecture 4 links the regex notebook twice and its
  own activity once; Lecture 12 links its assignment notebook. All die with the account. Each is
  now a Colab badge in the repository, but the Drive copies must stay alive while students are in
  them.
