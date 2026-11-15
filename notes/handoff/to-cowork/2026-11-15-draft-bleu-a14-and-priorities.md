# BLEU now means what A9 meant, A14 resumes on a GPU, and the course list takes P1 to P4

*From the repository session, written Sun Nov 15. Draft: not yet handed off. The baton goes to
Cowork with this file. Three things are asked of you, the first a repeat:*

1. *post the line you wrote for A14's page (asked Nov 9, section 1);*
2. *give every open item in the course task list a priority, P1 to P4 (section 2);*
3. *optionally, one line in the Final Project directions about BLEU (section 4).*

## 1. Repeat: post A14's page line

**Asked on Nov 9, with no reply since**, so it is raised again here. A14's notebook prints a
timing table for the write-up, and it has been on `course` since Nov 9. Your sentence for the
page, under **What to submit**, can go up as you wrote it. If the page already asks students to
note start and end times by hand, you and Eric choose which to keep. A student whose notebook
prints no table opened it before Nov 9; they can open the badge again, since every step resumes
from Drive, or note the times by hand.

## 2. Use P1 to P4 in the course task list

Eric, Nov 15: the course-side list, `torchlingo-private/notes/course/TASKS.md`, should rank its
items with the same priority levels the repository's `notes/TASKS.md` uses. Today it has none, so
neither side can tell at a glance what is urgent on the course side.

The levels are defined at the top of `notes/TASKS.md`, under **Priorities**, and they answer one
question: *who meets it, and when?* In short:

- **P1: a student, within about three weeks.** Something on a student's path in that window is
  broken, missing, or unverified as students will take it; or a dated promise falls inside it.
- **P2: a student, later this term, or the guards on the path**, including anything Eric asked
  for with no date.
- **P3: only the maintainers, at no set time.** If it were never done, would anyone but us notice?
- **P4: a student, the next offering.** It changes the path, but only for a later term.

A student's **path** is what they run, read for points, or are graded on: the assignment
notebooks and pages, what `course` serves, and the Final Project. Please read the full definitions
there rather than this summary, and keep them in one place: the course file can point to that
section instead of copying it.

What is asked:

- give every open item a priority, and sort the list by it, earliest date first within a level;
- in your reply, list any P1 items, so this side sees what is urgent on yours without reading
  the whole file.

## 3. Two fixes students have already received

Both are on `course`; a student gets them by deleting the runtime (Runtime → Disconnect and
delete runtime) and running the notebook again from the badge.

- **A14's Step 4 failed when it resumed on a GPU** (a student reported it on Discord, Nov 13).
  Resuming raised `TypeError: RNG state must be a torch.ByteTensor`. It is fixed on `course`.
  The checkpoint on Drive was never touched, so running Step 4 again picks up where it stopped.
  Eric sent the students a message the day it was fixed. A14's page and notebook need no
  change.
- **BLEU in A11 and A12 is now the BLEU A9 reported.** For a language written without spaces,
  A9 scores spBLEU, but the library function A11 and A12 call quietly switched to BLEU over
  single characters whenever it saw one such character, and ignored the tokenizer it was given.
  So A11's question 5, *"Do COMET and COMETKiwi agree with BLEU (as Assignment 9 reported it)"*,
  was comparing two different metrics for those students. Since Nov 14 both notebooks score
  BLEU the way A9 does, and the page's wording holds as written. **For grading:** a student
  whose language is written without spaces and who ran A11 or A12 before Nov 14 has BLEU
  numbers that differ from A9's for that reason alone, not from anything they did. Every other
  student's numbers are unchanged.

## 4. Optional: one line for the Final Project directions

The directions ask for "BLEU or chrF and COMET". A student who scores BLEU on a language written
without spaces, in their own code, should use spBLEU, as A9 did. If you want the directions to
say so, a line such as:

> *For a language written without spaces, score BLEU the way A9 did:
> `compute_bleu(hyps, refs, tokenize=bleu_tokenizer(written_without_spaces(refs)))`.*

Two other changes reach a Final Project, and need nothing from you:

- **A vocabulary built some other way** than the library builds them, with special-token ids
  that differ from the model's settings, used to translate every sentence to its maximum length
  without an error. Decoding now stops with a message naming the ids that differ.
- **The docs site has reference pages for training and for decoding**, which it lacked:
  byu-matrix-lab.github.io/torchlingo, under API Reference → Training and Decoding.

## 5. What moved on this side's task list

- **An architecture review of the library, done with the course first in mind,** became a dozen
  tasks. The ones that would change what a student's own code meets are held for the term
  boundary, so no Final Project breaks partway through.
- **Done since Nov 9:** the BLEU fix and the A14 resume fix above; the special-token check;
  decoding an LSTM with the fast beam search; the reference pages; a docs check that every
  import in the docs exists; and CI now says which notebooks it did not run.
- **Closed:** the check of A14's setup cells on real Colab, from the badge, which passed; and
  batching beam search across sentences, which stays out of scope.
- **Released:** TorchLingo 0.2.11 on PyPI carries all of the above for anyone outside the
  course. Students install from `course` and already have it.
