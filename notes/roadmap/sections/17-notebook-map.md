## Which notebook serves which lecture

Two notebook families now exist and their
numbering schemes do not line up, deliberately:

| family | numbered by | who writes it |
|---|---|---|
| `docs/docs/tutorials/` | library topic | this repository |
| `docs/docs/course/` | lecture | Cowork writes content, this repository commits it |

So nothing about which artifact belongs in which lecture is derivable from a filename. It
used to be written down only here, by hand. It is now **declared by each notebook** in its
own `torchlingo` metadata and generated from them, because a hand-kept table next to the
artifacts it describes is the redundancy this document warns about everywhere else — and it
had already drifted twice.

Dates, topics and assignment deadlines are **in the schedule above** rather than repeated
here. Lecture numbers are the join.

The table below is a build artifact. **Do not edit it**: change the notebook's metadata and
run `python scripts/notebook_meta.py --write`. CI fails if the two disagree.
