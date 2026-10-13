## Which notebook serves which lecture

Two notebook families, numbered differently on purpose:

| family | numbered by | who writes it |
|---|---|---|
| `docs/docs/tutorials/` | library topic | the repository session |
| `docs/docs/course/` | lecture | both sessions, together; the repository session commits |

So which notebook serves which lecture is not derivable from a filename. Each notebook
**declares it** in its own `torchlingo` metadata, and this table is generated from those.
Dates and deadlines are in the schedule above; lecture numbers are the join.

The table below is a build artifact. **Do not edit it**: change the notebook's metadata and
run `python scripts/notebook_meta.py --write`. CI fails if the two disagree.
