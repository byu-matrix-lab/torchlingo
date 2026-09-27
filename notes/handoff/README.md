# The hand-off channel

Two Claude sessions work on CS 479 and cannot message each other: this one, in the
repository, and a Cowork session that owns the course decks. This directory is the channel.

**One file per hand-off.** Eric, 2026-09-27: *"we should start creating separate files for
each new hand-off, and we should archive the old ones."*

## Layout

```
handoff/
  README.md            this file: the protocol
  briefing.md          standing context; read once, not every time
  to-cowork/           messages out, one file per hand-off
  from-cowork/         messages in, one file per hand-off
  archive/             the append-only logs this replaced
```

## Naming

```
YYYY-MM-DD-slug.md            2026-09-27-post-norm-and-the-split.md
YYYY-MM-DD-b-slug.md          a second hand-off the same day
```

ISO dates first, so the directory listing *is* the index in chronological order and nothing
has to maintain one. A second hand-off on the same date takes `-b`, a third `-c`. The slug
says what the hand-off is *about*, not that it is a hand-off — the directory already says
that.

## Rules

**A file is finished when it is written.** Do not edit a hand-off after the fact; if
something in it turns out to be wrong, say so in the next one. This is why the files are
small and dated: a correction is cheap when it is a new file, and dishonest when it is a
silent edit.

*One exception has been used, and it is recorded so the bar stays visible:* on 2026-09-26 a
past entry's date was corrected, because leaving two entries dated out of order was worse
than the error. The correction was announced in the next entry rather than made quietly.

**Reconcile `notes/TASKS.md` in the same sitting as any hand-off, in either direction.** A
hand-off is precisely when the lists go stale and the only moment both sides know what
changed. `CLAUDE.md` has the argument and the evidence for it.

**The Cowork session does not run git.** Changes arrive as requests in `from-cowork/`, and
committing, nav entries and pull requests happen here.

## Why this replaced two append-only logs

`to-cowork.md` reached twelve entries and `from-cowork.md` ten, and on 2026-09-27 alone the
incoming log grew by twelve — 1,657 lines in one file. Three things broke down at that size:

- **Finding one hand-off meant scrolling past all of them.** The newest-first ordering
  helped a reader and not a search.
- **Appending to the top of a long file is a conflict-prone edit** in the one document two
  sessions both write to.
- **Nothing could reference a single hand-off.** A task entry could cite the file but not
  the message, so "see the hand-off" meant "read all of it".

The archived files are kept whole rather than split into per-entry files. Splitting
twenty-two entries by hand would have risked the record to tidy it, and the record is the
thing worth protecting.
