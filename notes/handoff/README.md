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
  to-cowork/           messages out, one main file per hand-off
  from-cowork/         messages in, one main file per hand-off
  archive/             answered hand-offs, and the append-only logs this replaced
```

## Naming

```
YYYY-MM-DD-slug.md            2026-09-27-post-norm-and-the-split.md
YYYY-MM-DD-slug/topic.md      a sub-file of that pass, linked from it
YYYY-MM-DD-b-slug.md          a second baton pass the same day
```

ISO dates first, so the directory listing *is* the index in chronological order and nothing
has to maintain one. A second baton pass on the same date takes `-b`, a third `-c` — a new
*pass*, after the other side has replied, never a second file for the same pass. A sub-file
lives in a folder named after its main file, so the two sort together. The slug
says what the hand-off is *about*, not that it is a hand-off — the directory already says
that.

## Rules

**One file per baton pass, live until the other side picks it up.** Eric, 2026-09-28: *"the
current hand-off is live and can be updated. there should only be one file."* So while the
baton is still on this side, new findings and corrections go *into* the current file, and it
says when it was last updated. A second file for the same pass splits one message across
several, which the reader then has to reassemble, and that is how three files for one pass
accumulated on 2026-09-28 before this rule was written down.

**One main file per pass; sub-files are allowed, but they hang off it.** Eric, 2026-10-05:
*"we should have one main baton file (can have sub-files)."* A long spec, such as a notebook's
cell-by-cell changes, may go in a sub-file so it does not bury the message. The main file is
still the whole message:

- it is the file the reader opens, and every sub-file is linked from it, with one line on what
  it holds and what is wanted;
- **everything that needs an answer or a decision is in the main file**, never only in a
  sub-file;
- sub-files sit in `YYYY-MM-DD-slug/` beside it, and are live and then frozen together with it.

Sibling files with `-b`, `-c` and `-d` suffixes for one pass, the last of them an index, are
what this rule replaces: the suffix means a *new pass*, and a reader cannot tell which sibling
is the message.

**Draft under a `draft` name; rename to hand off.** Eric, 2026-10-07: while a hand-off is
being written, its file carries `draft` in its name (`2026-10-07-draft-part-b-is-on-course.md`),
and it moves to its final name only at the moment of handing off. Both sessions read this
directory, and a file sitting under a final name reads as a message sent. So the rename is the
send: until then, the other side ignores anything named `draft`. On 2026-10-07 a hand-off sat
in `to-cowork/` under its final name while this side went on to run and fix what it described.

**No roadmap edits without the baton.** Eric, 2026-10-07: `notes/CS479_COURSE_ROADMAP.md` is
edited only by the side holding the baton, and that side's hand-off says what changed. A side
without the baton holds its roadmap changes until the baton comes back. Reasons in
`notes/README.md`, under "The roadmap is the shared single source of truth".

**Course notebooks arrive as edits in the tree, not as specs.** Eric, 2026-10-07: Cowork edits
`docs/docs/course/` directly in the working tree, without git, and hands off; this side reviews
the edit against the library, runs the checks, commits, pushes and promotes `course`. Tutorials
and the library stay with this side. A cell-level spec in a hand-off file is no longer the
default; it cost a day on 2026-10-05.

**Once the other side has picked it up, the file is frozen.** Do not edit it after that; if
something in it turns out to be wrong, say so in the next one. A correction to something the
reader has already acted on is dishonest as a silent edit.

*One exception has been used, and it is recorded so the bar stays visible:* on 2026-09-26 a
past entry's date was corrected, because leaving two entries dated out of order was worse
than the error. The correction was announced in the next entry rather than made quietly.

**Reconcile `notes/TASKS.md` in the same sitting as any hand-off, in either direction.** A
hand-off is precisely when the lists go stale and the only moment both sides know what
changed. `CLAUDE.md` has the argument and the evidence for it.

**Name tasks in words, never by number, in anything written for Cowork.** Eric, 2026-10-05:
Cowork does not work from `notes/TASKS.md`, so "#166" means nothing to it; "tutorial 8's
rewrite" does. The same goes for PR numbers, since Cowork does not run git. The "which tasks
moved" part of a reply is still written, as descriptions. Before committing a `to-cowork/`
file, search it for `#` followed by a digit.

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
