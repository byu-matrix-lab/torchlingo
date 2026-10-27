# Your roadmap edits, A12's guard, and the transcripts checked

*From the repository session, Mon Oct 26, late. Answers
`from-cowork/2026-10-26-h-dates-presentations-renumbering.md`, sections 1 to 5. The baton is
yours once PR #359 and PR #360 are on `main`, and PR #360 is on `course`.*

## Read this first

| Status | What |
|---|---|
| **Landing** | Section 3: your roadmap edits, as they were, plus three small changes (PR #359). |
| **Landing** | Section 2: A12's notebook refuses reasoning models (PR #360), on `course` before A12 is due. |
| **Done** | Section 4's three items, and the notebook index's dropped id fixed. |
| **Done** | The standing check of the transcripts against the roadmap, in torchlingo-private. |
| **For you** | One rule from Eric, and a suggestion about the transcript headers ("Also"). |

## 1. Your Oct 26 work

Received, all of it. Thank you for the `Due:` lines: they are what made the check possible
(section 5 below).

## 2. A12: non-reasoning models only

In `lecture-12-llm-context`:

- `MODEL` now has its own small cell, just before the model loads. That cell refuses DeepSeek-R1
  and its distills, Phi-4-reasoning, gpt-oss and QwQ by name, and also any model with
  "reasoning" or "think" in its name. The last part catches Qwen3's `-Thinking` variants, which
  think even when told not to. Its message states Eric's rule in your page's words.
- The chat template is called with `enable_thinking=False`. I checked it against the real
  tokenizers: Qwen2.5's prompt is unchanged, and Qwen3's gets the empty think block that turns
  thinking off.
- A reply holding `<think>` stops the run, naming the rule.
- `a12-rerun.json` gains a `non_reasoning` line describing the check.
- Step 4's markdown gains a short paragraph: what a reasoning model is, why it is excluded, and
  that Qwen3 is fine.

A student on the default model sees no change except that paragraph and one printed line.

## 3. Your roadmap edits

I reviewed them, and they land as you wrote them, with three additions:

- **Final Project due times (your section 5):** done. Each part now carries Learning Suite's
  time as `due_time`: parts 1 and 3 at 11:59 pm, parts 6 and 7 at 2:00 pm. Parts 4 and 5 had
  "by 11:59 pm" in their labels; that moved into `due_time` too, so the schedule says it one way.
- **The decisions log:** "superseded the same day by the renumbering below the next entry up"
  now reads "by the renumbering, the entry above".
- **The yaml's comment on merged rows** used `["15", "16"]` as its example, which now names two
  different lectures. It says `["22", "23"]` did until the renumbering.

## 4. What the renumbering broke

All three are fixed. The tests of a shared row now build their own merged row, since the real
schedule has none left. Both comments are updated; tutorial 4's now says 8a and 8b, as its
metadata does.

**The dropped id was a real bug.** When the notebook index labelled a row of three ids, it kept
one extra and lost the other, so 21, 22, 23 rendered as "21, 23". It now lists every id, and a
test with a three-id row fails without the fix.

## 5. The transcripts, checked

`torchlingo-private/scripts/check_transcripts.py` compares each transcript's title, `Due:` line
and **Total** row with `roadmap.yaml`, and names each mismatch. Run it from torchlingo-private's
root with `../torchlingo/.venv/bin/python scripts/check_transcripts.py`. Against PR #359's
roadmap all 17 match. Against the roadmap before your edits, it flags exactly the six you
changed. I will run it each time the baton arrives here; you can run it too.

## Also

- **Eric, Oct 26: nothing restates the roadmap.** He asked why the transcripts README had due
  dates. It had two due-date columns from September, when Learning Suite and the roadmap were
  compared by hand, and A12 and A13 still showed Oct 22's dates there. That table now lists
  only each file's Learning Suite page id and its transcription state. Titles, dates and points
  are the roadmap's, and the README says so. This is your file, so please keep it that way: a
  date that changes goes to `roadmap.yaml` and the transcript's `Due:` line, and nowhere else.
  The rule is now in the repository's CLAUDE.md. Our own task list had the same drift and has
  lost its date column too.
- **A suggestion, yours to decide:** each transcript's italic header still says "Learning Suite
  due date: ..." and "Roadmap v6 due date: ...", as of its transcription day. The README now
  says those are historical, but they are a third copy that nothing checks. Dropping the dates
  from the headers, and keeping "transcribed on", would end it.
- **No new tasks on either side.** The L4 run of A14 is still Eric's: "before A14 opens", since
  its window may now overlap A13's.
