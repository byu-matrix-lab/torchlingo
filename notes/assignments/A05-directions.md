# Lecture 5 Assignment - Complete data cleaning pipeline

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Wed Sep 23, 10:00. Points: 30. Roadmap v6 due date: Wed Sep 23. Status: current. Rubric tables reformatted; otherwise verbatim.*

- Complete your data-cleaning pipeline, written in Python, and use it to clean your language data
- Download the Church TM data for your language from the shared folder (link on the Learning Suite page).
- Later for your class project, you can get more public data (e.g., see the OPUS link on the Content tab - Lecture 4), but for now, just use the Church data.
- For medium- and high-resource languages, prepare at least 200K bilingual pairs, but you can prepare all the data if you want. For low-resource languages (anything below 200K bilingual pairs), you must prepare all the data. Once you have a functional cleaning pipeline, the amount of data cleaned should not matter
- Combine all the data from multiple TMX files/sources into a single data set
- You can use one of the TMX Editors, if you want, to view and experiment with the data, but not as part of your pipeline
- You may use Python TMX packages to extract the data from the TMX files or write your own TMX parser. Perform all 16 cleaning steps (step 9 with a ? is optional). Remember: the cleaner your data, the better your MT!
- For each step, provide a brief description of what you did, including any utilities or regular expressions you used or code you wrote, and upload the descriptions and code to Learning Suite or to your subfolder in the shared folder.
- Upload your cleaning pipeline code together with your cleaned data as two sentence-aligned text files (English in one, your language in the other) to your subfolder in the shared folder. Make sure your last name and language are part of the filenames.
- IMPORTANT: Run your cleaned training data through the grader and fix/modify your pipeline until you pass the grader tests. You will have points taken off for cleaning steps that do not pass.

IMPORTANT REMINDERS:

- Be sure to upload descriptions of what you did in a separate document to LS
- Take plenty of time to spot-check your cleaned data thoroughly, looking for weird and/or uncaught character sequences and problems. >>> This cannot be overemphasized! <<<
- Always open your cleaned files in Notepad++ (or whatever) and do random searches for characters like “<“ or “&”, spot-check for alignment throughout the file, and just look for any kind of garbage.
- Look for and resolve inconsistent quote marks, escaped characters (&quot; &nbsp; &amp; …), etc.
- In the TMX files, you must check for and remove any extra CR’s (\r), LF’s (\n), or hex representation of characters (e.g., “&#xd”) in the text strings BEFORE you do the other cleaning steps. This is #1 in the list of cleaning steps. These will cause the number of source and target sentences not to match and the sentences will not be aligned (#15 in the list of cleaning steps).

Here are the cleaning steps you must perform. Refer to GILT Forum - TM Mgmt and Best Practices for details.

1. Detect and fix technical issues in the content – make sure no extra CRs or LFs!
2. Remove empty segments (source or target)
3. Normalize escaped characters/entities (pay attention to the “Note:”)
4. Normalize certain control characters/Normalize whitespaces
5. Normalize quotation marks
6. Removing tags that don’t affect the meaning
7. Identify and remove duplicates with no context for MT training purposes
8. Check if a segment contains mostly non-text content
9. Characters that do not match either the expected source or target language (? – do this only if you have lists of valid characters to use in your language and English)
10. Do not remove segments where source=target (you should remove these anyhow!)
11. Check unbalanced brackets (you should probably remove these, too)
12. Remove entries consisting of only punctuation, whitespace, or tags (like #8)
13. Remove segments that are too long (>100 words)
14. Remove segments that are too short (<3 words)
15. Misalignments (manual check)
16. Check sentence length ratios and remove if the ratio exceeds your threshold

Here are some helpful hints from previous classes' experiences:

- Always do searches in your “cleaned” data to confirm that you got everything (e.g., “&[a-z]+;” to find remaining entities, or “< ” to find missed tags)
- Possibly make your cleaning “stricter” to throw out some unhelpful training data at the expense of losing some good data
- Instead of looking for all kinds of specific tags, it would be easier and more comprehensive to use a regex like “<.+?>” to find all of them without missing any
- Replace \r, \n, \t, and tags with a blank, then remove multiple blanks – order of steps is important!
- Eliminate punctuation and normalize capitalization before checking for duplicates of both source & target
- Unique source with duplicate targets can be kept
- &nbsp; (non-breaking space = x00A0) must be replaced with a space (blank = x0020)
- Before training any MT system, remove the footnote numbers at the ends of sentences: “This is a sentence with a footnote.4” A better way to do this to also get the footnotes within sentences is to remove only digits that occur in between tags when you remove the tags.
- Please make sure your data is clean according to these guidelines

| RUBRIC: | Points |
|---|---|
| Cleaning steps completed (each step is 1 point) | 16 |
| Files aligned | 3 |
| 200K (or all) bilingual pairs prepared | 2 |
| Uploaded as two properly named sentence-aligned text files | 1 |
| Description of cleaning steps and cleaning pipeline code. | 8 |
| Total | 30 |
