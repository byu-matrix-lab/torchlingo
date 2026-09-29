# Lecture 4 Assignment - Initial steps in data cleaning pipeline

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Mon Sep 21, 10:00. Points: 20. Roadmap v6 due date: Mon Sep 21. Status: current. Rubric tables reformatted; otherwise verbatim.*

Download your language data from the shared folder (link on the Learning Suite page).

- Carefully examine your data, extract it from the TMX files using a Python script and make sure that the source and target sentences are all aligned properly, including removing any spurious characters (LFs, CRs, etc.) that are affecting the alignment.
- Write Python code to do at least three of the required cleaning steps and be ready to share any observations or issues you encountered in our next class.
- For medium- and high-resource languages, prepare at least 200K bilingual pairs, but you can prepare all the data if you want. For low-resource languages (anything below 200K bilingual pairs), you must prepare all the data.
- Combine all the data from multiple TMX files/sources into a single data set.

Very Important! Take plenty of time to spot-check your cleaned data thoroughly, looking for weird and/or uncaught character sequences and problems.

Upload your extracted data as two sentence-aligned text files (English in one, your language in the other) to your subfolder in the shared folder. The uploaded files must be free of any spurious characters and correctly sentence-aligned, but they do not yet need to be completely cleaned. Make sure your last name and language are part of the filenames.

- Upload to Learning Suite or to the same folder a description about which cleaning steps you attempted and if you were successful or what issues you encountered.

Of course, you can do Assignment 5, too, which is doing all the cleaning steps, and that would be great. But you must do at least the above 6 things by our next class so that we can discuss the issues you encountered and share/demonstrate how to resolve them.

IMPORTANT REMINDERS:

- Be sure to upload descriptions of what you did in a separate document to LS
- Very Important! Take plenty of time to spot-check your cleaned data thoroughly, looking for weird and/or uncaught character sequences and problems.
- Always open your cleaned files in Notepad++ (or some other efficient text editor) and do random searches for characters like “<“ or “&”, spot-check for alignment throughout the file, and just look for any kind of garbage.
- Look for and use code to resolve inconsistent quote marks, escaped characters (&quot; &nbsp; &amp; …), etc.
- In the TMX files, you must check for and remove any extra CR’s (\r), LF’s (\n), or hex representation of characters (e.g., “&#xd”) in the text strings BEFORE you do the other cleaning steps. This is #1 in the list of cleaning steps. These will cause the number of source and target sentences not to match and the sentences will not be aligned (#15 in the list of cleaning steps).

Here is the full list of cleaning steps you must perform. Refer to GILT Forum - TM Mgmt and Best Practices for details.

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

| RUBRIC | Points |
|---|---|
| Completed cleaning step 1 | 2 |
| Completed cleaning step 2 | 2 |
| Completed cleaning step 3 | 2 |
| Files aligned | 4 |
| 200K (or all) bilingual pairs prepared | 4 |
| Uploaded as two properly named sentence-aligned text files | 2 |
| Description of cleaning steps | 4 |
| Total | 20 |
