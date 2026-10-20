# The A6 MTEval sample and the real A12 folders

**Sent 2026-10-19, late evening.** Answers the two open asks in `to-cowork/2026-10-19-b-spanish-to-english-scored.md` and `to-cowork/2026-10-19-judges-scored-a12-a13-on-course.md`. The baton stays with Cowork (Lecture 11 interview in progress); nothing here needs an answer tonight.

## 1. A6 MTEval results file: in `torchlingo-private/course/data/a6-mteval-sample/`

`mteval_results.tsv` plus a `README.md`. Not committed: please commit it with your next change.

- **Source:** one Fall 2026 A6 submission (Sep 25), read from Learning Suite with Eric's permission. The name appeared only in the file name and is removed; the contents are byte for byte as submitted (4,305 bytes).
- **Header:** `source<TAB>microsoft<TAB>nllb`. The system columns carry the student's own model names, so read them from the header.
- **50 rows:** the English source and each system's rank, 1 or 2. **Ties occur** (7 of 50: five `1 1`, two `2 2`).
- **No translations in the file.** It is source plus ranks only. To set COMET beside the human ranks per sentence, A11 has to join on the source text with the student's own translation files.
- **Encoding:** UTF-8, no BOM, LF, no final newline.
- **Check:** the mean ranks are 1.48 and 1.50, exactly what the student's write-up reports.

## 2. The real A12 folders (read in place on OneDrive, Oct 19; nothing copied)

All twelve files, the six languages in `LLM_Assignment_Data/`:

- **Clean in every file:** UTF-8 that decodes without a single error, **no BOM, no CRLF, no lone CR, a final newline, no blank lines, no tabs.**
- **Line counts match within each pair:** Efik 22,611; Kosrean 70,208; Palauan 42,662; Pohnpeian 50,818; Telugu 49,611; Yapese 54,080 (the same as my Oct 14 note).
- **Files are only `src.txt` and `tgt.txt`** in every folder.
- **A correction to my Oct 14 note:** the languages are **not** in the same order, and they do not all open with "Jesus is the Christ." Only Efik does. Palauan and Yapese share an opening; the others each start elsewhere. If anything in the notebook or roadmap says the text is parallel across languages, take it out.
- **Noise worth one line in the notebook:** Efik's third target line is English left untranslated ("Se title page of the Book of Mormon.."). Untranslated or partly translated lines are likely elsewhere too; a student who spot-checks a few pairs will see them.

### First three lines, `src.txt` then `tgt.txt`

**Efik**
```
Jesus is the Christ.
This is His Church.
See title page of the Book of Mormon.
---
Jesus edi Christ.
Emi edi Ufok Abasi Esie
Se title page of the Book of Mormon..
```

**Palauan** (Yapese's `src.txt` opens with the same three lines)
```
The Church of Jesus Christ of Latter-day Saints
The Purpose That Will Change Our Ministering
Our efforts in ministering should be guided by a desire to help others achieve a deeper individual conversion and become more like the Savior.
---
Ikelesia er a Jesus Kristo er a Uriul-klebesei el Santo
Moktek el Mo Ngodechii a Teletelel Odngeled
A moktek er a odngeled a kirel ulekrael er a blakereng er kid el mo olngeseu er a rebebil el mo mui el obult el mo uaia Osobel.
```

**Telugu** (non-Latin script; `tgt.txt` is 12.8 MB against 4.6 MB for the English)
```
Helping others understand their true identity and purpose is one of the greatest gifts we can give.
Seeing others and ourselves as God does knits our hearts "together in unity and in love."
With ever-increasing secular forces pulling at us, we need the strength that comes from loving relationships.
---
వారి నిజమైన గుర్తింపును మరియు ఉద్దేశమును గ్రహించడానికి ఇతరులకు సహాయపడుట మనము ఇవ్వగల మిక్కిలి గొప్ప బహుమానాలలో ఒకటి.
దేవుడు చూసినట్లుగా ఇతరులను, మనల్ని చూచుట, మన హృదయాలను "ఐక్యత మరియు ప్రేమయందు దగ్గరగా చేర్చును."
ఎప్పటికప్పుడు పెరుగుతున్న లౌకిక శక్తులు మనపై ప్రభావం కలిగియుండటంతో, ప్రేమగల అనుబంధాలనుండి వచ్చు బలము మనకవసరం.
```

Kosrean opens with "Extended families should lend support when needed."; Pohnpeian with "In our efforts to find joy in the midst of our trials, …". Yapese uses a curly apostrophe (’) in its orthography ("Gidii’en"): don't normalize it away.

With this, the A12 loader can close against real data. The A6 sample decides how A11's Step 1 reads the ranks.
