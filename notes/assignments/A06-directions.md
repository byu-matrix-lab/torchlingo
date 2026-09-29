# Lecture 6 Assignment - Human vs automatic MT evaluation

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Mon Sep 28, 10:00. Points: 20 (27 with extra credit). Roadmap v6 due date: Mon Sep 28. Status: current. Rubric tables reformatted; otherwise verbatim.*

- Run at least 500 (1K would be better if you can – you may have to break them up into smaller chunks) random source sentences in your bilingual data through two MT systems of your choice (translate.google.com, bing.com/translator, deepl.com/en/translator, translate.yandex.com, etc. - try systems other than google if you haven't done so before and they support your target language)
- First (to avoid bias from seeing the BLEU/chrF scores):
  - Using MTEval, perform a human ranking evaluation on 50 sentences and their corresponding outputs from the two systems, obtaining the number of better and equivalent translations for each system, as well as the average ranking score for each system (Note: average scores closer to 1 are better than average scores closer to 2)
- Second:
  - Using the corresponding 500 (or 1K) target sentences from your bilingual data as references, apply the BLEU and chrF scoring tools at https://github.com/mjpost/sacrebleu and https://github.com/m-popovic/chrF to the output of each of the two systems.
  - You can earn extra credit by implementing and using your own simple BLEU score script instead of SacreBLEU. The code for this script would have to be included in what you upload to Learning Suite – do not copy SacreBLEU or use AI to write this code!
- Compare the BLEU and chrF scores to your human ranking results. Get an extra point for adding the chrF scores from SacreBLEU to your comparison and noting any differences between the Popovic and SacreBLEU versions.
- Write an analysis:
  - Did the BLEU and chrF score correlate with each other? (They won’t be the same, but do they correlate?) Why or why not?
  - How closely did the BLEU and chrF scores correlate with your human ranking, or did they not correlate at all?
  - Provide your thoughts/hypothesis on why they did or did not correlate.
- Submit to Learning Suite: the # of sentences machine translated (and by which systems), the BLEU and chrF scores from each tool for those sentences (compute the scores for all the sentences at once, not individually!), the manual ranking results (the .tsv output of MTEval along with the computed average rankings), and your written analysis as described above.

| RUBRIC: | Points |
|---|---|
| Provided the number of sentences translated and (by which systems) | 2 |
| Provided BLEU and chrF scores (including SacreBLEU's chrF score, with analysis) | 6 (7) |
| Provided manual ranking results (.tsv file from MTEval) with average ranking scores | 6 |
| Written analysis | 6 |
| (Extra Credit) Implemented BLEU scoring script | (6) |
| Total | 20 (27) |
