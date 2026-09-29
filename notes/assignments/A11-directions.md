# Lecture 11 Assignment - Install & run COMET models on assignment #6 sentences

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Wed Oct 14, 10:00. Points: 20 (+10 extra credit). Roadmap v6 due date: Mon Oct 19. Status: current in substance; due date not yet moved. Rubric tables reformatted; otherwise verbatim.*

1. Install COMET
2. With the sentences from assignment #6, and directing the output to a text file:
   - Run the default model on the output of each MT system, evaluating against the reference sentences
   - Run the COMET-QE-DA model on the same output from each MT system (without references) - If you're having problems, see the in-class activity on installing COMET.
3. Compare the BLEU and chrF scores (across all 500+ segments) and average human ranking scores (for 50+ segments) for each MT system you got previously to the COMET scores for each system
4. Submit a 1-page summary to Learning Suite, including:
   - # of sentences for both automatic and human scores, which two MT systems you used, BLEU scores, chrF scores, and MTEval average ranking scores (you should already have all this; but if not, please make sure you have BLEU and chrF scores for 500-1K sentences and the MTEval average ranking scores for 50+ sentences to compare with and that your results are clearly documented!)
   - Respective COMET scores (with and without reference) for each system and whether they correlate with the BLEU/chrF/Human scores, and your observations and analysis about why or why not. Since the default model scores will have a different distribution and average from the COMET-QE-DA scores, you should only check if the default model scores correlate with the BLEU/chrF/Human scores, and then check separately if the COMET-QE-DA scores correlate with the BLEU/chrF/Human scores.

Extra credit: Run xCOMET-XL (or –XXL), and include the results in your comparisons, indicating the degree of correlation with the other metrics and your human evaluation. Or, as an alternative, if you have API access to an LLM, you can prompt it to assign a DA score to each of the 500+ segments and report the average score and its correlation with the other metrics and your human evaluation.

Here are the resources for COMET:
- https://github.com/Unbabel/COMET
- https://unbabel.github.io/COMET/html/index.html

| RUBRIC | Points |
|---|---|
| Uploaded well-written 1-page summary | 4 |
| Included # of sentences for both automatic and human scores | 2 |
| Which two MT systems you compared | 2 |
| BLEU scores, chrF scores, and MTEval average ranking scores for each system | 2 |
| Respective COMET scores from both default model (with references) and COMET-QE-DA model (without references) for each system | 5 |
| Statement about whether both sets of COMET scores correlate with the BLEU/chrF/Human scores and observations and analysis about why or why not. | 5 |
| Total | 20 |
| Extra Credit: Compute, include, and indicate correlation for xCOMET scores or LLM DA scores | 10 |
