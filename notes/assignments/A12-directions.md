# Lecture 12 Assignment - Evaluating the Impact of Context on Low-Resource Language Translation

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Mon Oct 19, 10:00. Points: 15. Roadmap v6 due date: Mon Oct 26. Status: current in substance; due date not yet moved. Rubric tables reformatted; otherwise verbatim.*

Objective: The goal of this assignment is to explore how providing in-context examples affects the translation quality of a Large Language Model (LLM) for a low-resource language of your choice. You will use a pre-trained model from the HuggingFace Hub to translate sentences, iteratively adding more context, and measure the performance using the BLEU score.

Tasks:

1. Open the Jupyter notebook linked on the Learning Suite page to do the assignment
2. Select a Model: Choose a suitable model from the HuggingFace Hub. You can experiment with different models, but avoid models that were specifically trained for MT (like NLLB).
3. Load Data: Load the provided dataset from the linked folder for one of the target languages: Efik, Palauan, Pohnpeian, Yapese, Kosrean, or Kamba (Kikamba). Make sure to separate the sentences that you will sample for context from the sentences that you will use for inference.
4. Implement Iterative Translation: Create a loop that translates a set of test sentences. For each sentence, you will prepend a varying number of "context" sentences (e.g., 0, 5, 10, 20) to the input prompt. You should translate 50 sentences using each context size and calculate a BLEU score for the test set using each context size (meaning translate 50 sentences with no context, 50 with 5 examples, etc.).
5. Calculate BLEU Scores: For each level of context, calculate the BLEU score by comparing the model's translations against the provided reference translations.
6. Provide a description and analysis

Upload to your folder in the Assignment shared folder:

1. The files of sentences and their translations that you used as context for each level of context
2. Your test set
3. A report of the respective BLEU scores based on the different amounts of context provided, including in the description in #4 below.
4. A separate write-up containing a description and analysis of what you did, the results, which LLM you used, and the answers to the questions in the Jupyter notebook

| RUBRIC | Points |
|---|---|
| Uploaded the files of sentences and their translations that you used for each level of context | 5 |
| Uploaded test set | 3 |
| Report of the respective BLEU scores based on the different amounts of context provided | 2 |
| A description and analysis of what you did and the results, including which LLM you used and the answers to the questions in the Jupyter notebook | 5 |
| Total | 15 |
