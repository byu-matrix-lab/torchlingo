# Lecture 3 Assignment - Word Embeddings

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Wed Sep 16, 10:00. Points: 15. Roadmap v6 due date: Wed Sep 16. Status: current. Rubric tables reformatted; otherwise verbatim.*

(Use Python) Use a multilingual sentence embedding model to perform word-level comparisons on several words in your language (if it is supported by "distiluse-base-multilingual-cased-v2”).

1. Install sentence-transformers (Installation — Sentence Transformers documentation (sbert.net))
2. Pick 10 random words or short phrases in English and 10 corresponding translations for those words or phrases in your second language.
3. Embed each of those words/phrases using the sentence_transformers module (Quickstart — Sentence Transformers documentation (sbert.net)), specifically use the “distiluse-base-multilingual-cased-v2” model which supports 50+ languages
4. Calculate a similarity score for each word to each other word, comparing the two languages (example: Semantic Textual Similarity — Sentence Transformers documentation (sbert.net))
5. Provide a heat map matrix with labels. Example code for a heatmap is shown below.
6. Submit both your heat map and a paragraph or two describing whether or not the embeddings for the words and their translations are similar and any other patterns you notice in the similarity scores.

Here is an example of code creating a heat map.

```python
# Example code for a heat map

import seaborn as sns
import matplotlib.pyplot as plt

example_similarities = [
    [0.93, 0.2, 0.3],
    [0.4, 0.85, 0.6],
    [0.7, 0.8, 0.9],
]

example_labels_x = ["A", "B", "C"]
example_labels_y = ["X", "Y", "Z"]

# Create the heatmap
plt.figure(figsize=(5, 5))
ax = sns.heatmap(example_similarities, xticklabels=example_labels_x, yticklabels=example_labels_y, annot=True)

# Move x-axis tick labels to the top
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

# Rotate y-axis tick labels if needed
plt.xticks(rotation=45, ha='left')

plt.show()
```

(An example heat map for 8 words in English and Thai is shown on the Learning Suite page.)

| Rubric: | Points |
|---|---|
| Provided heat map with labels for the 10 word-level translations | 10 |
| Description | 5 |
| Total | 15 |
