## The through-line

The course is one long build. A student who keeps up finishes holding a cleaned bilingual
corpus, a trained NMT system, and a stack of measurements of it. Each assignment consumes the
previous artifact.

1. **Lectures 4 and 5** produce the corpus: Church translation-memory data, extracted from TMX,
   through a 16-step cleaning pipeline the student writes, delivered as two sentence-aligned
   text files. Medium- and high-resource languages: at least 200K pairs.
2. **Lecture 6** measures *other people's* systems on it: two commercial MT systems, human
   ranking in MTEval, then SacreBLEU and chrF.
3. **Lectures 7 through 9** build the student's own system: a toy model in class, then a real
   English-to-X model on 100K pairs, then the same model with SentencePiece so the two can be
   compared. Decoding as a choice is taught at Lecture 10, and the rule that comparisons must
   hold it constant is stated where A9 is set.
4. **Lectures 10 and 11** measure it properly, with COMET and COMET-QE, against the Lecture 6
   numbers.
5. **Lectures 12 through 14** go beyond it: an LLM prompted with in-context examples (A12, a
   different model, not the student's), then two ways to improve the student's own system,
   back-translation (A13) and a bidirectional multilingual model (A14).
6. **Lecture 15** steps outside text into speech.
7. **Lecture 16 onward** is context, research, history and the final project.

The pivot is Lecture 7. Before it the course is about data and measurement; after it, every
assignment assumes the student can train a model.
