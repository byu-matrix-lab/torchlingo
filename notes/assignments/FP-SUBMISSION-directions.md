# Final Project Submission

*Transcribed from the Learning Suite assignment description on 2026-09-29. Learning Suite due date: Thu Dec 10, 11:59 pm. Points: 40. Roadmap v6 due date: Thu Dec 10. Status: text says Wednesday, December 17; the due date field says Dec 10. Rubric tables reformatted; otherwise verbatim.*

Final Submission of Projects

Total Points: 40 (14.5% of final grade)

By the end of the day, 11:59pm (not midnight!) on Wednesday, December 17, you must submit the final version of all project deliverables, including your final write-up, to the OneDrive shared folder. YOU MUST SUBMIT BEFORE THIS DEADLINE - NO LATE SUBMISSIONS WILL BE ACCEPTED. The final version will include but is not limited to:

- The slide deck used during your presentation with any updates you may want to make after presenting. - 1 point
- A brief (1-2 min) *NARRATED* demo recording (e.g., of your system producing translations, evaluations, etc.) - 4 points
- Everything to run your systems
  - Well-documented code/scripts - include docstrings, comments on every code block, etc. - 5 points
  - All data sets - training, validation, and test - 5 points
  - All models you created - 5 points
  - A clearly written description or instructions about how to run your system given the components you have created or referenced (so that the TA or the professor can run it if we want to). - 10 points (we will try to run your system following your instructions, but the quality of the results will not affect the points)
- Your accurate time log. - at least 30 hours - 10 points

IMPORTANT REMINDER: At the end of the semester, you must remove all copies of Church data from your personal machines and from any COLAB accounts or other places, unless you have permission to do otherwise

Some final important tips to get more successful results:
- Make sure you understand your hyperparameters - try experimenting with different setting to choose the best ones.
- Use a Transformer model
- If you're using multiple corpora and/or multiple languages to train the same model, investigate and try different temperature settings.
- Train your models for at least 100K steps, or even better, to convergence (early stopping)
- Use all your available Church data and clean it well. Test and validation sets can be the same size as before.
- If you're using public corpora, about 2M sentence pairs is sufficient
