"""Lecture 19, executed from the notebook: post-editing measured by edit distance and TER.

The notebook declares ``exercises``, so CI's notebook run skips it: as shipped it stops at the
post-edits, and then at the prediction, until a student writes them. These tests run its cells
from the notebook file itself with both written in, and check what its markdown says of the
results, so the prose cannot drift from them: the professional's moved date costs 0.71 by edit
distance and 0.21 by TER, the reordered sentence 0.60, TER reproduces every published HTER, and
the two one-word post-edits of Step 6 get the same number.
"""

import unittest

from notebook_cells import COURSE, code_cells, run_cells, substitute

NOTEBOOK = COURSE / "lecture-19-post-editing.ipynb"
SKIPPED = ("%pip install",)

# A student's post-edits, as few changes as the meaning line asks for: sentence 4 needs only
# "It" to "He", and sentence 5 is left as it is.
POST_EDITED = {
    'POST_EDITS = [\n    "Johannes tells him that Jaani has a new carrot.",\n'
    '    "The interviews looked at around 90 million American TV viewers.",\n'
    '    "The idea changed owners and sold to 1783 Friedrich Wambmann.",\n'
    '    "It called for peace in Mali and Nigeria on 25 December 2012.",\n': (
        'POST_EDITS = [\n    "Johannes tells him that Jaanus has a new girlfriend.",\n'
        '    "The interview was watched by around 90 million American TV viewers.",\n'
        '    "The manor changed owners and was sold to Friedrich Wartmann in 1783.",\n'
        '    "He called for peace in Mali and Nigeria on 25 December 2012.",\n'
    )
}
PREDICTED = {'PREDICTION = ""': 'PREDICTION = "sentence 3, about 0.4"'}


def lecture_19_cells(post_edited: bool = True, predicted: bool = True):
    cells = [c for c in code_cells(NOTEBOOK) if not any(m in c[1] for m in SKIPPED)]
    if post_edited:
        cells = substitute(cells, POST_EDITED)
    if predicted:
        cells = substitute(cells, PREDICTED)
    return cells


class TestLecture19(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.namespace, cls.printed = run_cells(lecture_19_cells())
        cls.sentences = cls.namespace["SENTENCES"]

    def rates(self, number, post_edit_key="professional"):
        """Edit rate and TER of sentence ``number`` (1-based) against one of its post-edits."""
        sentence = self.sentences[number - 1]
        edit_rate, ter_rate = self.namespace["edit_rate"], self.namespace["ter_rate"]
        return (
            round(edit_rate(sentence["mt"], sentence[post_edit_key]), 2),
            round(ter_rate(sentence["mt"], sentence[post_edit_key]), 2),
        )

    def test_as_shipped_it_stops_at_the_post_edits(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_19_cells(post_edited=False))
        self.assertIn("Edit the MT outputs", str(stopped.exception))

    def test_then_it_stops_at_the_prediction(self):
        with self.assertRaises(AssertionError) as stopped:
            run_cells(lecture_19_cells(predicted=False))
        self.assertIn("Write your prediction", str(stopped.exception))

    def test_the_case_to_check_in_your_head_is_two_edits(self):
        self.assertIn("the cat sat -> the dog sat down: 2 edits", self.printed)

    def test_the_edit_distance_is_levenshtein_over_words(self):
        distance = self.namespace["edit_distance"]
        self.assertEqual(distance("a b c", "a b c"), 0)
        self.assertEqual(distance("a b c", ""), 3)
        # No move: moving c to the front is a deletion and an insertion.
        self.assertEqual(distance("a b c", "c a b"), 2)

    def test_ter_reproduces_every_published_hter(self):
        for number, sentence in enumerate(self.sentences, 1):
            with self.subTest(sentence=number):
                self.assertEqual(self.rates(number)[1], sentence["published_hter"])

    def test_the_moved_date_is_ten_edits_but_three_for_ter(self):
        """Step 5: 'ten edits in fourteen words, 0.71; TER ... 0.21'."""
        sentence = self.sentences[3]
        mt, theirs = sentence["mt"], sentence["professional"]
        self.assertEqual(self.namespace["edit_distance"](mt, theirs), 10)
        self.assertEqual(len(self.namespace["words"](theirs)), 14)
        self.assertEqual(self.rates(4), (0.71, 0.21))

    def test_the_reordered_sentence_costs_point_six(self):
        self.assertEqual(self.rates(5)[1], 0.60)

    def test_a_student_who_changed_only_it_to_he_made_one_edit(self):
        """Step 5: 'If you only changed It to He, ... with one edit'."""
        sentence, mine = self.sentences[3], self.namespace["POST_EDITS"][3]
        self.assertEqual(self.namespace["edit_distance"](sentence["mt"], mine), 1)

    def test_the_two_one_word_post_edits_get_the_same_number(self):
        """Step 6: a dropped article and a restored 'not', both 0.07."""
        edit_rate, ter_rate = self.namespace["edit_rate"], self.namespace["ter_rate"]
        for mt, post_edit in self.namespace["ONE_WORD"]:
            with self.subTest(post_edit=post_edit):
                self.assertEqual(self.namespace["edit_distance"](mt, post_edit), 1)
                self.assertEqual(round(edit_rate(mt, post_edit), 2), 0.07)
                self.assertEqual(round(ter_rate(mt, post_edit), 2), 0.07)

    def test_the_tables_print(self):
        self.assertIn("professional", self.printed)
        self.assertIn("You predicted: sentence 3, about 0.4", self.printed)


if __name__ == "__main__":
    unittest.main()
