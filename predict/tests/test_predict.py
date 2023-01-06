import unittest


import pandas as pd

from predict.predict.run import TextPredictionModel
from preprocessing.preprocessing import utils


def load_questions():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    #tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
      #      "php", "ruby-on-rails"]

    return titles

class TestPredict(unittest.TestCase):
        # TODO: CODE HERE

    def test_predict(self):
        path="/Users/boumahrat/Desktop/EPF/5A/from_poc_to_prod/poc-to-prod-capstone/train/data/artefacts/models/2023-01-03-12-37-20"
        prediction_object=TextPredictionModel.from_artefacts(path)
        #run a predict
        predict= prediction_object.predict(["salut j'ai faim"],top_k=1)

        # TODO: CODE HERE
            # assert that accuracy is equal to 1.0
        #self.assertEqual(accuracy, 1.0)
