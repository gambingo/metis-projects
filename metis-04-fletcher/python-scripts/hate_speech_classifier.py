"""
Combines two separtely trained classifiers into one classifier to label a tweet
as either contining or not containing hate speech.
---
J. Gambino
Nov. 2017
Metis Data Science Bootcamp
"""

import pandas as pd
import numpy as np


class hate_speech_classifier():
    def __init__(self, stage_one, stage_two):
        self.stage_one = stage_one
        self.stage_two = stage_two

    def predict(self, x_val, y1_test=None, y2_test=None):
        """
        Chain the two clasifiers. If truth labels have been provided, return
        the model's performance. If not, return all hate tweets found.
        """
        if y1_test is None and y2_test is None:
            y1_test = np.ones(len(x_val))
            y2_test = y1_test
            score_model = False
        else:
            score_model = True

        self.stage_one.predict(x_val, y1_test)
        if score_model:
            stage_one_score = self.stage_one.score

        # This section will not generalize well beyond this specific dataset.
        if score_model:
            df = pd.DataFrame(data=[y1_test, self.stage_one.pred],
                columns=['y1_test', 'stage_one.pred'])
            df = pd.DataFrame(y1_test).rename(columns={'class_second': 'y1_test'})
            df['stage_one.pred'] = self.stage_one.pred
            df['tweets'] = x_val
            df['y2_test'] = y2_test

            # x2_val = df['tweets'].loc[df['stage_one.pred'] == 0]
            # y2_test_ = df['y2_test'].loc[df['stage_one.pred'] == 0]
            x2_val = df['tweets'].loc[df['stage_one.pred'] == 1]
            y2_test_ = df['y2_test'].loc[df['stage_one.pred'] == 1]
        else:
            # x2_val = x_val[self.stage_one.pred == 0]
            x2_val = x_val[self.stage_one.pred == 1]
            y2_test_ = np.ones(len(x2_val))

        self.stage_two.predict(x2_val, y2_test_)
        if score_model:
            stage_two_score = self.stage_two.score
            return (stage_one_score, stage_two_score, stage_one_score*stage_two_score)
        else:
            # return tweets that contain hate speech
            return x2_val[self.stage_two.pred == 0]
