"""
This class handles the pipeline for classifying tweets. It performs the
following on training and testing data, calling fit_transform() where it should
and calling transform() where it should.
    1. General data cleaning
    2. NLP
    3. Dimensionality reduction
    4. Classification Model
---
J. Gambino
Metis Data Science Bootcamp
Nov. 2017
"""

import sys
modules = '/Users/Joe/Documents/Metis/Projects/metis-04-fletcher/python-scripts'
sys.path.append(modules)
from nlp_pipeline import *

import numpy as np
import xgboost as xgb

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


# Ignore data conversion warning from StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DataConversionWarning)


class tweet_classifier():
    def __init__(self, x, y):
        """
        At initialization, simply provide the feature and target data.
        Assumes the user has already done a train test split
        """
        # Assign Attributes
        self.x = x
        self.y = y


    def build_model(self,
                    ngram_range=(1,3),
                    max_features=5000,
                    max_df=0.6,
                    n_components = None,
                    vectorizer_method = 'tfidf',
                    reduction_method = 'SVD',
                    n_estimators=None,
                    boosted=True,
                    max_depth=3,
                    learning_rate=0.05):
        """
        Vectorize, reduce, and create model
        This method calls on fucntions defined in nlp_pipeline.py
        """
        self.vectorizer_method = vectorizer_method
        if n_components:
           n_components = n_components
        else:
           n_components = int(np.sqrt(max_features))

        # If n_estimators was not defined, set different default values
        # based on model method
        if not n_estimators:
            if boosted:
                n_estimators=300
            else:
                n_estimators=10

        data, self.vectorizer = make_doc_term_matrix(self.x,
            method=vectorizer_method,
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=max_df)

        if vectorizer_method == 'count':
            if reduction_method == 'SVD':
                self.ss = StandardScaler()
                data = self.ss.fit_transform(data)
            else:
                # Can't standardize with NMF. It will introduce negatives
                # Settle for normalization
                pass

        data, self.latent_space = dimensionality_reduction(n_components,
            data,
            method=reduction_method)

        if not boosted:
            self.model = RandomForestClassifier(n_estimators=n_estimators)
        else:
            self.model = xgb.XGBClassifier(max_depth=max_depth,
                                           n_estimators=n_estimators,
                                           learning_rate=learning_rate)
        self.model.fit(data, self.y)


    def predict(self, x_test, y_test):
        """
        Make and score predictions.
        In order to predict on the test set, must first transform the test set
        using the same vectorizer and same reduction matrix.
        """
        data = self.vectorizer.transform(x_test).toarray()

        if self.vectorizer_method == 'count':
            if reduction_method == 'SVD':
                data = self.ss.transform(data)
            else:
                pass

        data = self.latent_space.transform(data)


        self.pred = self.model.predict(data)
        # self.score = accuracy_score(self.pred, y_test)
        # self.score = precision_score(self.pred, y_test, average='weighted')
        self.score = f1_score(self.pred, y_test, average='weighted')
