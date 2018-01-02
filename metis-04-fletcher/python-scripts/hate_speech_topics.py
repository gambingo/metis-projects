"""
This class handles the pipeline for modeling topics of tweets. It could handles
any tweets given to it, but this will specifically analyze tweets filled with
hate speech.

Process:
    1. General data cleaning
    2. NLP
    3. Dimensionality reduction with LDA
    4. Clustering
---
J. Gambino
Metis Data Science Bootcamp
Nov. 2017
"""

import sys
modules = '/Users/Joe/Documents/Metis/Projects/metis-04-fletcher/python-scripts'
sys.path.append(modules)
from nlp_pipeline import *

# from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MeanShift

# Ignore data conversion warning from StandardScaler
import warnings
from sklearn.exceptions import DataConversionWarning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DataConversionWarning)


class tweet_topics():
    def __init__(self):
        pass


    def get_data(self):
        """
        Load, filter, and clean data.
        Here, we're looking at only tweets with hate speech.
        """
        data = load_data_from_mongo()
        self.tweets = data['tweet'].loc[data['class_second'] == 0]
        return clean_tweets(self.tweets)


    def fit(self,
            plot = False,
            vectorizer_method='count',
            ngram_range=(1,2),
            max_features=5000,
            max_df=0.6,
            reduction_method='SVD',
            n_components = 10):
        """
        Transform data, presumably with the method that worked best for the
        second stage classifier.
        """
        self.vectorizer_method = vectorizer_method

        data = self.get_data()
        data, self.vectorizer = make_doc_term_matrix(data,
            method=vectorizer_method,
            ngram_range=ngram_range,
            max_features=max_features,
            max_df=max_df)

        if vectorizer_method == 'count':
            if reduction_method == 'SVD':
                self.ss = StandardScaler()
                data = self.ss.fit_transform(data)
            else:
                pass

        data, latent_space = dimensionality_reduction(n_components,
            data,
            method=reduction_method)
        self.training_vectors = data
        self.latent_space = latent_space

        if plot:
            draw_dendro(data)


    def cluster(self):
        """Cluster tweets within the latent space"""
        pass




    def print_topics(self):
        """Print top n-grams of each topic (latent_space dimension)"""
        display_topics(self.latent_space,
                       self.vectorizer.get_feature_names(),
                       10)

    def print_samples(self, samples=5):
        """Prints the N tweets that are most representative of each topic."""
        U = self.training_vectors
        for ii in range(U.shape[1]):
            print('Topic ', ii)
            top_tweets = U[:,ii].argsort()[-(samples+1):-1][::-1]
            for index in top_tweets:
                print(self.tweets.iloc[index])
            print('')

    """
    def make_recommedation(self, tweet):
        tweet_vector = self.vectorizer.transform([tweet]).toarray()

        if self.vectorizer_method == 'count':
            if reduction_method == 'SVD':
                tweet_vector = self.ss.transform(tweet_vector)
            else:
                pass

        tweet_vector = self.latent_space.transform(tweet_vector)

        nn = NearestNeighbors(n_neighbors=6,
                              metric = 'euclidean',
                              algorithm = 'brute')
        nn.fit(self.training_vectors)
        recommendations = nn.kneighbors(tweet_vector)
        rec_indices = recommendations[1][0]

        print('Hateful Tweet:')
        print(tweet, '\n')
        for ii, index in enumerate(rec_indices[1:]):
            print('Similar Tweet: ', ii)
            print(self.tweets.iloc[index], '\n')
    """
