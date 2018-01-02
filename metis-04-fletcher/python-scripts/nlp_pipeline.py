"""
Defines functions that build the pipeline that takes raw tweets and returns
document-term matrix that has been mapped to a reduced latent space.

Should have a fit_transform method and a transform method for training versus
testing.
---
J. Gambino
Metis Data Science Bootcamp
Nov. 2017
"""

# Custom Libraries
import sys
sys.path.append('/Users/Joe/Documents/Metis/Projects/metis-04-fletcher/python-scripts')
from fletcher import *


# General
import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.cluster.hierarchy import dendrogram, linkage

# NLP
import re
import spacy
from spacy.en import STOP_WORDS as stopwords
from string import punctuation as punctuation
from nltk.tokenize.casual import TweetTokenizer

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD


# Text Cleaning
def remove_handles(tweet):
    """
    Each individual twitter handle will add too much variance to the data.
    Not yet sure if the handles should be removed or replaced with a dummy.
    """
    handle_pattern = '\B\@\w+'
    # return re.sub(handle_pattern, '@handle', tweet)
    return re.sub(handle_pattern, '', tweet)


def clean_tweets(X):
    """Apply all cleaning functions"""
    X = X.apply(remove_handles)
    X = X.apply(lambda x: re.sub('[0-9]', '', x))
    return X


def load_data_from_mongo():
    """Pulls tweet data from MongoDB and returns a pandas dataframe"""
    client = MongoClient()
    db = client['twitter-database']
    collection = db['hate-speech']
    loaded_data = pd.DataFrame(list(collection.find()))
    loaded_data.drop('_id', axis=1, inplace=True)
    return loaded_data


def load_and_split_data(stage=None, test_size=0.25):
    """
    Load tweet data and split into train and test sets
    ---
    0: Hate speech
    1: Offensive language but not hate speech
    2: Neither
    """
    df = load_data_from_mongo()

    target = df['class_second']
    tweets = clean_tweets(df['tweet'])

    if stage == 'dummy':
        return np.ones(len(target)), target

    def split(x, y, test_size=test_size):
        x_train, x_test, y_train, y_test = train_test_split(x, y,
            random_state = 42,
            test_size = test_size,
            stratify = y)
        return x_train, x_test, y_train, y_test

    # Stage 0
    x_train, x_val, y_train, y_val = split(tweets, target)

    if stage == 0:
        # y1_val = y_val.apply(lambda x: 0 if x<2 else 1)
        y1_val = y_val.apply(lambda x: 1 if x<2 else 0)
        # y2_val = y_val.apply(lambda x: int(x>0))
        y2_val = y_val.apply(lambda x: 1-int(x>0))
        return x_val, y1_val, y2_val

    elif stage == 1:
        # Classify between clean and offensive language
        tweets = x_train
        # target = y_train.apply(lambda x: 0 if x<2 else 1)
        # Make a true prediction be offensive language
        target = y_train.apply(lambda x: 1 if x<2 else 0)
        x_train, x_test, y_train, y_test = split(tweets, target)
        return x_train, x_test, y_train, y_test

    elif stage == 2:
        # Classify between hate speech and offensive but not hateful language
        tweets = x_train.loc[y_train != 2]
        target = y_train.loc[y_train != 2]
        # Make a true prediciton be hate speech
        target = target.apply(lambda x: 1-x)
        x_train, x_test, y_train, y_test = split(tweets, target)
        return x_train, x_test, y_train, y_test

    else:
        return x_train, x_val, y_train, y_val


# NLP
nlp = spacy.load('en')
def spacy_tokenizer(tweet):
    """
    Custom Tokenizer. Uses the default tokenizer from spaCy
    Not yet lemmatizing the data. Lemmatization can be problematic with tweets.
    """
    tokens = nlp(tweet)
    tokens = [word.lower_.strip() for word in tokens]
    tokens = [word for word in tokens if (word not in stopwords and word not in punctuation)]
    return tokens


def tweet_tokenizer(tweet):
    """
    Custom tokenizer that uses nltk's built in twitter Tokenizer
    """
    tt = TweetTokenizer(reduce_len=True, strip_handles=False)
    tokens = tt.tokenize(tweet)
    tokens = [word.lower().strip() for word in tokens]
    tokens = [word for word in tokens if (word not in stopwords and word not in punctuation)]
    tokens = [word for word in tokens if not re.match(r'\brt\b', word)]
    return tokens


def make_doc_term_matrix(x_train,
                         method='count',
                         tokenizer=tweet_tokenizer,
                         ngram_range=(1,2),
                         max_features=5000,
                         max_df=0.6):
    """
    Create document-term matrix.
    Record vectorizer object so it can transform the test data
    """
    if method == 'count':
        vectorizer = CountVectorizer(tokenizer=tweet_tokenizer,
                                          ngram_range=ngram_range,
                                          max_features=max_features,
                                          max_df=max_df)
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer(tokenizer=tweet_tokenizer,
                                          ngram_range=ngram_range,
                                          max_features=max_features,
                                          max_df=max_df)
    else:
        msg = "Supported vectorizer methods are 'count' or 'tfidf'"
        raise Exception(msg)

    doc_term_matrix = vectorizer.fit_transform(x_train).toarray()
    return doc_term_matrix, vectorizer


def dimensionality_reduction(n_components, doc_term_matrix, method='NMF'):
    """
    Reduce features with specified method.
    Default: Non-Negative Matrix Factorization
    """
    if method == "NMF":
        nmf = NMF(n_components = n_components)
        W = nmf.fit_transform(doc_term_matrix)
        return W, nmf
    elif method == "SVD":
        svd = TruncatedSVD(n_components = n_components)
        U = svd.fit_transform(doc_term_matrix)
        return U, svd
    else:
        raise Exception("Supported reduction methods are 'NMF' or 'SVD'")


def display_topics(latent_space, feature_names, no_top_words):
    """Prints the top words in each topic (latent_space dimension)"""
    for ii, topic in enumerate(latent_space.components_):
        print("\nDimension ", ii)
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))


# Clustering
def draw_dendro(data, l_method='complete', t_mode='mlab'):
    """
    This function was written by David Luther, Nov. 2016.
    ---
    Draws a dendrogram with given data to assist in cluster identification/
    selection.
    ---
    IN
    data: array of data to be clusetered (np array, list, etc.)
    link_method: method for calculating linkage for each new cluster, can be
        single, complete, average, weighted, centroid, median, or ward (str)
    trunc_mode: truncation mode, if any, can be None, mlab, lastp, or level
        (str)
    OUT
    No return
    """

    # plt.clf()
    Z = linkage(data, method=l_method, metric='euclidean')
    plt.figure(figsize=(12,8), dpi=200)
    dendrogram(Z, truncate_mode=t_mode, color_threshold=0.6*max(Z[:,2]))

    plt.xlabel('\nIndividual Tweets', fontsize=20)
    plt.ylabel('Linking Distance\n', fontsize=20)
    plt.tick_params(axis='x',
                    which='both',
                    bottom='off',
                    labelbottom='off')
    plt.savefig('../figures/dendrogram.png')
    plt.show();
