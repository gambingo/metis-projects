"""
Helper function for Project Fletcher
---
Metis Data Science Bootcamp
J. Gambino
November 2017
"""

# Libraries
import pickle
import itertools as it
from six import string_types
import matplotlib.pyplot as plt
import datetime

import sys
sys.path.append('/Users/Joe/Documents/Metis/Projects/metis-04-fletcher/python-scripts')
from tweet_classifier import *
from hate_speech_topics import *
from hate_speech_classifier import *


# Loading and Saving Data
def save_pickle(obj, filename):
    """pickles a python object"""
    filepath = '../data/'
    with open(filepath + filename, 'wb') as picklefile:
        pickle.dump(obj, picklefile)


def load_pickle(filename):
    """loads a pickled object"""
    filepath = '../data/'
    with open(filepath + filename, 'rb') as picklefile:
        obj = pickle.load(picklefile)
    return obj


# Model Tuning
def grid_search(x_train, y_train, x_test, y_test,
                params,
                print_steps=True,
                summary_plot=False,
                log_file=False,
                stage=None):
    """
    For every combination in params, build and score a model.
    ---
    KWargs:
        If plot is set to true, it will plot the scores overtime.
        This only works with one variable parameter.
    """
    print_ = print_steps
    plot_ = summary_plot

    if print_steps and log_file:
        old_stdout = sys.stdout
        timestamp = datetime.datetime.now().strftime("%I:%M%p_on_%d_%B_%Y")
        filename = '../python-scripts/log-files/grid-search-log_' + timestamp + '.txt'
        log_file = open(filename, 'w+')
        sys.stdout = log_file
        print(stage + '\n')

    def print_iff(s, print_):
        """only print when print_flag is enabled"""
        if print_:
            print(s)

    options = sorted(params)
    combinations = it.product(*(params[kwarg] for kwarg in options))

    commands = []
    try:
        for combo in combinations:
            string = ['model.build_model(']
            kwargs = []
            for ii, kwarg in enumerate(combo):
                if isinstance(kwarg, string_types):
                    statement = options[ii] + '=' + '"' + str(kwarg) + '"'
                else:
                    statement = options[ii] + '=' + str(kwarg)
                kwargs.append(statement)
            string.append(', '.join(kwargs))
            string.append(')')
            string = ''.join(string)
            commands.append(string)
    except:
        # generator is exhausted
        pass
    print_iff(f'Trying {len(commands)} differnt models.', print_)

    training_scores, testing_scores = [], []
    models = []
    max_score = 0
    best_params = ''
    for command in commands:
        try:
            model = tweet_classifier(x_train, y_train)
            print(command)
            exec(command)
            model.predict(x_train, y_train)
            training_scores.append(model.score)
            s = 'Training Score: ' + str(model.score)
            print_iff(s, print_)
            model.predict(x_test, y_test)
            testing_scores.append(model.score)
            if model.score > max_score:
                max_score = model.score
                best_params = command
            s = 'Testing Score: ' + str(model.score)
            print_iff(s, print_)
            print_iff('', print_)

            models.append(model)
        except:
            print('This model failed. Perhaps negative values were passed to NMF.\n')
            pass

    # Record Best Model
    print('\nBest Testing Score: ', max_score)
    print(best_params)

    # If the user only changed one parameter, plot the results.
    parameter = list(params.keys())
    if plot_:
        x = params[parameter[0]]
        plt.plot(x, training_scores, label='Training Scores')
        plt.plot(x, testing_scores,  label='Testing Scores')
        plt.xlabel(parameter[0])
        plt.ylabel('accuracy_score')
        plt.legend()
        plt.savefig('../figures/temp.png')

    if print_steps and log_file:
        sys.stdout = old_stdout
        log_file.close()

    return models[testing_scores.index(max(testing_scores))]
