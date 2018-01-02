"""
Helper Functions for Project McNulty
---
J. Gambino
October 2017
"""

# Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn import metrics

# Custom Modules
from cleaning_dictionaries import *


# Helper Functions
def convert_recidivism_to_int(entry):
    """
    POTENTIALLY DEPRECATED
    Used on a pandas dataframe in an apply method to convert recidivism values
    from strings to ints
    ---
    IN: string
    OUT: int
    """

    if entry == 'Yes':
        return 1
    else:
        return 0


def load_iowa_probation_data():
    """
    Loads parole data from https://catalog.data.gov/dataset/3-year-recidivism-for-offenders-admitted-to-probation
    ---
    IN: none
    OUT: pandas dataframe
    """
    filepath_parole = '../data/3-Year_Recidivism_for_Offenders_Admitted_to_Probation.csv'
    return pd.read_csv(filepath_parole)


def custom_train_test_split(df, test_size=0.3):
    """
    DEPRECATED
    A custom train_test_split function so that I can more easily retain race
    ---
    IN: args:   dataframe
        KWargs: size of testing set [0, 1]
    OUT: training dataframe, testing dataframe
    """
    df['seed'] = np.random.choice([0,1], size=df.shape[0],
                                         p=[test_size,
                                         1-test_size])

    df_train = df.loc[df['seed'] == 1]
    df_test  = df.loc[df['seed'] == 0]

    df.drop('seed', axis=1, inplace=True)


    df_train.drop('seed', axis=1, inplace=True)
    df_test.drop('seed', axis=1, inplace=True)

    return df_train, df_test


def pull_from_SQL(features, table_name='clean'):
    """
    Pulls the requested features from SQL, as well as recidivism and race
    """

    # SQL Engine
    engine_name = ('postgresql://Joe:' + os.environ['pw'] +
                   '@localhost:5432/mcnulty')
    engine = create_engine(engine_name)
    sql_command = (lambda x: 'SELECT ' + x + ' FROM '+ table_name + ';')

    x = pd.read_sql_query(sql_command(features[0]), engine)
    if len(features) > 1:
        for feature in features[1:]:
            df = pd.read_sql_query(sql_command(feature), engine)
            x = x.join(df)

    y = pd.read_sql_query(sql_command('reoffend'), engine)
    race = pd.read_sql_query(sql_command('race'), engine)
    ethnicity = pd.read_sql_query(sql_command('ethnicity'), engine)

    return x, y, race, ethnicity
