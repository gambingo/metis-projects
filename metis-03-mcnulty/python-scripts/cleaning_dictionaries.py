"""
Dictionaries used to convert coded data into text data.
"""
# Libraries
import os
from sqlalchemy import create_engine
import pandas as pd


# Helper Functions
def load_table(table_name):
    """Load raw dataset from PSQL"""
    engine_name = ('postgresql://Joe:' + os.environ['pw'] +
                   '@localhost:5432/mcnulty')
    engine = create_engine(engine_name)

    sql_command = 'SELECT * FROM ' + table_name
    df = pd.read_sql_query(sql_command, engine)
    return df

def return_column(col_name):
    """
    Accesses the cleaned data table stored in SQL
    """
    engine_name = ('postgresql://Joe:' + os.environ['pw'] +
                   '@localhost:5432/mcnulty')
    engine = create_engine(engine_name)

    sql_command = 'SELECT id, ' + col_name + ' FROM raw'
    df = pd.read_sql_query(sql_command, engine)

    return df


def write_to_csv(df, name):
    """ Write cleaned column to temporary csv file"""
    tempfile = '../data/temp_csv/temp.csv'
    df[['id', name]].to_csv(tempfile, sep=',', index=False)


# Targets
# Column 116 - Number of felony arrests while under supervision
# No dictionary needed
# Alternative Targets:
# Column 24 - Felony arrests while under supervision

# Features
# Column v7 - Number of conviction charges
# No dictionary needed

# Column v8 - Three digit offense code
offense_dict = {100: 'Homicides (undifferentiated)',
                110: 'Murder',
                120: 'Murder',
                130: 'Manslaughter',
                140: 'Reckless (including vehicular)',
                200: 'Rape (undifferentiated)',
                210: 'Forcible Rape',
                220: 'Statutory',
                300: 'Robbery (undifferentiated)',
                310: 'Armed robbery',
                320: 'Unarmed robbery',
                400: 'Assault',
                430: 'Attempted homicide',
                500: 'Burglary (undifferentiated)',
                510: 'Residential',
                520: 'Other structure',
                530: 'Burglary involving contact with victim or a weapon',
                600: 'Theft',
                700: 'Drug trafficking (undifferentiated)',
                710: 'Narcotics, cocaine, Schedules I & II',
                720: 'All other except cannabis',
                730: 'Cannabis',
                800: 'Miscellaneous',
                810: 'Sex related offenses',
                820: 'Weapons related',
                830: 'Drug possession',
                840: 'Possession of stolen property',
                850: 'Forgery',
                860: 'Fraud',
                870: 'Motor vehicular related offenses',
                880: 'Escape',
                998: 'Not ascertained'}

# Column v9 - Probationers' sex
sex_dict = {1: 'Male',
            2: 'Female',
            8: 'Unknown'}

# Column v10 - Percentage of time employed
employed_dict = {1: '>60 percent',
                 2: '40 to 60 percent',
                 3: '<40 percent',
                 8: 'unknown'}

# Column v13 - Educational level attained
edu_dict = {1: 'Grade school',
            2: 'Some high school',
            3: 'High school (GED)',
            4: 'Some college',
            5: 'College degree (including post grad)',
            8: 'Not ascertained'}

# Column v14 - Address changes in twelve months
address_dict = {1: 'None',
                2: 'One',
                3: 'Two or more',
                8: 'Not ascertained'}

# Column v15 - Drug Abuse History
drug_dict = {1: 'No drug abuse problem',
             2: 'Occasional abuse',
             3: 'Frequent abuse',
             8: 'Not ascertained'}

# Column v16 - Number of prior felony charges
prior_dict = {1: 'None',
              2: 'One',
              3: 'Two or more',
              8: 'Not ascertained'}

# Column v20 - Probation records reflect jail being imposed

# Column v24 - Number of felony arrests

# Column v99 - Age
# No dictionary needed

# Column v64 - Previously on Probation

# Column v130 - Jurisdiction

# Column v136 - Drug Treatment

# Column v140 - Previous Felony Type

# Column v113 - Race
race_dict = {1: 'White',
             2: 'Black',
             3: 'American Indian',
             4: 'Asian',
             5: 'Other',
             8: 'Unknown'}

# Column v114 - Ethnicity of Probationers

ethnicity_dict = {1: 'Hispanic',
                  2: 'Non-Hispanic',
                  8: 'Unknown'}
