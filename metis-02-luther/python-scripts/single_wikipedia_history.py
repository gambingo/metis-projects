"""
Scrapes the change history for any wikipedia page.
---
J. Gambino
Metis Project Luther
Sept. 27, 2017
"""

# Libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import pickle
import os.path
import time
import numpy as np

# Helper Functions
def make_soup(url):
    """
    For a given url, if the site is accessible, return soup
    ---
    IN: url as string
    OUT: dataframe
    """
    response = requests.get(url)
    if response.status_code != 200:
        # Site in inaccessible
        # TO-DO: Raise a error/warning and return an empty dataframe
        # For now, pass
        pass

    page = response.text
    soup = BeautifulSoup(page,"html5lib")
    return soup

def scrape_revisions(url):
    """
    Scrapes all visible revisions for the provied url. Must be provied the url
    for the revision history page.
    Reutrns df of revisions url for next next page of revisions
    ---
    IN: url
    OUT: (df, url)
    """

    # Scrape page and create dataframe
    soup = make_soup(url)
    df = pd.DataFrame(columns=['date', 'user', 'size', 'size_delta', 'minor_edit'])
    for revision in soup.find_all(class_="mw-history-histlinks"):

        # If the changes were reversed the date is displayed as crossed out.
        # A different class is assgined for this.
        try:
            date = revision.parent.find(class_="mw-changeslist-date").text
        except:
            date = revision.parent.find(class_="history-deleted").text

        # Convert date to datetime
        date = datetime.strptime(date, '%H:%M, %d %B %Y')

        # These next entris appear consistent (so far)
        user = revision.parent.find(class_='history-user').text
        size = revision.parent.find(class_="history-size").text
        size_delta = revision.parent.find(dir="ltr").text

        # This only exists for minor edits.
        try:
            minor_edit = revision.parent.find(class_="minoredit").text
        except:
            minor_edit = 0

        df = df.append(pd.Series({'date': date,
                                  'user': user,
                                  'size': size,
                                  'size_delta': size_delta,
                                  'minor_edit': minor_edit}),
                                  ignore_index=True)

    # Get url for next page of revisions
    try:
        for element in soup.find_all(class_="mw-nextlink"):
            suffix_older_500 = element['href']
        next_page = 'https://en.wikipedia.org' + suffix_older_500
    except:
        print('Company is young.')
        next_page = ''
        pass

    return (df, next_page)

def before_target_year(df, target_year):
    """
    Read in the dataframe of scraped history. Check if the earliest entry is before our target year.
    ---
    IN: pandas dataframe, int
    OUT: bool
    """
    earliest_date = df['date'].iloc[-1]
    return earliest_date.year < target_year-1

def pickle_dataframe(df, filename):
    """
    Saves a dataframe as a pickle file
    ---
    IN: anything, string
    OUT: picklefile
    """
    filepath = 'wikipedia-revision-logs-abbrev/'
    with open(filepath + filename, 'wb') as picklefile:
        pickle.dump(df, picklefile)

def already_scraped(abbrev):
    filename = 'revision-log-' + abbrev + '.pkl'
    filepath = 'wikipedia-revision-logs-abbrev/'
    return os.path.isfile(filepath + filename)

# Scrape Site
def history_df(abbrev, url, target_year):
    """
    For a specified wikipedia entry, scrapes all revision history until the specified target year
    ---
    IN: string, int
    OUT: pickled pandas dataframe
    """
    # Company Page
    soup = make_soup(url)
    #company_name = soup.find(class_="firstHeading").text

    # Navigate to history page
    history_suffix = soup.find(title="Past revisions of this page [h]").get('href')
    prefix = 'https://en.wikipedia.org'
    soup = make_soup(prefix + history_suffix)

    for element in soup.find_all(class_="mw-numlink"):
        if element.text == '500':
            most_recent_500 = prefix + element['href']

    # Naviage to a display limit of 500 and scrape first 500 revisions
    (df, next_page) = scrape_revisions(most_recent_500)
    print("Scraped 500 most recent revisions of {}'s page".format(abbrev))
    # Go to next page, append df, repeat until we've gone back far enough
    # next_page returns as an empty string if we've exhausted the compnay's history
    # before target_year
    while (not before_target_year(df, target_year)) and next_page:
        (df_next, next_page) = scrape_revisions(next_page)
        df = df.append(df_next)

    # Save file
    pickle_dataframe(df, 'revision-log-' + abbrev + '.pkl')
    print('Finished scraping revisions for {}.\n'.format(abbrev))
    time.sleep(np.random.uniform(1, 5)) # Pause
    # End Function

# Test the function
#IBM = 'https://en.wikipedia.org/wiki/IBM'
#history_df(IBM, 2010)
