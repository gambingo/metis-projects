"""
Contains the helper functions used for Project Luther.
---
J. Gambino
September 2017
Metis Bootcamp
"""

# Libraries used in Project Luther
import sys
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
import pickle
#from collections import deque

# Web Scraping
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import os

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

def nasdaq_companies(key='name'):
    """
    Returns a dictionary all companies, soreted by Market Cap, listed on the
    NASDAQ. Specify key to either be 'abbrev' or 'name'.
    ---
    IN: sting
    OUT: dictionary where keys are stock ticker and value is company name
    """
    filepath = '/Users/Joe/Documents/Metis/Projects/metis-two-Luther/data/nasdaq-companies/companylist.csv'
    df = pd.read_csv(filepath)
    df.sort_values('MarketCap', ascending=False, inplace=True)

    if key=='abbrev':
        return {key: value for key, value in zip(df['Symbol'], df['Name'])}
    elif key=='name':
        return {key: value for key, value in zip(df['Name'], df['Symbol'])}
    else:
        raise

def config_selenium():
    chromedriver = "/Applications/chromedriver"
    os.environ["webdriver.chrome.driver"] = chromedriver

    from selenium.webdriver.chrome.options import Options
    options = webdriver.ChromeOptions()
    options.add_argument("user-data-dir=/Users/Joe/Library/Application Support/Google/Chrome/Profile 2")

    return chromedriver, options
