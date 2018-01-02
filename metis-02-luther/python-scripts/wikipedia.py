"""
Scrapes the revision history for as many wikipedia pages provided
---
J Gambino
September 2017
Metis Bootcamp
"""

# Custom Libraries
import luther
import wikipedia_search
import single_wikipedia_history

# Libraries
import numpy as np
import timeit

# Get a dictionary of ticker abbreviations to company name.
ticker_dict = luther.nasdaq_companies('abbrev')
failed_companies = []
failed_counter = 0

for abbrev in ticker_dict.keys():
    #start = timeit.timeit()
    try:
        # Find the wikipedia page for that nasdaq ticker symbol
        (company_name, url) = wikipedia_search.find_company_page(abbrev)

        # Scrape the revision history of that page.
        single_wikipedia_history.history_df(abbrev, url, 2010)

    except:
        # Record the company that faild and move on.
        # For example, AABA failed because that companies abbreviation has changed.
        print('Failed to scrape {}'.format(company_name))
        #raise
        failed_companies.append(abbrev)
        failed_counter += 1
        if failed_counter >= 50:
            failed_counter = 0
            single_wikipedia_history.pickle_dataframe(failed_companies, 'failed_companies.pkl')

        pass

    #end = timeit.timeit()
    #print('Completeed in {} seconds.'.format(round(end - start, 5)))
