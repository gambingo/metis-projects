## Project Luther: Wikipedia Activity and a Company's Value  
Metis Data Science Bootcamp   
Joe Gambino  
September 2017  

### Overview  
When I was as student, not a single day went by that I did not lean on Wikipedia for help. The free site helped me more than any expensive text book. I have always admired the more than [one hundred thousand](https://en.wikipedia.org/wiki/Wikipedia_community) editors and have always been curious what motivates their work.

Do users edit only topics they are passionate about? Do current events spark sudden revisions? This led me to the idea that the edit activity on a company's page may have some connection to that company's stock price.

The answer was immediately apparent and was plainly no. But that didn't matter! Exploring the idea was a an exciting lesson in web scraping and a fun exploration of several linear regression models.

### Detailed Writeup
Please see my detailed write-up on this project, [here](https://www.gam.bingo/wikipedia/).  

### Code
#### Python Files  
Main File
1. `wikipedia.py`  
   Runs wikipedia_search.py on every ticker symbol listed on the NASDAQ.

Helper Files  
1. `wikipedia_search.py`  
   Searches a single ticker symbol on Wikipedia.org and navigates to the revision history.
2. `single_wikipedia_history.py`   
   Scrapes the revision history for a Wikipedia page.
3. `luther.py`  
   Various helper functions

#### Jupyter Notebooks
1. `Regression.ipynb`  
   Various linear regression models on all the data collected.
2. `combine-stock-wiki-trend-data.ipynb`  
   Scrapes Google trend data and joins with the merged Wikipedia/stock data
3. `wikipedia_and_quandl.ipynb`  
   Reads in scraped Wikipedia data and joins it with stock data from Quandl
