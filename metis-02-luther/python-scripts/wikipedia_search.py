"""
Searches a company abbrev. on Wikipedia and returns the url for the
company's page.
---
J. Gambino
September 2017
Metis Bootcamp
"""

# Libraries should be imported by luter.py
import luther
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

# Helper Functions
def search_and_enter(driver, element_name, search_term):
    search_box = driver.find_element_by_name(element_name)
    search_box.clear()
    search_box.send_keys(search_term)
    search_box.send_keys(Keys.RETURN)

def get_current_page_name(soup):
    company_name = soup.find(class_="firstHeading").text
    return company_name

def is_page_may_refer_to(soup):
    return 'may refer to:' in soup.find(class_="mw-parser-output").p.text

def find_link(soup):
    for element in soup.find_all('li'):
        if 'NASDAQ' in element.text:
            return 'https://en.wikipedia.org' + element.a['href']
        else:
            return None

# Main Function
def find_company_page(abbrev):
    """
    Searches a company's stock code on Wikipedia and returns the url
    of that company's page.
    ---
    IN: string
    OUT: string
    """
    chromedriver, options = luther.config_selenium()
    driver = webdriver.Chrome(chromedriver, chrome_options = options)
    wikipedia = 'https://www.wikipedia.org/'

    # Wrap everything in a try-except statment so we can close the driver
    try:
        driver.get(wikipedia)
        search_term = abbrev + ' (NASDAQ)'
        search_and_enter(driver, "search", search_term)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        if is_page_may_refer_to(soup):
            url = find_link(soup)
            company_name = get_current_page_name(luther.make_soup(url))
        else:
            url = driver.current_url
            company_name = get_current_page_name(soup)

        driver.close()
        print('Found Wikipedia page for {}'.format(company_name))
        return (company_name, url)
    except:
        driver.close()
        message = 'Crashed while searching {} on Wikipedia'.format(abbrev)
        raise Exception(message)
