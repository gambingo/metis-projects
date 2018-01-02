October 2017  

## Racial Bias: Machine Learning in the Criminal Justice System

Many states use classification algorithms to predict if a criminal's risk of recidivism. [They are used to inform decisions about who can be set free at every stage of the criminal justice system.](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) These algorithms have been shown to unintentionally induce a racial bias into the system.

ProPublica conducted a [thorough analysis](https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm) of one of the most common commercial algorithms, COMPAS. They found the COMPAS algorithm is biased towards white defendants and against black defendants. White defendants were twice as likely to be misclassified as lower risk then they later turned out to be, while black defendants were twice as likely to be misclassified as higher risk than they later turned out to be. In technical jargon, white defendants had a higher false negative rate and black defendants had a higher false positive rate.

ProPublica could only analyze the results of the COMPAS algorithm; they couldn't look under its hood. I wanted to look under the hood, and they only way to do that would be to create my own algorithm. So I set about designing my own classification model for predicting recidivism. Race would not be an input to my model, but I would examine my model's predictions for any racial bias.

For a detailed write-up on this project, please visit my [site](https://www.gam.bingo/racial-bias/).

### Code
#### Python Scripts
* `cleaning_dictionaries.py`
     * A collection of dictionaries used for decoding the raw data
* `mcnulty.py`
     * Various helper functions
 * `recidivism.py`
     * The class for building models off of chosen features. The user can specify a certain algorithm, or the class can try several and select the most precise. It will then generate confusion matrices and metrics filtered by race.

#### Notebooks
* `cleaning-with-pandas.ipynb`
    * Imports raw data from SQL, cleans and decodes various columns, and generates SQL commands to create a new table.
* `dev-recidivism-class.ipynb`
    * Notebook used while developing the `recidivism.py`. It demonstrates various uses of the class.

#### Web App
* `index.html`
    * Web page
* `predicting_recidividm.py`
    * Web page backend. Uses `recidivism.py`. Powered by Flask.

#### SQL
* `add_column.sql`
    * This function adds one column at a time to the `clean` table.
    * Using this function proved to be inefficient. It was way faster to clean the entire table in pandas and then upload a whole new table at once. But I wanted to see how to write a SQL function.
