November 2017  

### Flagging Hate Speech on Twitter
I love and am not at all addicted to Joke Twitter, a community of silly people who share silly jokes. But Joke Twitter has a problem: it is on Twitter. Twitter is filled with hateful, angry people who will harrass and attack you if they don't like what you have to say. These trolls can be espcially cruel to women. I wanted to build a tool that could flag incoming abusive tweets so that nice, silly people who like to make jokes (sometimes with a feminist slant) don't have to put up with such cruelty.

Please read a detailed write-up of this project, [here](https://www.gam.bingo/flagging-hate-speech-on-twitter)

### Data
[Automated Hate Speech Detection and the Problem of Offensive Language](https://github.com/t-davidson/hate-speech-and-offensive-language)  
Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017.

### Code
##### Python Scripts
1. `fletcher.py`
    Helper Functions for loading, testing, and saving models. Includes 
2. `nlp_pipeline.py`
    Helper functions specific to natural language processing (tokenizers, vecotrizers, etc.) shared by both the topic modeler and classifiers.
3. `hate_speech_topics.py`
    Contains the class `tweet_topics()`, which facilitates topic modelling of tweets containing hate speech.
4. `tweet_classifier.py`
    Contains the class `tweet_classifier()`, which facilitates building a classification model on supplied tweets and truth labels.
5. `hate_speech_classifier.py`
    Chains two instances of `tweet_classifier()` to build a tool that can flag incoming, hateful tweets.

##### Notebooks
1. `dev-exploring-data-sources.ipynb`
    Exploring options for data sources and early EDA.
2. `loading-data-into-mongoDB.ipynb`
    Moves data from a csv into a mongo database.
3. `dev-topic-modeler.ipnyb`
    Used for the development of the topic modeling.
4. `dev-classification-pipeline.ipynb`
    Used for the development and tuning of `tweet_classifier()` running a random forest model.
5. `dev-classificaiton-pipeline-XGBoost.ipynb`
    Used for the development and tuning of the `tweet_classifier()` running an XGBoost model.
6. `dev-twitter-firehose.ipynb`
    Testing the `hate_speech_classifier()` on replies sent to various twitter accounts.
