"""
J. Gambino
Metis Data Science Bootcamp
October, 2017
"""

# Libraries
from mcnulty import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class recidivism():
    """
    Builds a model to predict recidivism, then scores that model overall and
    per race/ethinic group.
    ---
    User provides features, target, and race data.
    """
    def __init__(self, web_app=False, model_type = 'Classification', algorithms = ['Logistic Regression']):
        """Declare type of model to build and set various status flags"""

        self.model_type = model_type
        if self.model_type not in ['Classification', 'Regression']:
            msg = 'Only suppored model types are "Classification" or "Regression"'
            raise Exception(msg)
            if self.model_type == 'Regression':
                msg = 'Predicting how long until someone reoffends has not yet been implemented.'
                raise NotImplementedError(msg)

        self.algorithms = algorithms
        self.supported_algoritms = {'KNN': KNeighborsClassifier(n_neighbors=3),
                                    'Logistic Regression': LogisticRegression(),
                                    'Naive Bayes': GaussianNB(),
                                    'SVM': SVC(),
                                    'Decision Tree': DecisionTreeClassifier(),
                                    'Random Forest': RandomForestClassifier()}
        for algorithm in self.algorithms:
            if algorithm not in self.supported_algoritms.keys():
                msg1 = 'Unsupprted algorithm: ' + algorithm
                msg2 = 'Supported algorithms are '
                for model in list(self.supported_algoritms.keys())[:-1]:
                    msg2 += model + ', '
                msg2 += 'and ' + list(self.supported_algoritms.keys())[-1] + '.'
                raise Exception(msg1 + ' ' + msg2)

        # Status Flags
        self._model_built = False
        self._predictions_made = False

        # Attributes
        self.web_app = web_app
        self.models = {}
        self.coef_ = None

        # POSSIBLE TO-DO
        # Function that renames target and race dataframes so I know the names

    def _record_race(self):
        """Pull unique values from the race data provided"""
        self.unique_races = self.race.iloc[:,0].value_counts().index.tolist()
        self.unique_ethnicities = self.ethnicity.iloc[:,0].value_counts().index.tolist()

    def _split(self, test_size):
        """
        Splits features and target into train and test sets.
        Drops race from the training sets.
        Saves race array corresponding to testing set.
        Standardizes all data with StandardScaler()
        """
        self.x_train, self.x_test, self.y_train, self.y_test, _, self.race_test, self.ethnicity_train, self.ethnicity_test = (
        train_test_split(self.features,
                         self.target,
                         self.race,
                         self.ethnicity,
                         test_size = test_size,
                         stratify = self.target,
                         random_state = 42))

        ss = StandardScaler()
        self.x_train = ss.fit_transform(self.x_train)
        self.x_test = ss.transform(self.x_test)

    def _build_classification_model(self):
        # Classification Helper Functions
        def dummify():
            """
            Turn categorical data into dummy variables. Drop first.
            All continuous data should be binned beforehand.
            """

            self.features = pd.get_dummies(self.features, drop_first=True)

        # Build Classification Model
        dummify()
        self._split(0.25)

        if self.web_app:
            # Use a proven model
            self.model = LogisticRegression(C=0.1, class_weight=None,
                                            dual=False, fit_intercept=True,
                                            intercept_scaling=1, max_iter=100,
                                            multi_class='ovr', n_jobs=1,
                                            penalty='l1', random_state=None,
                                            solver='liblinear', tol=0.0001,
                                            verbose=0, warm_start=False)
            self.model.fit(self.x_train, np.ravel(self.y_train))

        else:
            # Find the best model.
            # Try each model requested
            for algorithm in self.algorithms:
                self.models[algorithm] = {}
                if algorithm == 'Logistic Regression':
                    parameters = {"penalty": ["l1", "l2"],
                                  "C": [0.1, 1, 10, 100],
                                  "class_weight": ["balanced", None]}
                    print("parameters")
                    grid = GridSearchCV(LogisticRegression(),
                                        parameters,
                                        scoring=make_scorer(precision_score),
                                        n_jobs=-1)
                    print("grid")
                    grid.fit(self.x_train, np.ravel(self.y_train))
                    print("grid fit")

                    coef_ = ([(name, weight) for name, weight in
                             zip(self.features.columns,
                                 grid.best_estimator_.coef_[0])])
                    print("coef_")
                    self.logistic_coef_ = sorted(coef_,
                                                 key=lambda x: abs(x[1]),
                                                 reverse=True)
                    print("logistic_coef_")
                    self.models[algorithm]['model'] = grid.best_estimator_
                    self.models[algorithm]['score'] = grid.best_score_
                    print("recorded model")

                elif algorithm == 'Random Forest':
                    # feature_count = len(self.features.columns)
                    # parameters = {'max_features': [3, 4, 5, 6],
                    #               'max_depth': np.arange(feature_count-5, feature_count+1, 1)}
                    # grid = GridSearchCV(RandomForestClassifier(),
                    #                     parameters,
                    #                     scoring=make_scorer(precision_score),
                    #                     n_jobs=-1)
                    # grid.fit(self.x_train, np.ravel(self.y_train))
                    #
                    # self.models[algorithm]['model'] = grid.best_estimator_
                    # self.models[algorithm]['score'] = grid.best_score_

                    min_estimators = 50
                    max_estimators = 200
                    num_estimators = range(min_estimators, max_estimators+1)
                    oob_scores = []
                    model = RandomForestClassifier(warm_start=True,
                                                   oob_score=True)

                    for ii in num_estimators:
                        model.set_params(n_estimators=ii)
                        model.fit(self.x_train, np.ravel(self.y_train))
                        oob_scores.append(model.oob_score_)

                    self.num_estimators = num_estimators
                    self.oob_scores = oob_scores

                    forest_scores = zip(num_estimators, oob_scores)
                    forest_scores = sorted(forest_scores,
                                           key=lambda x: x[1],
                                           reverse=True)
                    n_estimators = forest_scores[0][0]
                    model.set_params(n_estimators=ii)
                    model.fit(self.x_train, np.ravel(self.y_train))

                    self.models[algorithm]['model'] = model
                    self.models[algorithm]['score'] = model.oob_score_


                else:
                    model = self.supported_algoritms[algorithm]
                    score = cross_val_score(model, self.x_train, np.ravel(self.y_train),
                                            scoring=make_scorer(precision_score))
                    self.models[algorithm]['model'] = model
                    self.models[algorithm]['score'] = np.mean(score)
                    # self.models[algorithm]['model'] = grid.best_estimator_
                    # self.models[algorithm]['score'] = grid.best_score_

            # Choose the best model
            rankings = sorted(self.algorithms, key=lambda x: self.models[x]['score'], reverse=True)
            if rankings[0] == 'Logistic Regression':
                self.model = self.models['Logistic Regression']['model']
            elif rankings[0] == 'Random Forest':
                self.model = self.models['Random Forest']['model']
            else:
                # cross_val_score does not save a built model
                # at least if it can, I'm not yet doing it that way
                self.model = self.models[rankings[0]]['model']
                self.model.fit(self.x_train, np.ravel(self.y_train))


    def _build_regression_model(self):
        pass

    def build_model(self, x, y, race, ethnicity):
        """Initalizes and fits either a classification or regression model"""

        self.features = x
        self.target = y
        self.race = race
        self.ethnicity = ethnicity

        self._record_race()

        if self.model_type == 'Classification':
            self._build_classification_model()
        else:
            self._build_regression_model()

        self._model_built = True
        print('internal: model built')

    def _predict(self):
        """Predict on the test set and collect results with race data"""
        if not self._model_built:
            raise Exception('You have to build a model first, ya dummy.')

        # For each model
        for algorithm in self.algorithms:
            model = self.models[algorithm]['model']
            predictions = model.predict(self.x_test)
            acc = accuracy_score(self.y_test, predictions)
            prec = precision_score(self.y_test, predictions)
            self.models[algorithm]['accuracy'] = acc
            self.models[algorithm]['precision'] = prec

        # For the best model
        predictions = self.model.predict(self.x_test)
        # probabilities =self.model.predict_proba(self.x_test)
        #print(self.model.classes_)
        # print(probabilities)
        self.results = self.y_test.join(self.race_test).join(self.ethnicity_test)
        self.results['predictions'] = predictions
        # self.results['probabilities'] = probabilities[:,1]
        # threshold = 0.6
        # self.results['strict_predictions'] = (
        # self.results['probabilities'].apply(lambda x: 1 if x>threshold else 0))
        self._predictions_made = True

    def score_model(self, filter_race='all'):
        """
        Filters by the give race/ethinc group and prints summary statitics
        of the model. If no race is specified, returns overall results.
        """
        if not self._predictions_made:
            self._predict()

            for algorithm in self.algorithms:
                print('\n' + algorithm)
                print('Accuracy: ', self.models[algorithm]['accuracy'])
                print('Precision: ', self.models[algorithm]['precision'])

        def create_confusion_matrix(df):
            """
            Creates confusion matrix for specified results
            """
            y_actual = df['reoffend']
            y_predicts = df['predictions']
            # y_predicts = df['strict_predictions']
            # margins=True adds sum row and column to matrix
            matrix = pd.crosstab(y_actual, y_predicts,
                                 rownames=['Actual'], colnames=['Predictions'],
                                 margins=False)
            matrix.rename(columns={0: 'No', 1: 'Yes'},
                          index = {0: 'No', 1: 'Yes'},
                          inplace=True)
            return matrix

        def false_rates(matrix):
            """
            From a supplied confusion matrix, calculates the FPR and FNR.
            ---
            tn: True Negative
            fn: False Negative
            fp: False Positive
            tp: True Positive

            fpr: False Positive Rate
            fnr: False Negative Rate
            """
            self.tn, self.fn, self.fp, self.tp, = 0, 0, 0, 0
            self.acc, self.prec, self.fpr, self.fnr = 0, 0, 0, 0

            def find_tn():
                self.tn = matrix['No'][0]

            def find_fn():
                self.fn = matrix['No'][1]

            def find_fp():
                self.fp = matrix['Yes'][0]

            def find_tp():
                self.tp = matrix['Yes'][1]

            def find_acc():
                self.acc = (self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
                self.acc = round(self.acc, 3)*100

            def find_prec():
                self.prec = self.tp/(self.tp+self.fp)
                self.prec = round(self.prec, 3)*100

            def find_fpr():
                self.fpr = self.fp/(self.fp + self.tn)
                self.fpr = round(self.fpr, 3)*100

            def find_fnr():
                self.fnr = self.fn/(self.fn + self.tp)
                self.fnr = round(self.fnr, 3)*100

            metric_funcs = ([find_tn, find_fn, find_fp, find_tp,
                             find_acc, find_prec, find_fpr, find_fnr])
            for func in metric_funcs:
                try:
                    func()
                except:
                    pass

            return [str(self.tp), str(self.fn), str(self.fp), str(self.tn), str(self.prec), str(self.fpr), str(self.fnr), str(self.acc)]

        if self.model_type == 'Classification':
            if filter_race == 'all':
                df = self.results
            elif filter_race == 'White':
                df = self.results.loc[(self.results['race'] == 'White') & (self.results['ethnicity'] == 'Non-Hispanic')]
            elif filter_race == 'Non-White':
                df = self.results.loc[(self.results['race'] != 'White') | (self.results['ethnicity'] != 'Non-Hispanic')]
            elif filter_race == 'Black':
                df = self.results.loc[(self.results['race'] == 'Black') & (self.results['ethnicity'] == 'Non-Hispanic')]
            else:
                df = self.results.loc[self.results['race'] == filter_race]

            matrix = create_confusion_matrix(df)
            #fpr, fnr = false_rates(matrix)

            #print(f'False Positive Rate {round(fpr, 4)}')
            #print(f'False Negative Rate {round(fnr, 4)}')
            return false_rates(matrix)
        else:
            pass
