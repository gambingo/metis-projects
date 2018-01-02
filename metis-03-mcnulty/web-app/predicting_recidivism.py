"""
J. Gambino
Metis Data Science Bootcamp
October 2017
---
Backend for the web-app demo.
Users can pick which features to build the model off of.
Then assess the model's fairness
"""
import sys
modules = '../python-scripts/'
sys.path.append(modules)

from mcnulty import *
from recidivism import recidivism
import flask


# Load entire clean dataset and extract column names
test, _, _, _ = pull_from_SQL(['*'])
test.drop(['case_weight',
           'index',
           'felony_count',
           'reoffend',
           'race',
           'convictions',
           'ethnicity'], axis=1, inplace=True)
available_features = list(test.columns)


# Initialize app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """Homepage for our Pary Prediction App"""
    with open("index.html", 'r') as viz_file:
        return viz_file.read()


@app.route("/build_radio_buttons", methods=["GET"])
def radio_buttons():
    """Sends list of available features to front end."""
    results = {"available_features": available_features}
    return flask.jsonify(results)


@app.route("/build_model", methods=["POST"])
def build():
    """Receives selected features and builds a model from them"""
    # print(flask.request.is_json)
    data = flask.request.get_json()
    selected_features = data["selected_features"]
    print(selected_features)

    x, y, race, ethnicity = pull_from_SQL(selected_features, table_name='clean')
    model = recidivism(web_app=True)
    model.build_model(x, y, race, ethnicity)
    print('Model Built')

    results = {}
    results['Everybody'] = model.score_model()
    results['White'] = model.score_model('White')
    results['Non-White'] = model.score_model('Non-White')
    results['Black'] = model.score_model('Black')
    print(results)

    return flask.jsonify(results)

app.run(host='0.0.0.0')
