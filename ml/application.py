import random

import aif360.sklearn.metrics

# Modellen
from sklearn import svm
from sklearn import tree
import numpy as np

from dash import Dash, html, dcc, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from flask import app

import hiring_ml as hire

# Base scenario
def load_scenario(scenario):
    environment = hire.setup_environment(scenario)
    training_data = hire.generate_training_data(environment, 100)
    test_data = hire.rename_goodness(hire.generate_test_data(environment, 100))

    trained_models = hire.train_models(training_data, models)
    predictions = hire.make_predictions(test_data, trained_models)

    dataframes_count_hired = hire.count_hired(predictions)
    cm = hire.generate_cm(predictions)

    scenarios_elements[scenario] = {'Human': {'cm-women-hired': cm[0],
                                              'cm-men-hired': cm[1],
                                              'df-hired': dataframes_count_hired[0]},
                                    'Decision tree': {'cm-women-hired': cm[2],
                                                      'cm-men-hired': cm[3],
                                                      'df-hired': dataframes_count_hired[1]},
                                    'Support vector machines': {'cm-women-hired': cm[4],
                                                                'cm-men-hired': cm[5],
                                                                'df-hired': dataframes_count_hired[2]},
                                    'Preview': hire.make_preview(test_data),
                                    }


app = Dash(__name__)

# Gebruikte modellen
models = [svm.SVC(), tree.DecisionTreeClassifier()]
scenarios_elements = {}

PREVIEW = "Preview_data"
DROPDOWN_MODELS = "Dropdown_models"
models_options = ["Human", "Decision tree", "Support vector machines"]
CM_GRAPH_WOMEN = "Confusion_matrix_model_women"
CM_GRAPH_MEN = "Confusion_matrix_model_men"
SCENARIO = "Scenario"
classification_labels = ["Not hired", "Hired"]
GRAPH_AMOUNT_HIRED = "Amount_hired_gender"

# Base
load_scenario('Base')

# Different distribution
load_scenario('Different distribution')

# Bias scenario
load_scenario('Bias')

# Initialisatie figuren
preview = scenarios_elements['Base']['Preview'].to_dict('records')
fig_amount_hired = px.histogram(scenarios_elements['Base']['Human']['df-hired'],
                                x=['Women', 'Men'], y='hired', labels={'x': 'Gender', 'y': 'Amount hired'})

fig_cm_women = px.imshow(scenarios_elements['Base']['Human']['cm-women-hired'],
                         labels=dict(x="Predicted", y="True"), x=classification_labels,
                         y=classification_labels, text_auto=True)
fig_cm_men = px.imshow(scenarios_elements['Base']['Human']['cm-men-hired'],
                       labels=dict(x="Predicted", y="True"), x=classification_labels,
                       y=classification_labels, text_auto=True)

app.layout = html.Div([
    html.H1("Highlighting and Mitigating Discrimination in Job Hiring Scenarios"),
    dcc.RadioItems(id=SCENARIO, options=['Base', 'Different distribution', 'Bias'], value='Base', inline=True),
    html.H4("Simulated applicants"),
    html.P("The following table is a sample of simulated applicants for a job. "
           "The purpose is to train a machine learning model on a hidden dataset and subsequently make a prediction for"
           " the following applicants whether or not they would be hired by the model."),
    html.H4("Train models"),
    dcc.Dropdown(id=DROPDOWN_MODELS, options=models_options, value=models_options[0], clearable=False),
    html.H4("Amount hired"),
    dcc.Graph(id=GRAPH_AMOUNT_HIRED, figure=fig_amount_hired),
    html.H4("Confusion matrix women"),
    dcc.Graph(id=CM_GRAPH_WOMEN, figure=fig_cm_women),
    html.H4("Confusion matrix men"),
    dcc.Graph(id=CM_GRAPH_MEN, figure=fig_cm_men),
])


@app.callback(
    [Output(GRAPH_AMOUNT_HIRED, "figure"), Output(CM_GRAPH_WOMEN, "figure"),
     Output(CM_GRAPH_MEN, "figure")],
    [Input(SCENARIO, "value"), Input(DROPDOWN_MODELS, "value")]
)
def update(scenario, model):
    global fig_cm_women, fig_cm_men, fig_amount_hired
    fig_amount_hired = px.histogram(scenarios_elements[scenario][model]['df-hired'],
                                    x=['Women', 'Men'], y='hired',
                                    labels={'x': 'Gender', 'y': 'Amount hired'})
    fig_cm_women = px.imshow(scenarios_elements[scenario][model]['cm-women-hired'],
                             labels=dict(x="Predicted", y="True"), x=classification_labels,
                             y=classification_labels,
                             text_auto=True)
    fig_cm_men = px.imshow(scenarios_elements[scenario][model]['cm-men-hired'],
                           labels=dict(x="Predicted", y="True"), x=classification_labels,
                           y=classification_labels,
                           text_auto=True)

    return [fig_amount_hired, fig_cm_women, fig_cm_men]


if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
