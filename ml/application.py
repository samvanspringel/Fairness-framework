import random

import aif360.sklearn.metrics

# Modellen
from sklearn import svm
from sklearn import tree
import numpy as np

from dash import Dash, html, dcc, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from flask import app

import hiring_baseline as hbase

app = Dash(__name__)

DROPDOWN_MODELS = "Dropdown_models"
models_options = ["Human", "Decision tree", "Support vector machines"]
CM_GRAPH_WOMEN = "Confusion_matrix_model_women"
CM_GRAPH_MEN = "Confusion_matrix_model_men"

environment = hbase.setup_environment()
training_data = hbase.generate_training_data(environment, 100)
test_data = hbase.generate_test_data(environment, 100)
preview = hbase.make_preview(test_data)

# Gebruikte modellen
models = [svm.SVC(), tree.DecisionTreeClassifier()]
trained_models = hbase.train_models(training_data, models)
cm = hbase.generate_cm(test_data, trained_models)

classification_labels = ["Hired", "Not hired"]
fig_cm_women = px.imshow(cm[0], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)
fig_cm_men = px.imshow(cm[1], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)

app.layout = html.Div([
    html.H1("Highlighting and Mitigating Discrimination in Job Hiring Scenarios"),
    html.H4("Simulated applicants"),
    html.P("The following table is a sample of simulated applicants for a job. "
           "The purpose is to train a machine learning model on a hidden dataset and subsequently make a prediction for"
           "the following applicants whether or not they would be hired by the model."),
    dash_table.DataTable(
        data=preview.to_dict('records'),
        columns=[{"name": i, "id": i} for i in preview.columns]
    ),
    html.H4("Train models"),
    dcc.Dropdown(id=DROPDOWN_MODELS, options=models_options, value=models_options[0], clearable=False),
    html.H4("Current prediction"),
    html.P("..."),

    html.H4("Confusion matrix women"),
    dcc.Graph(id=CM_GRAPH_WOMEN, figure=fig_cm_women),
    html.H4("Confusion matrix men"),
    dcc.Graph(id=CM_GRAPH_MEN, figure=fig_cm_men),
])

@app.callback(
    [Output(CM_GRAPH_WOMEN, "figure"), Output(CM_GRAPH_MEN, "figure")],
    [Input(DROPDOWN_MODELS, "value")]
)
def update_cm(model):
    global fig_women, fig_men
    if model == "Human":
        fig_women = px.imshow(cm[0], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)
        fig_men = px.imshow(cm[1], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)

    elif model == "Decision tree":
        fig_women = px.imshow(cm[2], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)
        fig_men = px.imshow(cm[3], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)

    elif model == "Support vector machines":
        fig_women = px.imshow(cm[4], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)
        fig_men = px.imshow(cm[5], labels=dict(x="Predicted", y="True"), x=classification_labels, y=classification_labels,
                   text_auto=True)

    return [fig_women, fig_men]

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
