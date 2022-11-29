import random

import aif360.sklearn.metrics
import numpy as np

from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff
from flask import app

import hiring_baseline as hbase

app = Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

graph = dcc.Graph(figure={})
title = dcc.Markdown(children='Discrimination in job hiring')
graph_chooser = dcc.Dropdown(options=['Bar graph'], value='Bar graph')


environment = hbase.setup_environment()
training_data = hbase.generate_training_data(environment, 100)
test_data = hbase.generate_test_data(environment, 100)
hbase.start_pipeline(training_data, test_data)

app.layout = dbc.Container([title, graph, graph_chooser])

@app.callback(
    Output(graph, 'figure'),
    Output(title, 'children'),
    Input(graph_chooser, 'value')
)

def update_graph(feature_name):
    fig = px.bar(data_frame=training_data, y=training_data[feature_name].value_counts())
    return fig, '#' + feature_name

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
