from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import *
import plotly.express as px
import plotly.graph_objects as go
import pandas


STEPS_SLIDER_1 = "Steps_1"


def get_upload_layout(steps_slider_upload, steps_slider_fairness, steps_slider_mitigate,
                                        steps_slider_compare, upload_button, checklist_sensitive_features, sunburst_plot
                                        , dropdown_models, hired_graph, cm_women, cm_men, fairness_graph, PREVIEW_HOME):
    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.H1("Highlighting & Mitigating Discrimination in Job Hiring Scenarios"),
            steps_slider_upload,
            html.Br(),
            html.H3("Upload your own dataset"),
            upload_button,
            html.Hr(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
    ])
    # Return tab layout
    return layout
