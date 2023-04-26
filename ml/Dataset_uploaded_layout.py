from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import *
from Upload_tab import *
import plotly.express as px
import plotly.graph_objects as go
import pandas


def get_dataset_uploaded_layout(steps_slider_upload, steps_slider_fairness, steps_slider_mitigate,
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
            steps_slider_fairness,
            html.Br(),
            html.H3("Upload your own dataset"),
            upload_button,
            html.Hr(),
            html.H2("Data sample"),
            html.P("Here is a sample of your uploaded data"),
            dash_table.DataTable(id=PREVIEW_HOME),
            horizontal_div([None, html.H5("Machine learning model"), dropdown_models, None,
                            html.H5("Sensitive feature"), checklist_sensitive_features],
                           width=[None, width2, width, None, width2, width],
                           space_width=space_width),
            html.Br(),
            html.H2("Visualisation"),
            html.Br(),
            horizontal_div([None, html.H4("\n \n  Distribution"),
                            html.P(" How the applicants are distributed based on sensitive features"),
                            None, html.H4("\n \n Qualified"),
                            html.P("How the applicants are distributed based on sensitive features")],
                           width=[None, width2, graph_width, None, width2, width],
                           space_width=space_width),
            horizontal_div([None, None, sunburst_plot, None, None, hired_graph],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.Br(),
            html.H2("Fairness"),
            fairness_graph,
            html.Hr(),
        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
    ])

    # Return tab layout
    return layout
