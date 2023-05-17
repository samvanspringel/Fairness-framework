from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import *
import plotly.express as px
import plotly.graph_objects as go
color_sequence = px.colors.qualitative.Safe

CHECKLIST_SENSITIVE_FEATURE_BASE = "Checklist_sensitive_feature_BASE"
sf_options = ["Gender", "Nationality", "Age", "Married"]


models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_BASE = "Dropdown_models_BASE"

CM_GRAPH = "Confusion matrix"

SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_BASE = "Amount_hired_gender_BASE"
GRAPH_FAIRNESS_NOTIONS_BASE = "Graph_fairness_notions_BASE"
GRAPH_ACCURACY_BASE = "Graph_accuracy_BASE"


def baseline_get_tab_dcc():
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_BASE, options=sf_options,
                                                 value=[sf_options[0]], inline=True, style={'display': 'block'})
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_BASE, options=models_options, value=models_options[1], clearable=False)
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_BASE)
    cm = dcc.Graph(id=CM_GRAPH)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_BASE)
    accuracy = dcc.Graph(id=GRAPH_ACCURACY_BASE)


    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.H1("Highlighting & Mitigating Discrimination in Job Hiring Scenarios"),
            html.Br(),
            html.H3("Data"),
            html.Hr(),
            dropdown_models,
            checklist_sensitive_features,
            html.H2("Visualisation"),
            html.Br(),
            horizontal_div([None, html.H4("\n \n  Distribution"),
                            html.P(" How the applicants are distributed based on sensitive features"),
                            None, html.H4("\n \n Qualified"),
                            html.P("How the applicants are distributed based on sensitive features")],
                           width=[None, width2, graph_width, None, width2, width],
                           space_width=space_width),
            horizontal_div([None, None, cm, None, None,  accuracy],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            hired_graph,
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


def baseline_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_BASE, "figure"),
         Output(CM_GRAPH, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_BASE, "figure"),
         Output(GRAPH_ACCURACY_BASE, "figure")],
        [Input(DROPDOWN_MODELS_BASE, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_BASE, "value")],
        suppress_callback_exceptions=True
    )
    def update(model, sensitive_features):

        # Base
        results = load_scenario('Base', sensitive_features, model)

        count_df = add_description_column(descriptive_age(descriptive_df(results['count_qualified_model'])),
                                          sensitive_features)
        fig_percentage_hired = px.bar(count_df, y='qualified', x='description', color_discrete_sequence=color_sequence)

        fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)

        fig_fairness = px.bar(results['fairness_notions'], y='Fairness notions',
                              color_discrete_sequence=color_sequence)


        fig_cm = px.imshow(results['confusion_matrix'],
                           labels=dict(x="Predicted", y="True"), x=classification_labels,
                           y=classification_labels,
                           text_auto=True, color_continuous_scale=color_sequence)

        fig_accuracy = px.bar(results['accuracy'], y="Model accuracy",
                              color_discrete_sequence=color_sequence)

        return [fig_percentage_hired, fig_cm, fig_fairness, fig_accuracy]
