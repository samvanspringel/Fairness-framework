from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import load_scenario
import plotly.express as px
import plotly.graph_objects as go

PREVIEW_BASE = "Preview_data_BASE"

CHECKLIST_SENSITIVE_FEATURE_BASE = "Checklist_sensitive_feature_BASE"
sf_options = ["Gender", "Origin"]


models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_BASE = "Dropdown_models_BASE"

CM_GRAPH_WOMEN_BASE = "Confusion_matrix_model_women_BASE"
CM_GRAPH_MEN_BASE = "Confusion_matrix_model_men_BASE"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_BASE = "Amount_hired_gender_BASE"
GRAPH_FAIRNESS_NOTIONS_BASE = "Graph_fairness_notions_BASE"
GRAPH_SUNBURST_BASE = "Graph_sunburst_BASE"


def baseline_get_tab_dcc():
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_BASE, options=sf_options, value=[sf_options[0]])
    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST_BASE)
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_BASE, options=models_options, value=models_options[1], clearable=False)
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_BASE)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_BASE)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_BASE)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_BASE)

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
            horizontal_div([None, html.H5("Machine learning model"), dropdown_models, None,
                            html.H5("Sensitive feature"), checklist_sensitive_features],
                           width=[None, width2, width, None, width2, width],
                           space_width=space_width),
            html.Br(),
            html.H2("Data sample"),
            html.P("Sample of 10 applicants"),
            dash_table.DataTable(id=PREVIEW_BASE),
            html.Br(),
            html.H2("Visualisation"),
            html.Br(),
            horizontal_div([None, html.H4("\n \n  Distribution"),
                            html.P(" How the applicants are distributed based on sensitive features"),
                            None, html.H4("\n \n Qualified"),
                            html.P("How the applicants are distributed based on sensitive features")],
                           width=[None, width2, graph_width, None, width2, width],
                           space_width=space_width),
            horizontal_div([None, None, sunburst_plot, None, None,  hired_graph],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            horizontal_div([None, html.H4("Confusion matrix women"), cm_women, None,
                            html.H4("Confusion matrix men"), cm_men],
                           width=[None, width2, width, None, width2, width],
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


def baseline_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_BASE, "figure"), Output(CM_GRAPH_WOMEN_BASE, "figure"),
         Output(CM_GRAPH_MEN_BASE, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_BASE, "figure"),
         Output(GRAPH_SUNBURST_BASE, "figure"), Output(PREVIEW_BASE, "data")],
        [Input(DROPDOWN_MODELS_BASE, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_BASE, "value")],
        suppress_callback_exceptions=True
    )
    def update(model, sensitive_features):

        # Base
        scenarios_elements = load_scenario('Base', sensitive_features, model)

        fig_percentage_hired = px.histogram(scenarios_elements['Base'][model]['df-hired'],
                                            x=['Women', 'Men'], y='qualified',
                                            labels={'x': 'Gender', 'y': 'Amount qualified'})
        fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)
        fig_cm_women = px.imshow(scenarios_elements['Base'][model]['cm-women-hired'],
                                 labels=dict(x="Predicted", y="True"), x=classification_labels,
                                 y=classification_labels,
                                 text_auto=True)
        fig_cm_women.update_layout(autosize=False, width=400)
        fig_cm_men = px.imshow(scenarios_elements['Base'][model]['cm-men-hired'],
                               labels=dict(x="Predicted", y="True"), x=classification_labels,
                               y=classification_labels,
                               text_auto=True)
        fig_cm_men.update_layout(autosize=False, width=400)
        fig_fairness = go.Figure(
            [go.Bar(x=['Statistical parity', 'Predictive equality', 'Equal opportunity', 'Accuracy'],
                    y=scenarios_elements['Base'][model]['fairness'])])

        fig_sunburst = px.sunburst(scenarios_elements['Base'][model]['df'],
                                   path=['gender', 'origin'], values='qualified')

        table = scenarios_elements['Base'][model]['df'].sample(10).to_dict("records")

        return [fig_percentage_hired, fig_cm_women, fig_cm_men, fig_fairness, fig_sunburst, table]
