from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import load_scenario
import plotly.express as px
import plotly.graph_objects as go

PREVIEW_DD = "Preview_data_DD"

CHECKLIST_SENSITIVE_FEATURE_DD = "Checklist_sensitive_feature_DD"
sf_options = ["Gender", "Origin"]

DROPDOWN_MODELS_DD = "Dropdown_models_DD"
models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]

CM_GRAPH_WOMEN_DD = "Confusion_matrix_model_women_DD "
CM_GRAPH_MEN_DD = "Confusion_matrix_model_men_DD "
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_DD = "Amount_hired_gender_DD "
GRAPH_FAIRNESS_NOTIONS_DD = "Graph fairness notions_DD "
GRAPH_SUNBURST_DD = "Graph_sunburst_DD"


def dd_get_tab_dcc():

    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_DD, options=models_options, value=models_options[1],
                                   clearable=False)
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_DD, options=sf_options, value=[sf_options[0]])
    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST_DD)
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_DD)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_DD)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_DD)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_DD)

    width = "30%"
    width2 = "10%"
    space_width = "2.5%"

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.Br(),
            html.H3("Data"),
            html.Hr(),
            horizontal_div([None, html.H4("Machine learning model"), dropdown_models, None,
                            html.H4("Sensitive feature"), checklist_sensitive_features],
                           width=[None, width2, width, None, width2, width],
                           space_width=space_width),
            html.H3("Sample"),
            html.P("Sample of 10 applicants"),
            dash_table.DataTable(id=PREVIEW_DD),
            horizontal_div([None, html.H4("Distribution"), sunburst_plot, None, html.H4("Qualified"), hired_graph],
                           width=[None, width2, width, None, width2, width],
                           space_width=space_width),
            horizontal_div([None, html.H4("Confusion matrix women"), cm_women, None,
                            html.H4("Confusion matrix men"), cm_men],
                           width=[None, width2, width, None, width2, width],
                           space_width=space_width), fairness_graph,
            html.Hr(),
        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
    ])

    # Return tab layout
    return layout


def dd_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_DD , "figure"), Output(CM_GRAPH_WOMEN_DD , "figure"),
         Output(CM_GRAPH_MEN_DD , "figure"), Output(GRAPH_FAIRNESS_NOTIONS_DD , "figure"),
         Output(GRAPH_SUNBURST_DD, "figure"), Output(PREVIEW_DD, "data")],
        [Input(DROPDOWN_MODELS_DD , "value"), Input(CHECKLIST_SENSITIVE_FEATURE_DD, "value")]
    )
    def update(model, sensitive_features):
        # Different distribution
        scenarios_elements = load_scenario('Different distribution', sensitive_features)

        fig_percentage_hired = px.histogram(scenarios_elements['Different distribution'][model]['df-hired'],
                                            x=['Women', 'Men'], y='qualified',
                                            labels={'x': 'Gender', 'y': 'Amount qualified'})
        fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)
        fig_cm_women = px.imshow(scenarios_elements['Different distribution'][model]['cm-women-hired'],
                                 labels=dict(x="Predicted", y="True"), x=classification_labels,
                                 y=classification_labels,
                                 text_auto=True)
        fig_cm_women.update_layout(autosize=False, width=400)
        fig_cm_men = px.imshow(scenarios_elements['Different distribution'][model]['cm-men-hired'],
                               labels=dict(x="Predicted", y="True"), x=classification_labels,
                               y=classification_labels,
                               text_auto=True)
        fig_cm_men.update_layout(autosize=False, width=400)
        fig_fairness = go.Figure(
            [go.Bar(x=['Statistical parity', 'Predictive equality', 'Equal opportunity', 'Accuracy'],
                    y=scenarios_elements['Different distribution'][model]['fairness'])])

        fig_sunburst = px.sunburst(scenarios_elements['Different distribution'][model]['df'],
                                   path=['gender', 'origin'], values='qualified')

        table = scenarios_elements['Different distribution'][model]['df'].sample(10).to_dict("records")

        return [fig_percentage_hired, fig_cm_women, fig_cm_men, fig_fairness, fig_sunburst, table]
