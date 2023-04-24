from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import load_scenario
import plotly.express as px
import plotly.graph_objects as go

PREVIEW_BIAS = "Preview_data_BIAS"

CHECKLIST_SENSITIVE_FEATURE_BIAS = "Checklist_sensitive_feature_BIAS"
sf_options = ["Gender", "Origin"]

DROPDOWN_MODELS_BIAS = "Dropdown_models_BIAS"
models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]

CM_GRAPH_WOMEN_BIAS = "Confusion_matrix_model_women_BIAS"
CM_GRAPH_MEN_BIAS = "Confusion_matrix_model_men_BIAS"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_BIAS = "Amount_hired_gender_BIAS"
GRAPH_FAIRNESS_NOTIONS_BIAS = "Graph_fairness_notions_BIAS"
GRAPH_SUNBURST_BIAS = "Graph_sunburst_BIAS"


def bias_get_tab_dcc():
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_BIAS, options=models_options, value=models_options[1],
                                   clearable=False)
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_BIAS, options=sf_options,
                                                 value=[sf_options[0]])
    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST_BIAS)
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_BIAS)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_BIAS)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_BIAS)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_BIAS)

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
            dash_table.DataTable(id=PREVIEW_BIAS),
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


def bias_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_BIAS, "figure"), Output(CM_GRAPH_WOMEN_BIAS, "figure"),
         Output(CM_GRAPH_MEN_BIAS, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_BIAS, "figure"),
         Output(GRAPH_SUNBURST_BIAS, "figure"), Output(PREVIEW_BIAS, "data")],
        [Input(DROPDOWN_MODELS_BIAS, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_BIAS, "value")], suppress_callback_exceptions=True
    )
    def update(model, sensitive_features):

        # Bias scenario
        scenarios_elements = load_scenario('Bias', sensitive_features)

        fig_percentage_hired = px.histogram(scenarios_elements['Bias'][model]['df-hired'],
                                            x=['Women', 'Men'], y='qualified',
                                            labels={'x': 'Gender', 'y': 'Amount qualified'})
        fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)
        fig_cm_women = px.imshow(scenarios_elements['Bias'][model]['cm-women-hired'],
                                 labels=dict(x="Predicted", y="True"), x=classification_labels,
                                 y=classification_labels,
                                 text_auto=True)
        fig_cm_women.update_layout(autosize=False, width=400)
        fig_cm_men = px.imshow(scenarios_elements['Bias'][model]['cm-men-hired'],
                               labels=dict(x="Predicted", y="True"), x=classification_labels,
                               y=classification_labels,
                               text_auto=True)
        fig_cm_men.update_layout(autosize=False, width=400)
        fig_fairness = go.Figure(
            [go.Bar(x=['Statistical parity', 'Predictive equality', 'Equal opportunity', 'Accuracy'],
                    y=scenarios_elements['Bias'][model]['fairness'])])

        fig_sunburst = px.sunburst(scenarios_elements['Bias'][model]['df'],
                                   path=['gender', 'origin'], values='qualified')

        table = scenarios_elements['Bias'][model]['df'].sample(10).to_dict("records")

        return [fig_percentage_hired, fig_cm_women, fig_cm_men, fig_fairness, fig_sunburst, table]
