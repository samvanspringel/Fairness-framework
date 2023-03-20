from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
import plotly.express as px
import plotly.graph_objects as go

PREVIEW = "Preview_data"
DROPDOWN_MODELS_BASE = "Dropdown_models_BASE"
models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
CM_GRAPH_WOMEN_BASE = "Confusion_matrix_model_women_BASE"
CM_GRAPH_MEN_BASE = "Confusion_matrix_model_men_BASE"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]
GRAPH_AMOUNT_HIRED_BASE = "Amount_hired_gender_BASE"
GRAPH_FAIRNESS_NOTIONS_BASE = "Graph_fairness_notions_BASE"


def baseline_get_tab_dcc(scenarios_elements):
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_BASE, options=models_options, value=models_options[0], clearable=False)
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_BASE)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_BASE)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_BASE)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_BASE)

    width = "30%"
    width2 = "6%"
    space_width = "2.5%"

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.Br(),
            html.H3("Data"),
            html.Hr(),
            dropdown_models, hired_graph, cm_women, cm_men, fairness_graph,
            html.Hr(),
        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        # dash_table.DataTable(id=HOME_TABLE),
        html.Br(),
    ])

    # Return tab layout
    return layout


def baseline_get_app_callbacks(app, scenarios_elements):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_BASE, "figure"), Output(CM_GRAPH_WOMEN_BASE, "figure"),
         Output(CM_GRAPH_MEN_BASE, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_BASE, "figure")],
        [Input(DROPDOWN_MODELS_BASE, "value")]
    )
    def update(model):

        fig_percentage_hired = px.histogram(scenarios_elements['Base'][model]['df-hired'],
                                            x=['Women', 'Men'], y='hired',
                                            labels={'x': 'Gender', 'y': 'Amount hired'}, width=700)
        fig_percentage_hired.update_layout(yaxis_title="Percentage hired", autosize=False)
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

        return [fig_percentage_hired, fig_cm_women, fig_cm_men, fig_fairness]
