from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import *
import plotly.express as px
color_sequence = px.colors.qualitative.Safe

PREVIEW_DD = "Preview_data_DD"

CHECKLIST_SENSITIVE_FEATURE_DD = "Checklist_sensitive_feature_DD"
sf_options = ["Gender", "Nationality", "Age", "Married"]

DROPDOWN_MODELS_DD = "Dropdown_models_DD"
models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]

CM_GRAPH = "Confusion_matrix_DD"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_DD = "Amount_hired_gender_DD "
GRAPH_FAIRNESS_DD = "Graph fairness notions_DD "
GRAPH_ACCURACY_DD = "Graph_fairness_DD"

GRAPH_AMOUNT_HIRED_DD_DATASET = "Graph fairness notions_DD_DATASET"
GRAPH_FAIRNESS_DD_DATASET = "Graph_fairness_DD_DATASET"

def dd_get_tab_dcc():
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_DD, options=sf_options,
                                                 value=[sf_options[0]], inline=True, style={'display': 'block'})
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_DD, options=models_options, value=models_options[1],
                                   clearable=False)
    hired_graph_model = dcc.Graph(id=GRAPH_AMOUNT_HIRED_DD)
    hired_graph_dataset = dcc.Graph(id=GRAPH_AMOUNT_HIRED_DD_DATASET)
    cm = dcc.Graph(id=CM_GRAPH)
    fairness_graph_model = dcc.Graph(id=GRAPH_FAIRNESS_DD)
    fairness_graph_dataset = dcc.Graph(id=GRAPH_FAIRNESS_DD_DATASET)
    accuracy = dcc.Graph(id=GRAPH_ACCURACY_DD)

    width = "30%"
    graph_width = "30%"
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

            html.H2("Model performance"),
            horizontal_div([None, None, cm, None, None, accuracy],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),

            html.H2("Proportional evaluation"),
            horizontal_div(
                [None, html.H4("\n \n  Dataset"), hired_graph_dataset, None, html.H4("\n \n  Model prediction"),
                 hired_graph_model],
                width=[None, width2, graph_width, None, width2, graph_width],
                space_width=space_width),
            html.H2("Fairness"),
            horizontal_div(
                [None, html.H4("\n \n  Dataset"), fairness_graph_dataset, None, html.H4("\n \n  Model prediction"),
                 fairness_graph_model],
                width=[None, width2, graph_width, None, width2, graph_width],
                space_width=space_width),
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
        [Output(GRAPH_AMOUNT_HIRED_DD, "figure"),
         Output(CM_GRAPH, "figure"), Output(GRAPH_FAIRNESS_DD, "figure"),
         Output(GRAPH_ACCURACY_DD, "figure"),
         Output(GRAPH_AMOUNT_HIRED_DD_DATASET, "figure"),
         Output(GRAPH_FAIRNESS_DD_DATASET, "figure")],
        [Input(DROPDOWN_MODELS_DD, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_DD, "value")],
        suppress_callback_exceptions=True
    )
    def update(model, sensitive_features):
        # DD
        results = load_scenario('Different distribution', sensitive_features, model)

        count_df = add_description_column(descriptive_age(descriptive_df(results['count_qualified_model'])),
                                          sensitive_features)

        count_df_dataset = add_description_column(descriptive_age(descriptive_df(results['count_qualified_dataset'])),
                                                  sensitive_features)

        fig_percentage_hired_model = px.bar(count_df, y='qualified', x='description',
                                            error_y='Standard deviation',
                                            color_discrete_sequence=color_sequence)

        fig_percentage_hired_model.update_layout(yaxis_title="Percentage qualified", autosize=False)

        fig_percentage_hired_dataset = px.bar(count_df_dataset, y='qualified', x='description',
                                              error_y='Standard deviation',
                                              color_discrete_sequence=color_sequence)
        fig_percentage_hired_dataset.update_layout(yaxis_title="Percentage qualified", autosize=False)

        fig_fairness_model = px.bar(results['fairness_notions_model'], y='Mean', error_y='Standard deviation',
                                    color_discrete_sequence=color_sequence)

        fig_fairness_dataset = px.bar(results['fairness_notions_dataset'], y='Mean', error_y='Standard deviation',
                                      color_discrete_sequence=color_sequence)

        fig_cm = px.imshow(results['confusion_matrix'],
                           labels=dict(x="Predicted", y="True"), x=classification_labels,
                           y=classification_labels,
                           text_auto=True, color_continuous_scale=color_sequence)

        fig_accuracy = px.bar(results['accuracy'], y="Model accuracy",
                              color_discrete_sequence=color_sequence)

        return [fig_percentage_hired_model, fig_cm, fig_fairness_model, fig_accuracy, fig_percentage_hired_dataset,
                fig_fairness_dataset]
