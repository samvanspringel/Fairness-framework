from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div
from Process_data import *
import plotly.express as px
import plotly.graph_objects as go
import pandas

STEPS_SLIDER = "Steps"

data_uploaded = False
uploaded_dataset = None
steps = {
    0: "Upload",
    1: "Check fairness",
    2: "Mitigate bias",
    3: "Compare"
}

PREVIEW_HOME = "Preview_data_HOME"

UPLOADED_DATASET = "uploaded-dataset"

CHECKLIST_SENSITIVE_FEATURE_HOME = "Checklist_sensitive_feature_HOME"
sf_options = ["Gender", "Origin"]

models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_HOME = "Dropdown_models_HOME "

CM_GRAPH_WOMEN_HOME = "Confusion_matrix_model_women_HOME"
CM_GRAPH_MEN_HOME = "Confusion_matrix_model_men_HOME"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_HOME = "Amount_hired_gender_HOME"
GRAPH_FAIRNESS_NOTIONS_HOME = "Graph_fairness_notions_HOME"
GRAPH_SUNBURST_HOME = "Graph_sunburst_HOME"

results = {}


def preview_df(df):
    df['gender'] = df['gender'].replace({1.0: "Male", 2.0: "Female"})
    df['origin'] = df['origin'].replace({1.0: "Belgian", 2.0: "FB", 3.0: "Foreign"})
    df['degree'] = df['degree'].replace({1.0: "Yes", 0.0: "No"})
    df['extra_degree'] = df['extra_degree'].replace({1.0: "Yes", 0.0: "No"})
    return df


def home_get_tab_dcc():
    steps_slider = dcc.Slider(id=STEPS_SLIDER, disabled=True, marks=steps, value=0)
    upload_button = dcc.Upload(
        id=UPLOADED_DATASET,
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'display': 'block'
        },
        # Allow multiple files to be uploaded
        multiple=False)
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_HOME, options=sf_options,
                                                 value=[sf_options[0]], inline=True, style= {'display': 'block'})
    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST_HOME, style= {'display': 'block'})
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_HOME, options=models_options, value=models_options[1],
                                   clearable=False, style= {'display': 'block'})
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_HOME)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_HOME)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_HOME)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_HOME, style= {'display': 'block'})

    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.H1("Highlighting & Mitigating Discrimination in Job Hiring Scenarios"),
            steps_slider,
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


def home_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_HOME, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_HOME, "figure"),
         Output(GRAPH_SUNBURST_HOME, "figure"), Output(PREVIEW_HOME, "data")],
        [Input(DROPDOWN_MODELS_HOME, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_HOME, "value"),
         Input(UPLOADED_DATASET, "filename")]
    )
    def update(model, sensitive_features, dataset_file_name):

        # Home
        global data_uploaded, uploaded_dataset, df_count_hired_model, results, preview

        if dataset_file_name is not None:
            data_uploaded = True
            uploaded_dataset = pandas.read_csv(dataset_file_name)
            preview = uploaded_dataset.sample(10).to_dict("records")
            results = pipeline(model, uploaded_dataset, sensitive_features)

            #fig_percentage_hired = px.histogram(results['count_qualified_model'], y='qualified')
            fig_percentage_hired = px.bar(results['count_qualified_model'], y='qualified')

            fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)

            fig_fairness = go.Figure(
            [go.Bar(x=['Statistical parity', 'Predictive equality', 'Equal opportunity', 'Accuracy'],
                    y=results['fairness_notions_model'])])

            fig_sunburst = px.sunburst(results['model_prediction'],
                                   path=['gender', 'origin'], values='qualified')

        return [fig_percentage_hired, fig_fairness, fig_sunburst, preview]

