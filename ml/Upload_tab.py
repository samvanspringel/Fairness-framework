from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
from Tabs import horizontal_div, TAB_UPLOAD
from Upload_layout import get_upload_layout
from Dataset_uploaded_layout import get_dataset_uploaded_layout
from Process_data import *
import plotly.express as px
import plotly.graph_objects as go
import pandas

STEPS_SLIDER = "Steps"

data_uploaded = False
uploaded_dataset = None
step_functions = {
    0: get_upload_layout,
    1: get_dataset_uploaded_layout,
    2: "Mitigate bias",
    3: "Compare"
}

steps = {
    0: "Upload",
    1: "Fairness",
    2: "Mitigate bias",
    3: "Compare"
}

PREVIEW_HOME = "Preview_data_UPLOAD"

UPLOADED_DATASET = "uploaded-dataset"

CHECKLIST_SENSITIVE_FEATURE_HOME = "Checklist_sensitive_feature_UPLOAD"
sf_options = ["Gender", "Origin"]

models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_HOME = "Dropdown_models_UPLOAD"

CM_GRAPH_WOMEN_HOME = "Confusion_matrix_model_women_UPLOAD"
CM_GRAPH_MEN_HOME = "Confusion_matrix_model_men_UPLOAD"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED_HOME = "Amount_hired_gender_UPLOAD"
GRAPH_FAIRNESS_NOTIONS_HOME = "Graph_fairness_notions_UPLOAD"
GRAPH_SUNBURST_HOME = "Graph_sunburst_UPLOAD"

results = {}

current_step = 0
def preview_df(df):
    df['gender'] = df['gender'].replace({1.0: "Male", 2.0: "Female"})
    df['origin'] = df['origin'].replace({1.0: "Belgian", 2.0: "FB", 3.0: "Foreign"})
    df['degree'] = df['degree'].replace({1.0: "Yes", 0.0: "No"})
    df['extra_degree'] = df['extra_degree'].replace({1.0: "Yes", 0.0: "No"})
    return df


def upload_get_tab_dcc():

    steps_slider_upload = dcc.Slider(id=STEPS_SLIDER, disabled=True, marks=steps, value=0)
    steps_slider_fairness = dcc.Slider(id=STEPS_SLIDER, disabled=True, marks=steps, value=1)
    steps_slider_mitigate = dcc.Slider(id=STEPS_SLIDER, disabled=True, marks=steps, value=2)
    steps_slider_compare = dcc.Slider(id=STEPS_SLIDER, disabled=True, marks=steps, value=3)

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
                                                 value=[sf_options[0]], inline=True, style={'display': 'block'})
    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST_HOME, style={'display': 'block'})
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_HOME, options=models_options, value=models_options[1],
                                   clearable=False, style={'display': 'block'})
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED_HOME)
    cm_women = dcc.Graph(id=CM_GRAPH_WOMEN_HOME)
    cm_men = dcc.Graph(id=CM_GRAPH_MEN_HOME)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_HOME, style={'display': 'block'})
    return step_functions[current_step](steps_slider_upload, steps_slider_fairness, steps_slider_mitigate,
                                        steps_slider_compare, upload_button, checklist_sensitive_features, sunburst_plot
                                        , dropdown_models, hired_graph, cm_women, cm_men, fairness_graph, PREVIEW_HOME)


def upload_get_app_callbacks(app):
    @app.callback(
        [Output(GRAPH_AMOUNT_HIRED_HOME, "figure"), Output(GRAPH_FAIRNESS_NOTIONS_HOME, "figure"),
         Output(GRAPH_SUNBURST_HOME, "figure"), Output(PREVIEW_HOME, "data")],
        [Input(DROPDOWN_MODELS_HOME, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_HOME, "value"),
         Input(UPLOADED_DATASET, "filename")]
    )
    def update(model, sensitive_features, dataset_file_name):

        # Home
        global data_uploaded, uploaded_dataset, df_count_hired_model, results, preview, current_step
        if dataset_file_name is not None:
            data_uploaded = True
            uploaded_dataset = pandas.read_csv(dataset_file_name)
            results = pipeline(model, uploaded_dataset, sensitive_features)

            preview = descriptive_df(uploaded_dataset).sample(10).to_dict("records")

            current_step += 1


            #fig_percentage_hired = px.histogram(results['count_qualified_model'], y='qualified')

            count_df = add_description_column(descriptive_df(results['count_qualified_model']), sensitive_features)

            fig_percentage_hired = px.bar(count_df, y='qualified', x='description')

            fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)

            fig_fairness = go.Figure(
            [go.Bar(x=['Statistical parity', 'Predictive equality', 'Equal opportunity', 'Accuracy'],
                    y=results['fairness_notions_model'])])

            fig_sunburst = px.sunburst(results['model_prediction'],
                                   path=['gender', 'origin'], values='qualified')
        return [fig_percentage_hired, fig_fairness, fig_sunburst, preview]

