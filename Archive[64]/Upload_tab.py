from dash import dcc, html, dash_table
from dash.dependencies import Input, Output

from Tabs import horizontal_div, TAB_UPLOAD
from Process_data import sensitive_feature_mapping
from Upload_layout import get_upload_layout
from Dataset_uploaded_layout import get_dataset_uploaded_layout
from Process_data import *
import plotly.express as px
import plotly.graph_objects as go
import pandas

# TODO
from dash.exceptions import PreventUpdate
import base64
import io

# TODO: ik stel voor dat je ofwel met de tabs werkt, ofwel zonder en op een andere manier de layout telkens verandert.
#  Hier heb je eigelijk de twee door elkaar gedaan. Ik zou aanraden om het met eventueel op de manier hieronder te doen
#  en je tabs behouden voor de start van de upload en je experimenten.

STEPS_SLIDER = "Steps"

data_uploaded = False
uploaded_dataset = None
model = None
sensitive_features = None
color_sequence = px.colors.qualitative.Safe

steps = {
    0: "Upload",
    1: "Setup",
    2: "Train model",  # TODO: added
    3: "Fairness",
    4: "Mitigate bias",
    5: "Compare"
}

# TODO: maak een aparte div aan voor in je tab om gemakkelijk in de huidige tab van layout te veranderen
UPLOAD_DIV = "upload_div"
NEXT_BUTTON_UPLOAD = "next_upload"
NEXT_BUTTON_TRAINING_RESULTS = "next_training_results"
NEXT_BUTTON_FAIRNESS = "next_fairness"
NEXT_BUTTON_MITIGATION_CHOICE = "next_mitigation_choice"

# TODO: zet in deze file/tab enkel de elementen en functies die door deze tab gebruikt worden
PREVIEW_HOME = "Preview_data_UPLOAD"

UPLOADED_DATASET = "uploaded-dataset"

CHECKLIST_SENSITIVE_FEATURE_HOME = "Checklist_sensitive_feature_UPLOAD"
sf_options = ["Gender", "Nationality", "Age", "Married"]

models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_HOME = "Dropdown_models_UPLOAD"

DROPDOWN_MITIGATION = "Dropdown_mitigation"
mitigation_options = ["Pre-processing: Sample Reweighing", "Post-processing: Calibrated Equalized Odds"]

CM_GRAPH = "Confusion_matrix"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED = "Amount_hired_gender_UPLOAD"
GRAPH_FAIRNESS_NOTIONS = "Graph_fairness_notions_UPLOAD"
GRAPH_SUNBURST = "Graph_sunburst_UPLOAD"
GRAPH_ACCURACY = "Graph_accuracy"

results = {}
results_mitigation = {}


def upload_get_tab_dcc():
    # TODO: beginpunt: dit is de eerste functie die opgeroepen wordt bij je tab
    steps_slider_upload = dcc.Slider(id=STEPS_SLIDER, disabled=True,
                                     marks=steps, value=0,
                                     min=min(steps.keys()),
                                     max=max(steps.keys()))  # TODO: dit is blijkbaar nodig om te updaten
    # TODO: ik denk dat je wel de style moet aanpassen aanezien de slider in het grijs wordt ingevuld

    layout = html.Div([
        ### Header under tab ###
        html.Div([  # TODO: dit is de volledige layout binnen je tab
            html.H1("Highlighting & Mitigating Discrimination in Job Hiring Scenarios"),
            steps_slider_upload,
            # TODO: zorg dat deze buiten de veranderende layout staat, zodat je heb telkens kan aanpassen
            html.Br(),
            html.Div(
                # TODO: dit is de layout waarbinnen je een dataset inlaadt, plots aanmaakt, ...
                #  Deze is degene die we telkens updaten/vervangen later
                step_upload(),
                id=UPLOAD_DIV),
        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
    ])
    # Return tab layout
    return layout


def step_upload():
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

    layout_children = [html.H3("Upload your own dataset"),
                       upload_button,
                       html.Hr(), ]
    return layout_children


def step_preview_select():
    checklist_sensitive_features = dcc.Checklist(id=CHECKLIST_SENSITIVE_FEATURE_HOME, options=sf_options,
                                                 value=[sf_options[0]], inline=True, style={'display': 'block'})
    dropdown_models = dcc.Dropdown(id=DROPDOWN_MODELS_HOME, options=models_options, value=models_options[1],
                                   clearable=False, style={'display': 'block'})

    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H3(f"Data preview"),
            dash_table.DataTable(id=PREVIEW_HOME, data=preview),
            html.Br(),
            checklist_sensitive_features,
            html.Br(),
            html.H3("Choose model"),
            dropdown_models,
            html.Hr(),
        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
        html.Button(id=NEXT_BUTTON_UPLOAD, children="Next")
    ]
    # Return tab layout
    return layout_children


def step_results_model():
    global results, sensitive_features

    fig_cm = px.imshow(results['confusion_matrix'][0],
                       labels=dict(x="Predicted", y="True"), x=classification_labels,
                       y=classification_labels,
                       text_auto=True, color_continuous_scale=color_sequence)

    fig_accuracy = px.bar(results['confusion_matrix'][1], y="Model accuracy", color_discrete_sequence=color_sequence)

    cm = dcc.Graph(id=CM_GRAPH, figure=fig_cm)
    accuracy = dcc.Graph(id=GRAPH_ACCURACY, figure=fig_accuracy)

    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"
    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H2(f"Machine learning model training"),
            html.P('A machine learning model was trained using a large percentage of your uploaded dataset. '
                   'Subsequently, its performance was tested using the rest of the data. '
                   'Check below to see the results. On the lef you will find a confusion matrix that lets you take '
                   'a look at the amount of candidates that the model evaluated differently compared to your dataset.'
                   'On the right there is a bar that tells you the accuracy of the model. You would want this '
                   'as close as possible to 100%. This way you can fully see the amount of bias the model is influenced'
                   'by using your dataset as training ground.'),
            html.Br(),
            horizontal_div([None, None, cm, None, None, accuracy],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            ########################
            html.Br(),
            html.Button(id=NEXT_BUTTON_TRAINING_RESULTS, disabled=False, children="Next"),
            html.Hr(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
    ]
    # Return tab layout
    return layout_children


def step_results_fairness():
    global results, sensitive_features

    count_df = add_description_column(descriptive_age(descriptive_df(results['count_qualified_model'])),
                                      sensitive_features)
    fig_percentage_hired = px.bar(count_df, y='qualified', x='description', color_discrete_sequence=color_sequence)

    fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)

    fig_fairness = px.bar(results['fairness_notions'], y='Fairness notions',
                          color_discrete_sequence=color_sequence)

    fig_sunburst = px.sunburst(descriptive_age(descriptive_columns(results['model_prediction'])), color_discrete_sequence=color_sequence,
                               path=list(map(lambda feature: sensitive_feature_mapping[feature], sensitive_features)),
                               values='qualified')

    sunburst_plot = dcc.Graph(id=GRAPH_SUNBURST, figure=fig_sunburst, style={'display': 'block'})
    hired_graph = dcc.Graph(id=GRAPH_AMOUNT_HIRED, figure=fig_percentage_hired)
    fairness_graph = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS, figure=fig_fairness, style={'display': 'block'})

    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"
    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H2(f"Fairness"),
            html.P('We tested the machine learning model in terms of fairness. Check below to see the results!'),
            html.H3(f"Plots"),
            html.P('Here you can see a sunburst plot that denotes the amount of candidates that were deemed qualified'
                   'based on their sensitive features. This plot can give you a sense of which applicants could be'
                   'treated unfairly. To get a more objective view the bar graph denotes the same amount, but '
                   'proportional to the amount of candidates present with those specific features. This way the '
                   'numbers cannot be influenced by having more candidates with a certain feature.'),
            horizontal_div([None, None, sunburst_plot, None, None, hired_graph],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.Br(),
            html.H3(f"Fairness notions"),
            html.P('Below you will find computations of certain fairness notions that illustrate the amount of '
                   'bias the model was influenced by. If you find that these values are too high, you will find '
                   'certain methods to try and mitigate this bias. Ultimately, you would want these values as close'
                   'as possible to zero.'),
            fairness_graph,
            ########################
            html.Br(),
            html.Button(id=NEXT_BUTTON_FAIRNESS, disabled=False, children="Next"),
            html.Hr(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
    ]
    # Return tab layout
    return layout_children

def step_choose_mitigation():
    dropdown_mitigating_technique = dcc.Dropdown(id=DROPDOWN_MITIGATION, options=mitigation_options,
                                                 value=mitigation_options[0], clearable=False,
                                                 style={'display': 'block'})

    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H3(f"Choose your mitigation technique."),
            dropdown_mitigating_technique,
            html.Br(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
        html.Button(id=NEXT_BUTTON_MITIGATION_CHOICE, disabled=False, children="Next"),
    ]
    # Return tab layout
    return layout_children

def step_results_mitigation():
    global results, results_mitigation, sensitive_features

    fig_sunburst_before = px.sunburst(descriptive_columns(results_mitigation['model_prediction']),
                                      color_discrete_sequence=color_sequence,
                               path=list(map(lambda feature: sensitive_feature_mapping[feature], sensitive_features)),
                               values='qualified')

    fig_sunburst_after = px.sunburst(descriptive_columns(results['model_prediction']),
                                      color_discrete_sequence=color_sequence,
                                      path=list(
                                          map(lambda feature: sensitive_feature_mapping[feature], sensitive_features)),
                                      values='qualified')

    sunburst_plot_before = dcc.Graph(id=GRAPH_SUNBURST, figure=fig_sunburst_before, style={'display': 'block'})
    sunburst_plot_after = dcc.Graph(id=GRAPH_SUNBURST, figure=fig_sunburst_after, style={'display': 'block'})

    count_df_before = add_description_column(descriptive_age(descriptive_df(results['count_qualified_model'])),
                                      sensitive_features)

    count_df_after = add_description_column(
        descriptive_age(descriptive_df(results_mitigation['count_qualified_model'])),
        sensitive_features)

    fig_percentage_hired_before = px.bar(count_df_before, y='qualified', x='description', color_discrete_sequence=color_sequence)
    fig_percentage_hired_after = px.bar(count_df_after, y='qualified', x='description', color_discrete_sequence=color_sequence)

    fig_percentage_hired_before.update_layout(yaxis_title="Percentage qualified", autosize=False)
    fig_percentage_hired_after.update_layout(yaxis_title="Percentage qualified", autosize=False)

    hired_graph_before = dcc.Graph(id=GRAPH_AMOUNT_HIRED, figure=fig_percentage_hired_before)
    hired_graph_after = dcc.Graph(id=GRAPH_AMOUNT_HIRED, figure=fig_percentage_hired_after)

    fig_fairness_before = px.bar(results['fairness_notions'], y='Fairness notions',
                          color_discrete_sequence=color_sequence)

    fig_fairness_after = px.bar(results_mitigation['fairness_notions'], y='Fairness notions',
                          color_discrete_sequence=color_sequence)



    fairness_graph_before = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS, figure=fig_fairness_before, style={'display': 'block'})
    fairness_graph_after = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS, figure=fig_fairness_after, style={'display': 'block'})

    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"
    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H2(f"Fairness"),
            html.P('We performed mitigation using your chosen pre- or postprocessing technique'
                   'to eliminate descrimination from the decisions.'
                   'Check below to compare the results to those of before to see if it worked!'),
            html.H3(f"Plots"),
            html.P('Here you can see a sunburst plot that denotes the amount of candidates that were deemed qualified'
                   'based on their sensitive features. This plot can give you a sense whether certain applicants were'
                   'now treated more fairly. To get a more objective look again at the bar graph on the right'
                   'the proportional view will give a more objective look on the matter.'),
            horizontal_div([None, None, sunburst_plot_before, None, None, sunburst_plot_after],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.Br(),
            horizontal_div([None, None, hired_graph_before, None, None, hired_graph_after],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.H3(f"Fairness notions"),
            html.P('Below you will find a comparison of the computations of the fairness notions from before and after'
                   'mitigation. '
                   'This will illustrate the best whether the mitigation worked.'
                   'If you find that these values are still too high, then the mitigation did not work as intended.'),
            horizontal_div([None, None, fairness_graph_before, None, None, fairness_graph_after],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            ########################
            html.Br(),
            html.Button(id=NEXT_BUTTON_FAIRNESS, disabled=False, children="Next"),
            html.Hr(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
    ]
    # Return tab layout
    return layout_children


def upload_get_app_callbacks(app):
    # TODO: Elk object mag maar 1 keer als een output voorkomen en het lijkt voor problemen te zorgen bij de div,
    #  dus ik gebruik de slider als trigger om de layout te updaten
    #   Elk andere callback onder deze zal dus de slider aanpassen om aan te geven dat de volgende stap getekend moet worden
    @app.callback(
        [Output(UPLOAD_DIV, "children")],
        [Input(STEPS_SLIDER, "value")],
    )
    def update_upload(step):
        # Upload file layout:
        # if step == 0:
        #     return [step_upload()]
        # Select model + features
        if step == 1:
            return [step_preview_select()]
        # Plot results & fairness
        elif step == 2:
            return [step_results_model()]
        elif step == 3:
            return [step_results_fairness()]
        elif step == 4:
            return [step_choose_mitigation()]
        elif step == 5:
            return [step_results_mitigation()]
        # ...

    @app.callback(
        [Output(STEPS_SLIDER, "value")],
        [Input(UPLOADED_DATASET, "filename"), Input(UPLOADED_DATASET, "contents")],
        # TODO: gebruik (ook) contents om de file in te laden, anders krijg je enkel de naam en niet het pad naar je file
    )
    def update_upload(dataset_file_name, dataset_contents):
        # TODO: om het gemakkelijk te houden, laat ik deze callback de layout aanpassen. Dus wanneer we niet kunnen
        #  inladen zoals verwacht, moet er niets gebeuren en mogen we ook niet naar de volgende stap
        if dataset_file_name is None:
            raise PreventUpdate

        else:
            global data_uploaded, uploaded_dataset, df_count_hired_model, results, preview, current_step
            print("Loading...", dataset_file_name)

            content_type, content_string = dataset_contents.split(',')
            decoded = base64.b64decode(content_string)
            uploaded_dataset = pandas.read_csv(io.StringIO(decoded.decode('utf-8')),
                                               index_col='Unnamed: 0')  # TODO: verwijder index kolom die werd opgeslagen
            data_uploaded = True

            preview = descriptive_df(uploaded_dataset).sample(10).to_dict("records")
            return [1]

    @app.callback(
        [Output(NEXT_BUTTON_UPLOAD, "disabled")],
        [Input(DROPDOWN_MODELS_HOME, "value"), Input(CHECKLIST_SENSITIVE_FEATURE_HOME, "value")],
    )
    def update_model_features(new_model, new_sensitive_features):
        global model, sensitive_features
        print("Updating model and sensitive features:", new_model, new_sensitive_features)
        model = new_model
        sensitive_features = new_sensitive_features

        # Ensure both are set
        if model is not None and sensitive_features is not None:
            next_button_disabled = False
        else:
            next_button_disabled = True
        return [next_button_disabled]

    @app.callback(
        [Output(STEPS_SLIDER, "value", allow_duplicate=True)],  # STEPS_SLIDER staat in een vorige Output ook
        [Input(NEXT_BUTTON_UPLOAD, "n_clicks"), Input(NEXT_BUTTON_UPLOAD, "disabled")], prevent_initial_call=True
    )
    def update_train_and_plot(n_clicks, disabled):
        global results
        if n_clicks is None or n_clicks == 0 or disabled:
            raise PreventUpdate
        print("Training...")
        results = pipeline(model, uploaded_dataset, sensitive_features)
        return [2]

    @app.callback(
        [Output(STEPS_SLIDER, "value", allow_duplicate=True)],
        [Input(NEXT_BUTTON_TRAINING_RESULTS, "n_clicks"), Input(NEXT_BUTTON_TRAINING_RESULTS, "disabled")],
        prevent_initial_call=True
    )
    def update_fairness_plot(n_clicks, disabled):
        if n_clicks is None or n_clicks == 0 or disabled:
            raise PreventUpdate
        print("Displaying fairness plots")
        return [3]

    @app.callback(
        [Output(STEPS_SLIDER, "value", allow_duplicate=True)],
        [Input(NEXT_BUTTON_FAIRNESS, "n_clicks"), Input(NEXT_BUTTON_FAIRNESS, "disabled")],
        prevent_initial_call=True
    )
    def update_mitigation_choice(n_clicks, disabled):
        global results_mitigation
        if n_clicks is None or n_clicks == 0 or disabled:
            raise PreventUpdate
        print("Choosing mitigation technique...")
        return [4]

    @app.callback(
        [Output(STEPS_SLIDER, "value", allow_duplicate=True)],
        [Input(NEXT_BUTTON_MITIGATION_CHOICE, "n_clicks"), Input(NEXT_BUTTON_MITIGATION_CHOICE, "disabled"),
         Input(DROPDOWN_MITIGATION, "value")],
        prevent_initial_call=True
    )
    def update_mitigation_plot(n_clicks, disabled, mitigation_technique):
        global results_mitigation
        if n_clicks is None or n_clicks == 0 or disabled:
            raise PreventUpdate
        print("Mitigating dataset...")
        results_mitigation = mitigation_pipeline(mitigation_technique,
                                                 uploaded_dataset,
                                                 results['model_prediction'],
                                                 sensitive_features,
                                                 results['fitted_model'])
        return [5]
