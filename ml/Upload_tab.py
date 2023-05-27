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


STEPS_SLIDER = "Steps"

data_uploaded = False
uploaded_dataset = None
model = None
sensitive_features = None
color_sequence = [px.colors.qualitative.Set3[4], px.colors.qualitative.Set3[3],
                  px.colors.qualitative.Set2[4], px.colors.qualitative.Set3[11]]

steps = {
    0: "Upload",
    1: "Setup",
    2: "Train model",
    3: "Fairness",
    4: "Mitigate bias",
    5: "Compare"
}

UPLOAD_DIV = "upload_div"
NEXT_BUTTON_UPLOAD = "next_upload"
NEXT_BUTTON_TRAINING_RESULTS = "next_training_results"
NEXT_BUTTON_FAIRNESS = "next_fairness"
NEXT_BUTTON_MITIGATION_CHOICE = "next_mitigation_choice"

PREVIEW_HOME = "Preview_data_UPLOAD"

UPLOADED_DATASET = "uploaded-dataset"

CHECKLIST_SENSITIVE_FEATURE_HOME = "Checklist_sensitive_feature_UPLOAD"
sf_options = ["Gender", "Nationality", "Age", "Married"]

models_options = ["Dataset", "Decision tree", "k-Nearest neighbours"]
DROPDOWN_MODELS_HOME = "Dropdown_models_UPLOAD"

DROPDOWN_MITIGATION = "Dropdown_mitigation"
mitigation_options = ["Pre-processing: Sample Reweighing",
                      "Post-processing: Calibrated Equalized Odds",
                      "Post-processing: Reject Option Classification",
                      # "Post-processing: Equalized Odds Optimization"
                      ]

SAVE_DATASETS_CHECK = "save_datasets"

CM_GRAPH = "Confusion_matrix"
SCENARIO = "Scenario"
classification_labels = ["Not qualified", "Qualified"]

GRAPH_AMOUNT_HIRED = "Amount_hired_gender_UPLOAD"
GRAPH_AMOUNT_HIRED_AM = "Amount_hired_gender_UPLOAD_AM"
GRAPH_FAIRNESS_NOTIONS = "Graph_fairness_notions_UPLOAD"
GRAPH_FAIRNESS_NOTIONS_AM = "Graph_fairness_notions_UPLOAD_AM"
GRAPH_SUNBURST = "Graph_sunburst_UPLOAD"
GRAPH_SUNBURST_AM = "Graph_sunburst_UPLOAD_AM"
GRAPH_ACCURACY = "Graph_accuracy"

results = {}
results_mitigation = {}


def upload_get_tab_dcc():
    steps_slider_upload = dcc.Slider(id=STEPS_SLIDER, disabled=True,
                                     marks=steps, value=0,

                                     min=min(steps.keys()),
                                     max=max(steps.keys()))

    layout = html.Div([
        ### Header under tab ###
        html.Div([
            html.H1("Highlighting & Mitigating Discrimination in Job Hiring Scenarios"),
            steps_slider_upload,
            html.Br(),
            html.Div(
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
                       html.P('Here you can upload your own dataset to do some experimenting. \n'
                              ' Machine learning algorithms are prone to bias in their training examples.'
                              ' It can cause them to discriminate due to the model reproducing patterns found in the '
                              'training data. '),
                       html.P(""),
                       html.P('The goal of this application is to visualize the impact of this'
                              ' potential bias in your dataset. We will train a machine learning algorithm that will '
                              'apply the patterns that it found in your own dataset.'
                              ' Subsequently, we will test this prediction for fairness. Depending on which sensitive'
                              ' features you choose, we construct all different groups with all possible for these. '
                              'Next, you will be able to choose which mitigation technique to use for your prediction.'
                              ' Again depending on which one you choose, the original dataset or the '
                              'models prediction is changed. '
                              'In the next step you will be shown how well the mitigation '
                              'worked. Depending on the technique, the fairness notion that is focussed on should be'
                              ' brought down to zero. '),
                       html.P('Upload your dataset now to see for yourself! '),
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
            html.H2(f"Data upload"),
            html.P('Below, you will find a sample of the data you chose to load. Each candidate is paired with '
                   'the evaluation that you deemed correct. Next, you should make a choice of sensitive features that '
                   'you may think candidates have been discriminated on. '),
            dash_table.DataTable(id=PREVIEW_HOME, data=preview),
            html.Br(),
            html.H3(f"Which sensitive features?"),
            html.P('Sensitive features are characteristics that should not influence a decision, but are '
                   'known to have an impact on whether candidates are considered qualified. '
                   'Here you can make your choice of which sensitive features to focus on. All possible combinations '
                   'of groups of candidates with these sensitive features are checked out.'),
            checklist_sensitive_features,
            html.Br(),
            html.H3("Machine learning model"),
            html.P('Choose your machine learning model. This model will use 80% of your data to learn from. The other '
                   '20% will be used to test the performance of the model. You can evaluate the performance in the '
                   'next step.'),
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
            html.H2(f"Machine learning"),
            html.H3(f"Confusion matrix"),
            html.P(f"When machine learning algorithms learn, they try to optimise their performance on the training "
                   f"set. This means that the test set is unknown territory for the model. When a new instance has "
                   f"no comparable one in the training set, the algorithm may not be able to correctly classify it, "
                   f"resulting in incorrect  predictions. Assuming we know the true outcome of the testing samples "
                   f"(your dataset) "
                   f"and every applicant has a positive or a  negative prediction, we can illustrate the mistakes "
                   f"the model made with "
                   f"a confusion matrix. In our job hiring scenario a positive outcome represents that the applicant "
                   f"is suitable for hiring or qualified, a negative means the applicant should be rejected."
                   f"\n The different elements of the confusion matrix:"),

            html.Li("TN: The true negatives. These instances where predicted negative when the actual value "
                    "was negative as well. In job hiring, this means that an applicant is predicted rejected and "
                    "the true value indicates the same. "),
            html.Br(),
            html.Li("FP: These instances are mistakes called false positives. Which means the actual value "
                    "was negative, but the machine learning model classified them as positive. This corresponds to an "
                    "applicant being predicted qualified when they should have been rejected."),
            html.Br(),
            html.Li("FN: Just as the false positives, these are mistakes as well. False negatives are "
                    "instances classified as negative when the actual value was positive. In the case of job hiring, "
                    "the applicant is predicted qualified when the true value indicates they should not be qualified."),
            html.Br(),
            html.Li("TP: The true positives. True positives are instances that where classified as true and "
                    "predicted true as well. In job hiring context, the applicant is predicted hired when the true "
                    "value indicates they should be hired as well."),
            html.H3(f"Accuracy"),
            html.P(f"Using the elements of the confusion matrix, the accuracy of the model can be computed."
                   f" The accuracy denotes the proportion of candidates the algorithm predicted correctly. \n The "
                   f"formula for the accuracy: "),
            html.P(style={'text-align': 'center'}, children=f"(TN + TP) / (TN + FP + FN + TP)"),

            html.H3(f"Model performance"),
            html.P('A machine learning model was trained using 80% of your uploaded dataset. '
                   'Subsequently, its performance was tested using the rest of the data. '
                   'Check below to see the results. On the left you will find a confusion matrix that lets you take '
                   'a look at the amount of candidates that the model evaluated differently compared to your dataset.'
                   ' On the right there is a bar graph that tells you the accuracy of the model. You would want this '
                   'as close as possible to 100%. This way you can fully see the amount of bias the model is influenced'
                   ' by using your dataset as training ground.'),
            html.Br(),
            horizontal_div([None,
                            None,
                            cm,
                            None,
                            None,
                            accuracy],
                           width=[None, width, graph_width, None, width, graph_width],
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
    fig_percentage_hired = px.bar(count_df, y='qualified', x='description', color_discrete_sequence=color_sequence,
                                  color='description',)

    fig_percentage_hired.update_layout(yaxis_title="Percentage qualified", autosize=False)

    fig_fairness = px.bar(results['fairness_notions'], y='Fairness notions',
                          color=results['fairness_notions'].index,
                          color_discrete_sequence=color_sequence)

    fig_sunburst = px.sunburst(descriptive_age(descriptive_columns(results['model_prediction'])),
                               color_discrete_sequence=color_sequence,
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
            html.P('We tested the machine learning model in terms of fairness. You can view fairness as '
                   'a way of quantifying how fair the machine learning model treated each candidate.'
                   ' Check below to see the results!'),
            html.H3(f"Total vs. Proportionally"),
            html.P('Here you can see a sunburst plot that denotes the total amount of candidates that were deemed '
                   'qualified based on their sensitive features. This plot can give you a sense of which applicants '
                   'could be treated unfairly. To get a more objective view the bar graph denotes the same amount, but '
                   'proportional to the amount of candidates present with those specific features. This way the '
                   'numbers cannot be influenced by having more candidates with a certain feature.'),
            horizontal_div([None, None, sunburst_plot, None, None, hired_graph],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.Br(),
            html.H3(f"Fairness notions"),
            html.P('Below you will find certain fairness notions that illustrate the amount of '
                   'bias the model was influenced by. Each fairness notion focuses on a different aspect of the data. '
                   'If you find that these values are too high, you will find '
                   'certain methods to try and mitigate this bias. Ultimately, you would want these values as close'
                   ' as possible to zero.'),

            html.H4(f"Group fairness notions"),
            html.Li("Statistical parity: When the statistical parity of two groups is equal, it means that both "
                    "groups have an equal acceptance rate. To satisfy statistical parity, the probability of being "
                    "predicted qualified should be the same for all groups. This fairness notion only looks at the "
                    "prediction of the model. This way it can detect bias in datasets without having to compare it to "
                    "a prediction. Because it doesn't use a comparison between datasets, it is suitable "
                    "when the outcomes of the initial dataset are unreliable. Statistical parity is "
                    "computed by this formula:"),
            html.P(style={'text-align': 'center'}, children=f"(TP + FP)/(TP + FP + FN + TN)"),
            html.Br(),
            html.Li("Equal opportunity: This fairness notion is a relaxation of the fairness notion equalized odds. "
                    "Equalized odds requires the sensitive attribute to be conditionally independent from the outcome. "
                    "However, satisfying equalized odds is difficult to implement and interpret in practice. "
                    "Therefore, we consider its two relaxed versions. The first relaxation is equal opportunity. In "
                    "equal opportunity the true positive rates (TPR) or sensitivity recall should be equal for all "
                    "groups. The formula for the true positive rates is the following:"),
            html.P(style={'text-align': 'center'}, children=f"TPR = TP/(TP + FN)"),

            html.Br(),
            html.Li("Predictive equality: The other relaxation of equalized odds that takes into account the "
                    "false positives is predictive equality. When false positives are important to the "
                    "fairness of decisions, predictive equality is a better option than equal opportunity. "
                    "Predictive equality demands the false positive rates (FPR) of two groups to be the same. "
                    ),
            html.P(style={'text-align': 'center'}, children=f"FPR = FP/(FP + TN)"),
            html.H4(f"Individual fairness notions"),
            html.Li("Consistency score: The consistency score is computed by checking the prediction of the "
                    "nearest neighbors or the most similar other candidates of a candidate. For each candidate"
                    " the percentages of the nearest neighbours that have the same prediction is "
                    "calculated. Taking the average of these values over all candidates results in the consistency "
                    "score. In the application the complement of this score is used to illustrate the "
                    "inconsistency in the dataset: "
                    ),
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
    save_datasets_check = dcc.Checklist(id=SAVE_DATASETS_CHECK, options=["Save test set, original prediction "
                                                                         "and mitigated prediction"])

    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H2(f"Mitigation"),
            html.H3(f"Which mitigation technique?"),
            html.H4(f"Pre-processing: Sample Reweighing"),
            html.P('In sample reweighing the dataset will be altered to minimize the effects of bias on the '
                   'decision-making of the algorithm. In this approach the outputs will not be changed, but each '
                   'candidate in the dataset is assigned a weight. Now we discuss how the weight are computed. '
                   'If our dataset contains no bias the sensitive features and the outcome of a candidate should be '
                   'conditionally independent. Encoded bias in the dataset results in a lower probability of having '
                   'sensitive features with a disadvantaged value and being considered qualified. The goal of '
                   'reweighing is compensating for this bias by assigning lower weights to candidates that were '
                   'favored or avoided. By adjusting these weights, the dataset can be transformed into being '
                   'completely unbiased.'),
            html.H4(f"Post-processing: Calibrated Equalized Odds"),
            html.P('This post-processing technique tries to mitigate equalized odds while maintaining the calibrated '
                   'probabilities of each group. In this framework calibrated equalized odds is used to ensure the '
                   'difference in predictive equality is brought down to zero. '
                   'To archieve this, the false positive rates of each group should be equalized while keeping the '
                   'probabilities of being considered qualified the same. Calibrated equalized odds will change the '
                   'outcomes of candidates to ensure these conditions.'),
            html.H4(f"Post-processing: Reject Option Classification"),
            html.P('This mitigation works by changing the outcomes of discriminated groups to positive and the '
                   'outcomes of favoured groups to negative. In our framework the reject '
                   'option classifier is used to ensure that the true positive rates of groups are equal. '
                   'This way it will bring the equal opportunity difference of the groups to zero. '),
            html.H3(f"Mitigate your model"),
            html.H4(f"Choose your mitigation technique"),
            dropdown_mitigating_technique,
            html.Br(),

        ], style={'position': 'sticky', "z-index": "999",
                  "width": "100%", 'background': '#FFFFFF', "top": "0%"}),
        ########################
        html.Br(),
        save_datasets_check,
        html.Br(),
        html.Button(id=NEXT_BUTTON_MITIGATION_CHOICE, disabled=False, children="Next"),
    ]
    # Return tab layout
    return layout_children


def step_results_mitigation():
    global results, results_mitigation, sensitive_features

    fig_sunburst_before = px.sunburst(descriptive_columns(results['model_prediction']),
                                      color_discrete_sequence=color_sequence,
                                      path=list(
                                          map(lambda feature: sensitive_feature_mapping[feature], sensitive_features)),
                                      values='qualified')

    fig_sunburst_after = px.sunburst(descriptive_columns(results_mitigation['mitigated_prediction']),
                                     color_discrete_sequence=color_sequence,
                                     path=list(
                                         map(lambda feature: sensitive_feature_mapping[feature], sensitive_features)),
                                     values='qualified')

    sunburst_plot_before = dcc.Graph(id=GRAPH_SUNBURST, figure=fig_sunburst_before, style={'display': 'block'})
    sunburst_plot_after = dcc.Graph(id=GRAPH_SUNBURST_AM, figure=fig_sunburst_after, style={'display': 'block'})

    count_df_before = add_description_column(descriptive_age(descriptive_df(results['count_qualified_model'])),
                                             sensitive_features)

    count_df_after = add_description_column(
        descriptive_age(descriptive_df(results_mitigation['count_qualified_mitigated'])),
        sensitive_features)

    fig_percentage_hired_before = px.bar(count_df_before, y='qualified', x='description',
                                         color='description',
                                         color_discrete_sequence=color_sequence)
    fig_percentage_hired_after = px.bar(count_df_after, y='qualified', x='description',
                                        color='description',
                                        color_discrete_sequence=color_sequence)

    fig_percentage_hired_before.update_layout(yaxis_title="Percentage qualified", autosize=False)
    fig_percentage_hired_after.update_layout(yaxis_title="Percentage qualified", autosize=False)

    hired_graph_before = dcc.Graph(id=GRAPH_AMOUNT_HIRED, figure=fig_percentage_hired_before)
    hired_graph_after = dcc.Graph(id=GRAPH_AMOUNT_HIRED_AM, figure=fig_percentage_hired_after)

    fig_fairness_before = px.bar(results['fairness_notions'], y='Fairness notions',
                                 color=results['fairness_notions'].index,
                                 color_discrete_sequence=color_sequence)

    fig_fairness_after = px.bar(results_mitigation['fairness_notions'],
                                color=results_mitigation['fairness_notions'].index, y='Fairness notions',
                                color_discrete_sequence=color_sequence)

    fairness_graph_before = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS, figure=fig_fairness_before, style={'display': 'block'})
    fairness_graph_after = dcc.Graph(id=GRAPH_FAIRNESS_NOTIONS_AM, figure=fig_fairness_after,
                                     style={'display': 'block'})

    width = "30%"
    graph_width = "40%"
    width2 = "10%"
    space_width = "2.5%"
    layout_children = [
        ### Header under tab ###
        html.Div([
            html.H2(f"Compare"),
            html.P('We performed mitigation using your chosen pre- or postprocessing technique '
                   'to eliminate discrimination from the decisions.'
                   ' Check below to compare the results to those of before to see if it worked!'),
            html.H3(f"Before vs. after"),
            html.P('Here you can see a sunburst plot that denotes the amount of candidates that were deemed qualified '
                   'based on their sensitive features. This plot can give you a sense whether certain applicants were '
                   'now treated more fairly. When there are more candidates with a certain feature this can results in'
                   ' a misleading view. When a dataset consists of more candidates with a certain characteristic, it '
                   'is to be expected that more of them will be evaluated qualified. However, this does not mean that'
                   ' other groups were discriminated against. The evaluation process could still have happened fairly.'
                   ''),
            horizontal_div([None, None, sunburst_plot_before, None, None, sunburst_plot_after],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.P('These bar graphs depict a more objective view of the data. It illustrates a proportional view '
                   'of the qualified candidates to their total representation in the dataset. When these bar graphs '
                   'look different it usually means that some groups were discriminated against.'
                   ' Beware of the scales of the graphs! It may look like there is a large difference when there is '
                   'actually not.'),
            horizontal_div([None, None, hired_graph_before, None, None, hired_graph_after],
                           width=[None, width2, graph_width, None, width2, graph_width],
                           space_width=space_width),
            html.P('It may be possible that the results are similar to the original computations. This is because '
                   'some mitigation techniques make sure that the amount of candidates deemed qualified from each '
                   'group are still the same. Take a look at the fairness notions below to see the difference.'),

            html.H3(f"Fairness notions"),
            html.P('Below you will find a comparison of the computations of the fairness notions from before and after'
                   'mitigation. '
                   'This will illustrate the best whether the mitigation worked.'
                   ' The fairness notion that your chosen technique focussed on, should have been brought down to'
                   'zero. You also have the option to save the original prediction and the mitigated prediction below. '
                   ' This way you can individually take a look at the evaluation of each candidate and where there are '
                   'any discrepancies.'),
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
    )
    def update_upload(dataset_file_name, dataset_contents):
        if dataset_file_name is None:
            raise PreventUpdate

        else:
            global data_uploaded, uploaded_dataset, df_count_hired_model, results, preview, current_step
            print("Loading...", dataset_file_name)

            content_type, content_string = dataset_contents.split(',')
            decoded = base64.b64decode(content_string)
            uploaded_dataset = pandas.read_csv(io.StringIO(decoded.decode('utf-8')))
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
         Input(DROPDOWN_MITIGATION, "value"), Input(SAVE_DATASETS_CHECK, "value")],
        prevent_initial_call=True
    )
    def update_mitigation_plot(n_clicks, disabled, mitigation_technique, save_datasets):
        global results_mitigation
        if n_clicks is None or n_clicks == 0 or disabled:
            raise PreventUpdate
        print("Mitigating dataset...")
        results_mitigation = mitigation_pipeline(uploaded_dataset,
                                                 results['simulator_evaluation'],
                                                 results['model_prediction'],
                                                 sensitive_features,
                                                 results['fitted_model'],
                                                 mitigation_technique)

        if save_datasets is not None and \
                save_datasets[0] == "Save test set, original prediction and mitigated prediction":
            results['simulator_evaluation'].to_csv(f"output/test_set_your_data.csv",
                                                   index=False)
            results_mitigation['original_prediction'].to_csv(f"output/original_model_prediction.csv",
                                                             index=False)
            results_mitigation['mitigated_prediction'].to_csv(f"output/mitigated_prediction.csv",
                                                              index=False)
        return [5]
