import random

import aif360.sklearn.metrics
import numpy as np
# import torch

import sklearn
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import pprint

# Modellen
from sklearn import svm
from sklearn import tree

from hiring.hire import HiringScenario

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.figure_factory as ff

import aif360 as aif


def convert_goodness(g):
    if g >= 5:
        return 1
    else:
        return 0


def make_df_women(prediction_human_women, predictions_svm_women, predictions_dt_women):
    # Voorspellingen voor vrouwen in dataset met de beoordeling simulator
    predictions_women = pd.DataFrame(prediction_human_women)

    predictions_women['svm'] = predictions_svm_women
    predictions_women['dt'] = predictions_dt_women

    print("\n ------- VROUWEN ---------")
    # print(predictions_women)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix

    true_hired = predictions_women['hired']
    pred_hired_dt = predictions_women['dt']
    pred_hired_svm = predictions_women['svm']

    print("\n ------- Confusion matrix decision tree ---------")
    print(confusion_matrix(true_hired, pred_hired_dt))

    print("\n ------- Confusion matrix support vector machines ---------")
    cm = confusion_matrix(true_hired, pred_hired_svm)
    print(cm)
    return cm


def make_df_men(prediction_human_men, predictions_svm_men, predictions_dt_men):
    # Voorspellingen voor vrouwen in dataset met de beoordeling simulator
    predictions_men = pd.DataFrame(prediction_human_men)

    predictions_men['svm'] = predictions_svm_men
    predictions_men['dt'] = predictions_dt_men

    print("\n ------- MANNEN ---------")
    # print(predictions_men)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix

    true_hired = predictions_men['hired']
    pred_hired_dt = predictions_men['dt']
    pred_hired_svm = predictions_men['svm']

    print("\n ------- Confusion matrix decision tree ---------")
    print(confusion_matrix(true_hired, pred_hired_dt))

    print("\n ------- Confusion matrix support vector machines ---------")
    cm = confusion_matrix(true_hired, pred_hired_svm)
    print(cm)
    return cm


def cross_val(training_x, training_y, clf, dt):
    # Cross-validation setup
    amt_folds = 20
    k_partitioning = KFold(n_splits=amt_folds, shuffle=False)

    model_mean_scores = {}

    # Cross-validation score berekenen
    score_svm = cross_val_score(clf, training_x, training_y, cv=k_partitioning, scoring='neg_root_mean_squared_error')
    model_mean_scores["Support Vector Machines:"] = repr(np.mean(score_svm))

    # print("\n ------- Score support vector machines ---------")
    # pprint.pprint(model_mean_scores)

    # Cross-validation score berekenen
    score_dt = cross_val_score(dt, training_x, training_y, cv=k_partitioning,
                               scoring='neg_root_mean_squared_error')
    model_mean_scores["Decision Tree:"] = repr(np.mean(score_dt))

    # print("\n ------- Score Decision Tree ---------")
    # pprint.pprint(model_mean_scores)


def rename_goodness(data):
    data['goodness'] = data['goodness'].map(convert_goodness)
    data.rename({'goodness': 'hired'}, axis=1, inplace=True)
    return data


def isolate_features(data):
    data.drop(['hired'], axis=1)
    return data


def isolate_prediction(data):
    return data[['hired']]


def print_info(data, string):
    print("\n -------" + string + "---------")

    print(data)


def pipeline(training_data, test_data):
    training_data = rename_goodness(training_data)
    test_data = rename_goodness(test_data)

    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_x)

    # print_info(training_data, "TRAINING DATA")
    print_info(training_x, "TRAINING FEATURES")
    # print_info(training_y, "TRAINING HIRINGS")

    # Support vector machines classifier
    clf = svm.SVC()
    # Decision Tree
    dt = tree.DecisionTreeClassifier()

    cross_val(training_x, training_y.values.ravel(), clf, dt)

    # Model fitten
    clf.fit(training_x, training_y.values.ravel())
    dt.fit(training_x, training_y.values.ravel())

    # Test features isoleren om voorspelling te doen met modellen
    test_data_women = test_data.loc[test_data['gender'] == 2]
    test_data_men = test_data.loc[test_data['gender'] == 1]

    # test_data_x = test_data.drop(['hired'], axis=1)

    test_data_x_women = isolate_features(test_data_women)
    test_data_x_men = isolate_features(test_data_men)

    # Menselijke beoordeling (geen bias)
    prediction_human_women = isolate_prediction(test_data_women)
    prediction_human_men = isolate_prediction(test_data_men)

    predictions_svm_women = clf.predict(test_data_x_women)
    predictions_svm_men = clf.predict(test_data_x_men)

    predictions_dt_women = dt.predict(test_data_x_women)
    predictions_dt_men = dt.predict(test_data_x_men)

    cm_women = make_df_women(prediction_human_women, predictions_svm_women, predictions_dt_women)

    cm_men = make_df_men(prediction_human_men, predictions_svm_men, predictions_dt_men)

    x = ['Hired', 'Not hired']
    y = ['Hired', 'Not hired']

    # change each element of z to type string for annotations
    cm_women_text = [[str(y) for y in x] for x in cm_women]
    fig = ff.create_annotated_heatmap(cm_women, x=x, y=y, annotation_text=cm_women_text, colorscale='Viridis')
    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Real value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()




if __name__ == '__main__':
    app = Dash(__name__)
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    env = HiringScenario(seed=seed)

    num_samples = 1000

    training_data = env.create_dataset(num_samples, show_goodness=True, rounding=5)
    test_data = env.create_dataset(num_samples, show_goodness=True, rounding=5)
    pipeline(training_data, test_data)

    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        dcc.Graph(
            id='example-graph',
        )
    ])
    app.run_server(debug=True, use_reloader=False)
