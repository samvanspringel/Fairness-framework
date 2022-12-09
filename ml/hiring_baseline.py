import random

import aif360.sklearn.metrics
import numpy as np
# import torch

import sklearn
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import confusion_matrix
import pprint



from hiring.hire import HiringScenario
import pandas as pd

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


def cross_val(training_data, models):
    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)

    # Cross-validation setup
    amt_folds = 20
    k_partitioning = KFold(n_splits=amt_folds, shuffle=False)

    model_mean_scores = {}

    for m in models:
        score = cross_val_score(m, training_x, training_y.values.ravel(), cv=k_partitioning, scoring='neg_root_mean_squared_error')
        model_mean_scores[m] = repr(np.mean(score))


def rename_goodness(data):
    data['goodness'] = data['goodness'].map(convert_goodness)
    data.rename({'goodness': 'hired'}, axis=1, inplace=True)
    return data


def isolate_features(data):
    return data.drop(['hired'], axis=1)


def isolate_prediction(data):
    return data[['hired']]


def print_info(data, string):
    print("\n -------" + string + "---------")
    print(data.head())


def gender_confusion_matrices(test_data, clf, dt):
    # Test features isoleren om voorspelling te doen met modellen
    test_data_women = test_data.loc[test_data['gender'] == 2]
    test_data_men = test_data.loc[test_data['gender'] == 1]

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
    # fig = ff.create_annotated_heatmap(cm_women, x=x, y=y, annotation_text=cm_women_text, colorscale='Viridis')
    # fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis = dict(title='x'),
                      # yaxis = dict(title='x')
    #                  )

    # # add custom xaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=0.5,
    #                         y=-0.15,
    #                         showarrow=False,
    #                         text="Predicted value",
    #                         xref="paper",
    #                         yref="paper"))
    #
    # # add custom yaxis title
    # fig.add_annotation(dict(font=dict(color="black", size=14),
    #                         x=-0.35,
    #                         y=0.5,
    #                         showarrow=False,
    #                         text="Real value",
    #                         textangle=-90,
    #                         xref="paper",
    #                         yref="paper"))
    #
    # # adjust margins to make room for yaxis title
    # fig.update_layout(margin=dict(t=50, l=200))
    #
    # # add colorbar
    # fig['data'][0]['showscale'] = True
    # fig.show()


def add_prediction_to_df(data, prediction, predicted_feature):
    data[predicted_feature] = prediction
    return data


def divide_data_on_feature(data):
    return [data[data['gender'] == 1], data[data['gender'] == 2]]


def apply_fairness_notion(sensitive_feature, priv_feature_value, test_data, models):
    #split_data = divide_data_on_feature(test_data)
    #data_men = split_data[0]
    #data_women = split_data[1]

    print("Difference in acceptance rate by human")
    print(aif360.sklearn.metrics.statistical_parity_difference(y_true=test_data, y_pred=test_data,
                                                         priv_group=test_data[
                                                                        sensitive_feature] == priv_feature_value,
                                                         pos_label=1))

    for m in models:
        print("Difference in acceptance rate by " + str(m))
        test_features = isolate_features(test_data)
        prediction = m.predict(test_features)
        model_prediction = add_prediction_to_df(test_features, prediction, 'hired')
        print(aif360.sklearn.metrics.statistical_parity_difference(y_true=test_data, y_pred=model_prediction,
                                                             priv_group=test_data[
                                                                            sensitive_feature] == priv_feature_value,
                                                             pos_label=1))

def generate_cm(test_data, trained_models):
    test_data = rename_goodness(test_data)
    cm = []

    test_data_men = test_data.loc[test_data['gender'] == 1]
    test_data_women = test_data.loc[test_data['gender'] == 2]

    men_true = isolate_prediction(test_data_men)
    women_true = isolate_prediction(test_data_women)

    women_x = isolate_features(test_data_women)
    men_x = isolate_features(test_data_men)

    cm_women_humans = confusion_matrix(women_true, women_true)
    cm_men_humans = confusion_matrix(men_true, men_true)
    cm.append(cm_women_humans)
    cm.append(cm_men_humans)

    for tm in trained_models:
        prediction_women = tm.predict(women_x)
        cm.append(confusion_matrix(women_true, prediction_women))
        prediction_men = tm.predict(men_x)
        cm.append(confusion_matrix(men_true, prediction_men))

    return cm

def train_models(training_data, models):
    training_data = rename_goodness(training_data)
    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)
    trained_models = []

    # Modellen fitten
    for m in models:
        trained_models.append(m.fit(training_x, training_y.values.ravel()))
    return trained_models


def pipeline(training_data, models):


    cross_val(training_data, models)

    trained_models = train_models(training_data, models)
    return trained_models

    # gender_confusion_matrices(test_data)

    # test_data = rename_goodness(test_data)
    # apply_fairness_notion('gender', 1, test_data, trained_models)


def generate_training_data(env, num_samples):
    return env.create_dataset(num_samples, show_goodness=True, rounding=5)

def generate_test_data(env, num_samples):
    return env.create_dataset(num_samples, show_goodness=True, rounding=5)

def make_preview(data):
    data = data.drop(['goodness'], axis=1).sample(20)
    return data

def setup_environment():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    return HiringScenario(seed=seed)
