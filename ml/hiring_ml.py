import random

import aif360.sklearn.metrics
import numpy as np
import pandas
# import torch

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix

from hiring import FeatureBias
from hiring.features import HiringFeature, Gender, GenderDescription

from hiring.hire import HiringScenario
import pandas as pd

import aif360 as aif


def convert_goodness(g):
    if g >= 5:
        return 1
    else:
        return 0


def cross_val(training_data, models):
    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)

    # Cross-validation setup
    amt_folds = 20
    k_partitioning = KFold(n_splits=amt_folds, shuffle=False)

    model_mean_scores = {}

    for m in models:
        score = cross_val_score(m, training_x, training_y.values.ravel(), cv=k_partitioning,
                                scoring='neg_root_mean_squared_error')
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


def add_prediction_to_df(data, prediction, predicted_feature):
    data[predicted_feature] = prediction
    return data


def divide_data_on_feature(data):
    return [data[data['gender'] == 1], data[data['gender'] == 2]]


def apply_fairness_notion(sensitive_feature, priv_feature_value, test_data, models):
    # split_data = divide_data_on_feature(test_data)
    # data_men = split_data[0]
    # data_women = split_data[1]

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


def generate_cm(predictions):
    cm = []

    df_human_prediction = predictions[0]

    human_prediction_men = df_human_prediction.loc[df_human_prediction['gender'] == 1]
    human_prediction_women = df_human_prediction.loc[df_human_prediction['gender'] == 2]

    men_true = isolate_prediction(human_prediction_men)
    women_true = isolate_prediction(human_prediction_women)

    for df_p in predictions:
        df_model_prediction_women = df_p.loc[df_p['gender'] == 2]
        df_model_prediction_men = df_p.loc[df_p['gender'] == 1]

        model_prediction_women = isolate_prediction(df_model_prediction_women)
        model_prediction_men = isolate_prediction(df_model_prediction_men)

        cm.append(confusion_matrix(y_true=women_true, y_pred=model_prediction_women))
        cm.append(confusion_matrix(y_true=men_true, y_pred=model_prediction_men))

    return cm


def make_count_df(counts):
    df = pandas.DataFrame(counts, ["Women", "Men"], columns=['hired'])
    return df


def count_hired(predictions):
    hired_counts = []

    for df_p in predictions:
        counts = [len(df_p[(df_p['gender'] == 2) & (df_p['hired'] == 1)]),
                  len(df_p[(df_p['gender'] == 1) & (df_p['hired'] == 1)])]
        dataframe_hired_model_count = make_count_df(counts)
        hired_counts.append(dataframe_hired_model_count)

    return hired_counts


def make_predictions(test_data, trained_models):
    predictions = [test_data]

    for tm in trained_models:
        test_x = isolate_features(test_data)
        prediction = tm.predict(test_x)
        dataframe_model = add_prediction_to_df(test_x, prediction, 'hired')
        predictions.append(dataframe_model)

    return predictions


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
    data = data.sample(10)
    return data


def setup_environment(scenario):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    if scenario == 'Base':
        return HiringScenario(seed=seed)
    elif scenario == 'Different distribution':
        return setup_diff_distribution_environment()
    elif scenario == 'Bias':
        return setup_bias_environment()


def setup_diff_distribution_environment():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    diff_distribution_environment = HiringScenario(seed=seed)

    diff_distribution_environment.description = "Historically less women"
    diff_distribution_environment.gender_desc = GenderDescription(prob_male=0.7, prob_female=0.3)

    return diff_distribution_environment


def setup_bias_environment():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    bias_environment = HiringScenario(seed=seed)

    bias_environment.description = "Men are more desirable than women"
    bias_environment.feature_biases = [FeatureBias(HiringFeature.gender, Gender.male, 2.0)]

    return bias_environment
