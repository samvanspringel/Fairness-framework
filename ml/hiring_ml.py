import random

import aif360.sklearn.metrics
import numpy as np
import pandas

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

from hiring import FeatureBias
from hiring.features import HiringFeature, Gender, GenderDescription, Origin, OriginDescription

from hiring.hire import HiringScenario
import pandas as pd


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
    data.rename({'goodness': 'qualified'}, axis=1, inplace=True)
    return data


def isolate_features(data):
    return data.drop(['qualified'], axis=1)


def isolate_prediction(data):
    return data[['qualified']]


def print_info(data, string):
    print("\n -------" + string + "---------")
    print(data.head())


def add_prediction_to_df(data, prediction, predicted_feature):
    data[predicted_feature] = prediction
    return data


def divide_data_on_feature(data):
    return [data[data['gender'] == 1], data[data['gender'] == 2]]


def change_types(data):
    return data.astype(str).astype(float).astype(int)


def calculate_fairness(predictions, sensitive_attributes, output):
    dataset_gt = change_types(predictions[0])

    fairness_metrics = []

    if len(sensitive_attributes) != 0:
        dataset = StandardDataset(df=dataset_gt,
                                  label_name=output,
                                  favorable_classes=[1],
                                  protected_attribute_names=sensitive_attributes,
                                  privileged_classes=[[1]])

        for p in predictions:
            prediction = p[['qualified']]
            dataset_model = dataset.copy()
            dataset_model.labels = prediction.values
            a = sensitive_attributes[0]
            i = dataset_model.protected_attribute_names.index(a)
            privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]}]
            unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]}]

            classification_metric = ClassificationMetric(dataset, dataset_model,
                                                         unprivileged_groups=unprivileged_groups,
                                                         privileged_groups=privileged_groups)

            model_fairness_metrics = [abs(classification_metric.statistical_parity_difference()),
                                      abs(classification_metric.false_positive_rate_difference()),
                                      abs(classification_metric.equal_opportunity_difference()),
                                      abs(classification_metric.accuracy())]

            fairness_metrics.append(model_fairness_metrics)
    else:
        fairness_metrics = [0, 0, 0, 0]

    return fairness_metrics


def generate_cm(predictions):
    cm = []

    df_dataset_prediction = predictions[0]

    dataset_prediction_men = df_dataset_prediction.loc[df_dataset_prediction['gender'] == 1]
    dataset_prediction_women = df_dataset_prediction.loc[df_dataset_prediction['gender'] == 2]

    men_true = isolate_prediction(dataset_prediction_men)
    women_true = isolate_prediction(dataset_prediction_women)

    for df_p in predictions:
        df_model_prediction_women = df_p.loc[df_p['gender'] == 2]
        df_model_prediction_men = df_p.loc[df_p['gender'] == 1]

        model_prediction_women = isolate_prediction(df_model_prediction_women)
        model_prediction_men = isolate_prediction(df_model_prediction_men)

        cm.append(confusion_matrix(y_true=women_true, y_pred=model_prediction_women))
        cm.append(confusion_matrix(y_true=men_true, y_pred=model_prediction_men))

    # TODO: Alle aantallen delen door totaal aantal van groep
    return cm


def make_percentage_df(percentages):
    percentages_rounded = list(map(lambda percentage: round(percentage, 2), percentages))
    df = pandas.DataFrame(percentages_rounded, ["Women", "Men"], columns=['qualified'])
    return df


def count_hired(predictions):
    hired_percentages = []

    for df_p in predictions:
        percentages = [len(df_p[(df_p['gender'] == 2) & (df_p['qualified'] == 1)]) / len(df_p[df_p['gender'] == 2]),
                       len(df_p[(df_p['gender'] == 1) & (df_p['qualified'] == 1)]) / len(df_p[df_p['gender'] == 1])]
        dataframe_hired_model_count = make_percentage_df(percentages)
        hired_percentages.append(dataframe_hired_model_count)

    return hired_percentages


def make_predictions(test_data, trained_models):
    predictions = [test_data]

    for tm in trained_models:
        test_x = isolate_features(test_data)
        prediction = tm.predict(test_x)
        # print(accuracy_score(test_x, prediction))
        dataframe_model = add_prediction_to_df(test_x, prediction, 'qualified')
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


def setup_environment(scenario, sensitive_features):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    if scenario == 'Base':
        return HiringScenario(seed=seed)
    elif scenario == 'Different distribution':
        return setup_diff_distribution_environment()
    elif scenario == 'Bias':
        return setup_bias_environment(sensitive_features)


def setup_diff_distribution_environment():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    diff_distribution_environment = HiringScenario(seed=seed)

    diff_distribution_environment.description = "Historically less women"
    diff_distribution_environment.gender_desc = GenderDescription(prob_male=0.7, prob_female=0.3)

    return diff_distribution_environment


def setup_bias_environment(sensitive_features):
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    bias_environment = HiringScenario(seed=seed)

    bias_environment.description = "Men are more desirable than women"
    if "Gender" in sensitive_features:
        print("gender bias added")
        bias_environment.feature_biases = [FeatureBias(HiringFeature.gender, Gender.male, 2.0)]
    if "Origin" in sensitive_features:
        print("origin bias added")
        bias_environment.feature_biases.append(FeatureBias(HiringFeature.origin, Origin.belgium, 2.0))

    return bias_environment
