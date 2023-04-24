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


def calculate_fairness(prediction, sensitive_attributes, output):
    dataset_gt = change_types(prediction)

    dataset = StandardDataset(df=dataset_gt,
                              label_name=output,
                              favorable_classes=[1],
                              protected_attribute_names=sensitive_attributes,
                              privileged_classes=[[1]])

    prediction = prediction[['qualified']]
    dataset_model = dataset.copy()
    dataset_model.labels = prediction.values
    a = sensitive_attributes[0]
    i = dataset_model.protected_attribute_names.index(a)
    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]}]

    classification_metric = ClassificationMetric(dataset, dataset_model,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    # TODO: Bug bij origin selecteren
    model_fairness_metrics = [abs(classification_metric.statistical_parity_difference()),
                              abs(classification_metric.false_positive_rate_difference()),
                              abs(classification_metric.equal_opportunity_difference()),
                              abs(classification_metric.accuracy())]


    return model_fairness_metrics


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


def make_percentage_df(df, total):
    df['total'] = total['count']
    df['qualified'] = df.apply(lambda row: row[len(df.columns)-2]/row[len(df.columns)-1], axis=1)
    df = df.drop(['total', 'count'], axis=1)
    return df


def count_hired(df_p, sensitive_features):
    qualified_dataframe = df_p[df_p['qualified'] == 1]
    qualified_result = qualified_dataframe.groupby(sensitive_features).size().reset_index().rename(columns={0: 'count'})

    total_result = df_p.groupby(sensitive_features).size().reset_index().rename(columns={0: 'count'})
    return make_percentage_df(qualified_result, total_result)


def make_prediction(test_data, trained_model):
    test_x = isolate_features(test_data)
    prediction = trained_model.predict(test_x)

    test_data_prediction = add_prediction_to_df(test_x, prediction, 'qualified')
    return [test_data, test_data_prediction]


def train_model(training_data, model):
    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)

    return model.fit(training_x, training_y.values.ravel())

def pipeline(training_data, models):
    cross_val(training_data, models)

    trained_models = train_model(training_data, models)
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

def make_nonbias_environment():
    seed = 1
    return HiringScenario(seed=seed)

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

    if 'gender' in sensitive_features:
        bias_environment.feature_biases.append(FeatureBias(HiringFeature.gender, Gender.male, 2.0))

    if 'origin' in sensitive_features:
        bias_environment.feature_biases.append(FeatureBias(HiringFeature.origin, Origin.belgium, 2.0))

    return bias_environment
