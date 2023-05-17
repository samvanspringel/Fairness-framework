import random

import aif360.sklearn.metrics
import numpy as np
import pandas

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# AIF360
from aif360.algorithms.preprocessing import *
from aif360.algorithms.postprocessing import *
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

from sklearn.metrics import accuracy_score

from hiring import FeatureBias
from hiring.features import HiringFeature, Gender, Nationality
from hiring.features import ApplicantGenerator

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
    if 'qualified' in data.columns:
        return data.drop(['qualified'], axis=1)
    else:
        return data


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


def calculate_fairness(simulator_df, prediction_df, sensitive_attributes, output):
    dataset = StandardDataset(df=prediction_df,
                              label_name=output,
                              favorable_classes=[1],
                              protected_attribute_names=sensitive_attributes,
                              privileged_classes=[[0]])

    dataset_model = StandardDataset(df=simulator_df,
                                    label_name=output,
                                    favorable_classes=[1],
                                    protected_attribute_names=sensitive_attributes,
                                    privileged_classes=[[0]])

    a = sensitive_attributes[0]
    i = dataset_model.protected_attribute_names.index(a)
    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]}]

    classification_metric = ClassificationMetric(dataset, dataset_model,
                                                 unprivileged_groups=unprivileged_groups,
                                                 privileged_groups=privileged_groups)

    fairness = {"Fairness notions": [abs(classification_metric.statistical_parity_difference()),
                                     abs(classification_metric.false_positive_rate_difference()),
                                     abs(classification_metric.equal_opportunity_difference()),
                                     1 - abs(classification_metric.consistency(n_neighbors=20).flat[0])]}

    fairness_df = pd.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality', 'Equal opportunity',
                                                     'Inconsistency'])
    return fairness_df


def sample_reweighing(df, sensitive_features, output):
    dataset_model = StandardDataset(df=df,
                                    label_name=output,
                                    favorable_classes=[1],
                                    protected_attribute_names=sensitive_features,
                                    privileged_classes=[[0]])
    a = sensitive_features[0]
    i = dataset_model.protected_attribute_names.index(a)
    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]}]

    reweighing = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

    reweighing.fit(dataset=dataset_model)

    print(reweighing.transform(dataset_model))

    return


def calibrated_equalized_odds(training_data, test_data, fitted_model, sensitive_features):
    print(sensitive_features)
    ceo = CalibratedEqualizedOdds(prot_attr=sensitive_features, cost_constraint='fpr')

    post_processor = PostProcessingMeta(estimator=fitted_model, postprocessor=ceo, prefit=True, val_size=18000)

    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)

    post_processor.fit(training_x, training_y)

    return make_prediction(test_data, post_processor)


def generate_cm(dataset, prediction):
    test_y = dataset[['qualified']]
    prediction_y = prediction[['qualified']]
    cm = confusion_matrix(test_y, prediction_y)

    accuracy = {"Model accuracy": [accuracy_score(test_y, prediction_y)]}
    accdf = pandas.DataFrame(data=accuracy)

    return [cm, pandas.DataFrame(data=accuracy)]


def make_percentage_df(df, total):
    df['total'] = total['count']
    df['qualified'] = df.apply(lambda row: row[len(df.columns) - 2] / row[len(df.columns) - 1], axis=1)
    df = df.drop(['total', 'count'], axis=1)
    return df


def count_hired(df_p, sensitive_features):
    qualified_dataframe = df_p[df_p['qualified'] == 1]
    qualified_result = qualified_dataframe.groupby(sensitive_features).size().reset_index().rename(columns={0: 'count'})

    total_result = df_p.groupby(sensitive_features).size().reset_index().rename(columns={0: 'count'})

    return make_percentage_df(qualified_result, total_result)


def make_prediction(test_data, trained_model):
    simulator_evaluation = test_data.copy()
    labels_y = test_data[['qualified']]
    test_x = isolate_features(test_data)
    p = trained_model.predict(test_x)

    prediction = add_prediction_to_df(test_x, p, 'qualified')

    return [simulator_evaluation, prediction]


def train_model(training_data, model):
    training_x = isolate_features(training_data)
    training_y = isolate_prediction(training_data)

    return model.fit(training_x, training_y.values.ravel())


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


def setup_environment(scenario, sensitive_features, seed):
    if scenario == 'Base':
        applicant_generator = ApplicantGenerator(seed=seed, csv="../hiring/data/belgian_population.csv")
        env = HiringScenario(seed=seed, applicant_generator=applicant_generator, threshold=5)
        return env
    elif scenario == 'Different distribution':
        return setup_diff_distribution_environment(seed)
    elif scenario == 'Bias':
        return setup_bias_environment(sensitive_features, seed)


def setup_diff_distribution_environment(seed):
    applicant_generator = ApplicantGenerator(seed=seed, csv="../hiring/data/belgian_pop_diff_dist.csv")

    env = HiringScenario(seed=seed, applicant_generator=applicant_generator, threshold=5)

    return env


def setup_bias_environment(sensitive_features, seed):
    applicant_generator = ApplicantGenerator(seed=seed, csv="../hiring/data/belgian_population.csv")
    bias_env = HiringScenario(seed=seed, applicant_generator=applicant_generator, threshold=5)

    if 'gender' in sensitive_features:
        bias_env.feature_biases.append(FeatureBias(HiringFeature.gender, Gender.male, 2.0))

    if 'nationality' in sensitive_features:
        bias_env.feature_biases.append(FeatureBias(HiringFeature.nationality, Nationality.belgian, 2.0))

    if 'age' in sensitive_features:
        bias_env.feature_biases.append(FeatureBias(HiringFeature.age, lambda age: age < 50, 2.0))

    if 'married' in sensitive_features:
        bias_env.feature_biases.append(FeatureBias(HiringFeature.married, lambda married: married == 0, 2.0))

    return bias_env
