import random

import aif360.sklearn.metrics
import numpy as np
import pandas

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree

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
    a = isolate_features(simulator_df)
    b = isolate_features(prediction_df)

    c = a.reset_index(drop=True) == b.reset_index(drop=True)
    print(c)

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


def sample_reweighing(df, test_set, prediction_model, sensitive_features, output, model):
    dataset_model = StandardDataset(df=df,
                                    label_name=output,
                                    favorable_classes=[1],
                                    protected_attribute_names=sensitive_features,
                                    privileged_classes=[[0.0] for a in sensitive_features])

    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]
                          for i, a in enumerate(dataset_model.protected_attribute_names)}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]
                            for i, a in enumerate(dataset_model.protected_attribute_names)}]

    reweighing = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    reweighing.fit(dataset=dataset_model)
    reweigh_dataset = reweighing.transform(dataset_model)
    reweigh_df, new_dict = reweigh_dataset.convert_to_dataframe()

    # TODO: check je of dit een nieuw model van hetzelfde type als model traint of het model zelf her-traint?
    #  Ik denk dat het een nieuw is, wat hetgene is dat we willen in dit geval
    from sklearn.neighbors import KNeighborsClassifier
    if not isinstance(model, KNeighborsClassifier):
        new_model = model.fit(isolate_features(reweigh_df), isolate_prediction(reweigh_df),
                              sample_weight=new_dict["instance_weights"])
    else:
        # TODO: KNN heebft geen sample weight voor fit, maar we kunnen de predictie (tenminste gedeeltelijk)
        #  beinvloeden a.d.h.v. de volgorde van de data op basis van de sample weights
        knn_df = reweigh_df.copy()
        knn_df["instance_weights"] = new_dict["instance_weights"]
        knn_df = knn_df.sort_values(["instance_weights"], ascending=[False])
        knn_df = knn_df.drop(columns=["instance_weights"])
        new_model = model.fit(isolate_features(knn_df), isolate_prediction(knn_df))
    # TODO: voor de dataset geval als je model, wat doe je dan?

    prediction = new_model.predict(isolate_features(test_set))
    predict_df = test_set.copy()
    predict_df[output] = prediction

    return prediction_model, predict_df


def calibrated_equalized_odds(full_dataset, df, prediction, sensitive_features, output, model):
    dataset = StandardDataset(df=df,
                              label_name=output,
                              favorable_classes=[1],
                              protected_attribute_names=sensitive_features,
                              privileged_classes=[[0]])

    pred = StandardDataset(df=prediction,
                           label_name=output,
                           favorable_classes=[1],
                           protected_attribute_names=sensitive_features,
                           privileged_classes=[[0]])

    a = sensitive_features[0]
    i = pred.protected_attribute_names.index(a)
    privileged_groups = [{a: pred.privileged_protected_attributes[i]}]
    unprivileged_groups = [{a: pred.unprivileged_protected_attributes[i]}]

    ceo = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups, cost_constraint='fpr')

    ceo.fit(dataset_true=dataset, dataset_pred=pred)

    prediction_ceo = ceo.predict(dataset=pred).convert_to_dataframe()[0]

    predict_df = isolate_features(df)
    predict_df[output] = prediction_ceo[output]

    return prediction, predict_df, df


def reject_option_classification(full_dataset, df, prediction, sensitive_features, output, model):
    dataset = StandardDataset(df=df,
                              label_name=output,
                              favorable_classes=[1],
                              protected_attribute_names=sensitive_features,
                              privileged_classes=[[0]])

    pred = StandardDataset(df=prediction,
                           label_name=output,
                           favorable_classes=[1],
                           protected_attribute_names=sensitive_features,
                           privileged_classes=[[0]])

    a = sensitive_features[0]
    i = pred.protected_attribute_names.index(a)
    privileged_groups = [{a: pred.privileged_protected_attributes[i]}]
    unprivileged_groups = [{a: pred.unprivileged_protected_attributes[i]}]

    roc = RejectOptionClassification(privileged_groups=privileged_groups,
                                     unprivileged_groups=unprivileged_groups,
                                     metric_name="Equal opportunity difference")

    roc.fit(dataset_true=dataset, dataset_pred=pred)

    prediction_roc = roc.predict(dataset=pred).convert_to_dataframe()[0]

    predict_df = isolate_features(df)
    copy = prediction_roc.copy()
    predict_df[output] = copy[output].values.tolist()

    return prediction, prediction_roc, dataset.convert_to_dataframe()[0]


def equalized_odds_optimization(full_dataset, df, prediction, sensitive_features, output, model):
    dataset_model = StandardDataset(df=df,
                                    label_name=output,
                                    favorable_classes=[1],
                                    protected_attribute_names=sensitive_features,
                                    privileged_classes=[[0.0] for a in sensitive_features])

    prediction_model = StandardDataset(df=prediction,
                                       label_name=output,
                                       favorable_classes=[1],
                                       protected_attribute_names=sensitive_features,
                                       privileged_classes=[[0.0] for a in sensitive_features])

    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]
                          for i, a in enumerate(dataset_model.protected_attribute_names)}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]
                            for i, a in enumerate(dataset_model.protected_attribute_names)}]

    eq_pp = EqOddsPostprocessing(privileged_groups=privileged_groups, unprivileged_groups=unprivileged_groups)

    eq_pp.fit(dataset_true=dataset_model, dataset_pred=prediction_model)

    prediction_roc = eq_pp.predict(dataset=prediction_model).convert_to_dataframe()

    return prediction, prediction_roc[0]


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
