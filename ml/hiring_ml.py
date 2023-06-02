import random

import aif360.sklearn.metrics
import numpy as np
import pandas

from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# AIF360
from aif360.algorithms.preprocessing import *
from aif360.algorithms.postprocessing import *
from aif360.datasets import StandardDataset
from aif360.sklearn.metrics import *
from aif360.sklearn.postprocessing import *
from aif360.sklearn.datasets import standardize_dataset

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
    if len(sensitive_attributes) > 1:
        copy_df = simulator_df.copy()
        copy_df['sensitive'] = np.ones(len(simulator_df))
        copy_df['sensitive'] = np.where(copy_df[sensitive_attributes].eq(0.0).all(1, skipna=True), 0.0, 1.0)

        copy_pred = prediction_df.copy()
        copy_pred['sensitive'] = np.ones(len(prediction_df))
        copy_pred['sensitive'] = np.where(copy_pred[sensitive_attributes].eq(0.0).all(1, skipna=True), 0.0, 1.0)

        spd = abs(aif360.sklearn.metrics.statistical_parity_difference(
            y_true=copy_df.set_index('sensitive')[output],
            y_pred=copy_pred.set_index('sensitive')[output].to_numpy(),
            priv_group=0.0,
            prot_attr='sensitive',
            pos_label=1))

        pe = abs(false_positive_rate_error(y_true=copy_df.set_index('sensitive')[output],
                                           y_pred=copy_pred.set_index('sensitive')[
                                               output].values.ravel(),
                                           pos_label=1))

        eo = abs(equal_opportunity_difference(y_true=copy_df.set_index('sensitive')[output],
                                              y_pred=copy_pred.set_index('sensitive')[
                                                  output].values.ravel(),
                                              priv_group=0.0,
                                              prot_attr='sensitive',
                                              pos_label=1))
    else:
        spd = abs(aif360.sklearn.metrics.statistical_parity_difference(
            y_true=simulator_df.set_index(sensitive_attributes)[output],
            y_pred=prediction_df.set_index(sensitive_attributes)[output].to_numpy(),
            priv_group=0.0,
            prot_attr=sensitive_attributes,
            pos_label=1))

        pe = abs(false_positive_rate_error(y_true=simulator_df.set_index(sensitive_attributes)[output],
                                           y_pred=prediction_df.set_index(sensitive_attributes)[
                                               output].values.ravel(),
                                           pos_label=1))

        eo = abs(equal_opportunity_difference(y_true=simulator_df.set_index(sensitive_attributes)[output],
                                              y_pred=prediction_df.set_index(sensitive_attributes)[
                                                  output].values.ravel(),
                                              priv_group=0.0,
                                              prot_attr=sensitive_attributes,
                                              pos_label=1))

    cs = 1 - abs(consistency_score(isolate_features(prediction_df), isolate_prediction(prediction_df).values.ravel(),
                                   n_neighbors=20).flat[0])

    fairness = {"Fairness notions": [spd, pe, eo, cs]}

    fairness_df = pd.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality', 'Equal opportunity',
                                                     'Inconsistency'])

    return fairness_df


def sample_reweighing(df, test_set, prediction_model, sensitive_features, output, model):
    dataset_model = StandardDataset(df=df,
                                    label_name=output,
                                    favorable_classes=[1],
                                    protected_attribute_names=sensitive_features,
                                    privileged_classes=[[0] for a in sensitive_features])

    privileged_groups = [{a: dataset_model.privileged_protected_attributes[i]
                          for i, a in enumerate(dataset_model.protected_attribute_names)}]
    unprivileged_groups = [{a: dataset_model.unprivileged_protected_attributes[i]
                            for i, a in enumerate(dataset_model.protected_attribute_names)}]

    reweighing = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    reweighing.fit(dataset=dataset_model)
    reweigh_dataset = reweighing.transform(dataset_model)
    reweigh_df, new_dict = reweigh_dataset.convert_to_dataframe()

    from sklearn.neighbors import KNeighborsClassifier
    if not isinstance(model, KNeighborsClassifier):
        new_model = model.fit(isolate_features(reweigh_df), isolate_prediction(reweigh_df),
                              sample_weight=new_dict["instance_weights"])
    else:
        knn_df = reweigh_df.copy()
        knn_df["instance_weights"] = new_dict["instance_weights"]
        knn_df = knn_df.sort_values(["instance_weights"], ascending=[False])
        knn_df = knn_df.drop(columns=["instance_weights"])
        new_model = model.fit(isolate_features(knn_df), isolate_prediction(knn_df))

    prediction = new_model.predict(isolate_features(test_set))
    predict_df = test_set.copy()
    predict_df[output] = prediction

    return prediction_model, predict_df


def calibrated_equalized_odds(full_dataset, df, prediction, sensitive_features, output, model):
    dataset = StandardDataset(df=df,
                              label_name=output,
                              favorable_classes=[1],
                              protected_attribute_names=sensitive_features,
                              privileged_classes=[[0.0] for a in sensitive_features])

    pred = StandardDataset(df=prediction,
                           label_name=output,
                           favorable_classes=[1],
                           protected_attribute_names=sensitive_features,
                           privileged_classes=[[0.0] for a in sensitive_features])

    privileged_groups = [{a: dataset.privileged_protected_attributes[i]
                          for i, a in enumerate(dataset.protected_attribute_names)}]
    unprivileged_groups = [{a: dataset.unprivileged_protected_attributes[i]
                            for i, a in enumerate(dataset.protected_attribute_names)}]

    ceo = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                         unprivileged_groups=unprivileged_groups, cost_constraint='fpr')

    ceo.fit(dataset_true=dataset, dataset_pred=pred)

    prediction_ceo = ceo.predict(dataset=pred).convert_to_dataframe()[0]

    return prediction, prediction_ceo


def is_unprivileged(row, sensitive_features):
    for sf in sensitive_features:
        print(row[sf])
        if row[sf] == 1.0:
            return True
    return False


def reject_option_classification(full_dataset, df, prediction, sensitive_features, output, model):
    if model is None:
        model = KNeighborsClassifier(n_neighbors=3)
    if len(sensitive_features) > 1:
        copy_df = df.copy()
        copy_df['sensitive'] = np.ones(len(df))
        copy_full = full_dataset.copy()
        copy_full['sensitive'] = np.ones(len(full_dataset))
        copy_df['sensitive'] = np.where(copy_df[sensitive_features].eq(0.0).all(1, skipna=True), 0.0, 1.0)
        copy_full['sensitive'] = np.where(copy_full[sensitive_features].eq(0.0).all(1, skipna=True), 0.0, 1.0)

        roc = RejectOptionClassifier(prot_attr='sensitive', threshold=0.5, margin=0.1)

        pp = PostProcessingMeta(estimator=model, postprocessor=roc, val_size=0.1)

        pp.fit(X=isolate_features(copy_full.set_index('sensitive')),
               y=isolate_prediction(copy_full).values.ravel(), pos_label=1,
               priv_group=0.0)

        pred = pp.predict(isolate_features(copy_df.set_index('sensitive')))

        pred_df = isolate_features(df)

        pred_df[output] = pred

        return prediction, pred_df

    else:
        roc = RejectOptionClassifier(prot_attr=sensitive_features, threshold=0.5, margin=0.1)

        pp = PostProcessingMeta(estimator=model, postprocessor=roc, val_size=0.1)

        pp.fit(X=isolate_features(full_dataset.set_index(sensitive_features)),
               y=isolate_prediction(full_dataset).values.ravel(), pos_label=1,
               priv_group=0.0)

        pred = pp.predict(isolate_features(df.set_index(sensitive_features)))

        pred_df = isolate_features(df)

        pred_df[output] = pred

        return prediction, pred_df


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
