import hiring_ml as hire
import pandas
import numpy as np

# Modellen
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# Mitigation
from aif360.algorithms.preprocessing import *
from aif360.algorithms.postprocessing import *

# Gebruikte modellen
models = [KNeighborsClassifier(n_neighbors=3), tree.DecisionTreeClassifier()]

model_mapping = {
    'Decision tree': tree.DecisionTreeClassifier(),
    'k-Nearest neighbours': KNeighborsClassifier(n_neighbors=3)
}

mitigation_mapping = {
    "Pre-processing: Sample Reweighing": hire.sample_reweighing,
    "Post-processing: Calibrated Equalized Odds": hire.calibrated_equalized_odds,
    "Post-processing: Reject Option Classification": hire.reject_option_classification,
}

sensitive_feature_mapping = {
    'Gender': "gender",
    'Nationality': "nationality",
    'Age': "age",
    'Married': "married"
}

SPLIT_PERCENTAGE = 0.1
N = 2000
AMOUNT_OF_SEEDS = 1


def combine_row(row, sensitive_features):
    s = ""
    for i in sensitive_features:
        s += row[i] + ", "
    return s


def add_description_column(df, sf):
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    df['description'] = df.apply(lambda row: combine_row(row, sensitive_features), axis=1)
    return df


def descriptive_age(data):
    df = data.copy()

    if 'age' in df.columns:
        df['age'] = df['age'].replace([0, 1], ['Under 50', 'Over 50'])

    return df


def descriptive_columns(data):
    df = data.copy()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace([0, 1], ['Male', 'Female'])
    if 'nationality' in df.columns:
        df['nationality'] = df['nationality'].replace([0, 1], ['Belgian', 'Foreign'])
    if 'degree' in df.columns:
        df['degree'] = df['degree'].replace([0, 1], ['No', 'Yes'])
    if 'extra_degree' in df.columns:
        df['extra_degree'] = df['extra_degree'].replace([0, 1], ['No', 'Yes'])
    if 'married' in df.columns:
        df['married'] = df['married'].replace([0, 1], ['Not married', 'Married'])
    return df


def descriptive_df(data):
    df = data.copy()
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace([0, 1], ['Male', 'Female'])
    if 'nationality' in df.columns:
        df['nationality'] = df['nationality'].replace([0, 1], ['Belgian', 'Foreign'])
    if 'degree' in df.columns:
        df['degree'] = df['degree'].replace([0, 1], ['No', 'Yes'])
    if 'extra_degree' in df.columns:
        df['extra_degree'] = df['extra_degree'].replace([0, 1], ['No', 'Yes'])
    if 'married' in df.columns:
        df['married'] = df['married'].replace([0, 1], ['Not married', 'Married'])
    if 'qualified' in df.columns:
        df['qualified'] = df['qualified'].replace([0, 1], ['No', 'Yes'])
    return df


def train_model(data, model):
    trained_model = hire.train_model(data, model_mapping[model])
    return trained_model


def make_prediction_with_model(model, test_data):
    prediction = hire.make_prediction(test_data, model)
    return prediction


def compute_fairness(simulator_eval, prediction, sf, output_label):
    return hire.calculate_fairness(simulator_eval, prediction, sf, output_label)


def map_age(df):
    if 'age' in df.columns:
        df.loc[df['age'] < 50, 'age'] = 0.0
        df.loc[df['age'] >= 50, 'age'] = 1.0
    return df


def average(list):
    return sum(list) / len(list)


def pipeline(model, data, sf):
    results = {}
    data = map_age(data)
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    test_data = data.iloc[:int(len(data) * SPLIT_PERCENTAGE), :]
    training_data = data.iloc[int(len(data) * SPLIT_PERCENTAGE):, :]

    if model == 'Dataset':
        results['simulator_evaluation'] = test_data
        results['model_prediction'] = test_data
        results['fairness_notions'] = compute_fairness(test_data, test_data, sensitive_features,
                                                       'qualified')
        results['count_qualified_model'] = hire.count_hired(test_data, sensitive_features)
        results['confusion_matrix'] = hire.generate_cm(test_data, test_data)
        results['fitted_model'] = 'Dataset'
    else:
        trained_model = train_model(training_data, model)
        prediction = make_prediction_with_model(trained_model, test_data)

        simulator_evaluation = prediction[0]
        model_prediction = prediction[1]

        results['simulator_evaluation'] = simulator_evaluation
        results['model_prediction'] = model_prediction
        results['fairness_notions_unbiased'] = compute_fairness(simulator_evaluation, simulator_evaluation,
                                                                sensitive_features, 'qualified')
        results['fairness_notions'] = compute_fairness(simulator_evaluation, model_prediction, sensitive_features,
                                                       'qualified')
        results['count_qualified_unbiased'] = hire.count_hired(simulator_evaluation, sensitive_features)

        results['count_qualified_model'] = hire.count_hired(model_prediction, sensitive_features)
        results['confusion_matrix'] = hire.generate_cm(simulator_evaluation, model_prediction)
        results['fitted_model'] = trained_model

    return results


def mitigation_pipeline(full_dataset, dataset, prediction, sf, fitted_model, technique):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))

    if fitted_model == 'Dataset':
        results['original_prediction'] = dataset
        results['mitigated_prediction'] = dataset
        results['fairness_notions'] = compute_fairness(dataset, dataset,
                                                       sensitive_features, 'qualified')
        results['count_qualified_mitigated'] = hire.count_hired(dataset, sensitive_features)

    else:
        mitigation = mitigation_mapping[technique](full_dataset, dataset, prediction, sensitive_features, 'qualified',
                                                   fitted_model)

        model_prediction = mitigation[0]
        mitigated_prediction = mitigation[1]

        results['original_prediction'] = model_prediction
        results['mitigated_prediction'] = mitigated_prediction
        results['fairness_notions'] = compute_fairness(dataset, mitigated_prediction,
                                                       sensitive_features, 'qualified')

        results['count_qualified_mitigated'] = hire.count_hired(mitigated_prediction, sensitive_features)

    return results


def combine_count_df(count_dataframes):
    count = count_dataframes[0]

    for d in range(1, AMOUNT_OF_SEEDS):
        count['qualified'] = count['qualified'] + count_dataframes[d]['qualified']
    count['qualified'] = count['qualified'] / AMOUNT_OF_SEEDS
    return count


def combine_confusion_matrices(confusion_matrices):
    cm = confusion_matrices[0]
    for c in range(1, AMOUNT_OF_SEEDS):
        for i in range(2):
            for j in range(2):
                cm[i][j] = cm[i][j] + confusion_matrices[c][i][j]
    for i in range(2):
        for j in range(2):
            cm[i][j] = (cm[i][j] / AMOUNT_OF_SEEDS)
    return cm


def load_scenario(scenario, sf, model):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))

    fairness = {"Fairness notions": [0, 0, 0, 0]}

    fairnesses_model = pandas.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality',
                                                              'Equal opportunity', 'Inconsistency'])

    fairnesses_dataset = pandas.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality',
                                                                'Equal opportunity', 'Inconsistency'])

    fairnesses_mitigation = pandas.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality',
                                                                   'Equal opportunity', 'Inconsistency'])

    accuracy = 0

    counts_model = pandas.DataFrame()
    counts_dataset = pandas.DataFrame()
    counts_mitigated = pandas.DataFrame()

    confusion_matrices = []

    for i in range(0, AMOUNT_OF_SEEDS):
        print(f"Processing for seed {i}")
        environment = hire.setup_environment(scenario, sensitive_features, i)
        candidates = environment.create_dataset(N, show_goodness=True, rounding=5)

        data = hire.rename_goodness(candidates)
        data = map_age(data)
        test_data = data.iloc[:int(len(data) * SPLIT_PERCENTAGE), :]
        training_data = data.iloc[int(len(data) * SPLIT_PERCENTAGE):, :]

        trained_model = train_model(training_data, model)
        prediction = make_prediction_with_model(trained_model, test_data)

        simulator_evaluation = prediction[0]
        model_prediction = prediction[1]

        # Model prediction
        fairness_notions_seed = compute_fairness(simulator_evaluation, model_prediction,
                                                 sensitive_features, 'qualified')
        fairnesses_model[f'Fairness notions_{i}'] = fairness_notions_seed['Fairness notions']

        if i == 0:
            counts_model = hire.count_hired(model_prediction, sensitive_features)
        else:
            counts_model[f'Counts_{i}'] = hire.count_hired(model_prediction, sensitive_features)['qualified']

        cm_accuracy = hire.generate_cm(simulator_evaluation, model_prediction)

        confusion_matrices.append(cm_accuracy[0])

        accuracy += cm_accuracy[1]

        # Dataset
        fairness_dataset = compute_fairness(simulator_evaluation, simulator_evaluation,
                                            sensitive_features, 'qualified')

        fairnesses_dataset[f'Fairness notions_{i}'] = fairness_dataset['Fairness notions']

        if i == 0:
            counts_dataset = hire.count_hired(simulator_evaluation, sensitive_features)
        else:
            counts_dataset[f'Counts_{i}'] = hire.count_hired(simulator_evaluation, sensitive_features)['qualified']

        if scenario == 'Bias':
            mitigation = hire.reject_option_classification(data, simulator_evaluation, model_prediction,
                                                        sensitive_features, 'qualified', trained_model)
            mitigated_prediction = mitigation[1]

            fairness_mitigation = compute_fairness(simulator_evaluation, mitigated_prediction,
                                                   sensitive_features, 'qualified')

            fairnesses_mitigation[f'Fairness notions_mitigation{i}'] = fairness_mitigation['Fairness notions']

            if i == 0:
                counts_mitigated = hire.count_hired(mitigated_prediction, sensitive_features)
            else:
                counts_mitigated[f'Counts_{i}'] = hire.count_hired(mitigated_prediction, sensitive_features)['qualified']

    #####################################################################################################
    ########################################### MODEL ##############################################
    #####################################################################################################

    confusion_matrix = combine_confusion_matrices(confusion_matrices)

    fairnesses_model = fairnesses_model.drop(['Fairness notions'], axis=1)
    fm_copy_statistics = fairnesses_model.copy()
    fairnesses_model['Mean'] = fm_copy_statistics.mean(axis=1)
    fairnesses_model['Standard deviation'] = fm_copy_statistics.std(axis=1, numeric_only=True)
    results['fairness_notions_model'] = fairnesses_model

    results['accuracy'] = accuracy / AMOUNT_OF_SEEDS

    average_df_model = counts_model.copy()
    for sf in sensitive_features:
        average_df_model = average_df_model.drop([sf], axis=1)
    df_model_statistics_copy = average_df_model.copy()
    average_df_model['qualified'] = df_model_statistics_copy.mean(axis=1)
    average_df_model['Standard deviation'] = df_model_statistics_copy.std(axis=1, numeric_only=True)

    counts_model['qualified'] = average_df_model['qualified']
    counts_model['Standard deviation'] = average_df_model['Standard deviation']

    for c in counts_model.columns:
        if not (c in sensitive_features or c == 'qualified' or c == 'Standard deviation'):
            counts_model = counts_model.drop(c, axis=1)

    results['count_qualified_model'] = counts_model

    results['confusion_matrix'] = confusion_matrix

    #####################################################################################################
    ########################################### DATASET ##############################################
    #####################################################################################################

    fairnesses_dataset = fairnesses_dataset.drop(['Fairness notions'], axis=1)
    fd_copy_statistics = fairnesses_dataset.copy()
    fairnesses_dataset['Mean'] = fd_copy_statistics.mean(axis=1)
    fairnesses_dataset['Standard deviation'] = fd_copy_statistics.std(axis=1, numeric_only=True)
    results['fairness_notions_dataset'] = fairnesses_dataset

    average_df_dataset = counts_dataset.copy()
    for sf in sensitive_features:
        average_df_dataset = average_df_dataset.drop([sf], axis=1)
    df_dataset_statistics_copy = average_df_dataset.copy()
    average_df_dataset['qualified'] = df_dataset_statistics_copy.mean(axis=1)
    average_df_dataset['Standard deviation'] = df_dataset_statistics_copy.std(axis=1, numeric_only=True)

    counts_dataset['qualified'] = average_df_dataset['qualified']
    counts_dataset['Standard deviation'] = average_df_dataset['Standard deviation']

    for c in counts_dataset.columns:
        if not (c in sensitive_features or c == 'qualified' or c == 'Standard deviation'):
            counts_dataset = counts_dataset.drop(c, axis=1)

    results['count_qualified_dataset'] = counts_dataset

    #####################################################################################################
    ########################################### MITIGATION ##############################################
    #####################################################################################################
    if scenario == 'Bias':
        average_df_mitigated = counts_mitigated.copy()
        for sf in sensitive_features:
            average_df_mitigated = average_df_mitigated.drop([sf], axis=1)
        df_mitigated_statistics_copy = average_df_mitigated.copy()
        average_df_mitigated['qualified'] = df_mitigated_statistics_copy.mean(axis=1)
        average_df_mitigated['Standard deviation'] = df_mitigated_statistics_copy.std(axis=1, numeric_only=True)

        counts_mitigated['qualified'] = average_df_mitigated['qualified']
        counts_mitigated['Standard deviation'] = average_df_mitigated['Standard deviation']

        for c in counts_mitigated.columns:
            if not (c in sensitive_features or c == 'qualified' or c == 'Standard deviation'):
                counts_mitigated = counts_mitigated.drop(c, axis=1)

        results['count_qualified_mitigated'] = counts_mitigated

        fairnesses_mitigation = fairnesses_mitigation.drop(['Fairness notions'], axis=1)
        fm_mitigation_copy_statistics = fairnesses_mitigation.copy()
        fairnesses_mitigation['Mean'] = fm_mitigation_copy_statistics.mean(axis=1)
        fairnesses_mitigation['Standard deviation'] = fm_mitigation_copy_statistics.std(axis=1, numeric_only=True)
        results['fairness_notions_model_after_mitigation'] = fairnesses_mitigation

    return results
