import hiring_ml as hire
import pandas
import numpy as np

# Modellen
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# Gebruikte modellen
models = [KNeighborsClassifier(n_neighbors=3), tree.DecisionTreeClassifier()]

model_mapping = {
    'Decision tree': tree.DecisionTreeClassifier(),
    'k-Nearest neighbours': KNeighborsClassifier(n_neighbors=3)
}

sensitive_feature_mapping = {
    'Gender': "gender",
    'Nationality': "nationality",
    'Age': "age",
    'Married': "married"
}

SPLIT_PERCENTAGE = 0.1
N = 20000
AMOUNT_OF_SEEDS = 30


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


def preprocess_dataset(test_data):
    return hire.preprocess_test_data(test_data)


def map_age(df):
    if 'age' in df.columns:
        df.loc[df['age'] < 50, 'age'] = 0
        df.loc[df['age'] >= 50, 'age'] = 1
    return df


def average(list):
    return sum(list) / len(list)


def pipeline(model, data, sf):
    results = {}
    data = map_age(data)
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    test_data = data.iloc[:int(len(data) * SPLIT_PERCENTAGE), :]
    training_data = data.iloc[int(len(data) * SPLIT_PERCENTAGE):, :]

    trained_model = train_model(training_data, model)
    prediction = make_prediction_with_model(trained_model, test_data)

    simulator_evaluation = prediction[0]
    model_prediction = prediction[1]

    results['simulator_evaluation'] = simulator_evaluation
    results['model_prediction'] = model_prediction
    results['fairness_notions_unbiased'] = compute_fairness(simulator_evaluation, model_prediction, sensitive_features,
                                                            'qualified')
    results['fairness_notions'] = compute_fairness(simulator_evaluation, model_prediction, sensitive_features,
                                                   'qualified')
    results['count_qualified_unbiased'] = hire.count_hired(simulator_evaluation, sensitive_features)

    results['count_qualified_model'] = hire.count_hired(model_prediction, sensitive_features)
    results['confusion_matrix'] = hire.generate_cm(simulator_evaluation, model_prediction)
    results['fitted_model'] = trained_model

    return results


def mitigation_pipeline(technique, dataset, prediction_df, sf, fitted_model):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))

    test_data = dataset.iloc[:int(len(dataset) * SPLIT_PERCENTAGE), :]
    training_data = dataset.iloc[int(len(dataset) * SPLIT_PERCENTAGE):, :]

    prediction = hire.sample_reweighing(dataset, sensitive_features, 'qualified', fitted_model)  # TODO: added fitted_model
    simulator_evaluation = prediction[0]
    mitigated_prediction = prediction[1]

    results['simulator_evaluation'] = simulator_evaluation
    results['model_prediction'] = mitigated_prediction
    results['fairness_notions'] = compute_fairness(simulator_evaluation, mitigated_prediction, sensitive_features,
                                                   'qualified')

    results['count_qualified_model'] = hire.count_hired(mitigated_prediction, sensitive_features)

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
            cm[i][j] = cm[i][j]/AMOUNT_OF_SEEDS
    return cm


def load_scenario(scenario, sf, model):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))

    fairness = {"Fairness notions": [0, 0, 0, 0]}

    fairness_df = pandas.DataFrame(data=fairness, index=['Statistical parity', 'Predictive equality',
                                                         'Equal opportunity', 'Inconsistency'])
    accuracy = 0

    count_dataframes = []
    confusion_matrices = []

    for i in range(0, AMOUNT_OF_SEEDS):
        print(f"Processing for seed {i}")
        environment = hire.setup_environment(scenario, sensitive_features, i)
        candidates = environment.create_dataset(10000, show_goodness=True, rounding=5)

        data = hire.rename_goodness(candidates)
        data = map_age(data)
        test_data = data.iloc[:int(len(data) * SPLIT_PERCENTAGE), :]
        training_data = data.iloc[int(len(data) * SPLIT_PERCENTAGE):, :]

        trained_model = train_model(training_data, model)
        prediction = make_prediction_with_model(trained_model, test_data)

        simulator_evaluation = prediction[0]
        model_prediction = prediction[1]

        fairness_notions_seed = compute_fairness(simulator_evaluation, model_prediction,
                                                 sensitive_features, 'qualified')

        fairness_df['Fairness notions'] = fairness_df['Fairness notions'] + fairness_notions_seed['Fairness notions']

        count_df = hire.count_hired(model_prediction, sensitive_features)
        cm_accuracy = hire.generate_cm(simulator_evaluation, model_prediction)

        confusion_matrices.append(cm_accuracy[0])

        accuracy += cm_accuracy[1]

        count_dataframes.append(count_df)

    count = combine_count_df(count_dataframes)

    confusion_matrix = combine_confusion_matrices(confusion_matrices)

    fairness_df['Fairness notions'] = fairness_df['Fairness notions'] / AMOUNT_OF_SEEDS

    results['fairness_notions'] = fairness_df

    results['accuracy'] = accuracy / AMOUNT_OF_SEEDS

    results['count_qualified_model'] = count

    results['confusion_matrix'] = confusion_matrix

    return results
