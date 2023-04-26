import hiring_ml as hire

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
    'Origin': "origin"
}

SPLIT_PERCENTAGE = 0.2


def combine_row(row, sensitive_features):
    s = ""
    for i in sensitive_features:
        s += row[i] + ", "
    return s


def add_description_column(df, sf):
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    df['description'] = df.apply(lambda row: combine_row(row, sensitive_features), axis=1)
    return df


def descriptive_df(df):
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace([1, 2], ['Male', 'Female'])
    if 'origin' in df.columns:
        df['origin'] = df['origin'].replace([1, 2, 3], ['Belgian', 'FB', 'Foreign'])
    if 'degree' in df.columns:
        df['degree'] = df['degree'].replace([0, 1], ['No', 'Yes'])
    if 'extra_degree' in df.columns:
        df['extra_degree'] = df['extra_degree'].replace([0, 1], ['No', 'Yes'])
    if 'qualified' in df.columns:
        df['qualified'] = df['qualified'].replace([0, 1], ['No', 'Yes'])

    return df


def train_model(data, model):
    trained_model = hire.train_model(data, model_mapping[model])
    return trained_model


def make_prediction_with_model(model, test_data):
    prediction = hire.make_prediction(test_data, model)
    return prediction


def compute_fairness(prediction, sf, output_label):
    return hire.calculate_fairness(prediction, sf, output_label)


def pipeline(model, data, sf):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    test_data = data.iloc[:int(len(data) * SPLIT_PERCENTAGE), :]
    training_data = data.iloc[int(len(data) * SPLIT_PERCENTAGE):, :]
    trained_model = train_model(training_data, model)
    prediction = make_prediction_with_model(trained_model, test_data)
    unbiased_evaluation = prediction[0]
    model_prediction = prediction[1]

    results['unbiased_evaluation'] = unbiased_evaluation
    results['model_prediction'] = model_prediction
    results['fairness_notions_unbiased'] = compute_fairness(unbiased_evaluation, sensitive_features, 'qualified')
    results['fairness_notions_model'] = compute_fairness(model_prediction, sensitive_features, 'qualified')
    results['count_qualified_unbiased'] = hire.count_hired(unbiased_evaluation, sensitive_features)
    results['count_qualified_model'] = hire.count_hired(model_prediction, sensitive_features)
    return results


def load_scenario(scenario, sf, model):
    results = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    environment = hire.setup_environment(scenario, sensitive_features)

    training_data = hire.rename_goodness(hire.generate_training_data(environment, 1000))
    test_data = hire.rename_goodness(hire.generate_test_data(environment, 1000))

    trained_model = hire.train_model(training_data, model_mapping[model])
    predictions = hire.make_prediction(test_data, trained_model)

    output_label = "qualified"
    fairness_notions = hire.calculate_fairness(predictions[1], sensitive_features, output_label)

    dataframes_count_hired = hire.count_hired(predictions[1], sensitive_features)
    # cm = hire.generate_cm(predictions)

    scenarios_elements[scenario] = {'Dataset': {  # 'cm-women-hired': cm[0],
        # 'cm-men-hired': cm[1],
        'df-hired': dataframes_count_hired[0],
        'fairness': fairness_notions[0],
        'df': predictions[0]
    },

        'Decision tree': {  # 'cm-women-hired': cm[2],
            # 'cm-men-hired': cm[3],
            'df-hired': dataframes_count_hired[1],
            'fairness': fairness_notions[1],
            'df': predictions[1]
        },
        'k-Nearest neighbours': {  # 'cm-women-hired': cm[4],
            # 'cm-men-hired': cm[5],
            'df-hired': dataframes_count_hired[2],
            'fairness': fairness_notions[2],
            'df': predictions[2]
        },
    }
    return scenarios_elements
