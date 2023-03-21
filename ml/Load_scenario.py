import hiring_ml as hire

# Modellen
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

# Gebruikte modellen
models = [KNeighborsClassifier(n_neighbors=3), tree.DecisionTreeClassifier()]

sensitive_feature_mapping = {
    'Gender': "gender",
    'Origin': "origin"
}

def load_scenario(scenario, sf):
    scenarios_elements = {}
    sensitive_features = list(map(lambda feature: sensitive_feature_mapping[feature], sf))
    environment = hire.setup_environment(scenario, sensitive_features)
    training_data = hire.generate_training_data(environment, 1000)
    test_data = hire.rename_goodness(hire.generate_test_data(environment, 1000))

    trained_models = hire.train_models(training_data, models)
    predictions = hire.make_predictions(test_data, trained_models)

    output_label = "qualified"
    fairness_notions = hire.calculate_fairness(predictions, sensitive_features, output_label)

    dataframes_count_hired = hire.count_hired(predictions)
    cm = hire.generate_cm(predictions)

    scenarios_elements[scenario] = {'Dataset': {'cm-women-hired': cm[0],
                                                'cm-men-hired': cm[1],
                                                'df-hired': dataframes_count_hired[0],
                                                'fairness': fairness_notions[0],
                                                'df': predictions[0]
                                                },

                                    'Decision tree': {'cm-women-hired': cm[2],
                                                      'cm-men-hired': cm[3],
                                                      'df-hired': dataframes_count_hired[1],
                                                      'fairness': fairness_notions[1],
                                                      'df': predictions[1]
                                                      },
                                    'k-Nearest neighbours': {'cm-women-hired': cm[4],
                                                             'cm-men-hired': cm[5],
                                                             'df-hired': dataframes_count_hired[2],
                                                             'fairness': fairness_notions[2],
                                                             'df': predictions[2]
                                                             },
                                    }
    return scenarios_elements