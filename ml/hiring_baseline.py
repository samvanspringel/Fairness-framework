import random

import numpy as np
# import torch

import sklearn
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import pprint

# Modellen
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor as mlpReg
from sklearn import tree

from hiring.hire import HiringScenario

def convert_goodness(g):
    if (g >= 5): return 1
    else: return 0

def pipeline(training_data, test_data):
    print("\n ------- Training Data ---------")
    print(training_data.head())

    # Aparte lijst voor features en targets
    training_x = training_data.drop(['goodness'], axis=1)
    print("\n ------- Training Features ---------")
    print(training_x.head())

    print("\n ------- Training Targets ---------")
    training_y = training_data[['goodness']]
    print(training_y.head())

    # Cross-validation setup
    amt_folds = 20
    k_partitioning = KFold(n_splits=amt_folds, shuffle=False)

    model_mean_scores = {}

    # Lineare regressie LOS
    lin_reg = linear_model.LinearRegression()

    # Cross-validation score berekenen
    score_lin_reg = cross_val_score(lin_reg, training_x, training_y, cv=k_partitioning, scoring='neg_root_mean_squared_error')
    model_mean_scores["Least Ordinary Squares:"] = repr(np.mean(score_lin_reg))

    print("\n ------- Score lineaire regressie ---------")
    pprint.pprint(model_mean_scores)

    # Neuraal Netwerk
    nn = mlpReg(solver='lbfgs',
                  nesterovs_momentum=False,
                  learning_rate='invscaling',
                  hidden_layer_sizes=10,
                  epsilon=0.0000001,
                  beta_2=0.7,
                  beta_1=0.2,
                  batch_size=10,
                  alpha=0.00001,
                  activation='relu',
                  max_iter=100000)
    scores_nn = cross_val_score(nn, training_x, training_y.values.ravel(), cv=k_partitioning,
                                scoring='neg_root_mean_squared_error')
    model_mean_scores["Neural Network"] = repr(np.mean(scores_nn))

    print("\n ------- Score neuraal netwerk ---------")
    pprint.pprint(model_mean_scores)


    # Decision Tree
    dt = tree.DecisionTreeRegressor()

    # Cross-validation score berekenen
    score_dt = cross_val_score(dt, training_x, training_y, cv=k_partitioning,
                                    scoring='neg_root_mean_squared_error')
    model_mean_scores["Decision Tree:"] = repr(np.mean(score_dt))

    print("\n ------- Score Decision Tree ---------")
    pprint.pprint(model_mean_scores)


    # Test features isoleren om voorspelling te doen met modellen
    test_data_x = test_data.drop(['goodness'], axis=1)

    # Menselijke beoordeling (geen bias)
    prediction_human = test_data[['goodness']]

    # Model voorspelling maken
    lin_reg.fit(training_x, training_y.values.ravel())
    nn.fit(training_x, training_y.values.ravel())
    dt.fit(training_x, training_y.values.ravel())

    predictions_lin_reg = lin_reg.predict(test_data_x)
    predictions_nn = nn.predict(test_data_x)
    predictions_dt = dt.predict(test_data_x)

    # Voorspellingen in dataset met de beoordeling simulator
    predictions = pd.DataFrame(prediction_human)
    predictions['lin_reg'] = predictions_lin_reg
    predictions['nn'] = predictions_nn
    predictions['dt'] = predictions_dt

    #RENAMING
    predictions['goodness'] = predictions['goodness'].map(convert_goodness)
    predictions.rename({'goodness': 'Human hired'}, axis=1, inplace=True)

    predictions['lin_reg'] = predictions['lin_reg'].map(convert_goodness)
    predictions.rename({'lin_reg': 'Linear regression hired'}, axis=1, inplace=True)

    predictions['nn'] = predictions['nn'].map(convert_goodness)
    predictions.rename({'nn': 'Neural network hired'}, axis=1, inplace=True)

    predictions['dt'] = predictions['dt'].map(convert_goodness)
    predictions.rename({'dt': 'Decision tree hired'}, axis=1, inplace=True)

    print("\n ------- VOORSPELLINGEN ---------")
    print(predictions)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    true_hired = predictions['Human hired']
    pred_hired_dt = predictions['Decision tree hired']
    pred_hired_nn = predictions['Neural network hired']
    pred_hired_lin_reg = predictions['Linear regression hired']

    print("\n ------- Confusion matrix decision tree ---------")
    print(confusion_matrix(true_hired, pred_hired_dt))

    print("\n ------- Confusion matrix neural network ---------")
    print(confusion_matrix(true_hired, pred_hired_nn))

    print("\n ------- Confusion matrix linear regression ---------")
    print(confusion_matrix(true_hired, pred_hired_lin_reg))

if __name__ == '__main__':
    # Example baseline for Machine Learning
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    env = HiringScenario(seed=seed)

    num_samples = 100

    training_data = env.create_dataset(num_samples, show_goodness=True, rounding=5)
    test_data = env.create_dataset(num_samples, show_goodness=True, rounding=5)
    pipeline(training_data, test_data)
