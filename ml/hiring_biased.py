import random

import sklearn
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import pprint

# Modellen
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor as mlpReg

import numpy as np
# import torch

from hiring import FeatureBias
from hiring.features import HiringFeature, Gender, GenderDescription
from hiring.hire import HiringScenario


def pipeline(training_data):
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

if __name__ == '__main__':
    # Example feature bias for Machine Learning
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    env = HiringScenario(seed=seed)

    # Historically less women
    env.description = "Historically less women"
    env.gender_desc = GenderDescription(prob_male=0.7, prob_female=0.3)
    # Men were considered better than Women with the same features besides gender: give higher score
    env.feature_biases = [FeatureBias(HiringFeature.gender, Gender.male, 1.0)]

    num_samples = 100

    training_data = env.create_dataset(num_samples, show_goodness=True, rounding=5)
    pipeline(training_data)
