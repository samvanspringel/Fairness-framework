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
