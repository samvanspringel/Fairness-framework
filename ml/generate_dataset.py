import random

import numpy as np

from hiring import FeatureBias, hire
from hiring.features import ApplicantGenerator, HiringFeature, Gender, Nationality
# import torch

from hiring.hire import HiringScenario
from ml import hiring_ml

if __name__ == '__main__':
    # Example baseline for Machine Learning
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)

    # Normal distribution
    applicant_generator = ApplicantGenerator(seed=seed, csv="../hiring/data/belgian_population.csv")

    # More men
    #applicant_generator = ApplicantGenerator(seed=seed, csv="../hiring/data/belgian_pop_diff_dist.csv")

    env = HiringScenario(seed=seed, applicant_generator=applicant_generator, threshold=5)

    # Gender bias
    env.feature_biases.append(FeatureBias(HiringFeature.gender, Gender.male, 2.0))

    # Nationality bias
    env.feature_biases.append(FeatureBias(HiringFeature.nationality, Nationality.belgian, 2.0))

    # Age bias
    env.feature_biases.append(FeatureBias(HiringFeature.age, lambda age: age < 50, 2.0))

    # Marital bias
    env.feature_biases.append(FeatureBias(HiringFeature.married, lambda married: married == 0, 2.0))

    num_samples = 20000

    samples = hiring_ml.rename_goodness(env.create_dataset(num_samples, show_goodness=True, rounding=5))

    samples.to_csv(f"datasets/bias_gender_nationality_married_age.csv", index=False)
    #samples.to_csv(f"datasets/base_gender_age_nationality_married.csv")
