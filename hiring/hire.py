from hiring import State, Scenario
from hiring.features import *
import numpy as np


class HiringActions(Enum):
    """Enumeration for all actions in the hiring setting"""
    reject = 0
    hire = 1


class HiringScenario(Scenario):
    """A scenario for generating data for a hiring setting."""

    def __init__(self, description="Baseline Hiring Scenario", seed=None, episode_length=1000,
                 applicant_generator=None, threshold=5):
        # Super call
        super(HiringScenario, self).__init__(seed)
        # Scenario Setup
        self.description = description
        self.applicant_generator = ApplicantGenerator(seed=self.seed) \
            if applicant_generator is None else applicant_generator
        self.features = [HiringFeature.nationality, HiringFeature.age, HiringFeature.gender, HiringFeature.married,
                         HiringFeature.degree, HiringFeature.extra_degree, HiringFeature.experience]
        self.nominal_features = [HiringFeature.gender, HiringFeature.married, HiringFeature.nationality]
        self.numerical_features = [f for f in self.features if f not in self.nominal_features]
        self.feature_biases = []
        self.reward_biases = []
        self.goodness_noise = GOODNESS_NOISE
        self.noise_reject = NOISE_REJECT
        self.noise_hire = NOISE_HIRE
        self.goodness_clip = (0, 10)
        self.threshold = threshold

        self._state = None
        self._t = 0
        self.episode_length = episode_length
        # reject/hire
        self.actions = [a for a in HiringActions]

    def reset(self):
        self._t = 0
        self._state = self.generate_sample()
        return self._state

    def step(self, action):
        goodness = self.calc_goodness(self._state)
        rewards = self.calculate_rewards(self._state, goodness)
        reward = rewards[HiringActions(action)]
        next_state = self.generate_sample()
        self._state = next_state
        self._t += 1
        done = self._t > self.episode_length
        info = {
            "goodness": goodness,
            "true_action": 1 if rewards[HiringActions.hire] > rewards[HiringActions.reject] else 0
        }

        return next_state, reward, done, info

    def generate_sample(self):
        # Create a sample
        sample = self.applicant_generator.sample()
        state = State(sample)
        return state

    def calc_goodness(self, candidate_state: State):
        nationality, age, gender, married, degree, extra_degree, experience = candidate_state.get_features(self.features)

        # younger individuals better? Could be hired at a lower salary + can work for longer after hiring
        # => maximum age would be worst as they will retire sooner
        max_age = 65
        age_factor = 0.15
        # degree is better?
        degree_factor = 1
        # extra degree is better?
        extra_degree_factor = 1.5
        # experience is better?
        experience_factor = 1 / 4

        # Score
        noise = self.rng.normal() * self.goodness_noise
        goodness = ((max_age - age) * age_factor +
                    degree * degree_factor +
                    extra_degree * extra_degree_factor +
                    experience * experience_factor +
                    degree * (experience * experience_factor) +
                    extra_degree * (experience * experience_factor) +
                    degree * extra_degree * (degree_factor * extra_degree_factor * experience_factor)
                    + noise)

        # Add bias
        for bias in self.feature_biases:
            goodness += bias.get_bias(candidate_state)

        #
        goodness = goodness / 2.5
        goodness = np.clip(goodness, *self.goodness_clip)
        return goodness

    def calculate_rewards(self, sample: State, goodness):
        # Reward for hiring
        reward_noise = self.rng.normal() * self.noise_hire
        reward_hire = (goodness - self.threshold) + reward_noise

        # Add bias
        for bias in self.reward_biases:
            reward_hire += bias.get_bias(sample)

        # Reward for rejecting candidate (depends on scenario)
        # If reward_hiring = -10 => goodness says don't hire, then reward_reject = 10 +- noise
        # reject_noise = self.rng.normal() * self.noise_reject
        reward_reject = -reward_hire  # + reject_noise
        # Return the rewards
        rewards = {HiringActions.reject: reward_reject, HiringActions.hire: reward_hire}
        return rewards

    def similarity_metric(self, state1: State, state2: State, distance="HMOM", alpha=1.0, exp=True):
        num1 = np.array(self._normalise_features(state1, self.numerical_features))
        nom1 = np.array(state1.get_features(self.nominal_features))
        num2 = np.array(self._normalise_features(state2, self.numerical_features))
        nom2 = np.array(state2.get_features(self.nominal_features))

        # Heterogeneous Euclidean-Overlap Metric (HEOM)
        if distance == 'HEOM':
            d = np.sum(np.abs(num1 - num2)) + np.sum(nom1 != nom2)
        # Heterogeneous Manhattan-Overlap Metric (HMOM)
        elif distance == 'HMOM':
            d = np.sum((num1 - num2) ** 2) + np.sum(nom1 != nom2)
        else:
            raise ValueError(f"Expected distance: HEOM or HMOM, got: {distance}")

        return np.exp(-alpha * d) if exp else d

    def _normalise_features(self, state: State, features: List[HiringFeature] = None):
        new_values = self.applicant_generator.normalise_features(state.sample_dict, features)
        new_values = np.array([new_values[f] for f in features])
        return new_values

    def get_individual(self, state: State):
        return state.sample_dict
