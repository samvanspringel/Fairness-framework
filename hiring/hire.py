from hiring import State, Scenario
from hiring.features import *
import numpy as np


class HiringActions(Enum):
    """Enumeration for all actions in the hiring setting"""
    reject = 0
    hire = 1


class HiringScenario(Scenario):
    """A scenario for generating data for a hiring setting."""

    def __init__(self, description="Baseline Hiring Scenario", seed=None, episode_length=1000):
        # Super call
        super(HiringScenario, self).__init__(seed)
        # Scenario Setup
        self.description = description
        self.gender_desc = GenderDescription()
        self.age_desc = AgeDescription()
        self.degree_desc = DegreeDescription()
        self.extra_degree_desc = ExtraDegreeDescription()
        self.experience_desc = ExperienceDescription()

        # Uitbreiding
        self.origin_desc = OriginDescription()

        self.features = [HiringFeature.gender, HiringFeature.age, HiringFeature.degree,
                         HiringFeature.extra_degree, HiringFeature.experience,
                         HiringFeature.origin]
        self.feature_biases = []
        self.reward_biases = []
        self.goodness_noise = GOODNESS_NOISE
        self.noise_reject = NOISE_REJECT
        self.noise_hire = NOISE_HIRE
        self.goodness_clip = (0, 10)
        self.threshold = 5

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
        gender, gender_feature = self.gender_desc.generate_and_feature(self.rng)
        age, age_feature = self.age_desc.generate_and_feature(self.rng, gender)
        degree, degree_feature = self.degree_desc.generate_and_feature(self.rng, gender)
        extra_degree, extra_degree_feature = self.extra_degree_desc.generate_and_feature(self.rng, gender, degree)
        experience, experience_feature = self.experience_desc.generate_and_feature(self.rng, gender, age)

        origin, origin_feature = self.origin_desc.generate_and_feature(self.rng)


        # Create a sample
        sample = {
            gender_feature: gender,
            age_feature: age,
            degree_feature: degree,
            extra_degree_feature: extra_degree,
            experience_feature: experience,
            origin_feature: origin
        }
        state = State(sample)
        return state

    def calc_goodness(self, candidate_state: State):
        gender, age, degree, extra_degree, experience, origin = candidate_state.get_features(self.features)

        # younger individuals better? Could be hired at a lower salary + can work for longer after hiring
        # => maximum age would be worst as they will retire sooner
        max_age = max(self.age_desc.max_age.values())
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
                    extra_degree_factor * extra_degree_factor +
                    experience * experience_factor +
                    degree * (experience * experience_factor) +
                    extra_degree * (experience * experience_factor) +
                    degree * extra_degree * (degree_factor * extra_degree_factor * experience_factor)
                    + noise)

        # Add bias
        for bias in self.feature_biases:
            goodness += bias.get_bias(candidate_state)

        #
        goodness = goodness / 1.5
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

    def similarity_metric(self, state1: State, state2: State):
        """The similarity metric between states. Using Euclidean distance,
        with increased distance for degrees to indicate they're important attributes as it separates
        candidates of similar age/experience a lot more"""
        features1 = state1[HiringFeature.gender].value * 0.5, state1[HiringFeature.age] * 0.1, state1[HiringFeature.experience] * 0.1, \
                    state1[HiringFeature.degree] * 10, state1[HiringFeature.extra_degree] * 1.5
        features2 = state2[HiringFeature.gender].value * 0.5, state2[HiringFeature.age] * 0.1, state2[HiringFeature.experience] * 0.1, \
                    state2[HiringFeature.degree] * 0.5, state2[HiringFeature.extra_degree] * 1.5

        # for f1, f2 in zip(features1, features2):
        #     print(f1, f2, (f1 - f2) ** 2)
        d = np.sqrt(np.sum([(f1 - f2) ** 2 for f1, f2 in zip(features1, features2)]))
        return d

    def get_individual(self, state: State):
        return state.sample_dict


class MDPState(State):
    """A sample from a scenario"""

    def __init__(self, sample):
        # Super call
        super(MDPState, self).__init__(sample)
        self.sample_dict = sample

    def __str__(self):
        features = [key.name if isinstance(key, Enum) else key for key in self.sample_dict.keys()]
        vals = self.to_array()
        lst = [f"{f}: {v}" for f, v in zip(features, vals)]
        s = ", ".join(lst)
        return f"<{s}>"


class HiringHistory(object):
    """A history of encountered states"""

    def __init__(self, actions, features, n_bins=10, has_ground_truth=True):
        self.actions = actions
        self.features = features
        self.n_bins = n_bins
        self.hist = {f: [] for f in features}

    def add_sample(self, state):  # , action, ground_truth=None):
        # if self.has_ground_truth and ground_truth is None:
        #     raise ValueError(f"History requires ground truth")

        for feature, value in state.sample_dict.items():
            self.hist[feature].append(value)

    def update_state(self, state):
        self.add_sample(state)
        for feature, values in self.hist.items():
            new_values = [v.value if isinstance(v, Enum) else v for v in values]
            if isinstance(self.n_bins, dict):
                n_bin_edges, n_bins = self.n_bins[feature]
            else:
                n_bins = self.n_bins
                n_bin_edges = None
            counts, bin_edges = np.histogram(new_values, bins=n_bins, density=True)
            counts = counts * np.diff(bin_edges)
            for i, (count, bin) in enumerate(zip(counts, bin_edges)):
                if isinstance(self.n_bins, dict):
                    if n_bin_edges != 0:
                        state.sample_dict[f"{feature.name}_bin_{i}_edge"] = bin
                    state.sample_dict[f"{feature.name}_bin_{i}_count"] = count
                else:
                    state.sample_dict[f"{feature.name}_bin_{i}_count"] = count
                    # Binary features don't need additional bin edge information
                    if feature not in [HiringFeature.gender, HiringFeature.degree, HiringFeature.extra_degree]:
                        state.sample_dict[f"{feature.name}_bin_{i}_edge"] = bin
        return MDPState(state.sample_dict)


class HiringScenarioMDP(HiringScenario):
    def __init__(self, description="Baseline Hiring Scenario", seed=None, n_bins=10, episode_length=1000):
        # Super call
        super(HiringScenarioMDP, self).__init__(description, seed, episode_length)
        self.n_bins = n_bins
        self.history = HiringHistory([HiringActions.reject, HiringActions.hire],
                                     [HiringFeature.gender, HiringFeature.age, HiringFeature.degree,
                                      HiringFeature.extra_degree, HiringFeature.experience,
                                      HiringFeature.origin], n_bins=self.n_bins)

    def generate_sample(self):
        # Get MDP state
        state = super(HiringScenarioMDP, self).generate_sample()
        mdp_state = self.history.update_state(state)
        return mdp_state

    def get_individual(self, state: State):
        features = [HiringFeature.gender, HiringFeature.age, HiringFeature.degree, HiringFeature.extra_degree,
                    HiringFeature.experience, HiringFeature.origin]
        individual = {feature: state[feature] for feature in features}
        return individual
