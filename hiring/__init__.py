from enum import Enum
from typing import List
import types
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.random import Generator


class Feature(Enum):
    """The feature for a scenario"""
    pass


class FeatureDescription(object):
    """A feature and its description"""
    def __init__(self, feature: Feature, name=None):
        self.feature = feature
        self.name = self.feature.name if name is None else name

    def generate(self, rng: Generator, *args):
        """Generate a value for the feature"""
        raise NotImplementedError

    def generate_and_feature(self, rng: Generator, *args):
        """Generate a value and return it with the feature"""
        return self.generate(rng, *args), self.feature


class State(object):
    """A sample from a scenario"""
    def __init__(self, sample):
        self.sample_dict = sample

    def __str__(self):
        features = [key.name if isinstance(key, Enum) else key for key in self.sample_dict.keys()]
        vals = self.to_array()
        lst = [f"{f}: {v}" for f, v in zip(features, vals)]
        s = ", ".join(lst)
        return f"<{s}>"

    def __getitem__(self, feature: Feature):
        """Get the value of a given feature"""
        return self.sample_dict[feature]

    def to_array(self):
        """Return the state as a numpy array of the values"""
        a = []
        for v in self.sample_dict.values():
            if isinstance(v, Enum):
                a.append(v.value)
            else:
                a.append(v)
        # return np.array(a, dtype=object)
        return np.array(a, dtype=float)
        # return np.array(list(self.sample_dict.values()), dtype=float)

    def to_vector_dict(self):
        """Return the state as a dictionary"""
        d = {}
        for k, v in self.sample_dict.items():
            if isinstance(v, Enum):
                d[k.name] = v.value
            elif isinstance(k, Enum):
                d[k.name] = v
            else:
                d[k] = v
        return d

    def get_feature_names(self, no_hist=False):
        """Return the names of the features"""
        if no_hist:
            return [feature.name for feature in self.sample_dict.keys() if isinstance(feature, Feature)]
        else:
            return [feature.name if isinstance(feature, Feature) else feature for feature in self.sample_dict.keys()]

    def get_features(self, features: List[Feature]):
        """Get the values of the requested features"""
        values = [self[feature] for feature in features]
        return values

    def get_state_features(self, no_hist=False):
        """Return the features of the state"""
        if no_hist:
            return [feature for feature in self.sample_dict.keys() if isinstance(feature, Feature)]
        else:
            return [feature if isinstance(feature, Feature) else feature for feature in self.sample_dict.keys()]


class FeatureBias(object):
    """Bias on goodness score for a given feature"""
    def __init__(self, feature, feature_value, bias):
        self.feature = feature
        self.feature_value = feature_value
        self.bias = bias

    def get_bias(self, state: State):
        """Get the amount of bias to add to the goodness score for the given state"""
        # feature_value is a list of allowed values
        if isinstance(self.feature_value, Iterable) and not isinstance(self.feature_value, str):
            add_bias = lambda v: v in self.feature_value
        # The feature_value is a function
        elif isinstance(self.feature_value, types.FunctionType):
            add_bias = self.feature_value
        # Only if equal to the given feature value
        else:
            add_bias = lambda v: v == self.feature_value

        if add_bias(state[self.feature]):
            return self.bias
        else:
            return 0


class Scenario(object):
    """A scenario for generating data for a given setting"""
    def __init__(self, seed=None):
        # The random generator for the scenario
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)

    def generate_sample(self):
        """Generate a sample"""
        raise NotImplementedError

    def calc_goodness(self, sample: State):
        """Calculate the goodness score for a given sample"""
        raise NotImplementedError

    def calculate_rewards(self, sample: State, goodness):
        """Calculate the rewards for taking different actions in the current state, given the goodness score"""
        raise NotImplementedError

    def step(self, t):
        """Sample a state and return the rewards for corresponding actions in the scenario"""
        state = self.generate_sample()
        goodness = self.calc_goodness(state)
        rewards = self.calculate_rewards(state, goodness)
        return state, rewards

    def create_dataset(self, num_samples, show_goodness=False, show_rewards=False, rounding=None):
        """Generate a dataset with the given number of samples."""
        dataset = []
        features = None
        for t in range(num_samples):
            sample = self.generate_sample()
            entry = list(sample.to_array())
            if features is None:
                features = sample.get_feature_names()
                if show_goodness:
                    features.append("goodness")
                if show_rewards:
                    features.append("rewards")
            if show_goodness or show_rewards:
                goodness = self.calc_goodness(sample)
                if show_goodness:
                    entry.append(goodness)
                if show_rewards:
                    rewards = self.calculate_rewards(sample, goodness)
                    new_rewards = {k.name: (v if rounding is None else round(v, rounding)) for k, v in rewards.items()}
                    entry.append(new_rewards)
            dataset.append(np.array(entry, dtype=object))
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        dataset = pd.DataFrame(np.array(dataset), columns=features)
        return dataset
