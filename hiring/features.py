from enum import auto, Enum
from typing import List

import numpy as np
from numpy.random import Generator
import pandas as pd

from hiring import Feature


############
# Features #
############
class HiringFeature(Feature):
    """The features for a hiring scenario"""
    age = auto()
    gender = auto()
    degree = auto()
    extra_degree = auto()
    experience = auto()
    #
    married = auto()
    nationality = auto()


class Gender(Enum):
    """Enumeration for the gender"""
    male = 0  # auto()
    female = auto()


class Nationality(Enum):
    """Enumeration for the nationality"""
    belgian = 0
    foreign = auto()


ALL_NATIONALITIES = [n for n in Nationality]

GOODNESS_NOISE = 0.5
FEMALE_BIAS = -0.2

NOISE_HIRE = 0.1
NOISE_REJECT = 0.5


########################
# Feature Descriptions #
########################
class ApplicantGenerator(object):
    """An applicant generator

    Attributes:
        csv: The dataset containing the weighted population.
        seed: The random seed
    """

    def __init__(self,
                 csv="data/belgian_population.csv",
                 seed=None):
        self.df = pd.read_csv(csv, index_col=None) if csv is not None else None
        self.seed = seed
        self.rng = np.random.default_rng(seed=self.seed)
        # Not all features are included in dataframe
        self.features = [HiringFeature.nationality, HiringFeature.age, HiringFeature.gender, HiringFeature.degree,
                         HiringFeature.extra_degree, HiringFeature.married]
        self.df_features = [f.name for f in self.features]

    @staticmethod
    def _extract_features(row, features):
        return {f: row[f.name] for f in features}

    @staticmethod
    def _add_enumerations(sample):
        new_sample = sample.copy()
        for feature, value in sample.items():
            if feature == HiringFeature.gender:
                new_sample[feature] = Gender(value)
            elif feature == HiringFeature.nationality:
                new_sample[feature] = Nationality(value)
        return new_sample

    def _sample(self, n=1):
        """Sample dataframe"""
        indices = self.rng.choice(self.df.index, size=n, replace=True, p=self.df["w"])
        samples_df = self.df.iloc[indices]
        return samples_df.drop(columns=["w"])

    def sample(self, n=1):
        """Sample the given number of applicants"""
        samples_df = self._sample(n)

        applicants = []
        for _, row in samples_df.iterrows():
            applicant = self._extract_features(row, [HiringFeature.age, HiringFeature.gender,
                                                     HiringFeature.degree, HiringFeature.extra_degree,
                                                     HiringFeature.nationality, HiringFeature.married])
            # Add experience
            applicant, degree_idx = self._add_experience(applicant)

            # Add completed applicant to results
            applicant = self._add_enumerations(applicant)
            applicants.append(applicant)

        return applicants[0] if n == 1 else applicants

    def _add_experience(self, applicant):
        # Add experience
        max_work_experience = applicant[HiringFeature.age] - 18  # No experience before 18
        degree_idx = 0
        if applicant[HiringFeature.degree]:
            max_work_experience -= 3
            degree_idx = 1
        if applicant[HiringFeature.extra_degree]:
            max_work_experience -= 2
            degree_idx = 2
        max_work_experience = max(0, max_work_experience)
        # Linearly increasing probability towards max experience given their years of experience
        years = np.arange(max_work_experience + 1)
        prob = (years + 1) / np.sum(years + 1)  # Take 0 years into account
        experience = self.rng.choice(years, p=prob)
        applicant[HiringFeature.experience] = experience
        return applicant, degree_idx

    @staticmethod
    def normalise_feature(feature: HiringFeature, value):
        if feature == HiringFeature.gender:
            return value.value / len(Gender) if isinstance(value, Gender) else value
        elif feature == HiringFeature.nationality:
            return value.value / len(Nationality) if isinstance(value, Nationality) else value
        elif feature == HiringFeature.age:
            return value / (65 - 18)
        elif feature == HiringFeature.experience:
            return value / (65 - 18)
        else:
            return value

    def normalise_features(self, applicant, features: List[HiringFeature] = None):
        if features is not None:
            new_values = {f: self.normalise_feature(f, applicant[f]) for f in features}
        else:
            new_values = {f: self.normalise_feature(f, v) for f, v in applicant.items()}
        return new_values

    def fit_model(self, path, feature_dist):
        """Fit a new model from given CSV with new feature distributions and save in path

        Example single feature:
            generator.fit_model(path, feature_dist=(HiringFeature.gender, {0: 0.7, 1: 0.3}))

        Example feature combination:
            generator.fit_model(path_dd, feature_dist=[(HiringFeature.gender, HiringFeature.married),
                                {(0, False): 0.25, (0, True): 0.3, (1, False): 0.2, (1, True): 0.25}])

        """
        new_df = self.df.copy()

        # Single feature
        is_single = False
        if isinstance(feature_dist, tuple):
            f, p = feature_dist
            p = {(k, ): v for k, v in p.items()}
            feature_dist = [(f, ), p]
            is_single = True

        # Feature combination
        features, probs = feature_dist
        names = [f.name for f in features]
        # Get current probs
        view = self.df[[*names, "w"]]
        current = view.groupby(names).sum()['w'].to_dict()
        if is_single:
            current = {(k, ): v for k, v in current.items()}
        print(f"current {names}: {current}")
        print(f"requested {names}: {probs}")

        for feat_vals, feat_probs in probs.items():
            f_view = [new_df[f.name] == v for f, v in zip(features, feat_vals)]
            if len(feat_vals) > 1:
                f_view = np.multiply(*f_view)
            else:
                f_view = f_view[0]
            # Scale probabilities to give values
            new_df.loc[f_view, "w"] = (probs[feat_vals] / current[feat_vals]) * new_df.loc[f_view, "w"]

        new_view = new_df[[*names, "w"]]
        new_values = new_view.groupby(names).sum()
        new = new_values['w'].to_dict()
        print(f"new {names}: {new}")

        self.df = new_df
        self.df.to_csv(path, index=False)

    def load_model(self, path):
        """Load model from given path"""
        self.df = pd.read_csv(path, index_col=None)

    def print_model(self):
        # Per feature
        print("Single feature:")
        for feature in self.features:
            view = self.df[[feature.name, "w"]]
            # values = view.groupby([feature.name]).sum()["w"].to_dict()
            values = view.groupby([feature.name]).sum().to_dict()
            print(f"{feature.name}: {values}")
        # Combination of features TODO multiple?
        print("\nFeature combinations:")
        for i, f1 in enumerate(self.features):
            for f2 in self.features[i+1:]:
                view = self.df[[f1.name, f2.name, "w"]]
                # values = view.groupby([f1.name, f2.name]).sum()["w"].to_dict()
                values = view.groupby([f1.name, f2.name]).sum().to_dict()
                print(f"{f1.name} & {f2.name}: {values}")


if __name__ == '__main__':
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 120 * 2)

    ag = ApplicantGenerator(seed=0)
    # ag.print_model()

    # Different distributions
    ag_dd = ApplicantGenerator(seed=0)
    path_dd = "data/belgian_pop_diff_dist.csv"
    # ag_dd.fit_model(path_dd, feature_dist=(HiringFeature.gender, {0: 0.7, 1: 0.3}))
    ag_dd.load_model(path_dd)

    # Different distributions 2, combination of features
    ag_dd2 = ApplicantGenerator(seed=0)
    path_dd2 = "data/belgian_pop_diff_dist2.csv"
    # ag_dd2.fit_model(path_dd2, feature_dist=[(HiringFeature.gender, HiringFeature.married),
    #                                          {(0, False): 0.25, (0, True): 0.3, (1, False): 0.2, (1, True): 0.25}])
    ag_dd2.load_model(path_dd2)

    n_samples = 100
    samples = []
    for (name, model) in [("base", ag), ("70-30% gender", ag_dd), ("gender married combo", ag_dd2)]:
        appl = model.sample(n=n_samples)
        appl = pd.DataFrame(appl)
        appl = appl.rename(columns={hf: hf.name for hf in HiringFeature})
        appl["gender"] = [v.name for v in appl["gender"]]
        appl["nationality"] = [v.name for v in appl["nationality"]]
        appl["group"] = name
        samples.append(appl)
        print(appl)
    exit()

    # # Compare models
    # import plotly.express as px
    # import dash
    # from dash import html, dcc
    #
    # features = [HiringFeature.nationality, HiringFeature.gender, HiringFeature.degree, HiringFeature.extra_degree,
    #             HiringFeature.married, HiringFeature.age]
    #
    # full_df = pd.concat(samples, ignore_index=True)
    # figures = []
    # for feature in ag.features:
    #     fig = px.histogram(full_df, x=feature.name, color="group", barmode="group", title=f"Feature {feature.name}")
    #     figures.append(fig)
    #
    # # Combination of features
    # for i, f1 in enumerate(ag.features):
    #     for f2 in ag.features[i+1:]:
    #         if f1 == f2:
    #             continue
    #         fig = px.scatter(full_df, x=f1.name, y=f2.name, color="group",
    #                          marginal_x="violin", marginal_y="violin",
    #                          title=f"Features {f1.name} and {f2.name}")
    #         figures.append(fig)
    # figures = [dcc.Graph(figure=fig) for fig in figures]
    #
    # app = dash.Dash()
    # app.layout = html.Div(figures)
    #
    # app.run_server(debug=False, use_reloader=False)
