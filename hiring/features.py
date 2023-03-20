from enum import auto, Enum
from numpy.random import Generator

from hiring import Feature, FeatureDescription


############
# Features #
############
class HiringFeature(Feature):
    """The features for a hiring scenario"""
    degree = auto()
    extra_degree = auto()
    experience = auto()
    age = auto()
    gender = auto()
    origin = auto()


class Gender(Enum):
    """Enumeration for the gender"""
    male = auto()
    female = auto()

class Origin(Enum):
    """Enumeration for the origine"""
    belgium = auto()
    foreign_background = auto()
    foreign = auto()

# TODO: Data van 2013 zoeken
PROB_BELGIUM = 0.666
PROB_FOREIGN_BACKGROUND = 0.206
PROB_FOREIGN = 0.128

#https://bestat.statbel.fgov.be/bestat/crosstable.xhtml?view=8b82b79e-4696-45ec-a082-254756db6be6
# 2013 referentie jaar
# Some default values which can be used for a baseline scenario
PROB_MALE = 0.5024
PROB_FEMALE = 0.4976

AGE_MALE = range(18, 66)
AGE_FEMALE = range(18, 66)

# MANNEN
# Onderwijsniveau laag: 1158164
# Onderwijsniveau midden: 1439308
# Onderwijsniveau hoog: 1048457
# TOTAAL: 3645929
P_DEGREE_MALE = 0.3948 + 0.2876
P_E_DEGREE_MALE = 1048457 / 1048457 + 1439308


# VROUWEN
# Onderwijsniveau laag: 1047297
# Onderwijsniveau midden: 1324730
# Onderwijsniveau hoog: 1239118
# TOTAAL: 3611145
P_DEGREE_FEMALE = 0.3668
P_E_DEGREE_FEMALE = 0.3431


P_E_DEGREE_MALE = 0.2
P_E_DEGREE_FEMALE = 0.6


EXPERIENCE_MALE = range(0, 10)
EXPERIENCE_FEMALE = range(0, 10)
P_EXPERIENCE_MALE = (0.3, 0.2, 0.15, 0.15, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01)
P_EXPERIENCE_FEMALE = (0.3, 0.2, 0.15, 0.15, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01)

GOODNESS_NOISE = 0.5
FEMALE_BIAS = -0.2

NOISE_HIRE = 0.1
NOISE_REJECT = 0.5


########################
# Feature Descriptions #
########################
class GenderDescription(FeatureDescription):
    """Gender: Male of Female"""
    def __init__(self, prob_male=PROB_MALE, prob_female=PROB_FEMALE):
        # Super call
        super(GenderDescription, self).__init__(feature=HiringFeature.gender)
        #
        self.prob_male = prob_male
        self.prob_female = prob_female
        # Normalise probabilities in case counts were given instead
        self.prob_male = self.prob_male / (self.prob_male + self.prob_female)
        self.prob_female = 1 - self.prob_male

    def generate(self, rng: Generator, *args):
        """Generate a gender"""
        gender = rng.choice([Gender.male, Gender.female], p=[self.prob_male, self.prob_female])
        return gender


class AgeDescription(FeatureDescription):
    """Age of hiring: between 18 and 65 (inclusive)

    Attributes:
        range_male: The age range of male candidates.
        range_female: The age range of female candidates.
        prob_male: (Optional) The probabilities over the ages in the age range of males.
        prob_female: (Optional) The probabilities over the ages in the age range of females.
    """
    def __init__(self, range_male=AGE_MALE, range_female=AGE_FEMALE, prob_male=None, prob_female=None):
        # Super call
        super(AgeDescription, self).__init__(feature=HiringFeature.age)
        #
        self.ranges = {Gender.male: range_male, Gender.female: range_female, }
        self.probabilities = {Gender.male: prob_male, Gender.female: prob_female, }
        self.max_age = {Gender.male: max(range_male), Gender.female: max(range_female), }
        self.min_age = {Gender.male: min(range_male), Gender.female: min(range_female), }

    def generate(self, rng: Generator, gender: Gender = None):
        """Generate an age given the gender"""
        gender_range = self.ranges[gender]
        range_prob = self.probabilities[gender]
        age = rng.choice(gender_range, p=range_prob)
        return age


class DegreeDescription(FeatureDescription):
    """Degree: Either has a degree or not.

    Attributes:
        prob_male: (Optional) The probability of a male having a degree.
        prob_female: (Optional) The probability of a female having a degree.
    """
    def __init__(self, prob_male=P_DEGREE_MALE, prob_female=P_DEGREE_FEMALE):
        # Super call
        super(DegreeDescription, self).__init__(feature=HiringFeature.degree)
        #
        self.probabilities = {Gender.male: prob_male, Gender.female: prob_female, }

    def generate(self, rng: Generator, gender: Gender = None):
        """Generate a degree"""
        degree = rng.binomial(1, self.probabilities[gender])
        return degree


class ExtraDegreeDescription(FeatureDescription):
    """Extra Degree: Either has a degree or not.

    Attributes:
        prob_male: (Optional) The probability of a male having a degree.
        prob_female: (Optional) The probability of a female having a degree.
    """
    def __init__(self, prob_male=P_E_DEGREE_MALE, prob_female=P_E_DEGREE_FEMALE):
        # Super call
        super(ExtraDegreeDescription, self).__init__(feature=HiringFeature.extra_degree)
        #
        self.probabilities = {Gender.male: prob_male, Gender.female: prob_female, }

    def generate(self, rng: Generator, gender: Gender = None, default_degree: int = None):
        """Generate a degree"""
        degree = rng.binomial(1, self.probabilities[gender])
        # Can only have extra degree if it has a (normal) degree
        if not default_degree:
            degree = 0
        return degree


class ExperienceDescription(FeatureDescription):
    """Experience: How many years of experience a candidate has.

    Attributes:
        Attributes:
        range_male: The years range of male candidates.
        range_female: The years range of female candidates.
        prob_male: (Optional) The probabilities over the years in the experience range of males.
        prob_female: (Optional) The probabilities over the years in the experience range of females.
    """
    def __init__(self, range_male=EXPERIENCE_MALE, range_female=EXPERIENCE_FEMALE,
                 prob_male=P_EXPERIENCE_MALE, prob_female=P_EXPERIENCE_FEMALE):
        # Super call
        super(ExperienceDescription, self).__init__(feature=HiringFeature.experience)
        #
        self.ranges = {Gender.male: range_male, Gender.female: range_female, }
        self.probabilities = {Gender.male: prob_male, Gender.female: prob_female, }

    def generate(self, rng: Generator, gender: Gender = None, age: int = None):
        """Generate years of experience"""
        gender_range = self.ranges[gender]
        range_prob = self.probabilities[gender]
        # Choose an age according to the given gender and age probabilities
        experience = rng.choice(gender_range, p=range_prob)
        # Experience must come after 18 years
        if age - experience < 18:
            experience = max(age - 18, 0)
        return experience



class OriginDescription(FeatureDescription):
    """Nationality"""
    def __init__(self, prob_bel=PROB_BELGIUM, prob_fb=PROB_FOREIGN_BACKGROUND, prob_foreign=PROB_FOREIGN):
        # Super call
        super(OriginDescription, self).__init__(feature=HiringFeature.origin)
        #
        self.prob_bel = prob_bel
        self.prob_fb = prob_fb
        self.prob_foreign = prob_foreign

    def generate(self, rng: Generator, *args):
        """Generate a nationality"""
        nationality = rng.choice([Origin.belgium, Origin.foreign_background, Origin.foreign],
                            p=[self.prob_bel, self.prob_foreign, self.prob_fb])
        return nationality
