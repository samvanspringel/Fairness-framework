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
    nationality = auto()
    migration_background = auto()


class Gender(Enum):
    """Enumeration for the gender"""
    male = auto()
    female = auto()

class Nationality(Enum):
    """Enumeration for the origine"""
    Belgium = auto()
    France = auto()
    Romania = auto()
    Yugoslavia = auto()
    Netherlands = auto()
    Poland = auto()
    Spain = auto()
    Morocco = auto()
    Russia = auto()
    Turkey = auto()
    Italy = auto()


PROB_BEL = 0.87

PROB_RUS = 0.01
PROB_ROM = 0.01
PROB_FRA = 0.02
PROB_NET = 0.02
PROB_POL = 0.01
PROB_SPA = 0.007
PROB_MOR = 0.02
PROB_YUG = 0.007
PROB_TUR = 0.016
PROB_ITA = 0.01

P_MIGRATION_BACKGROUND = 0.33

# Some default values which can be used for a baseline scenario
PROB_MALE = 0.5
PROB_FEMALE = 0.5

AGE_MALE = range(18, 66)
AGE_FEMALE = range(18, 66)

P_DEGREE_MALE = 0.4
P_DEGREE_FEMALE = 0.4

P_E_DEGREE_MALE = 0.4
P_E_DEGREE_FEMALE = 0.4

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


class MigrationBackgroundDescription(FeatureDescription):
    """Migration background: Either has a different origine or not.

    Attributes:
        prob_yes: (Optional) The probability of a person having a different origine.
    """
    def __init__(self, prob_mb=P_MIGRATION_BACKGROUND):
        # Super call
        super(MigrationBackgroundDescription, self).__init__(feature=HiringFeature.migration_background)
        #
        self.probabilities = {Gender.male: prob_mb, Gender.female: prob_mb, }

    def generate(self, rng: Generator, gender: Gender = None, default_background: int = None):
        """Generate a background"""
        background = rng.binomial(1, self.probabilities[gender])
        return background


class NationalityDescription(FeatureDescription):
    """Nationality"""
    def __init__(self, prob_bel=PROB_BEL, prob_fra=PROB_FRA, prob_net=PROB_NET, prob_tur=PROB_TUR, prob_rom=PROB_ROM,
                 prob_spa=PROB_SPA, prob_ita=PROB_ITA, prob_yug=PROB_YUG, prob_rus=PROB_RUS, prob_pol=PROB_POL,
                 prob_mor=PROB_MOR):
        # Super call
        super(NationalityDescription, self).__init__(feature=HiringFeature.nationality)
        #
        self.prob_bel = prob_bel
        self.prob_fra = prob_fra
        self.prob_net = prob_net
        self.prob_tur = prob_tur
        self.prob_rom = prob_rom
        self.prob_spa = prob_spa
        self.prob_ita = prob_ita
        self.prob_yug = prob_yug
        self.prob_rus = prob_rus
        self.prob_pol = prob_pol
        self.prob_mor = prob_mor

    def generate(self, rng: Generator, *args):
        """Generate a nationality"""
        nationality = rng.choice([Nationality.Belgium, Nationality.Italy, Nationality.Netherlands, Nationality.Spain,
                             Nationality.Spain, Nationality.Romania, Nationality.Morocco, Nationality.Russia,
                             Nationality.Poland, Nationality.Yugoslavia, Nationality.France],
                            p=[self.prob_bel, self.prob_fra, self.prob_net, self.prob_tur, self.prob_rom, self.prob_spa,
                               self.prob_ita, self.prob_yug, self.prob_rus, self.prob_pol, self.prob_mor])
        return nationality
