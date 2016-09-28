import numpy as np


class Gittins(object):
    """Uses a pre-calculated Gittins Index approximation as input

    :param gittins_input: Gittins index, (n x n) array
            rows: a count, i.e. number of successes.
            columns: b count, i.e. number of failures.
    :param parameters: array containing true parameters of bandit arms
    :param prior: If no prior specified, uniform prior is used
    """
    def __init__(self, gittins_input, parameters, prior=None):
        self.gittins_input = gittins_input  # (n x n) array
        self.parameters = parameters
        self.gittins_index = np.zeros(len(parameters))  # Gittins index approximation
        self.name = 'gittins'

        if not isinstance(prior, type(None)):  # If prior specified
            if len(prior) != len(parameters):
                print "Prior specified incorrectly!"
                print "Prior must be of dimensions: Number_of_arms x 2 "
            for arm in range(0, len(parameters)):
                n = sum(prior[arm])
                mean = prior[arm][0]/float(n)
                self.a[arm] = prior[arm][0]
                self.counts[arm] = n
                self.values[arm] = mean
                self.update_gittins(arm)

        else:  # Assume uniform prior
            self.a = np.ones(len(parameters))
            self.counts = 2*np.ones(len(parameters))  # Uniform prior: counts = success + failure = 1 + 1 = 2
            self.values = 0.5*np.ones(len(parameters))  # Uniform Prior
            for arm in range(0, len(parameters)):
                self.update_gittins(arm)

    def initialize(self, initial_pulls=1):
        """Needed for simulations"""
        pass

    def select_arm(self):
        max_arms = np.argwhere(self.gittins_index == np.amax(self.gittins_index))  # identify arm indices with max value
        max_arms = max_arms.flatten().tolist()
        return np.random.choice(max_arms)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.a[arm] += reward
        old_value = self.values[arm]
        new_value = ((n-1)/float(n))*old_value + (1/float(n))*reward
        self.values[arm] = new_value
        self.update_gittins(arm)

    def update_gittins(self, arm):
        a = self.a[arm]  # Number of successes on arm
        b = self.counts[arm] - a  # Number of failures on arm
        try:
            self.gittins_index[arm] = self.gittins_input[a, b]
        except:
            print("Error in accessing Gittins Index input")