import numpy as np


class Thompson(object):
    """Thompson Sampling algorithm for choosing arms in a MAB problem.

    Algorithm uses Bayes' rule to update beliefs over the arm parameters.
    The beliefs for each arm follow a Beta distribution over the parameter
    space [0,1].
    An arm is selected by drawing an element from each of the posterior
    distributions, and then choosing the arm with the highest draw.

    :param parameters: List of Bernoulli probabilities of each arm.

    """

    def __init__(self, parameters):
        self.n_arms = len(parameters)
        self.values = np.zeros(len(parameters))  # Expected values of arms
        self.counts = np.zeros(len(parameters))  # Number of times arms pulled
        self.parameters = parameters  # True Bernoulli probabilities of arms
        self.draw_sample = np.zeros(self.n_arms)
        self.name = 'thompson'

    def initialize(self, initial_pulls=1):
        """Reset arm statistics to initial position and initialize them with specified number of pulls.

        :param initial_pulls: Number of times each arm is initially pulled

        """
        self.values = np.random.binomial(initial_pulls, self.parameters, self.n_arms) / float(initial_pulls)
        self.counts = initial_pulls * np.ones(len(self.parameters))

    def select_arm(self):
        """Outputs an integer indexing the arm choice.

        Given the statistics of arm counts and values, beliefs are formed
        according to the beta distribution where the counts and values act
        as the two parameters. Recall that values are the sample estimates
        of the Bernoulli parameters:
        alpha = number of success = count * value
        beta = number of failures = counts - alpha

        :return Index of chosen arm.

        """
        alpha = self.counts * self.values
        beta = self.counts - alpha
        self.draw_sample = np.random.beta(alpha+1, beta+1, self.n_arms)
        return np.argmax(self.draw_sample)

    def update(self, arm, reward):
        """Update arm statistics.

        :param arm: Arm index that was pulled.
        :param reward: Arm reward from pull.

        """
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        new_value = ((n-1)/float(n)) * old_value + (1/float(n)) * reward
        self.values[arm] = new_value
