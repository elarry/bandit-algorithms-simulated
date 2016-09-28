import numpy as np


class UCB1(object):
    """UBC1 algorithm for choosing arms in a MAB problem.

    Choose the arm with the highest upper Chernoff-Hoeffding bound.
    Reference: Auer, Cesa-Bianchi, and Fischer (2002) "Finite-time Analysis of the Multiarmed Bandit Problem"

    :param parameters: List of Bernoulli probabilities of each arm.

    """

    def __init__(self, parameters):
        self.n_arms = len(parameters)
        self.values = np.zeros(len(parameters))  # Expected values of arms
        self.counts = np.zeros(len(parameters))  # Number of times arms pulled
        self.parameters = parameters  # True Bernoulli probabilities of arms
        self.ucb = np.zeros(len(parameters))  # Upper Chernoff-Hoeffding bounds
        self.name = 'ubc1'

    def initialize(self, initial_pulls=1):
        """Reset arm statistics to initial position and initialize them with specified number of pulls.

        :param initial_pulls: Number of times each arm is initially pulled

        """
        initial_draws = np.random.binomial(initial_pulls, self.parameters, self.n_arms)/float(initial_pulls)
        self.values = initial_draws
        self.counts = initial_pulls * np.ones(len(self.parameters))
        # Compute the Upper Chernoff-Hoeffding bounds
        self.ucb = initial_draws + np.sqrt(2 * np.log(sum(self.counts)) / self.counts)

    def select_arm(self):
        """Outputs an integer indexing the arm choice.

        :return Index of chosen arm.

        """
        max_arms = np.argwhere(self.ucb == np.amax(self.ucb))  # Identify arm indices with max value
        max_arms = max_arms.flatten().tolist()
        return np.random.choice(max_arms)

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
        # Compute the Upper Chernoff-Hoeffding bound
        self.ucb[arm] = self.values[arm] + np.sqrt(2*np.log(sum(self.counts))/float(n))
