import numpy as np


class EpsilonGreedy(object):
    """Epsilon Greedy algorithm for choosing arms in a MAB problem.

    With probability (1-epsilon), choose arm with highest expected value.
    With probability epsilon, explore random arm.

    :param parameters: List of Bernoulli probabilities of each arm.
    :param epsilon: Probability of choosing a random arm.

    """

    def __init__(self, parameters, epsilon=0.1):
        self.n_arms = len(parameters)
        self.values = np.zeros(len(parameters))  # Expected values of arms
        self.counts = np.zeros(len(parameters))  # Number of times arms pulled
        self.parameters = parameters  # True Bernoulli probabilities of arms
        self.epsilon = epsilon
        self.name = "epsilon_greedy"

    def initialize(self, initial_pulls=1):
        """Reset arm statistics to initial position and initialize them with specified number of pulls.

        :param initial_pulls: Number of times each arm is initially pulled

        """
        self.values = np.random.binomial(initial_pulls, self.parameters, self.n_arms)/float(initial_pulls)
        self.counts = initial_pulls * np.ones(len(self.parameters))

    def select_arm(self):
        """Outputs an integer indexing the arm choice.

        :return Index of chosen arm.

        """
        max_arms = np.argwhere(self.values == np.amax(self.values))  # Identify arm indices with max value
        max_arms = max_arms.flatten().tolist()
        if np.random.uniform() > self.epsilon:
            return np.random.choice(max_arms)  # Exploitation: Choose index number from maximal indices
        else:
            return np.random.randint(self.n_arms)  # Exploration: Choose random index

    def update(self, arm, reward):
        """Update arm statistics.

        :param arm: Arm index that was pulled.
        :param reward: Arm reward from pull.

        """
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        new_value = ((n-1)/float(n)) * old_value + (1.0/float(n)) * reward
        self.values[arm] = new_value
