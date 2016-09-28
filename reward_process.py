"""Generates a simulated reward process"""

import numpy as np


class RewardProcess(object):
    """Simulates reward processes, by drawing row vectors
    of length horizon, for each arm indexed by the row.
    """

    def __init__(self, parameters, horizon):
        self.n_arms = len(parameters)  # Number of arms
        self.horizon = horizon         # Length of arm pulls in simulation
        self.parameters = parameters   # Vector containing the Bernoulli parameters of all the arms
        self.simulation = np.zeros([self.n_arms, horizon])  # Number of simulations

    def bernoulli(self, safe=False):
        """Simulate all arms which are Bernoulli independent processes. Include the possibility of a safe arm.

        :param safe: If True, then the first element of the vector containing
                     the Bernoulli parameters (self.parameters), is designated
                     as the safe arm, and pays a certain payoff of self.parameters[0] every period.
                     If False, every arm is a risky arm.
        """
        if safe == True:
            k = 1
            self.simulation[0, :] = self.parameters[0]*np.ones(self.horizon)
        else:
            k = 0
        for i in range(k, self.n_arms):
            self.simulation[i, :] = 1*(np.random.uniform(0, 1, self.horizon) < self.parameters[i])
        return self.simulation
