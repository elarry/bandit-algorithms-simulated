import numpy as np


class BrezziLai(object):
    """Gittins Approximation algorithm for choosing arms in a MAB problem.

    Algorithm based on Brezzi and Lai (2002) "Optimal learning and experimentation in bandit problems."
    The algorithm provides an approximation of the Gittins index, by specifying
    a closed-form expression, which is a function of the discount factor, and
    the number of successes and failures associated with each arm.

    :param parameters: List of Bernoulli probabilities of each arm.
    :param discount: Discount factor
    :param prior: Prior beliefs over Bernoulli parameters governing each arm.
        Beliefs specified by Beta distribution with two parameters [alpha,beta],
        where, alpha = number of success, beta = number of failures.

    """

    def __init__(self, parameters, discount=0.95, prior=None):
        self.n_arms = len(parameters)
        self.values = np.zeros(len(parameters))  # Expected values of arms
        self.counts = np.zeros(len(parameters))  # Number of times arms pulled
        self.parameters = parameters  # True Bernoulli probabilities of arms
        self.prior = prior  # Prior beliefs over Bernoulli probabilities of arms
        self.discount = discount  # Discount factor
        self.gittins_index = np.zeros(len(parameters))  # Gittins index approximation
        self.name = 'gittins_brezzi_lai'

    def initialize(self, initial_pulls=1):
        """If a prior is specified, then initialize arm statistics to correspond
        to these beliefs. If a prior is not specified, then we initialize arms
        with the number of pulls specified by *initial_pulls*.

        Note, that we need the (beliefs over) initial values of the arms to be
        bounded away from zero. Otherwise, an arm with value = 0 will never be chosen.

        :param initial_pulls: Number of initial pulls per arm.

        """
        if not isinstance(self.prior, type(None)):  # If prior specified
            if len(self.prior) != len(self.parameters):
                print "Prior specified incorrectly!"
                print "Prior must be of dimensions: [*n_arms*, 2] "
            for arm in range(0, len(self.parameters)):
                n = sum(self.prior[arm])
                mean = self.prior[arm][0]/float(n)
                self.counts[arm] = n
                self.values[arm] = mean
                self.gittins_approximation(arm)  # Calculate initial Gittins Index approximation

        else:  # Assume uniform prior
            self.counts = 2 * np.ones(len(self.parameters))  # Uniform prior: counts = success + failure = 1 + 1 = 2
            self.values = 0.5 * np.ones(len(self.parameters))  # Uniform Prior
            for arm in range(0, len(self.parameters)):
                self.gittins_approximation(arm)  # Calculate initial Gittins Index approximation

    def select_arm(self):
        max_arms = np.argwhere(self.gittins_index == np.amax(self.gittins_index))  # Identify arm indices with max value
        max_arms = max_arms.flatten().tolist()
        return np.random.choice(max_arms)

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        old_value = self.values[arm]
        new_value = ((n-1)/float(n)) * old_value + (1/float(n)) * reward
        self.values[arm] = new_value
        self.gittins_approximation(arm)  # Calculate Gittins Index approximation

    def gittins_approximation(self, arm):
        p_hat = self.values[arm]
        v_arm = p_hat*(1 - p_hat)
        v_mean = p_hat*(1 - p_hat)/float(self.counts[arm] + 1)  # With 1 in denominator: beta variance; without frequentist variance.
        c = - np.log(self.discount)
        self.gittins_index[arm] = p_hat + np.sqrt(v_mean) * self.phi( v_mean/float(c * v_arm) )  # Store Gittins Index approximation

    @staticmethod
    def phi(s):
        if s > 15:
            return ( 2*np.log(s) - np.log(np.log(s)) - np.log(16*np.pi) )**0.5
        elif 5 < s <= 15:
            return 0.77 - 0.58*s**(-0.5)
        elif 1 < s <= 5:
            return 0.63 - 0.26*s**(-0.5)
        elif 0.2 < s <= 1:
            return 0.49 - 0.11*s**(-0.5)
        elif 0 <= s <= 0.2:
            return np.sqrt(s/2.0)
        else:
            print s, 'Domain error in Brezzi_Lai Phi-function'
            return 0.0
