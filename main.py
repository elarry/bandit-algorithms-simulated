"""Runs simulations and compares the performance of several multi-armed bandit (MAB) algorithms.

The reward process for each bandit arm is described by a sequence of iid Bernoulli
trials with a fixed parameter describing the probability of a success. The
algorithms try to balance exploration for the best arm with exploitation of the
arm that is myopically optimal.

We plot the performance statistics of a pair of algorithms which allows us to
compare their relative performance with respect to a number of criteria (cumulative
reward, regret, frequency of correct arm choice, consistency)
 """

import plots_and_statistics as ra
import simulations
from bandit_algorithms.gittins_index_computation import gittins_index
from bandit_algorithms.gittins import *
from bandit_algorithms.gittins_brezzi_lai import *
from bandit_algorithms.thompson import *
from bandit_algorithms.ucb1 import *
from bandit_algorithms.epsilon_greedy import *
from bandit_algorithms.naive import *

__author__ = 'Ilari'

# Simulation parameters
parameters = [0.1, 0.12, 0.15, 0.2]  # True bernoulli parameters of bandit arms
n_sims = 200
horizon = 500
delta = 0.95  # Discount factor
grid_gittins = 1000  # Increase to get more precise Index approximation


"""Example 1: Compare Thompson Sampling algorithm with UCB1"""
algorithms_2 = [Thompson(parameters),
                UCB1(parameters)]  # Choose and initialize bandit_algorithms
results_raw = simulations.simulate_algorithms(algorithms_2, parameters, n_sims, horizon)  # Run simulation
df = ra.simulation_statistics(results_raw, parameters)  # Collect data on simulations
ra.plot_results(df, parameters)  # Plot simulation results
"""Note: Label "correct_arm" refers to the fraction of times that
the best arm is chosen, where the averaging is across simulations."""


"""Example 2: Compare Gittins Index approximation to the Brezzi-Lai Gittins Approximation Algorithm"""
g, v = gittins_index(n=550, grid=grid_gittins, discount=delta, value=True, df=False)  # Pre-Calculate Gittins Index
algorithms_1 = [Gittins(g, parameters),
                BrezziLai(parameters, discount=delta)]  # Choose and initialize bandit_algorithms

# Run simulations and collect data
results_raw = simulations.simulate_algorithms(algorithms_1, parameters, n_sims, horizon)
df = ra.simulation_statistics(results_raw, parameters)

# Plot simulation results
ra.plot_results(df, parameters)


"""Plot the value functions associated with the Gittins index"""
# Plot Gittins index and associated value function
ra.plot_value_functions(v, g, discount=delta, normalized=True)
ra.plot_gittins_index(g)

"""Plot number successes needed to keep Gittins Index value approximately at 0.5, as the number of failures ranges over [0, 40]"""
ra.plot_gittins_isoquant(g, 0.5, 40)

ra.plot_gittins_index_sliced(g, 0.1)  # Horizontal cross-section at Index value=0.1
ra.plot_gittins_surplus(g)
