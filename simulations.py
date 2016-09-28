"""Runs the main simulations which compare the performance of various bandit
 algorithms

We pre-draw simulated Bernoulli reward sequences for each arm separately of length
indicated by the horizon of the simulation. Each algorithm does not know the
realization of this sequence chooses and arm, and hence reward, to sample.
"""

import numpy as np
import reward_process as rp
import time
import copy

__author__ = 'Ilari'


def simulate_algorithms(algorithms, parameters, n_sims, horizon):

    results = {}
    best_arm = np.argmax(parameters)

    for algorithm in algorithms:
        start_time = time.time()
        # Initiate data collection arrays:
        chosen_arms = np.zeros([n_sims, horizon])
        cum_rewards = np.zeros([n_sims, horizon])
        cum_regret = np.zeros([n_sims, horizon])  # The realized regret compared to pulling the best ar
        expected_values = np.zeros([n_sims, horizon, len(parameters)])


        for sim in range(0, n_sims):
            algo = copy.deepcopy(algorithm)
            """Note: Deepcopy circumvents the need to re-initialize the arms for
             bandit_algorithms with priors in every simulation loop"""

            algo.initialize()  # Does nothing for algorithms which start with priors

            simulate = rp.RewardProcess(parameters, horizon)  # Draw reward process for each arm
            rewards = simulate.bernoulli()

            for t in range(0, horizon):
                chosen_arm = algo.select_arm()
                chosen_arms[sim, t] = chosen_arm

                reward = rewards[chosen_arm, t]
                if t > 0:
                    cum_rewards[sim, t] = cum_rewards[sim, t-1] + reward
                    cum_regret[sim, t] = cum_regret[sim, t-1] + (parameters[best_arm] - reward)
                else:
                    cum_rewards[sim, t] = reward
                    cum_regret[sim, t] = parameters[best_arm] - reward

                algo.update(chosen_arm, reward)
                expected_values[sim, t, :] = algo.values

        # Collect results into dictionary
        results_algo = {'cumulative_regret': cum_regret,
                   'cumulative_rewards': cum_rewards,
                   'chosen_arms': chosen_arms,
                   'expected_values': expected_values}
        name = algo.name
        results[name] = results_algo
        print "Elapsed time of simulation: ", time.time() - start_time, name
    return results
