"""Functions that compute performance statistics given the raw date of simulations.
Various plotting functions are specified below
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D
from cycler import cycler

__author__ = 'Ilari'


def simulation_statistics(results_raw, parameters):
    """Compute performance statistics for chosen bandit algorithms.

    Input of form:
    results_raw[algo][statistic] is a matrix of dimension (simulation x horizon)

    We take the mean of each column; i.e take mean over performed simulations.
    Rows contain simulations in results_raw[algorithm][statistic]
    Columns contain time step in results_raw[algorithm][statistic]

    :param results_raw: A Nested dictionary.
    :param parameters: Array of true Bernoulli parameters governing the arms


    """
    n_sims = next(next(results_raw.itervalues()).itervalues()).shape[0]  # Extract number of simulations from input
    results = {}
    for algo in results_raw:  # iterate over dictionary keys
        result = {}
        result['cumulative_rewards'] = np.mean(results_raw[algo]['cumulative_rewards'], axis=0)
        result['expected_values'] = np.mean(results_raw[algo]['expected_values'], axis=0)
        result['cumulative_regret'] = np.mean(results_raw[algo]['cumulative_regret'], axis=0)
        # Compute the fraction of times optimal arm chosen at each time step:
        result['correct_arm'] = (results_raw[algo]['chosen_arms']
                                 == np.argmax(parameters)).sum(axis=0)/float(n_sims)
        results[algo] = result
    df = pd.DataFrame.from_dict(results, orient='index')  # 'index' parameter orders algorithm results under statistics ("cumulative_regret", etc.)

    return df


def plot_results(df, parameters):
    """Main plots for algorithm comparisons

    Plot Simulation results using four statistics:
         1. Average Cumulative Rewards across simulations at each time step.
         2. Average frequency of choosing best arm across simulations at each time step.
         3. Average Cumulative Regret across simulations at each time step.
         4. Average Estimated Expected Values across simulations at each time step.

    :param df:
    :param parameters: Array of true Bernoulli parameters governing the arms

    Dimensions df[statistic][bandit_algorithms][horizon, n_arms, n_players]
            statistic: ['cumulative_rewards', 'correct_arm', 'cumulative_regret',
                        'expected_values']
            bandit_algorithms: e.g. ['UBC1', 'gittins']
            [horizon, n_arms, n_players]: Numpy array
    """
    fig, ax = plt.subplots(2, 2, sharex='col')
    fig.suptitle("True parameters of arms: " + str(parameters), fontsize=14, fontweight='bold')

    styles = ['-', '--', '-.', ':']
    colors = ['c', 'm', 'y', 'k']  # Number of colors should be matched with number of arms
    n_arms = df['expected_values'][0].shape[1]
    n_algorithms = df.shape[0]
    names = [df['expected_values'].index[k] + " arm: " +
             str(i) for k in range(0, n_algorithms) for i in range(0, n_arms)]

    for idx, statistic in enumerate(df.columns):
        ax.item(idx).set_title(statistic)
        for idx_alg, algorithm in enumerate(df[statistic].index):
            ax.item(idx).set_xlabel("Time Step")
            if statistic != 'expected_values':
                ax.item(idx).plot(df[statistic][algorithm], label=algorithm)
                ax.item(idx).legend(loc='upper left')
            else:
                ax.item(idx).set_prop_cycle(cycler('color', colors))
                ax.item(idx).plot(df[statistic][algorithm], styles[idx_alg])
                ax.item(idx).legend(names, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0)


def plot_value_functions(value, gittins_index=None, discount=None, normalized=False):
    """Plot value functions associated with the Gittins Index.

    Plot the value function V(a,b,r) which is computed in module gittins_computation.
    We fix a, and plot V(a,b,r) for several b values, as r varies. (r is the bernoulli parameter of the safe arm)
    (Plotting function Works on both Numpy Arrays and DataFrames. The indices might differ by one)

    Arguments:
        value: Value function computed through gittins_computation module
        gittins_index: If False, then only the Value functions will be plotted. If True then dots will be
                        plotted where the safe arm and the risky arm cross, which indicated the Gittins Index value.
        discount: discount factor
        normalized: If True, then we normalize all payoffs to their per-period equivalents, i.e. we multiply
                    payoffs by (1/(1-delta)).

    """
    fig, ax = plt.subplots(1, 2, sharex='col')
    fig.suptitle("Value Functions with Risky and Certain Arm; n = " +
                 str(value.shape[0] + 1), fontsize=14, fontweight='bold')
    a_anchor = [2, 20]
    b_range = range(0, 41, 10)
    b_range[0] = 1

    for idx, a in enumerate(a_anchor):
        ax.item(idx).set_title("Risky Arm Parameters: Fix a = " + str(a) + ";   Varying b in range: [" + str(b_range[0]) + "," + str(b_range[-1]) + "]")
        ax.item(idx).set_xlabel("r: Bernoulli Parameter of Safe Arm")
        for b in b_range:
            if normalized:
                ax.item(idx).plot((1 - discount)*value[a, b, :], label=str(b))
            else:
                ax.item(idx).plot(value[a, b, :], label=str(b))
            ax.item(idx).legend(title="Values of b", loc='lower right')
            ax.item(idx).set_xticklabels(map(str, np.linspace(0, 1, 6)))
            if type(gittins_index) != None:
                git_idx = min(np.argwhere((1/float(1 - discount))*
                                          gittins_index[a, b] <= value[a, b, :]))  # Find crossing point of risky and safe arm
                if normalized:
                    ax.item(idx).plot(git_idx, gittins_index[a, b], 'ro')
                else:
                    ax.item(idx).plot(git_idx, (1/float(1 - discount))*gittins_index[a, b], 'ro')


def plot_gittins_index(gittins):
    """Plot Gittins Index gittins(a,b) for values of (a,b)
    (Works on both Numpy Arrays and DataFrames. The indices might differ by one)"""
    a_grid = np.arange(1, gittins.shape[0] + 1)
    b_grid = np.arange(1, gittins.shape[0] + 1)
    a_m, b_m = np.meshgrid(a_grid, b_grid)

    fig = plt.figure()
    fig.suptitle("Gittins Indices for (a,b)", fontsize=14, fontweight='bold')
    ax = plt.gca(projection='3d')
    ax.plot_surface(a_m, b_m, gittins, alpha=0.5, rstride=10, cstride=10)
    # Latter two arguments control precision of plot
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('Gittins Index')


def plot_gittins_index_sliced(gittins, iso):
    """Plot Gittins Index gittins(a,b) for values of (a,b)
    (Works on both Numpy Arrays and DataFrames. The indices might differ by one)"""
    a_grid = np.arange(1, gittins.shape[0]+1)
    b_grid = np.arange(1, gittins.shape[0]+1)
    a_m, b_m = np.meshgrid(a_grid, b_grid)
    gittins_sliced = np.zeros([gittins.shape[0], gittins.shape[0]])
    for b in range(0, gittins.shape[0]):
        for a in range(0, gittins.shape[0]):
            if gittins[a, b] >= iso:
                gittins_sliced[a, b] = iso
            else:
                gittins_sliced[a, b] = gittins[a, b]

    fig = plt.figure()
    fig.suptitle("Gittins Indices for (a,b), sliced at " + str(iso), fontsize=14, fontweight='bold')
    ax = plt.gca(projection='3d')
    ax.plot_surface(a_m, b_m, gittins_sliced, alpha=1, rstride=10, cstride=10)
    # Latter two arguments control precision of plot

    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('Gittins Index')


def plot_gittins_isoquant(gittins, iso, bmax):
    """Plot locus of (a,b) such that gittins(a,b) = iso
    We plot the locus as a = f(b)
    """
    locus = np.zeros(gittins.shape[0])
    for b in range(0, gittins.shape[0]):
        for a in range(0, gittins.shape[0]):
            if gittins[a, b] >= iso:
                locus[b] = a
                break

    fig = plt.figure()
    fig.suptitle("Gittins Isoquant = " + str(iso), fontsize=14, fontweight='bold')
    plt.plot(np.arange(bmax), locus[:bmax])
    plt.xlabel('b')
    plt.ylabel('a')


def plot_gittins_surplus(g):
    """Shows the exrta value that Gittins Index adds to the myopic expectation a/(a+b).
    This extra value indicates the additional value of exploration which the Gittins Index captures.
    """

    p_grid = np.linspace(1, g.shape[0], g.shape[0])
    pa, pb = np.meshgrid(p_grid, p_grid)
    p = pa/(pa + pb)

    p = p.T
    for col in range(0, p.shape[0]):
        for row in range(0, p.shape[0]):
            if col + row >= g.shape[0]:
                p[col, row] = 0

    fig = plt.figure()
    fig.suptitle("Gittins Indices for (a,b)", fontsize=14, fontweight='bold')

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_title("Myopic expectation a/(a + b)")
    ax.view_init(30, 250)
    ax.plot_surface(pa, pb, p, alpha=0.5, rstride=10, cstride=10)
    # Latter two arguments control precision of plot

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.set_title("Gittins residual surplus: Gittins - Myopic")
    ax.view_init(20, 210)
    ax.plot_surface(pa[:50, :50], pb[:50, :50], (g - p)[:50, :50], alpha=0.5, rstride=10, cstride=10)
    # Latter two arguments control precision of plot
