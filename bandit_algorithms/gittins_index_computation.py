"""Computes an approximated Gittins Index, using backward induction for a fixed horizon length.

For more see Gittins etal. 2011 MAB Allocation Indices, 2nd ed.
For example the authors list precomputed index values for some parameter specifications on p.265.
"""

import numpy as np
import pandas as pd
import time

__author__ = 'Ilari'


def initial_approximation(pulls, discount, grid_n):
    """Approximate the initial values for the value function to begin backward induction.

    Pulls specifies the total number of bandit arm pulls and observations from which backward
    induction is used to compute the index values for any distribution of discrete binary
    observations. Success denoted by a, and failure denoted by b.

    Assumptions 1 <= a,b <= pulls - 1, so we assume at least one observation of success and failure.

    :param pulls: scalar
    :param discount: Discount factor from interval (0,1)
    :param grid_n

    :return gittins, values: Initialized index array and value array
    """

    values = np.zeros([pulls-1, pulls-1, grid_n])  # Store V(a=k, b=n-k, r) in values[k,n-1,:] as k varies
    gittins = np.zeros([pulls-1, pulls-1])  # Store Gittins(a=k, b=n-k) in gittins[k,n-1] as k varies

    a_grid = np.arange(1, pulls)
    r_grid = np.linspace(0, 1, grid_n)

    initial_gittins = a_grid/float(pulls)  # Initial Gittins Approximation to start Backward Induction
    gittins[0:pulls, pulls - 2] = initial_gittins  # Record initial Gittins approximation

    for idx_a, a in enumerate(a_grid):
        values[idx_a, pulls - 2, :] = (1.0/(1 - discount)) *\
                                      np.maximum(r_grid, a/float(pulls))  # Record initial Value approximation

    return gittins, values


def recursion_step(value_n, r_grid, discount):
    """One-step backward induction computing the value function and the Gittins Index.

     See for instance Gittins etal 2011, or Powell and Ryzhov 2012, for recursion step details.
     """

    n = value_n.shape[0]
    r_len = r_grid.shape[0]
    value_n_minus_1 = np.zeros([n - 1, r_len])  # Value function length reduced by 1
    gittins_n_minus_1 = np.zeros(n - 1)  # Value function length reduced by 1
    for k in range(0, n - 1):
        a = k + 1        # a in range [1,n-1]
        b = n - k - 1    # b in range [1,n-1]
        value_n_minus_1[k, :] = np.maximum((r_grid/float(1 - discount)),
                                           (a/float(n))*(1 + discount*value_n[k + 1, :]) +
                                           (b/float(n))*discount*value_n[k, :]
                                           )
        try:
            # Find first index where Value = (Value of Safe Arm)
            idx_git = np.argwhere((r_grid/float(1 - discount)) == value_n_minus_1[k, :]).flatten()
            gittins_n_minus_1[k] = 0.5*(r_grid[idx_git[0]] + r_grid[idx_git[0] - 1])  # Take average
        except:
            print "Error in finding Gittins index"

    return gittins_n_minus_1, value_n_minus_1


def recursion_loop(pulls, discount, grid_n):
    """This produces the value functions and Gittins indices by backward induction"""

    r_grid = np.linspace(0, 1, grid_n)
    gittins, values = initial_approximation(pulls, discount, grid_n)
    n = pulls - 2  # Note that the 2 comes from (1) the initial approximation and (2) python indexing
    while n >= 1:
        g, v = recursion_step(values[:n + 1, n, :], r_grid, discount)
        values[:n, n-1] = v
        gittins[:n, n-1] = g
        n -= 1
    return gittins, values


def reformat_gittins(g, v=None):
    """Reformat output.

    We reformat the result to get the results in a similar form
    as in (Gittins etal 2011, Powell and Ryzhov 2012), except that we store:
    Success count denoted by a in rows
    Failure count denoted by b in columns
    """

    start_time = time.time()
    n = g.shape[0]
    g_reformat = np.zeros(g.shape)

    for row in range(0, n):
        g_reformat[row, :n-row] = g[row, row:]

    try:
        v_reformat = np.zeros(v.shape)
        for row in range(0, n):
            v_reformat[row, :n-row, :] = v[row, row:, :]
        print "Elapsed time in Gittins Index Reformatting: ", time.time() - start_time
        return g_reformat, v_reformat
    except:
        print "Elapsed time in Gittins Index Reformatting: ", time.time() - start_time
        return g_reformat


def gittins_index(n=500, grid=1000, discount=0.9, value=False,  df=False):
    """Compute Gittins indices and value functions.

    Comment: To get the results to match up, with See Gittins etal. (2011, p.265)
    we need a fairly fine grid: approx 5000 grid points, equates the results.

    :param n: Number of pulls, from which to start backward induction
    :param grid: Number of grid points to use for safe arm
    :param discount: discount factor used to compute Gittins Index
    :param value: If True, function return value functions, in addition to Gittins Index
    :param df: If True, function returns a Pandas DataFrame of the results in addition to Numpy Arrays

    :return g: Gittins index, (n x n) array
               rows: a count, i.e. number of successes.
               columns: b count, i.e. number of failures.
     :return v: Value function, (n x n x grid) array
                rows: a count, i.e. number of successes.
                columns: b count, i.e. number of failures.
                dimension 3: r grid for the bernoulli parameter of the certain arm.
    """

    print "Computing Gittins Index..."
    start_time = time.time()
    g, v = recursion_loop(n, discount, grid)
    print "Elapsed time in Gittins Index Calculation: ", time.time() - start_time

    if df == False:
        if value == False:
            g_reformat = reformat_gittins(g, v)
            return g_reformat
        else:
            g_reformat, v_reformat = reformat_gittins(g, v)
            return g_reformat, v_reformat
    else:
        if value == True:
            g_reformat, v_reformat = reformat_gittins(g, v)
            df_g = pd.DataFrame(g_reformat, index=range(1, n), columns=range(1, n))
            df_v = pd.Panel(v_reformat, items=range(1, n), major_axis=range(1, n), minor_axis=range(0, grid))
            return g_reformat, df_g, v_reformat, df_v
        else:
            g_reformat = reformat_gittins(g)
            df_g = pd.DataFrame(g_reformat, index=range(1, n), columns=range(1, n))
            return g_reformat, df_g
