# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:04:06 2021

@author: jkcle
"""

import numpy as np


def generate_paths(s_0, mu, sigma, periods, steps, num_paths):
    """This generates geometric Brownian Motion paths of 
    prices/values/levels of an asset/portfolio/index using Monte Carlo 
    simulation.

    CAUTION: this produces a num_paths x (steps + 1) array. Using large values
    can generate matrices too large to fit in memory, or can cause slow downs.

    Parameters
    ----------
    s_0 : float
        Initial value.
    mu : float
        The mean compound return over a period.
    sigma : float
        The volatility (standard deviation) of returns over a period.
    periods : float
        Number of periods being simulated.
    steps : TYPE
        Total number of .
    num_paths : TYPE
        The number of simulated paths to generate.

    Returns
    -------
    paths : numpy array, shape(num_paths, steps + 1)
        The simulated paths.

    """
    dt = periods / steps
    B_t = np.random.standard_normal(size=(num_paths, steps))
    B_t = (B_t - B_t.mean()) / B_t.std()
    paths = (mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*B_t
    paths = np.insert(paths, 0, 0, axis=1)
    paths = paths.cumsum(axis=1)
    paths = np.exp(paths)

    return paths * s_0
