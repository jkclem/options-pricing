# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 01:06:05 2022

@author: jkcle
"""
import numpy as np


def generate_std_norm(n, m, anti_paths=True, mo_match=True):
    """Generate a matrix of random draws from a standard normal distribution.

    Relies on Yves Hilpisch's code.

    Parameters
    ----------
    n : int
        The number of rows in the random matrix.
    m : int
        The number of columns in the random matrix.
    anti_paths : bool
        Whether or not to set half (must be even number of columns) of the
        random column vectors as the negative of the other half i.e. mirror
        half the paths. Default is True.
    mo_match : bool
        Whether or not to standardize the first and second moments (mean and
        standard deviation) of the random matrix. Default is True.

    Returns
    -------
    std_norms : numpy.ndarray
        A matrix of pseudo-random draws from a N(0, 1) distribution.

    """

    if anti_paths is True:
        assert m % 2 == 0, 'Argument "anti_paths" is only valid for ' \
                           'even numbers!'
        std_norms = np.random.standard_normal(size=(n, int(m/2)))
        std_norms = np.concatenate((std_norms, -std_norms), axis=1)
    else:
        std_norms = np.random.standard_normal(size=(n, m))
    if mo_match is True:
        std_norms = (std_norms - std_norms.mean()) / std_norms.std()

    return std_norms


def generate_gbm_paths(spot0, r, vol, periods, steps, num_paths, 
                       anti_paths=False, mo_match=True):
    """This generates geometric Brownian Motion paths of 
    prices/values/levels of an asset/portfolio/index using Monte Carlo 
    simulation.

    Relies on Yves Hilpisch's code.

    CAUTION: this produces a num_paths x (steps + 1) array. Using large values
    can generate matrices too large to fit in memory, or can cause slow downs.

    Parameters
    ----------
    spot0 : float
        Initial value.
    r : float
        The risk-free rate.
    vol : float
        The volatility (standard deviation) of returns over a period.
    periods : float
        Number of periods being simulated.
    steps : int
        Total number of time steps to break up the simulation over.
    num_paths : int
        The number of simulated paths to generate.
    anti_paths : bool
        Whether to use anti-paths in the Monte Carlo simulation. Default is
        True.
    mo_match : bool
        Whether to use moment matching in the Monte Carlo simulation. Default
        is True

    Returns
    -------
    paths : numpy ndarray, shape(steps + 1, num_paths)
        The simulated paths.

    """
    paths = np.zeros(shape=(steps+1, num_paths))
    paths[0, :] = spot0
    dt = periods / steps
    B_t = generate_std_norm(steps, num_paths, anti_paths, mo_match)
    for t in range(1, steps + 1):
        paths[t, :] = paths[t-1, :] * np.exp(
            (r - 0.5 * vol**2)*dt + vol*np.sqrt(dt)*B_t[t-1, :]
        )
    
    return paths
