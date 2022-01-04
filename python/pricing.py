# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:04:06 2021

@author: jkcle

Following Yves Hilpisch's code in Python for Finance.
"""

import numpy as np


def generate_std_norm(rows, columns, anti_paths=True, mo_match=True):
    """
    

    Parameters
    ----------
    rows : int
        DESCRIPTION.
    columns : int
        DESCRIPTION.
    anti_paths : bool
        DESCRIPTION.
    mo_match : bool
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if anti_paths is True:
        assert columns % 2 == 0, 'Argument "anti_paths" is only valid for ' \
                                  'even numbers!'
        std_norms = np.random.standard_normal(size=(rows, int(columns/2)))
        std_norms = np.concatenate((std_norms, -std_norms), axis=1)
    else:
        std_norms = np.random.standard_normal(size=(rows, columns))
    if mo_match is True:
        std_norms = (std_norms - std_norms.mean()) / std_norms.std()
    return std_norms


def generate_paths(s_0, mu, sigma, periods, steps, num_paths, 
                   anti_paths=False, mo_match=True):
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
    paths : numpy array, shape(steps + 1, num_paths)
        The simulated paths.

    """
    paths = np.zeros(shape=(steps+1, num_paths))
    paths[0, :] = s_0
    dt = periods / steps
    B_t = generate_std_norm(steps, num_paths, anti_paths, mo_match)
    for t in range(1, steps + 1):
        paths[t, :] = paths[t-1, :] * np.exp(
            (mu - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*B_t[t-1, :]
        )
    
    return paths


def price_european_mc(K, s_0, mu, sigma, periods, steps, num_paths, option, 
                      anti_paths=False, mo_match=True):
    """Estimates the value and standard deviation of a European option. Just
    for educational purposes, as an analytic formula exists.

    Parameters
    ----------
    K : float
        The strick price of the option.
    s_0 : float
        Initial value.
    mu : float
        The mean compound return over a period.
    sigma : float
        The volatility (standard deviation) of returns over a period.
    periods : float
        Number of periods being simulated.
    steps : int
        Total number of time steps to break up the simulation over.
    num_paths : int
        The number of simulated paths to generate.
    option : str
        The type of option. Valid arguments are "call" and "put".
    anti_paths : bool
        Whether to use anti-paths in the Monte Carlo simulation. Default is
        True.
    mo_match : bool
        Whether to use moment matching in the Monte Carlo simulation. Default
        is True

    Returns
    -------
    tuple
        The Monte Carlo estimate and standard deviation of the value of a
        European call option.

    """
    assert option in ['call', 'put'], 'Valid arguments for option are ' \
                                      '"call" and "put"!'

    paths = generate_paths(
        s_0, mu, sigma, periods, steps, num_paths, anti_paths, mo_match
        )

    if option == 'call':
        payoffs = np.maximum(paths[-1, :] - K, 0)
    else:
        payoffs = np.maximum(K - paths[-1, :], 0)

    return (np.exp(-mu * periods) * np.mean(payoffs), 
            np.exp(-mu * periods) * np.std(payoffs))
