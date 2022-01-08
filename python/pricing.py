# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 23:04:06 2021

@author: jkcle

Following Yves Hilpisch's code in Python for Finance.
"""

import numpy as np


def generate_std_norm(rows, columns, anti_paths=True, mo_match=True):
    """Generate a matrix of random draws from a standard normal distribution.

    Following Yves Hilpisch's code in Python for Finance.

    Parameters
    ----------
    rows : int
        The number of rows in the random matrix.
    columns : int
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


def generate_paths(s_0, r, sigma, periods, steps, num_paths, 
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
    r : float
        The risk-free rate.
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
    paths : numpy ndarray, shape(steps + 1, num_paths)
        The simulated paths.

    """
    paths = np.zeros(shape=(steps+1, num_paths))
    paths[0, :] = s_0
    dt = periods / steps
    B_t = generate_std_norm(steps, num_paths, anti_paths, mo_match)
    for t in range(1, steps + 1):
        paths[t, :] = paths[t-1, :] * np.exp(
            (r - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*B_t[t-1, :]
        )
    
    return paths


def price_european_mc(K, s_0, r, sigma, periods, steps, num_paths, option, 
                      anti_paths=False, mo_match=True):
    """Estimates the value and standard deviation of an European option. Just
    for educational purposes, as an analytic formula exists.

    Parameters
    ----------
    K : float
        The strick price of the option.
    s_0 : float
        Initial value.
    r : float
        The risk-free rate.
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
    float
        The Monte Carlo estimate of the value of a European option.

    """
    assert option in ['call', 'put'], 'Valid arguments for option are ' \
                                      '"call" and "put"!'

    paths = generate_paths(
        s_0, r, sigma, periods, steps, num_paths, anti_paths, mo_match
        )

    if option == 'call':
        payoffs = np.maximum(paths[-1, :] - K, 0)
    else:
        payoffs = np.maximum(K - paths[-1, :], 0)

    return np.exp(-r * periods) * np.mean(payoffs)


def price_american_mc(K, s_0, r, sigma, periods, steps, num_paths, option, 
                      anti_paths=False, mo_match=True):
    """Estimates the value of an American option. Using Least-Squares Monte 
    Carlo.

    Parameters
    ----------
    K : float
        The strick price of the option.
    s_0 : float
        Initial value.
    r : float
        The risk-free rate.
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
    float
        The Monte Carlo estimate of the value of an American call option.

    """
    assert option in ['call', 'put'], 'Valid arguments for option are ' \
                                      '"call" and "put"!'

    df = np.exp(-r * periods/steps)
    paths = generate_paths(
        s_0, r, sigma, periods, steps, num_paths, anti_paths, mo_match
        )

    if option == 'call':
        payoffs = np.maximum(paths - K, 0)
    else:
        payoffs = np.maximum(K - paths, 0)

    V = np.copy(payoffs)

    for t in range(steps - 1, 0, -1):
        reg = np.polyfit(paths[t, :], V[t+1, :] * df, 3)
        C = np.polyval(reg, paths[t, :])
        V[t, :] = np.where(C > payoffs[t], V[t+1, :] * df, payoffs[t, :])

    return df * np.mean(V[1, :])
