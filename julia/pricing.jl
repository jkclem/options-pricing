using Random, Distributions


function generate_paths(s_0, mu, sigma, periods, steps, num_paths)
    #=This generates geometric Brownian Motion paths of 
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

    =#
    dt = periods / steps
    B_t = randn((num_paths, steps))
    return B_t - mean(B_t)
end

generate_paths(100, 0.15, 0.2, 1, 10, 5)
