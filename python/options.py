# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:01:39 2022

@author: jkcle

Code draws from Yves Hilpisch's code in Python for Finance and Ben Gimpert's
code in pyfin (https://github.com/someben/pyfin/blob/master/pyfin).
"""
import numpy as np
from scipy.stats import norm

from orthopolyreg import OrthoPolyReg
from stochastic import generate_gbm_paths


def check_option_params(opt_type, 
                        spot0, 
                        strike, 
                        r, 
                        vol, 
                        exercise,
                        year_delta,
                        div_yield
                        ):
    assert opt_type in ['call', 'put'], 'Valid arguments for opt_type ' \
        f'are "call" and "put". You passed {opt_type}.'
    assert isinstance(spot0, int) or isinstance(spot0, float), 'Valid ' \
        f'arguments for spot0 are of type int or float. You passed {spot0} ' \
        f'of type {type(spot0)}.'
    assert isinstance(strike, int) or isinstance(strike, float), '' \
        'Valid arguments for strike are of type int or float. You passed ' \
        f'{strike} of type {type(strike)}.'
    assert isinstance(r, float), 'Valid arguments for r are of type ' \
        f'float. You passed {r} of type {type(r)}.'
    assert isinstance(vol, float), 'Valid arguments for vol are of ' \
        f'type float. You passed {vol} of type {type(vol)}.'
    assert exercise in ['american', 'european'], 'Valid arguments for ' \
        f'exercise are "american" and "european". You passed {exercise}.'
    assert isinstance(year_delta, float) and (year_delta > 0), 'Valid ' \
        'arguments for year_delta are positive floats.'
    assert isinstance(div_yield, float) or div_yield is None, 'Valid ' \
        'arguments for div_yield are of type float. You passed ' \
        f'{div_yield} of type {type(div_yield)}.'
    return


class Option(object):
    """
    
    Initalized Attributes
    ---------------------
    opt_type : str
        Either "call" or "put".
    spot0 : int or float
        The initial value of the asset.
    strike : int or float
        The strike price of the option.
    r : float
        The risk-free rate.
    vol : float
        The standard deviation of the asset.
    exercise : str
        Type of exercise of the option. Valid arguments are "american" and
        "european".
    year_delta : float
        A postive float measuring the time to expiration in years.
    year_delta : float
        The time between start_date and expire_date measured in years.
    div_yield : float
        Yield of the asset.

    Methods
    -------
    copy: Returns an object of class Option with the same attributes.

    """
    def __init__(self,
                 opt_type,
                 spot0, strike,
                 r,
                 vol,
                 exercise,
                 year_delta,
                 div_yield=None
                 ):
        check_option_params(
            opt_type=opt_type,
            spot0=spot0,
            strike=strike,
            r=r,
            vol=vol,
            exercise=exercise,
            year_delta=year_delta,
            div_yield=div_yield
            )
        self.opt_type = opt_type
        self.spot0 = spot0
        self.strike = strike
        self.r = r
        self.vol = vol
        self.exercise = exercise
        self.year_delta=year_delta
        self.div_yield = div_yield if div_yield is not None else 0.
        return


    def copy(self):
        return Option(
            opt_type=self.opt_type,
            spot0=self.spot0,
            strike=self.strike,
            r=self.r,
            vol=self.vol,
            exercise=self.exercise,
            year_delta=self.year_delta,    
            div_yield=self.div_yield
            )


    def value_option(self, method='mc', **kwargs):
        """Values an option using a passed in method."""

        assert method in ['mc', 'bs'], 'Valid method arguments are "mc" ' \
            'and "bs".'

        self.__valuation_method = method

        if method == 'mc':
            value = self.__value_mc(**kwargs)

        elif method == 'bs':
            assert self.exercise == 'european', 'The exercise type must be ' \
                '"european" to use Black-Scholes to price the option.'
            value = self.__black_scholes()

        return value


    def __value_mc(self, 
                   steps, 
                   num_paths, 
                   anti_paths=False, 
                   mo_match=True,
                   save_paths=False,
                   seed=None,
                   calc_greeks=True,
                   sens_degree=2
                   ):
        """Value the option using Monte Carlo simulation of Geometric Brownian
        Motion.

        steps : int
            Total number of time steps to break up the simulation over.
        num_paths : int
            The number of simulated paths to generate.
        anti_paths : bool
            Whether to use anti-paths in the Monte Carlo simulation. Default
            is True.
        mo_match : bool
            Whether to use moment matching in the Monte Carlo simulation. 
            Default is True
        save_paths : bool
            Whether or not to save the paths to the Option object. Default is
            False.
        seed : NoneType or int
            A random seed. Default is None
        calc_greeks : bool
            Whether to estimate the Greeks or not.
        sens_degree : int
            If 2 > sens_degree >= 1, calculate everything but Gamma. If
            sens_degree >= 2, calculate Gamma, as well.

        Returns
        -------
        float
            The Monte Carlo estimate of the value of a European option.
    
        """
        if seed is not None:
            np.random.seed(seed)

        if self.exercise == 'american':
            self.value = self.__value_american_mc(
                            steps=steps, 
                            num_paths=num_paths, 
                            anti_paths=anti_paths, 
                            mo_match=mo_match,
                            save_paths=save_paths
                            )
        else:  # If the exercise is 'european'
            self.value = self.__value_european_mc(
                            steps=steps, 
                            num_paths=num_paths, 
                            anti_paths=anti_paths, 
                            mo_match=mo_match,
                            save_paths=save_paths
                            )

        if calc_greeks:
            self. __calc_greeks_mc(sens_degree=sens_degree,
                                   steps=steps, 
                                   num_paths=num_paths, 
                                   anti_paths=anti_paths, 
                                   mo_match=mo_match,
                                   seed=seed
                                   )

        return self.value


    def __black_scholes(self):
        """Uses Black-Scholes to value the option.

        Relies on Ben Gimpert's code.

        Returns
        -------
        None.

        """
        div_yield = 0 if self.div_yield is None else self.div_yield

        sqrt_mat = self.year_delta ** 0.5
        d1 = ((np.log(self.spot0 / self.strike)
               + (self.r - div_yield + 0.5 * self.vol**2) 
               * self.year_delta) / (self.vol * sqrt_mat))
        d2 = d1 - self.vol * (self.year_delta ** 0.5)
        d1_pdf = norm.pdf(d1)
        riskless_disc = np.exp(-self.r * self.year_delta)
        yield_disc = np.exp(-div_yield * self.year_delta)
        if self.opt_type == 'call':
            d1_cdf = norm.cdf(d1)
            d2_cdf = norm.cdf(d2)
            delta = yield_disc * d1_cdf
            value = self.spot0 * delta - riskless_disc * self.strike * d2_cdf
            theta = (-yield_disc * (self.spot0 * d1_pdf * self.vol) 
                     / (2 * sqrt_mat) 
                     - self.r * self.strike * riskless_disc * d2_cdf 
                     + div_yield * self.spot0 * yield_disc * d1_cdf)
            rho = self.strike * self.year_delta * riskless_disc * d2_cdf
    
        else:  # self.opt_type == 'put':
            neg_d1_cdf = norm.cdf(-d1)
            neg_d2_cdf = norm.cdf(-d2)
            delta = -yield_disc * neg_d1_cdf
            value = riskless_disc*self.strike*neg_d2_cdf + self.spot0*delta
            theta = (-yield_disc * (self.spot0 * d1_pdf * self.vol) 
                     / (2 * sqrt_mat) + self.r*self.strike 
                     * riskless_disc * neg_d2_cdf - div_yield 
                     * self.spot0 * yield_disc * neg_d1_cdf)
            rho = -self.strike * self.year_delta * riskless_disc * neg_d2_cdf
    
        vega = self.spot0 * yield_disc * d1_pdf * sqrt_mat
        gamma = yield_disc * (d1_pdf / (self.spot0 * self.vol * sqrt_mat))

        self.value, self.delta, self.gamma = value, delta, gamma
        self.theta, self.vega, self.rho = theta, vega, rho
        self.__greeks_calculated = True

        return value
        

    
    def __value_european_mc(self, 
                            steps, 
                            num_paths, 
                            anti_paths=False, 
                            mo_match=True,
                            save_paths=False
                            ):
        """Estimates the value of an European option. An analytic formula 
        exists and is prefered. Assumes Geometric Brownian Motion.

        Relies on Yves Hilpisch's code.
    
        Parameters
        ----------
        steps : int
            Total number of time steps to break up the simulation over.
        num_paths : int
            The number of simulated paths to generate.
        anti_paths : bool
            Whether to use anti-paths in the Monte Carlo simulation. Default
            is True.
        mo_match : bool
            Whether to use moment matching in the Monte Carlo simulation. 
            Default is True
        save_paths : bool
            Whether or not to save the paths to the Option object. Default is
            False.
    
        Returns
        -------
        float
            The Monte Carlo estimate of the value of a European option.
    
        """
    
        paths = generate_gbm_paths(
            self.spot0, self.r - self.div_yield, self.vol, self.year_delta, 
            steps, num_paths, anti_paths, mo_match
            )
    
        if self.opt_type == 'call':
            payoffs = np.maximum(paths[-1, :] - self.strike, 0)
        else:
            payoffs = np.maximum(self.strike - paths[-1, :], 0)

        if save_paths:
            self.sim_paths = paths
    
        return np.exp(-self.r * self.year_delta) * np.mean(payoffs)
    
    
    def __value_american_mc(self, 
                            steps, 
                            num_paths,
                            degree=7,
                            anti_paths=False, 
                            mo_match=True,
                            save_paths=False
                            ):
        """Estimates the value of an American option using Least-Squares 
        Monte Carlo. Assumes Geometric Brownian Motion.

        Relies on Yves Hilpisch's code.
    
        Parameters
        ----------
        steps : int
            Total number of time steps to break up the simulation over.
        num_paths : int
            The number of simulated paths to generate.
        degree : int
            The degree of fit in LSM.
        anti_paths : bool
            Whether to use anti-paths in the Monte Carlo simulation. Default
            is True.
        mo_match : bool
            Whether to use moment matching in the Monte Carlo simulation. 
            Default is True.
        save_paths : bool
            Whether or not to save the paths to the Option object. Default is
            False.
    
        Returns
        -------
        float
            The Monte Carlo estimate of the value of an American call option.
    
        """
        K = self.strike
        spot0 = self.spot0
        r = self.r - self.div_yield
        vol = self.vol
        periods = self.year_delta
        opt_type = self.opt_type

        df = np.exp(-r * periods/steps)
        paths = generate_gbm_paths(
            spot0, r, vol, periods, steps, num_paths, anti_paths, mo_match
            )
    
        if opt_type == 'call':
            payoffs = np.maximum(paths - K, 0)
        else:
            payoffs = np.maximum(K - paths, 0)

        V = payoffs[-1]
    
        reg = OrthoPolyReg()
        for t in range(steps - 2, 0, -1):
            pos_payoff_indices = np.where(V > 0)
            y = V[pos_payoff_indices] * df
            X = paths[t][pos_payoff_indices]
            
            reg.fit(X, y, degree=degree)
            C = reg.predict(paths[t])
            """
            reg = np.polyfit(X, y, deg=degree)
            C = np.polyval(reg, paths[t])
            """ 
            V = np.where(C > payoffs[t], V * df, payoffs[t])

        if save_paths:
            self.sim_paths = paths

        return df*np.mean(V)


    def greeks(self):
        assert self.__greeks_calculated, "The greeks are not calculated."
        return {
            'delta': self.delta, 
            'theta': self.theta, 
            'rho': self.rho,
            'vega': self.vega,
            'gamma': self.gamma
            }


    def __calc_greeks_mc(self,
                         sens_degree,
                         steps, 
                         num_paths, 
                         anti_paths=False, 
                         mo_match=True,
                         seed=None
                         ):
        """
        """
        delta, theta, rho, vega, gamma = None, None, None, None, None

        if sens_degree >= 1:
            if seed is not None:
                np.random.seed(seed)
            delta = self.sensitivity(
                'spot0', 
                self.spot0 * 1.0e-05,
                sens_degree=sens_degree - 1,
                method='mc',
                steps=steps, 
                num_paths=num_paths, 
                anti_paths=anti_paths, 
                mo_match=mo_match,
                seed=seed
                )
            if seed is not None:
                np.random.seed(seed)
            theta = -self.sensitivity(
                'year_delta', 
                0.001, 
                sens_degree=sens_degree - 1,
                method='mc',
                steps=steps, 
                num_paths=num_paths, 
                anti_paths=anti_paths, 
                mo_match=mo_match,
                seed=seed
                )
            if seed is not None:
                np.random.seed(seed)
            rho = self.sensitivity(
                'r',
                0.0001,
                sens_degree=sens_degree - 1,
                method='mc',
                steps=steps, 
                num_paths=num_paths, 
                anti_paths=anti_paths, 
                mo_match=mo_match,
                seed=seed
                )
            if seed is not None:
                np.random.seed(seed)
            vega = self.sensitivity(
                'vol',
                0.001, 
                sens_degree=sens_degree - 1,
                method='mc',
                steps=steps, 
                num_paths=num_paths, 
                anti_paths=anti_paths, 
                mo_match=mo_match,
                seed=seed
                )

            self.delta = delta
            self.theta = theta
            self.rho = rho
            self.vega = vega

        if sens_degree >= 2:
            if seed is not None:
                np.random.seed(seed)
            gamma = self.sensitivity(
                'spot0',
                self.spot0 * 0.05,
                opt_measure='delta',
                sens_degree=sens_degree - 1,
                method='mc',
                steps=steps, 
                num_paths=num_paths, 
                anti_paths=anti_paths, 
                mo_match=mo_match,
                seed=seed
                )
            self.gamma = gamma

        self.__greeks_calculated = True

        return


    def sensitivity(self, opt_param, opt_param_bump, method, 
                    opt_measure='value',**method_kwargs):
        up_opt = self.copy()
        setattr(
            up_opt, opt_param, getattr(up_opt, opt_param) + opt_param_bump
            )
        down_opt = self.copy()
        setattr(
            down_opt, opt_param, getattr(down_opt, opt_param) - opt_param_bump
            )
        _ = up_opt.value_option(method=method, **method_kwargs)
        up_measure = getattr(up_opt, opt_measure)
        _ = down_opt.value_option(method=method, **method_kwargs)
        down_measure = getattr(down_opt, opt_measure)
        return (up_measure - down_measure) / (2 * opt_param_bump)


###
# Testing Functions
###

from datetime import datetime
from matplotlib import pyplot as plt


def plot_sim_paths(steps, 
                   option, 
                   title='Simulated Price Paths for {exercise} {opt_type} Option', 
                   xlab='Time (Start = 0, Expiry = 1)', 
                   ylab='Asset Price', 
                   add_strike=True,
                   **kwargs
                   ):

    if title == 'Simulated Price Paths for {exercise} {opt_type} Option':
        title = title.format(
            exercise=option.exercise.upper(),
            opt_type=option.opt_type.upper()
            )

    x = np.arange(0, steps + 1) / steps
    plt.plot(x, option.sim_paths)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)

    if add_strike:
        plt.hlines(
            option.strike, xmin=np.min(x), xmax=np.max(x), zorder=len(x) + 1,
            **kwargs
            )
        plt.legend()
    
    plt.show();

    return


def plot_value_vs_strike(strike_delta,
                         opt_type, 
                         spot0,
                         r, 
                         vol, 
                         exercise,
                         year_delta,
                         method='mc',
                         **kwargs
                         ):
    strike_range = np.arange(spot0 - strike_delta, spot0 + strike_delta + 1)
    values = []
    for strike_price in strike_range:
        temp_option = Option(
        opt_type, spot0, int(strike_price), r, vol, exercise, year_delta
        )
        temp_val = temp_option.value_option(method, **kwargs)
        values.append(temp_val)
    plt.plot(strike_range, values)
    plt.title(f'{exercise.upper()} {opt_type.upper()} Value vs. Strike '\
              f'(r = {r}, Vol = {vol})')
    plt.xlabel('Strike Price')
    plt.ylabel('Option Value')
    plt.show();
    return
