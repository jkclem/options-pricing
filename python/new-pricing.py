# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:01:39 2022

@author: jkcle
"""
from datetime import datetime
import numpy as np


def generate_std_norm(n, m, anti_paths=True, mo_match=True):
    """Generate a matrix of random draws from a standard normal distribution.

    Following Yves Hilpisch's code in Python for Finance.

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


def check_option_params(opt_type, 
                        spot0, 
                        strike, 
                        r, 
                        vol, 
                        exercise,
                        start_date,
                        expire_date,
                        instr_yield
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
    assert isinstance(start_date, datetime), 'Valid arguments for ' \
        'start_date are of time datetime.datetime. You passed ' \
        f'{start_date} of type {type(start_date)}.'
    assert isinstance(expire_date, datetime), 'Valid arguments for ' \
        'start_date are of time datetime.datetime. You passed ' \
        f'{expire_date} of type {type(expire_date)}.'
    assert isinstance(instr_yield, float) or instr_yield is None, 'Valid ' \
        'arguments for instr_yield are of type float. You passed ' \
        f'{instr_yield} of type {type(instr_yield)}.'
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
    start_date : datetime.datetime
        The date at which you are pricing the option.
    expire_date : datetime.datetime
        The date at which the option expires
    year_delta : float
        The time between start_date and expire_date measured in years.
    instr_yield : float
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
                 start_date,
                 expire_date,
                 instr_yield=None
                 ):
        check_option_params(
            opt_type=opt_type,
            spot0=spot0,
            strike=strike,
            r=r,
            vol=vol,
            exercise=exercise,
            start_date=start_date,
            expire_date=expire_date,
            instr_yield=instr_yield
            )
        self.opt_type = opt_type
        self.spot0 = spot0
        self.strike = strike
        self.r = r
        self.vol = vol
        self.exercise = exercise
        self.start_date=start_date,
        self.expire_date=expire_date,
        self.year_delta=(expire_date - start_date).days / 365
        self.instr_yield = instr_yield
        return


    def copy(self):
        return Option(
            opt_type=self.opt_type,
            spot0=self.spot0,
            strike=self.strike,
            r=self.r,
            vol=self.vol,
            exercise=self.exercise,
            instr_yield=self.instr_yield
            )



start_date = datetime(year=2022, month=1, day=1)
expire_date = datetime(year=2023, month=1, day=1)
a = Option('call', 100., 105., 0.015, 0.25, 'european', start_date, expire_date)











