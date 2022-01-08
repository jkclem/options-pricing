# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 21:01:39 2022

@author: jkcle
"""


def check_option_params(opt_type, spot0, strike, r, vol, exercise, instr_yield):
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
    assert isinstance(instr_yield, float), 'Valid arguments for ' \
        f'instr_yield are of type float. You passed {instr_yield} of type ' \
        f'{type(instr_yield)}.'
    return


class Option(object):
    """
    
    Attributes
    ----------
    opt_type : str
    spot0 : int or float
    strike : int or float
    r : float
    vol : float
    exercise : str
    instr_yield : float

    Methods
    -------
    copy: Returns an object of class Option with the same attributes.

    """
    def __init__(self, opt_type, spot0, strike, r, vol, exercise, instr_yield):
        check_option_params(
            opt_type, spot0, strike, r, vol, exercise, instr_yield
            )
        self.opt_type = opt_type
        self.spot0 = spot0
        self.strike = strike
        self.r = r
        self.vol = vol
        self.exercise = exercise
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















