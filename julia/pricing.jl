#=
DX Analytics
Framework Classes and Functions
dx_frame.py

DX Analytics is a financial analytics library, mainly for
derviatives modeling and pricing by Monte Carlo simulation

(c) Dr. Yves J. Hilpisch
The Python Quants GmbH

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program. If not, see http://www.gnu.org/licenses/.

This is a translation of Yves Hilpisch's Python code into Julia.
=#
using Dates, Random, Distributions, Interpolations

# Helper functions


function get_year_deltas(time_list, day_count=365.)
    #= Return vector of floats with time deltas in years.
    Initial value normalized to zero.

    Parameters
    ==========
    time_list : Vector{DateTime}
        collection of datetime objects
    day_count : Float64
        number of days for a year (to account for different conventions)

    Results
    =======
    delta_list : Array{Float64}
        year fractions
    =#

    delta_list = Array{Float64}(undef, length(time_list))
    start = minimum(time_list)
    for i in 1:length(time_list)
        days = convert(Dates.Day, time_list[i] - start).value
        delta_list[i] = days / day_count
    end

    return delta_list
end


function sn_random_numbers(shape, antithetic=true, moment_matching=true, fixed_seed=false)
    #= Return an array of shape "shape" with (pseudo-) random numbers
    which are standard normally distributed.
    Parameters
    ==========
    shape : Tuple{Int64} (o, n, m)
        generation of array with shape (o, n, m)
    antithetic : Bool
        generation of antithetic variates
    moment_matching : Bool
        matching of first and second moments
    fixed_seed : Bool
        flag to fix the seed
    Results
    =======
    ran : Array{Float64} (o, n, m) 
        array of (pseudo-)random numbers
    =#
    #if fixed_seed is True:
    #    np.random.seed(1000)
    if antithetic == true
        ran = randn((shape[1], shape[2], Int(shape[3] / 2)))
        ran = cat(ran, -ran, dims=3)
    else
        ran = randn(shape)
    end

    if moment_matching == true
        ran = ran .- mean(ran)
        ran = ran ./ std(ran)
    end

    if shape[1] == 1
        ran = ran[1, :, :]
    end

    return ran
end


# Discounting Structs

struct constant_short_rate
    #= Struct for constant short rate discounting.
    Attributes
    ==========
    name : string
        name of the object
    short_rate : float (positive)
        constant rate for discounting

    Associated Functions
    =======
    get_discount_factors
        get discount factors given a list/array of datetime objects
        or year fractions
    =#
    name::String
    short_rate::Float64
end


function get_discount_factors(constant_short_rate_obj, time_list, dtobjects=true)
    if dtobjects == true
        dlist = get_year_deltas(time_list)
    else
        dlist = time_list
    end
    discount_factors = exp.(constant_short_rate_obj.short_rate * sort(-dlist))
    return time_list, discount_factors
end


struct deterministic_short_rate
    #= Stuct for discounting based on deterministic short rates,
    derived from a term structure of zero-coupon bond yields
    Attributes
    ==========
    name : string
        name of the object
    yield_list : list/array of (time, yield) tuples
        input yields with time attached

    Associated Functions
    =======
    get_interpolated_yields :
        return interpolated yield curve given a time list/array
    get_forward_rates :
        return forward rates given a time list/array
    get_discount_factors :
        return discount factors given a time list/array
    =#
    name::String
    yield_list::Array
end

function get_interpolated_yields(deterministic_short_rate_obj, time_list, dtobjects=true)
    #= time_list either list of datetime objects or list of
    year deltas as decimal number (dtobjects=False)
    =#
    if dtobjects == true
        tlist = get_year_deltas(time_list)
    else
        tlist = time_list
    end

    dtuple = tuple(get_year_deltas(deterministic_short_rate_obj.yield_list[:, 1]))
    yield_vec = convert(Vector{Float64}, deterministic_short_rate_obj.yield_list[:,2])

    if length(dtuple) <= 3
        yield_spline = interpolate(dtuple, yield_vec, Gridded(Linear()))
    else
        yield_spline = interpolate(dtuple, yield_vec, (BSpline(Cubic(Natural(OnGrid())))))
    end
    yield_spline_ex = extrapolate(yield_spline, Line())
    yield_curve = yield_spline_ex(tlist)
    yield_deriv = Interpolations.ChainRulesCore.rrule(yield_spline_ex, tlist)[2].x[1]
    yield_mat = hcat(time_list, yield_curve, yield_deriv)
    return yield_mat
end

dates = [DateTime(2021, 1, 1)]
yields = [0.015]
for i in 1:365:5
    append!(dates_abc, dates_abc + Day(i))
    append!(yields, yields * 1.05)
end

dates_alt = [DateTime(2021, 1, 1)]
for i in 1:365
    append!(dates_abc, dates_abc + Day(i))
end

time_list = get_year_deltas(dates)
time_list_alt = get_year_deltas(dates_alt)
yield_arr = cat(dates, yields, dims=2)
yield_list = cat(time_list, yields, dims=2)

a = deterministic_short_rate("my_market", yield_arr)

b = get_interpolated_yields(a, dates_alt, true)

yield_vec = convert(Vector{Float64}, yield_list[:,2])


using Plots

scatter(time_list_alt, b[:, 2])
scatter!(time_list, yields)
plot!(time_list_alt, b[:, 2])


#=interpolate(tuple(yield_list[:, 1]), yield_list[:, 2], (BSpline(Cubic(Natural(OnGrid())))))
[reshape(b[1], 1, :); reshape(b[2], 1, :)]
# Discounting classes

class deterministic_short_rate(object):
    ''' Class for discounting based on deterministic short rates,
    derived from a term structure of zero-coupon bond yields
    Attributes
    ==========
    name : string
        name of the object
    yield_list : list/array of (time, yield) tuples
        input yields with time attached
    Methods
    =======
    get_interpolated_yields :
        return interpolated yield curve given a time list/array
    get_forward_rates :
        return forward rates given a time list/array
    get_discount_factors :
        return discount factors given a time list/array
    '''

    def __init__(self, name, yield_list):
        self.name = name
        self.yield_list = np.array(yield_list)
        if np.sum(np.where(self.yield_list[:, 1] < 0, 1, 0)) > 0:
            raise ValueError('Negative yield(s).')

    def get_interpolated_yields(self, time_list, dtobjects=True):
        ''' time_list either list of datetime objects or list of
        year deltas as decimal number (dtobjects=False)
        '''
        if dtobjects is True:
            tlist = get_year_deltas(time_list)
        else:
            tlist = time_list
        dlist = get_year_deltas(self.yield_list[:, 0])
        if len(time_list) <= 3:
            k = 1
        else:
            k = 3
        yield_spline = sci.splrep(dlist, self.yield_list[:, 1], k=k)
        yield_curve = sci.splev(tlist, yield_spline, der=0)
        yield_deriv = sci.splev(tlist, yield_spline, der=1)
        return np.array([time_list, yield_curve, yield_deriv]).T

    def get_forward_rates(self, time_list, paths=None, dtobjects=True):
        yield_curve = self.get_interpolated_yields(time_list, dtobjects)
        if dtobjects is True:
            tlist = get_year_deltas(time_list)
        else:
            tlist = time_list
        forward_rates = yield_curve[:, 1] + yield_curve[:, 2] * tlist
        return time_list, forward_rates

    def get_discount_factors(self, time_list, paths=None, dtobjects=True):
        discount_factors = []
        if dtobjects is True:
            dlist = get_year_deltas(time_list)
        else:
            dlist = time_list
        time_list, forward_rate = self.get_forward_rates(time_list, dtobjects)
        for no in range(len(dlist)):
            factor = 0.0
            for d in range(no, len(dlist) - 1):
                factor += ((dlist[d + 1] - dlist[d]) *
                           (0.5 * (forward_rate[d + 1] + forward_rate[d])))
            discount_factors.append(np.exp(-factor))
        return time_list, discount_factors


# Market environment class

class market_environment(object):
    ''' Class to model a market environment relevant for valuation.
    Attributes
    ==========
    name: string
        name of the market environment
    pricing_date : datetime object
        date of the market environment
    Methods
    =======
    add_constant :
        adds a constant (e.g. model parameter)
    get_constant :
        get a constant
    add_list :
        adds a list (e.g. underlyings)
    get_list :
        get a list
    add_curve :
        adds a market curve (e.g. yield curve)
    get_curve :
        get a market curve
    add_environment :
        adding and overwriting whole market environments
        with constants, lists and curves
    '''

    def __init__(self, name, pricing_date):
        self.name = name
        self.pricing_date = pricing_date
        self.constants = {}
        self.lists = {}
        self.curves = {}

    def add_constant(self, key, constant):
        self.constants[key] = constant

    def get_constant(self, key):
        return self.constants[key]

    def add_list(self, key, list_object):
        self.lists[key] = list_object

    def get_list(self, key):
        return self.lists[key]

    def add_curve(self, key, curve):
        self.curves[key] = curve

    def get_curve(self, key):
        return self.curves[key]

    def add_environment(self, env):
        for key in env.curves:
            self.curves[key] = env.curves[key]
        for key in env.lists:
            self.lists[key] = env.lists[key]
        for key in env.constants:
            self.constants[key] = env.constants[key]
=#