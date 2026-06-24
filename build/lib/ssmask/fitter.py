from .fit_funcs import lorentzian
from scipy.optimize import curve_fit

def fit_lorentzian(x, y, p0):
    popt, pcov = curve_fit(lorentzian, x, y, p0)
    return popt, pcov