import numpy as np

def lorentzian(f, f0, A, w):
    x = 2*(f-f0)/w
    return A/(1 + x**2)