import numpy as np
from scipy.constants import h, k

# Resistance and inductance values from Shibo's note.
t_Al_Shibo = np.array([20, 30, 40, 50, 100, 200])*1e-9
Rs_Al_Shibo = np.array([1.33, .682, .362, .241, .0691, .0208])
Ls_Al_Shibo = np.array([1.42, .74, .4, .28, .11, .075])*1e-12 # H/sq

# Single resistance and inductance note from Andrew in Shibo's note.
t_Al_Andrew = 40e-9
Rs_Al_Andrew = .785
Ls_Al_Andrew = .54e-12

# Resistance and inductance values from TIM paper.
# https://doi.org/10.1007/s10909-022-02755-3
t_Al_tim = np.array([20, 30, 40])*1e-9
Rs_Al_tim = np.array([1.818, 0.848, 0.581])
Tc_tim = np.array([1.31, 1.29, 1.2])
Delta0_tim = 1.76*k*Tc_tim
Ls_Al_tim = h*Rs_Al_tim/(2*np.pi**2 * Delta0_tim)

def fit_Rs_t_Shibo():
    """
    Performs a linear fit to the log-log data
    of Rs (y-axis) vs. thickness (x-axis).
    
    Returns:
    poly: linear fit result from np.polyfit
    """
    log_t = np.log(t_Al_Shibo)
    log_Rs = np.log(Rs_Al_Shibo)
    poly = np.polyfit(log_t, log_Rs, deg=1)
    return poly
    

def get_Rs_from_t_Shibo(t):
    """
    Gets Rs value for a given thickness
    using Shibo's data.
    
    Parameters:
    t: Aluminum thickness in meters
    
    Returns:
    Rs: Sheet resistance in Ohms/sq
    """
    poly = fit_Rs_t_Shibo()
    Rs = np.exp(np.polyval(poly, np.log(t)))
    return Rs

def get_t_from_Rs_Shibo(Rs):
    """
    Gets the thickness value to get a desired Rs
    using Shibo's data.
    
    Parameters:
    Rs: Sheet resistance in Ohms/sq
    
    Returns:
    t: Aluminum thickness in meters
    """
    poly = fit_Rs_t_Shibo()
    t_Al_samp = np.geomspace(20e-9, 200e-9, 1000)
    log_Rs_fit = np.polyval(poly, np.log(t_Al_samp))
    t = np.interp(-np.log(Rs), -log_Rs_fit, t_Al_samp)
    return t

def get_Ls_from_Rs(Rs, Tc):
    """
    Computes sheet inductance using the BCS equation
    for superconducting gap energy, and the thin-film
    and T<<Tc limits for the sheet inductance.
    
    Parameters:
    Rs: Sheet resistance in Ohms/sq
    
    Returns:
    Ls: Sheet inductance in Henries/sq
    """
    Delta0 = 1.76*k*Tc
    hbar = h/(2*np.pi)
    Ls = Rs*hbar/(np.pi*Delta0)
    return Ls

# log_Ls = np.log(Ls_Al_Shibo)
# poly_Ls = np.polyfit(log_t, log_Ls, deg=2)
# log_Ls_fit = np.polyval(poly_Ls, np.log(t_Al_samp))
# Ls_fit = np.exp(log_Ls_fit)

# log_t_tim = np.log(t_Al_tim)
# log_Rs_tim = np.log(Rs_Al_tim)
# poly_tim = np.polyfit(log_t_tim, log_Rs_tim, deg=2)
# t_Al_samp = np.geomspace(20e-9, 200e-9, 1000)
# log_Rs_fit_tim = np.polyval(poly_tim, np.log(t_Al_samp))
# Rs_fit_tim = np.exp(log_Rs_fit_tim)
# Ls_fit_tim = Rs_fit_tim * h/(2*np.pi**2 * np.mean(Delta0_tim))

# fig, (ax0, ax1) = plt.subplots(figsize=(7,3), nrows=1, ncols=2)
# ax0.scatter(t_Al_Shibo*1e9, Rs_Al_Shibo, marker='s', label='Peter Day')
# # ax0.scatter(t_Al_Andrew*1e9, Rs_Al_Andrew, marker='^')
# ax0.scatter(t_Al_tim*1e9, Rs_Al_tim, marker='o', label='TIM 2021')
# ax0.plot(t_Al_samp*1e9, Rs_fit, 'k--')
# ax0.plot(t_Al_samp*1e9, Rs_fit_tim, color='k', ls='dotted')
# ax0.set(xlabel='Al thickness [nm]', ylabel='Rs [Ohm/sq]', xscale='log', yscale='log')
# ax0.grid()
# ax1.scatter(t_Al_Shibo*1e9, Ls_Al_Shibo*1e12, marker='s', label='Peter Day')
# # ax1.scatter(t_Al_Andrew*1e9, Ls_Al_Andrew*1e12, marker='^')
# ax1.scatter(t_Al_tim*1e9, Ls_Al_tim*1e12, marker='o', label='TIM 2021')
# ax1.plot(t_Al_samp*1e9, Ls_fit*1e12, 'k--')
# ax1.plot(t_Al_samp*1e9, Ls_fit_tim*1e12, color='k', ls='dotted')
# ax1.set(xlabel='Al thickness [nm]', ylabel='Ls [pH/sq]', xscale='log', yscale='log')
# ax1.grid()

# ax0.legend(loc='upper right')
# ax1.legend(loc='upper right')
# # ax0.set(xlim=[15, 50], ylim=[.2, 2.5])
# # ax1.set(xlim=[15, 50], ylim=[.2, 2.5])

# plt.tight_layout()
# plt.show()