import numpy as np
from scipy import constants
from scipy.integrate import cumulative_trapezoid
from scipy.special import ellipk
from scipy.optimize import fsolve

eps0 = constants.epsilon_0
Z0vac = (constants.mu_0/eps0)**.5
c = constants.c
h = constants.h
hbar = h / (2*np.pi)
k_B = constants.k


###########################
### M I C R O S T R I P ###
###########################

def get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq):
    '''
    Calculates transmission line parameters of a microstrip line, with the option
    to include a higher-permittivity material in place of the vacuum (i.e., the superstrate).
    All formulae are from "Accurate models for microstrip computer-aided design",
    doi: 10.1109/MWSYM.1980.1124303

    Note that it will return complex values if eps_sup > eps_sub, but the real parts
    still seem to agree well with Sonnet simulations.
    Z0_tot will generally have a complex part anyway if Rsq > 0.

    Parameters:
        eps_sub <float>: relative permittivity of substrate
        eps_sup <float>: relative permittivity of superstrate
        w_strip <float>: width of center line in m
        h_sub <float>: height of substrate in m
        Rsq <float>: sheet resistance of center line in Ohms/sq
        freq <float>: frequency in Hz
    Returns:
        eps_eff <float>: effective dielectric constant of approximate TEM mode
        L <float>: inductance in H/m
        C <float>: capacitance in F/m
        Z0_tot <float>: impedance of approximate TEM mode
    '''

    # effective permittivity of approximate TEM mode
    eps_rel = eps_sub/eps_sup 
    u = w_strip / h_sub
    a = 1 + 1/49*np.log((u**4+(u/52)**2)/(u**4+.432)) + 1/18.7*np.log(1+(u/18.1)**3)
    b = 0.564*((eps_rel-0.9)/(eps_rel+3))**.053
    eps_eff = (eps_rel+1)/2 + (eps_rel-1)/2 * (1+10/u)**(-a*b)
    eps_eff *= eps_sup

    # impedance of approximate TEM mode
    f = 6 + (2*np.pi-6)*np.exp(-(30.666/u)**.7528)
    Z0 = Z0vac/(2*np.pi*np.sqrt(eps_eff)) * np.log(f/u + np.sqrt(1+(2/u)**2))
    
    # inductance and capacitance per length
    vph = c/np.sqrt(eps_eff)
    L = Z0/vph
    C = 1/(Z0*vph)
    
    # impedance after taking into account sheet resistance
    R = Rsq/w_strip
    omega = 2*np.pi*freq
    Z0_tot = np.sqrt((R+1j*omega*L)/(1j*omega*C))
    
    return eps_eff, L, C, Z0_tot

def get_sc_mstrip_params_thinfilm(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq, Tc):
    '''
    Same as get_mstrip_params(), with the addition of kinetic inductance.
    Assumes a thin superconducting film (i.e. w_strip << penetration depth).
    Kinetic inductance is calculated using eq. (47) of JZ12 (10.1146/annurev-conmatphys-020911-125022)

    Parameters:
        see get_mstrip_params()
        Tc <float>: superconducting critical temperature in K
    Returns:
        see get_mstrip_params()
        Lkin <float>: kinetic inductance in H/m
    '''
    eps_eff, L, C, Z0_tot = get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, 0, freq)
    Delta0 = 1.76*k_B*Tc
    Lkin = hbar*Rsq/(np.pi*Delta0*w_strip)
    Ltot = L + Lkin
    Z0_tot = np.sqrt(Ltot/C)
    
    return eps_eff, L, Lkin, C, Z0_tot

def get_sc_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Lsq, freq):
    '''
    Same as get_mstrip_params(), with the addition of kinetic inductance.
    Does not assume a thin-film geometry.

    Parameters:
        see get_mstrip_params()
        Lsq <float>: kinetic inductance in H/sq
        Tc <float>: superconducting critical temperature in K
    Returns:
        see get_mstrip_params()
        Lkin <float>: kinetic inductance in H/m
    '''
    eps_eff, L, C, Z0_tot = get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, 0, freq)
    Lkin = Lsq/w_strip
    Ltot = L + Lkin
    Z0_tot = np.sqrt(Ltot/C)
    
    return eps_eff, L, Lkin, C, Z0_tot

def get_voltage_along_tapered_mstrip(x, eps_sub, eps_sup, w_strip, h_sub, Rsq, freq, do_back_prop=False):
    """
    Esimates the voltage along a microstrip line of varying width, 
    using the small reflections model (Pozar, Microwave Engineering, Chapter 5.8).
    Voltage is normalized to a value of 1 at x = 0.
    
    The effect of resistive loss is treated using a first-order approximation
    of the complex propagation coefficient: exp(-gamma * x) ~= 1 - gamma * dx.
    Thus, the voltage change due to lossless phase propagation + resistive loss is
        dV = -V(x) * gamma * dx
    at each discrete step of length dx along the microstrip line.
        
    The effect of reflections is treated in a similar way. At each step, 
    the differential reflection coefficient (Pozar 5.65) is
        dGamma = dZ / 2Z.
    Thus, the forward "loss" due to the differential reflected voltage is
        dV = V(x) * (dZ / 2Z) * dx.
    
    In the small reflections model, we assume that only the first reflection is important.
    Thus, we propagate the reflected voltage wave backwards along the entire length of line
    that was already traversed before the reflection. This can be enabled or disabled
    using the "do_back_prop" parameter.
    Note that the small reflections model becomes less accurate the more sharp the 
    changes in impedance per length.
    
    Parameters:
    x (array-like): Position array along the line, in meters.
    do_back_prop (bool): If True, compute the voltage of each differential reflected wave
        along the reverse length of line that had been traversed before the reflection.
    All others: See get_mstrip_params. Lengths are in meters, frequency in Hz.
        w_strip should be (array-like) with the same length as x.
        All other parameters should be scalars.
        
    Returns:
    V (array_like): voltage along the microstrip line. Length is 1 less than len(x).
    """
    # Resistance per length along the line
    R = Rsq/w_strip # Ohm m-1
    
    # Calculate impedances (Z0) and propagation coefficients (gamma).
    eps, L, C, Z0 = get_mstrip_params(eps_sub, eps_sup, w_strip, h_sub, Rsq, freq)
    omega = 2*np.pi*freq
    gamma = ((R + 1j*omega*L)*1j*omega*C)**.5
    gamma_prop = gamma[:-1]
    
    # Calculate the differential reflection coefficients.
    dZ0 = np.diff(Z0)
    dGamma = - 1/(2*Z0[:-1]) * dZ0
    
    # Propagate the voltage (and optional reflections) along the line.
    V = np.zeros(gamma_prop.shape, dtype='complex128')
    V[0] = 1
    dx = np.diff(x)
    for iix in range(1, V.shape[0]):
        V0 = V[iix-1]
        # Differential voltage change due to reflection
        Vrefl = V0 * dGamma[iix]
        # Differential voltage change due to normal propagation + resistive loss
        Vprop = V0 * gamma_prop[iix]*dx[iix]
        # The voltage at this point (x+dx) is the last voltage plus the voltage changes
        V[iix] = V0 - Vprop - Vrefl
        
        if do_back_prop:
            # Add the reflected voltages, times the phase + amplitude change during normal propagation
            # backwards down the line, to the voltages backwards down the line.
            gamma_back_prop = np.flip(gamma_prop[:iix])
            xback = np.flip(x[iix-1] - x[:iix])
            phase = cumulative_trapezoid(gamma_back_prop, xback)
            phase = np.flip(phase)
            V[1:iix] += Vrefl * np.exp(1j*phase)
            
    return V
        

##############
### C P W ####
##############
        
def get_cpw_params(eps_sub, eps_sup, center_width, gap_width, ground_height, Rsq, freq):
    '''
    Calculates transmission line parameters of a CPW line, with the option
    to include a higher-permittivity material in place of the vacuum (i.e., the superstrate).
    All formulae are from "Transmission Line Design Handbook" by Brian C. Wadell, page 79.

    Parameters:
        eps_sub <float>: relative permittivity of substrate
        eps_sup <float>: relative permittivity of superstrate
        center_width <float>: width of center line in m
        gap_width <float>: width of gap between center line and the ground planes
        ground_height <float>: height of substrate between the CPW and a ground plane beneath all the CPW top layer
        Rsq <float>: sheet resistance of center line in Ohms/sq
        freq <float>: frequency in Hz
    Returns:
        eps_eff <float>: effective dielectric constant of TEM mode
        L <float>: inductance in H/m
        C <float>: capacitance in F/m
        Z0_tot <float>: impedance of approximate TEM mode
    '''
    eps_rel = eps_sub/eps_sup
    a = center_width
    b = a + 2*gap_width
    k = a/b
    kprime = (1-k**2)**.5
    k1 = np.tanh(np.pi*a/(4*ground_height)) / np.tanh(np.pi*b/(4*ground_height))
    k1prime = (1-k1**2)**.5
    K = ellipk(k)
    Kprime = ellipk(kprime)
    K1 = ellipk(k1)
    K1prime = ellipk(k1prime)
    
    # effective relative permittivity
    eps_eff = (1 + eps_rel*Kprime*K1/(K*K1prime)) / (1 + Kprime*K1/(K*K1prime))
    eps_eff *= eps_sup
    
    # Wave impedance
    Z0 = 60*np.pi / (eps_eff**.5 * (K/Kprime + K1/K1prime))
    
    # inductance and capacitance per length
    vph = c/np.sqrt(eps_eff)
    L = Z0/vph
    C = 1/(Z0*vph)
    
    # impedance after taking into account sheet resistance
    R = Rsq/center_width
    omega = 2*np.pi*freq
    Z0_tot = np.sqrt((R+1j*omega*L)/(1j*omega*C))
    
    return eps_eff, L, C, Z0_tot


###########################
### C A P A C I T O R S ###
###########################

def get_IDC_capacitance(W, S, eps_eff, l):
    """
    Get the capacitance of an IDC.
    Using equations 7.7-7.9 of Inder Bahl - Lumped Elements for RF and Microwave Circuits.
    
    Parameters:
    W: Finger width of side 1
    S: Finger width of side 2
    eps_eff: Effective dielectric constant of microstrip of width W
    l: Finger length in microns
    
    Returns:
    C: Capacitance in Farads per number of finger pairs N. 
        Multiply by (N-1) to get capacitance in F.
    """
    a = W/2
    b = (W+S)/2
    k = np.tan(a*np.pi/(4*b))**2
    kprime = (1-k**2)**.5
    
    K = ellipk(k)
    Kprime = ellipk(kprime)
    K_ratio = K/Kprime
                
    C = 1e-15 / (18*np.pi) * eps_eff * K_ratio * l
    return C

def solve_for_capacitances(f0, Qc, L, Z0, guess=(1e-11, 1e-13)):
    """
    Solves for the KID capacitance C and the effective coupling 
    capacitance Ceff needed to achieve a desired resonant frequency f0
    coupling quality factor Qc, using the following equations
    (First one is Equation 2.45 in Pete Barry's thesis).
    
    Qc = 2 * C / (omega0 * Z0 * Ceff**2)
    
    omega0 = 1 / (L * (C + Ceff))**0.5
    
    Ceff = Cc * Cg / (Cc + Cg)
    
    Cc is the shunt capacitance between the KID and the readout feedline,
    and Cg is the shunt capacitance between the KID and ground.
    
    Parameters:
    f0: Desired readout frequency in Hz.
    Qc: Desired coupling quality factor.
    L: Total inductance of the KID in Henries.
    Z0: Impedance of the readout microstrip in Ohms.
    guess (array-like, length=2): Guesses for C and Ceff.
    
    Returns:
    C: KID capacitance in Farads.
    Ceff: Effective coupling capacitance in Farads.
    """
    omega0 = f0*2*np.pi
    
    def equations(p):
        C, Ceff = p
        return [
            omega0 - 1/(L*(C+Ceff))**.5,
            Qc - 8*C/(omega0*Z0*Ceff**2)
        ]
    
    C, Ceff =  fsolve(equations, guess)
    return C, Ceff
    
    