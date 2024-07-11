import numpy as np
import skrf as rf
from kid233.model.transmission_line import *

def ABCD2S(A, B, C, D, Z0):
    '''
    Function to convert ABCD-matrix to S-matrix.
    
    Parameters: A, B, C, D, Z0.
    Returns: 2x2xf array S-matrix.
    '''
    S11 = (1.0*A+B/Z0-C*Z0-D)/(A+B/Z0+C*Z0+D)
    S12 = 2.0*(A*D-B*C)/(A+B/Z0+C*Z0+D)
    S21 = 2.0/(A+B/Z0+C*Z0+D)
    S22 = (-1.0*A+B/Z0-C*Z0+D)/(A+B/Z0+C*Z0+D)
    return np.array([[S11, S12],[S21, S22]])

def SpectralChannel3PortNetwork(freq, Z0, fres, Qc1, Qc2, Qloss):
    '''
    Creates an skrf Newtork object representing a 3 port spectral channel.
    Ports 1,2 = transmission line ports
    Port 3 = output line which will go to a detector

    Parameters:
        freq <array>: array of frequencies in Hz
        Z0 <float>: transmission line impedance in Ohm
        fres <float>: filter resonant frequency in Hz
        Qc1 <float>: transmission line to filter coupling quality factor
        Qc2 <float>: filter to output line coupling quality factor
        Qloss <float>: filter internal quality factor
    Returns:
        Ntwk <skrf.Network>: Network object for the filter
    '''
    x = (freq-fres)/fres
    Qr = 1/(1/Qc1 + 1/Qc2 + 1/Qloss)
    qi = 1/(1/Qloss+1/Qc2)
    s11 = -Qr/(Qc1*(1+2j*Qr*x))
    s21 = (Qr/qi+2j*Qr*x)/(1+2j*Qr*x)
    s31 = (2*Qc1*Qc2)**.5 / (Qc2+Qc1*(1+Qc2/Qloss))
    s31 /= 1+2j*Qr*x


    s31 = 1. - Q_r / (Qc1 * (1. + 2.j *Q_r * x))
    s11 = Q_r / (Qc1 * (1. + 2.j *Q_r * x))
    ZL = Z0 / (2. / s31 - 2.)
    s22 = (ZL - 0.5 * Z0) / (0.5 * Z0 + ZL)
    s12 = eff * np.sqrt(1. - np.conj(s31) * s31 - np.conj(s11) * s11)
            #I think the last term is necessary for power correspondance
            #and so I can treat it as a Z0 port when networking
    S = np.array([[s11, s12, s31], #now using symmetry/reciprocity to fill out
                [s12, s22, s12], #from calculated values
                [s31, s12, s11]])
    S = np.moveaxis(S, -1, 0)
    Band = rf.Frequency.from_f(freq, unit='Hz')
    Ntwk = rf.Network(frequency = Band, s = S, z0 = Z0)

    return Ntwk