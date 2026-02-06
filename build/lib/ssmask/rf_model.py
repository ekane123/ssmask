import numpy as np
import skrf as rf

from ssmask.constants import c

def ABCD2S(A, B, C, D, Z0):
    '''
    Function to convert ABCD-matrix to S-matrix.
    
    Arguments: A, B, C, D, Z0.
    Returns: 2x2xf array S-matrix.
     
    Notes: Transposition to fxnxn format is done externally.
    '''
    S11 = (A+B/Z0-C*Z0-D)/(A+B/Z0+C*Z0+D)
    S12 = 2.0*(A*D-B*C)/(A+B/Z0+C*Z0+D)
    S21 = 2.0/(A+B/Z0+C*Z0+D)
    S22 = (-A+B/Z0-C*Z0+D)/(A+B/Z0+C*Z0+D)
    return np.array([[S11, S12],[S21, S22]])

def SpectralChannel3PortNetwork(Band, Z0, fres, Qc1, Qc2, Qloss):
    '''
    Creates a 3-port network representing a mm-wave filter.
    Port1 = feedline, Port2 = filter output to detector, Port3 = other side of feedline.
    The expressions for the S-parameters are from doi:10.1109/TTHZ.2021.3095429
    
    Parameters:
        Band: skrf.Frequency object containing the frequencies at which S-params are calculated
        Z0 <float>: Port impedance for all 3 ports
        fres <float>: filter center frequency in Hz
        Qc1 <float>: coupling quality factor between feedline and filter
        Qc2 <float>: coupling quality factor between filter and output to detector
        Qloss <float>: resonator loss quality factor
    Returns:
        Ntwk: skrf.Network object representing the filter
    '''
    x = (Band.f-fres)/fres
    Qr = 1/(1/Qc1 + 1/Qc2 + 1/Qloss)
    denom = 1 + 2j*Qr*x
    s11_0 = -Qr/Qc1
    s21_0 = (2*Qc1*Qc2)**.5/(Qc2 + Qc1*(1+Qc2/Qloss))
    s11 = s11_0/denom
    s31 = 1 + s11
    s21 = s21_0/denom
    # calculate reflection coefficient at detector port (2).
    # eta21 = fraction of transmitted power at port 2 which
    # gets coupled to the feedline rather than dissipated in the resonator.
    eta21 = Qloss/(Qc1+Qloss)
    # note that this method of calculating S22 only gives us the magnitude
    # of the reflection and not the phase.
    s22 = np.sqrt(1-2*s21*np.conj(s21)/eta21)
    # Use the symmetry of ports 1 and 3 and the reciprocity of the network
    # (i.e., symmetric S-matrix) to fill out the remaining entries.
    S = np.array([
        [s11, s21, s31],
        [s21, s22, s21],
        [s31, s21, s11]
    ])
    S = np.moveaxis(S, -1, 0)
    Ntwk = rf.Network(frequency = Band, s = S, z0 = Z0)
    return Ntwk

def TransmissionLineLossy(Band, length, Z0, epsr, lossTan=0):   
    '''
    Creates a network object representing a lossy transmission line.
    Note: we assume that loss is dominated by dielectric loss
    
    Parameters:
        Band: skrf.Frequency object containing the frequencies at which S-params are calculated
        length <float>: physical length of transmission line
        Z0 <float>: characteristic impedance
        epsr <float>: relative permittivity
        lossTan <float>: dielectric loss tangent
    Returns:
        Ntwk: skrf.Network object representing the transmission line
    '''
    beta = 2.0*np.pi*np.sqrt(epsr)*Band.f/c # real propagation constant
    alpha = np.pi*np.sqrt(epsr)*Band.f/c*lossTan # attenuation constant in Np/m
    gamma = alpha + 1j*beta # complex propagation constant (with attenuation)
    
    # construct ABCD matrix of the lossy line
    A = np.cosh(gamma*length)
    B = Z0*np.sinh(gamma*length)
    C = 1/Z0*np.sinh(gamma*length)
    D = np.cosh(gamma*length)
    
    S = ABCD2S(A, B, C, D, Z0) # convert lossy line ABCD to S-parameters
    S = np.moveaxis(S, -1, 0)
    Ntwk = rf.Network(frequency=Band, s=S, z0=Z0)
    return Ntwk

def FilterBankLossy(Band, fres, Qc1, Qc2, Qloss, Z0, physSep, epsr, lossTan=0):
    '''
    Creates a filterbank represented by a cascaded series of spectral channels and lossy transmission lines.
    
    Parameters:
        Band: skrf.Frequency object containing the frequencies at which S-params are calculated
        fres <np.array>: array of filter resonant frequencies in Hz
        Qc1 <np.array>: array of coupling quality factors between the feedline and each filter
        Qc2 <np.array>: array of coupling quality factors between each filter and its output line
        Qloss <np.array>: array of resonator loss quality factors
        Z0 <float>: impedance of feedline and output lines
        physSep <float>: number of wavelengths separation between successive filters
        epsr <float>: relative permittivity
        lossTan <float>: dielectric loss tangent for transmission line sections connecting the filters
    Returns:
        Ntwk: skrf.Network object representing the filterbank
    '''    
    # initialize current network to the first spectral channel
    CurrentNtwk = SpectralChannel3PortNetwork(Band, Z0, fres[0], Qc1[0], Qc2[0], Qloss[0])
        
    # loop to create filter bank with arbitrary # of channels and create network
    for i in range(len(fres)):
        if i < len(fres)-1:
            # resonant frequencies and quality factors for current and next SCs
            fres_current = fres[i]    
            fres_nxt, Qc1_nxt, Qc2_nxt, Qloss_nxt = fres[i+1], Qc1[i+1], Qc2[i+1], Qloss[i+1]
            
            # create Network object for next SC
            NextSC = SpectralChannel3PortNetwork(Band, Z0, fres_nxt, Qc1_nxt, Qc2_nxt, Qloss_nxt)
            
            # create interconnecting transmission line
            lambda_current = c/fres_current
            lineLength = physSep*lambda_current
            TLine = TransmissionLineLossy(Band, lineLength, Z0, epsr, lossTan)
            
            # connect current network to the transmission line
            N = CurrentNtwk.nports
            InterNtwk = rf.connect(CurrentNtwk, N-1, TLine, 0)
            
            # connect current network to the next SC
            N = InterNtwk.nports
            CurrentNtwk = rf.connect(InterNtwk, N-1, NextSC, 0)             
    
    return CurrentNtwk