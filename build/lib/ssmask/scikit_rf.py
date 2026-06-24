import numpy as np
import skrf as rf
from scipy.constants import c
from tqdm.auto import tqdm

def ABCD2S(A, B, C, D, Z0):
    '''
    Adapted from the Ph.D. thesis of George Che, 2018.
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

def get_2port_shunt_Sparams(Z1, Z3, Y2):
    '''
    Gets S-parameters for a 2-port network with the following geometry:
    -----------------
            |
   Z1      Y2      Z3
            |
    -----------------
    
    Parameters:
        Z1: First port impedance
        Z3: Second port impedance
        Y2: Shunt admittance.
    
    Returns:
        S = ((S11, S13), (S31, S33))
    '''
    Z2 = 1/Y2
    Z = Z2*np.ones((2,2))
    F = np.array([[np.real(Z1)**-.5, 0], [0, np.real(Z3)**-.5]])
    Finv = np.array([[np.real(Z1)**.5, 0], [0, np.real(Z3)**.5]])
    Z0 = np.array([[Z1, 0], [0, Z3]])
    S = np.matmul(np.linalg.inv(Z+Z0), Finv)
    S = np.matmul(Z-np.conj(Z0), S)
    S = np.matmul(F, S)
    return S


def get_3PortJunction_Sparams(Z0, Z1):
    '''
    Gets the S-parameters for 3 transmission lines joined at a point.
    Ports 1 and 2 have impedance Z0, and port 3 has impedance Z1.
    
    Parameters:
        Band: skrf.Frequency object containing the frequencies at which S-params are calculated
        Z0: Impedance of ports 1 and 3.
        Z1: Impedance of port 2.
    
    Returns:
        S: S-parameters
    '''
    S = np.zeros((3,3), dtype=complex)
    # Calculate S-parameters between ports 1 and 2
    thisS = get_2port_shunt_Sparams(Z0, Z1, 1/Z0)
    S[0:2, 0:2] = thisS
    # calculate S-parameters between ports 1,2 and 3
    thisS = get_2port_shunt_Sparams(Z0, Z0, 1/Z1)
    S[0,2] = thisS[0,1]
    S[2,0] = thisS[1,0]
    S[1,2] = S[0,2]
    S[2,1] = S[2,0]
    S[2,2] = thisS[1,1]
    
    return S

def SpectralChannel3PortNetwork_ExtraLine(
    Band, length, extra_length, Z0, Z1, C, epsr, lossTan=0
    ):
    """
    Create a Network representing a 3-port resonant filter.
    All ports and transmission lines have impedance Z0.
    The Network geometry is as follows:
    
     port 2
    |      |
    |      |    __________________    | C |    ____________    | C |
    o----------|__ extra_length __|---|   |---|__ length __|---|   |---
    |      |                          |   |                    |   |  port 1
    |      |
    |      o-----------------------------------------------------------
    |      |
    |      |
     port 0
     
    The transmission line between the two capacitors is a half-wavelength
    resonator. In this model, we replace it with an equivalent RLC shunt 
    to ground, which is approximately correct near its resonant frequency.
     
    Parameters:
    Band (skrf.Frequency): frequencies to simulate the Network at, in Hz.
    length: length of resonant half-wave filter in meters.
    extra_length: length of extra transmission line in meters.
    Z0: impedance of all ports and transmission lines in Ohms.
    C: coupling capacitance.
    epsr: relative permittivity of the transmission line.
    lossTan: dielectric loss tangent of the transmission line. lossTan = 1/Qloss.
    
    Returns:
    Ntwk: skrf.Network object representing the 3-port network.
    """
    
    alpha = np.pi*np.sqrt(epsr)*Band.f/c*lossTan # attenuation constant in Np/m    
    Qloss = 1/lossTan # Loss Q of the resonator
    wavelen = 2*length
    f0 = c / (wavelen * epsr**.5)
    R = Z0/(alpha * length) # Shunt resistance of the equivalent resonator

    x0 = (Band.f-f0)/f0
    omega = 2*np.pi*Band.f
    
    ### Create a 2-port Network for the extra length of transmission line.
    Ntwk_extra_TL = TransmissionLineLossy(Band, extra_length, Z1, epsr, lossTan)
    
    ### Create a 2-port Network for the capacitor-coupled shunt resonance.
    Yr = (1 + 2j*Qloss*x0)/R
    Zc = 1/(1j*omega*C)
    A = 1 + Zc*Yr
    B = 2*Zc + Zc**2 * Yr
    C = Yr
    D = 1 + Zc*Yr
    S = ABCD2S(A, B, C, D, Z0)
    S = np.moveaxis(S, -1, 0)
    Ntwk_res = rf.Network(frequency=Band, s=S, z0=Z0)
    
    ### Create a 3-port Network for the junction.
    S = get_3PortJunction_Sparams(Z0, Z0)
    S = np.array([S for _ in x0])
    Ntwk_junction = rf.Network(frequency=Band, s=S, z0=Z0)
    
    ### Join the Networks together.
    Ntwk = rf.network.connect(Ntwk_extra_TL, 1, Ntwk_res, 0)
    Ntwk = rf.network.connect(Ntwk_junction, 1, Ntwk, 0)
    
    return Ntwk
    
def FilterbankLossy_ExtraLine(
    Band, lengths, extra_length, Z0, Z1, Cs, 
    epsr, physSep, lossTan=0, verbose=False
    ):
    """
    Creates a filterbank represented by a cascaded series of 
    spectral channels and lossy transmission lines.
    """
    v = c/epsr**.5
    
    # initialize current network to the first spectral channel
    CurrentNtwk = SpectralChannel3PortNetwork_ExtraLine(
        Band, lengths[0], extra_length, Z0, Z1, Cs[0], epsr, lossTan
    )
    # loop to create filter bank with arbitrary # of channels and create network
    pbar = range(len(lengths)-1)
    if verbose:
        pbar = tqdm(pbar, leave=False)
    for i in pbar:
        length_current = lengths[i]    
        length_nxt, C_nxt = lengths[i+1], Cs[i+1]
        
        # create Network object for next SC
        NextSC = SpectralChannel3PortNetwork_ExtraLine(
            Band, length_nxt, extra_length, Z0, Z1, C_nxt, epsr, lossTan
        )
        # create interconnecting transmission line
        lambda_current = length_current * 2
        lineLength = physSep*lambda_current
        TLine = TransmissionLineLossy(Band, lineLength, Z0, epsr, lossTan)
        
        # connect current network to the transmission line
        N = CurrentNtwk.nports
        InterNtwk = rf.network.connect(CurrentNtwk, N-1, TLine, 0)
        
        # connect current network to the next SC
        N = InterNtwk.nports
        CurrentNtwk = rf.network.connect(InterNtwk, N-1, NextSC, 0)           
    
    return CurrentNtwk


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

def FilterBankLossy(Band, fres, Qc1, Qc2, Qloss, Z0, physSep, epsr, lossTan=0, verbose=False):
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
    v = c/epsr**.5
    # initialize current network to the first spectral channel
    CurrentNtwk = SpectralChannel3PortNetwork(Band, Z0, fres[0], Qc1[0], Qc2[0], Qloss[0])
    
    # loop to create filter bank with arbitrary # of channels and create network
    pbar = range(len(fres)-1)
    if verbose:
        pbar = tqdm(pbar, leave=False)
    for i in pbar:
        # resonant frequencies and quality factors for current and next SCs
        fres_current = fres[i]    
        fres_nxt, Qc1_nxt, Qc2_nxt, Qloss_nxt = fres[i+1], Qc1[i+1], Qc2[i+1], Qloss[i+1]
        
        # create Network object for next SC
        NextSC = SpectralChannel3PortNetwork(Band, Z0, fres_nxt, Qc1_nxt, Qc2_nxt, Qloss_nxt)
        
        # create interconnecting transmission line
        lambda_current = v/fres_current
        lineLength = physSep*lambda_current
        TLine = TransmissionLineLossy(Band, lineLength, Z0, epsr, lossTan)
        
        # connect current network to the transmission line
        N = CurrentNtwk.nports
        InterNtwk = rf.network.connect(CurrentNtwk, N-1, TLine, 0)
        
        # connect current network to the next SC
        N = InterNtwk.nports
        CurrentNtwk = rf.network.connect(InterNtwk, N-1, NextSC, 0)           
    
    return CurrentNtwk

######################################################################
######################################################################
######################################################################

# Functions below here are adapted directly from George Che's thesis,
# with minor changes to clean up the code a bit.
# George's approach to generating a spectral channel is to
# generate a 3-port S-matrix with equal port impedances Z0,
# then renormalize the port 2 impedance to the impedance of the resonant filter.
# I found that this is at least 10x slower than my approach, probably 
# due to the step which renormalizes the port impedances.
# My approach does not give the correct magnitude of power through each filter,
# but it is off by a very nearly constant factor which depends on Qc1, Qc2, Qloss.

def TransmissionLineLossy_gche(Band, length, Z0, epsr, lossTan=0, nPorts=2):   
    '''
    Adapted from the Ph.D. thesis of George Che, 2018.
    Creates a network object representing a lossy transmission line.
    Note: we assume that loss is dominated by dielectric loss
    '''
    beta = 2.0*np.pi*np.sqrt(epsr)*Band.f/c # real propagation constant
    alpha = np.pi*np.sqrt(epsr)*Band.f/c*lossTan # attenuation constant in Np/m
    gamma = alpha + 1j*beta # complex propagation constant (with attenuation)
    
    # construct ABCD matrix of the lossy line
    A = np.cosh(gamma*length)
    B = Z0*np.sinh(gamma*length)
    C = 1/Z0*np.sinh(gamma*length)
    D = np.cosh(gamma*length)
    
    # convert lossy line ABCD to S-parameters
    S_2port = ABCD2S(A, B, C, D, Z0)
    S11 = S_2port[0, 0]
    S12 = S_2port[0, 1]
    S21 = S_2port[1, 0]
    S22 = S_2port[1, 1]
    
    if nPorts == 2:
        S_2port = np.array([[S11, S12], [S21, S22]]).transpose(2, 0, 1)
        return rf.Network(frequency=Band, s=S_2port, z0=Z0)
    elif nPorts == 3:
        S13 = np.sqrt(1 - np.conj(S11)*S11 - np.conj(S12)*S12)
        S31 = S13
        theta23 = np.pi/2 - beta*length
        S23 = np.abs(S13) * np.exp(1j*theta23)
        S32 = S23
        S33 = np.zeros(np.size(Band.f))

        S_3port = np.array([
            [S11, S12, S13],
            [S21, S22, S23],
            [S31, S32, S33]
        ]).transpose(2, 0, 1)
        
        return rf.Network(frequency=Band, s=S_3port, z0=Z0)

def SpectralChannelLossy_gche(Band, Z0, fres, Qc, Qdet, Qloss, 
                         approach=1):
    '''
    Adapted from the Ph.D. thesis of George Che, 2018.
    Function to create a network object representing a spectral channel
    as either a 2-port or 3-port network. Incorporates lossy dielectric
    as Qloss.

    Arguments:
        Band      : frequency band
        Z0        : characteristic impedance
        fres      : resonant frequency
        Qc        : coupling Q
        Qdet      : internal Q
        Qloss     : loss Q
        approach  :
            "1" for 3-port representation all referenced to Z0
                and terminated in ZL
            "2" for 3-port representation with port 2
                referenced to ZL
            "3" for 2-port representation

    Returns:
        Spectral channel network object
    '''
    x = (Band.f - fres) / fres

    # shunt impedance of the entire resonator
    ZL = Z0 / 2 * Qc * (1 / Qdet + 1 / Qloss) + 1j * Z0 * Qc * x

    # 3-port network with all ports referenced to 50 Ohm;
    # terminate port 2 with ZL
    if approach == 1:
        S_3port = np.ones((np.size(x), 3, 3))

        # create 3-port S-matrix for the 50 Ohm matched
        # network
        S_3port[:, 0, 0] = -1.0 / 3
        S_3port[:, 0, 1] = 2.0 / 3
        S_3port[:, 0, 2] = 2.0 / 3

        S_3port[:, 1, 0] = 2.0 / 3
        S_3port[:, 1, 1] = -1.0 / 3
        S_3port[:, 1, 2] = 2.0 / 3

        S_3port[:, 2, 0] = 2.0 / 3
        S_3port[:, 2, 1] = 2.0 / 3
        S_3port[:, 2, 2] = -1.0 / 3

        # create a network object for the 50 Ohm matched
        # network
        MatchedNtwrk = rf.Network(frequency=Band, \
                                  s=S_3port, z0=Z0)

        # create 1-port S-matrix for the resonator load
        S11_L = (ZL - Z0) / (ZL + Z0)

        Resonator = rf.Network(frequency=Band, \
                               s=S11_L, z0=Z0)

        Ntwk = rf.network.connect(MatchedNtwrk, 2, Resonator, 0)
        return Ntwk
    
    # 3-port network with all ports referenced to 50 Ohm;
    # renormalize so port 2 is referenced to ZL
    elif approach == 2:
        S_3port = np.ones((np.size(x), 3, 3))

        # create 3-port S-matrix for the 50 Ohm matched
        # network
        S_3port[:, 0, 0] = -1.0 / 3
        S_3port[:, 0, 1] = 2.0 / 3
        S_3port[:, 0, 2] = 2.0 / 3

        S_3port[:, 1, 0] = 2.0 / 3
        S_3port[:, 1, 1] = -1.0 / 3
        S_3port[:, 1, 2] = 2.0 / 3

        S_3port[:, 2, 0] = 2.0 / 3
        S_3port[:, 2, 1] = 2.0 / 3
        S_3port[:, 2, 2] = -1.0 / 3

        # create a network object for the 50 Ohm matched
        # network
        Ntwk = rf.Network(frequency=Band, s=S_3port, \
                        z0=Z0)

        # create port reference impedance matrix
        Zmatrix = np.empty([len(Band.f), 3], dtype=complex)
        Zmatrix[:, 2] = Z0
        Zmatrix[:, 0] = Z0
        Zmatrix[:, 1] = ZL

        Ntwk.renormalize(Zmatrix, s_def='power')
        return Ntwk

    elif approach == 3:
        # create 2-port S-matrix for the network
        S11 = Z0 / (2 * ZL + Z0)
        S22 = S11
        S12 = 2 * ZL / (2 * ZL + Z0)
        S21 = S12

        S_2port = np.array([
            [S11, S12],
            [S21, S22]
        ]).transpose(2, 0, 1)

        return rf.Network(frequency=Band, s=S_2port, \
                        z0=Z0)

def FilterBankLossy_gche(Band, Data, Z0=50.0, physSep=0.25,
        epsr=11.7, lossTan=0.0, approach=3, verbose=True):
    '''
    Adapted from the Ph.D. thesis of George Che, 2018.
    
    Data = [fres, Qc1, Qc2, Qloss]
    '''
    v = c/epsr**.5
    # initialize current network to the first spectral
    # channel
    CurrentNtwk = SpectralChannelLossy_gche(Band, Z0, Data[0,0],
        Data[1,0], Data[2,0], Data[3,0], approach)
    
    # loop to create filter bank with arbitrary # of
    # channels and create network
    pbar = np.arange(np.shape(Data)[1])
    if verbose:
        pbar = tqdm(pbar, leave=False)
    for i in pbar:

        if i < np.shape(Data)[1]-1:
            # resonant frequencies and quality factors for
            # current and next SCs
            fres_current = Data[0,i]
            fres_nxt = Data[0,i+1]; Qc_nxt = Data[1,i+1]
            Qdet_nxt = Data[2,i+1]; Qloss_nxt = Data[3,i+1]

            # create Network object for next SC
            NextSC = SpectralChannelLossy_gche(Band, Z0,
                    fres_nxt, Qc_nxt, Qdet_nxt, Qloss_nxt,
                            approach)

            # create interconnecting transmission line
            lambda_current = v/fres_current
            # lambda_nxt = v/fres_nxt
            lineLength = physSep*lambda_current
            # lineLength = physSep*(lambda_current+
            #     lambda_nxt)/2.0
            TLine = TransmissionLineLossy_gche(Band,
                    lineLength, Z0, epsr, lossTan)

            # connect current network to the transmission
            # line
            N = CurrentNtwk.nports
            InterNtwk = rf.network.connect(CurrentNtwk, N-1,
                    TLine, 0)

            # connect current network to the next SC
            N = InterNtwk.nports
            CurrentNtwk = rf.network.connect(InterNtwk, N-1,
                    NextSC, 0)
    
    return CurrentNtwk


"""
def FilterBankLossy_partial_gche(Band, fres, Qc1, Qc2, Qloss, Pcutoff=1e-2, Z0=50.0, physSep=0.25,
                            epsr=11.7, lossTan=0.0, verbose=True):
    '''
    Creates a filterbank represented by a cascaded series of spectral channels and lossy transmission lines.
    Only connects spectral channels to each other whose profiles overlap.
    Assumes the same Qc1, Qc2, and Qloss for all channels.
    
    Parameters:
        Band: skrf.Frequency object containing the frequencies at which S-params are calculated
        Pcutoff <float>: Cutoff power (normalized to a maximum of 1) for interlinking channels.
        fres: Array of filter resonant frequencies.
        Qc1, Qc2, Qloss: Input coupling, output coupling, and loss quality factors.
        Z0 <float>: impedance of feedline
        physSep <float>: number of wavelengths separation between successive filters
        epsr <float>: relative permittivity
        lossTan <float>: dielectric loss tangent
    Returns:
        Ntwks: list of skrf.Network objects representing the filterbank
        ixs_in_ntwks: index of each channel in each list
    '''
    Qtot = 1/(1/Qc1 + 1/Qc2 + 1/Qloss)
    v = c/epsr**.5
    
    # Group spectral channels together that are sufficiently overlapped.
    # One group per spectral channel.
    # The higher the value of Pcutoff, the smaller the groups will be.
    ix_groups = []
    ixs_in_ntwks = np.zeros(len(fres), dtype=int)
    Qtot = 1/(1/Qc1 + 1/Qc2 + 1/Qloss)
    dx = (1/Pcutoff - 1)**.5 / (2*Qtot)
    for ii in range(len(fres)):
        dxs = abs(fres - fres[ii])/fres[ii]
        ixs = np.where(dxs<dx)[0]
        ix_groups.append(ixs)
        ixs_in_ntwks[ii] = np.where(dxs[ixs]==0)[0][0]
    
    Ntwks = []
    
    pbar0 = range(len(fres))
    if verbose:
        pbar0 = tqdm(pbar0, leave=False)
    
    # ii0 = index for Network groups
    for ii0 in pbar0:
        ixs = ix_groups[ii0]
        ii1 = 0
            
        # ii1 = index through spectral channels within a Network group
        for ii1 in range(len(ixs)-1):
            ix = ixs[ii1]
            
            if ii1 == 0:
                if ix == 0:
                    # If this spectral channel happens to be the first one in the filterbank,
                    # just initialize the spectral channel with no transmission line preceding it.
                    Ntwk = SpectralChannelLossy(Band, Z0, fres[ix], Qc1[ix], Qc2[ix], Qloss[ix], approach=2)
                else:
                    # If this spectral channel is not the first in the filterbank,
                    # get the length of transmission line to this spectral channel.
                    conn_ix = 0
                    lineLength = 0
                    while conn_ix < ix:
                        this_lambda = v/fres[conn_ix]
                        lineLength += physSep * this_lambda
                        conn_ix += 1
                        
                    # Create Network object for the starting transmission line
                    Ntwk = TransmissionLineLossy(Band, lineLength, Z0, epsr, lossTan)
                    
                    # Create Network object for the first spectral channel
                    SChan = SpectralChannelLossy(Band, Z0, fres[ix], Qc1[ix], Qc2[ix], Qloss[ix], approach=2)

                    # Connect the transmission line to the spectral channel
                    N = Ntwk.nports
                    Ntwk = rf.network.connect(Ntwk, N-1, SChan, 0)
                    
            else:
                SChan = SpectralChannelLossy(Band, Z0, fres[ix], Qc1[ix], Qc2[ix], Qloss[ix], approach=2)
                N = Ntwk.nports
                Ntwk = rf.network.connect(Ntwk, N-1, SChan, 0, num=1)
                
            # Generate the transmission line to the next overlapping spectral channel
            conn_ix = ix
            lineLength = 0
            while conn_ix < ixs[ii1+1]:
                this_lambda = v/fres[conn_ix]
                lineLength += physSep * this_lambda
                conn_ix += 1
            TLine = TransmissionLineLossy(Band, lineLength, Z0, epsr, lossTan)
            N = Ntwk.nports
            Ntwk = rf.network.connect(Ntwk, N-1, TLine, 0, num=1)
                
            ii1 += 1
                
        ix = ixs[-1]
        this_SC = SpectralChannel3PortNetwork(Band, Z0, fres[ix], Qc1[ix], Qc2[ix], Qloss[ix])
        N = Ntwk.nports
        Ntwk = rf.network.connect(Ntwk, N-1, this_SC, 0, num=1)
        Ntwks.append(Ntwk)
        
    return Ntwks, ixs_in_ntwks
"""