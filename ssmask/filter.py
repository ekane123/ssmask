import phidl.geometry as pg
import numpy as np

c = 3e8

def make_filter(f0, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0 <float>: Desired filter center frequency in Hz.
        eps_eff <float>: effective dielectric constant of the microstrip TEM mode
        bend_radius <float>: radius in microns of 90 degree bends
        h_staple <float>: height of the vertical (non bendy) part of the half-wave staple
        w_mstrip <float>: microstrip width in microns
        Qc_gap <float>: coupling gap distance in microns between the staple and the output line
    Returns:
        D <phidl.device_layout.Device>: Device object representing the filter
    '''
    totlen = c/(f0*eps_eff**.5)/2 * 1e6
    l_staple = (totlen-h_staple-np.pi*bend_radius)/2
    l_open = l_staple

    # create Device objects
    D = pg.Device()

    A = pg.arc(radius = bend_radius, width = w_mstrip, theta = 90, layer = 0)

    R0 = pg.Device('rect')
    points =  [(0, 0), (l_staple, 0), (l_staple, w_mstrip), (0, w_mstrip)]
    R0.add_polygon(points)
    R0.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R0.add_port(name = '2', midpoint = [l_staple,w_mstrip/2], width = w_mstrip, orientation = 0)

    R1 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, h_staple), (0, h_staple)]
    R1.add_polygon(points)
    R1.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R1.add_port(name = '2', midpoint = [w_mstrip/2,h_staple], width = w_mstrip, orientation = 90)

    R2 = pg.Device('rect')
    points =  [(0, 0), (l_staple-A.xsize, 0), (l_staple-A.xsize, w_mstrip), (0, w_mstrip)]
    R2.add_polygon(points)
    R2.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R2.add_port(name = '2', midpoint = [l_staple-A.xsize,w_mstrip/2], width = w_mstrip, orientation = 0)

    R3 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, l_open), (0, l_open)]
    R3.add_polygon(points)
    R3.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R3.add_port(name = '2', midpoint = [w_mstrip/2,l_open], width = w_mstrip, orientation = 90)

    # create references
    arc0 = D << A
    arc1 = D << A
    rect0 = D << R0
    rect1 = D << R1
    rect2 = D << R0
    rect3 = D << R2
    arc2 = D << A
    arc3 = D << A
    rect4 = D << R3

    # move around the references and connect them together
    arc0.connect(port = 1, destination = rect0.ports['2'])
    rect1.connect(port = '1', destination = arc0.ports[2])
    arc1.connect(port = 1, destination = rect1.ports['2'])
    rect2.connect(port = '2', destination = arc1.ports[2])
    rect3.move((arc0.xsize, rect2.ymax+Qc_gap))
    arc2.connect(port = 2, destination = rect3.ports['1'])
    arc3.connect(port = 1, destination = rect3.ports['2'])
    rect4.connect(port = '1', destination = arc2.ports[1])

    return D

def make_filterbank(f0s, spacing, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0s <array>: array of resonant frequencies in Hz
        spacing <float>: physical spacing between filters as a fraction of the wavelength
        Other parameters: see make_filter()
    Returns:
        D_bank <phidl.device_layout.Device>: Device object representing the filterbank
    '''
    D_filts = []
    ref_filts = []
    D_bank = pg.Device()
    for ii in range(len(f0s)):
        f0 = f0s[ii]
        D_filt = make_filter(f0, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap)
        D_filts.append(D_filt)
        ref = D_bank << D_filt
        ref_filts.append(ref)
        
        if ii > 0:
            filt_dist = c/(f0*eps_eff**.5) * spacing * 1e6
            ref.move((ref_filts[ii-1].xmin-filt_dist, 0))
            
    D_bank.move((-D_bank.xmin, 0))
    return D_bank