import phidl.geometry as pg
import numpy as np

c = 3e8

def make_filter(f0, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap, layer):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0 <float>: Desired filter center frequency in Hz.
        eps_eff <float>: effective dielectric constant of the microstrip TEM mode
        bend_radius <float>: radius in microns of 90 degree bends
        h_staple <float>: height of the vertical (non bendy) part of the half-wave staple
        w_mstrip <float>: microstrip width in microns
        Qc_gap <float>: coupling gap distance in microns between the staple and the output line
        layer: layer of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the filter
    '''
    totlen = c/(f0*eps_eff**.5)/2 * 1e6
    l_staple = (totlen-h_staple-np.pi*bend_radius)/2
    l_open = l_staple

    # create Device objects
    D = pg.Device()

    A = pg.arc(radius = bend_radius, width = w_mstrip, theta = 90, layer=layer)

    R0 = pg.Device('rect')
    points =  [(0, 0), (l_staple, 0), (l_staple, w_mstrip), (0, w_mstrip)]
    R0.add_polygon(points, layer=layer)
    R0.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R0.add_port(name = '2', midpoint = [l_staple,w_mstrip/2], width = w_mstrip, orientation = 0)

    R1 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, h_staple), (0, h_staple)]
    R1.add_polygon(points, layer=layer)
    R1.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R1.add_port(name = '2', midpoint = [w_mstrip/2,h_staple], width = w_mstrip, orientation = 90)

    R2 = pg.Device('rect')
    points =  [(0, 0), (l_staple-A.xsize, 0), (l_staple-A.xsize, w_mstrip), (0, w_mstrip)]
    R2.add_polygon(points, layer=layer)
    R2.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R2.add_port(name = '2', midpoint = [l_staple-A.xsize,w_mstrip/2], width = w_mstrip, orientation = 0)

    R3 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, l_open), (0, l_open)]
    R3.add_polygon(points, layer=layer)
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
    rect4.connect(port = '1', destination = arc3.ports[2])

    return D

def make_filterbank(f0s, spacing, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap, layer):
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
        D_filt = make_filter(f0, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap, layer=layer)
        D_filts.append(D_filt)
        ref = D_bank << D_filt
        ref_filts.append(ref)
        
        if ii > 0:
            filt_dist = c/(f0*eps_eff**.5) * spacing * 1e6
            ref.move((ref_filts[ii-1].xmin-filt_dist, 0))
            
    D_bank.move((-D_bank.xmin, 0))
    return D_bank

def make_kid(w0, wtrans, wfin, h0, l0, ltrans, lfin, final_gap, final_h,
             llip, wC, hC, Cgap, w_readout, layer):
    '''
    Creates a phidl Device of a lumped-element KID with a tapered microstrip inductor
    and a parallel plate capacitor. The mm-wave signal is to be injected into the center of the inductor.

    Parameters (all lengths are in microns):
        w0: starting inductor linewidth 
        wtrans: linewidth of transition section
        wfin: final inductor linewidth
        h0: height of starting inductor section
        l0: total length of starting inductor section
        ltrans: length of transition section
        lfin: length of final inductor section
        final_gap: distance between the two lines of the inductor when they insert into the KID PPC plates
        final_h: height of the capacitor plates above the horizontal length of the inductor
        llip: length of the lip put into one of the final inductor lines to equalize their lengths
        wC: total width of the two capacitor plates together
        hC: height of each capacitor plate
        Cgap: gap between the capacitor plates
        w_readout: width of the readout line which will couple the KID PPC to the readout PPC
        layer: layer of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the KID
    '''

    # initialize Device objects
    D = pg.Device()
    C0 = pg.C(width=w0, size=((l0-h0)/2-w0/2, h0-w0), layer=layer)
    R0 = pg.rectangle(size=(ltrans, wtrans), layer=layer)
    R0.add_port(name='1', midpoint=(0,wtrans/2), orientation=180)
    R0.add_port(name='2', midpoint=(ltrans,wtrans/2), orientation=0)
    R1 = pg.rectangle(size=(lfin-h0-final_h, wfin), layer=layer)
    R1.add_port(name='1', midpoint=(0,wfin/2), orientation=180)
    R1.add_port(name='2', midpoint=(R1.xsize,wfin/2), orientation=0)
    R2 = pg.rectangle(size=(wfin, h0+final_h-(w0-wfin)/2), layer=layer)
    R2.add_port(name='1', midpoint=(0,wfin/2), orientation=180)
    R2.add_port(name='2', midpoint=(wfin/2,R2.ysize), orientation=90)
    R3 = pg.rectangle(size=(lfin-h0-final_h-final_gap-llip+wfin, wfin), layer=layer)
    R3.add_port(name='1', midpoint=(0,wfin/2), orientation=180)
    R3.add_port(name='2', midpoint=(R3.xsize-wfin/2,0), orientation=-90)
    C1 = pg.C(width=wfin, size=((final_gap+h0)/2-wfin, llip-wfin), layer=layer)
    R4 = pg.rectangle(size=(wfin, wfin+final_h+(w0-wfin)/2), layer=layer)
    R4.add_port(name='1', midpoint=(wfin/2,0), orientation=-90)
    R4.add_port(name='2', midpoint=(wfin/2,R4.ysize), orientation=90)
    R5 = pg.rectangle(size=(wC/2-Cgap/2, hC), layer=layer)
    R5.add_port(name='1', midpoint=(R5.xsize+(Cgap-final_gap-wfin)/2, 0), orientation=-90)
    R6 = pg.rectangle(size=(wC/2-Cgap/2, hC), layer=layer)
    R6.add_port(name='1', midpoint=((final_gap-Cgap+wfin)/2, 0), orientation=-90)

    # create references
    startc = D << C0
    rect0 = D << R0
    rect1 = D << R0
    rect2 = D << R1
    rect3 = D << R2
    rect4 = D << R3
    lipc = D << C1
    rect5 = D << R4
    rect6 = D << R5
    rect7 = D << R6

    # connect references together
    rect0.connect(port='1', destination=startc.ports[1])
    rect1.connect(port='1', destination=startc.ports[2])
    rect2.connect(port='1', destination=rect1.ports['2'])
    rect3.connect(port='1', destination=rect2.ports['2'])
    rect4.connect(port='1', destination=rect0.ports['2'])
    lipc.connect(port=1, destination=rect4.ports['2'])
    rect5.connect(port='1', destination=lipc.ports[2])
    rect6.connect(port='1', destination=rect5.ports['2'])
    rect7.connect(port='1', destination=rect3.ports['2'])

    # add ports for connecting the KID to mm-wave signal and readout coupling capacitor
    D.add_port(name='readout', midpoint=(rect7.xmax, rect7.ymin+w_readout/2), orientation=0)
    D.add_port(name='mmwave', midpoint=(startc.xmin, startc.ymax/2), orientation=180)

    return D

def make_readout_ppc(wCc, hCc, Ccgap, lin, lout, wms, layer):
    '''
    Creates a phidl Device of a parallel plate capacitor for readout.

    Parameters (all lengths are in microns):
        wCc: total width of both plates of the PPC
        hCc: height of each plate of the PPC
        Ccgap: gap between the capacitor plates
        lin: input length of microstrip coming from the KID PPC
        lout: output length of microstrip going to the readout thru line
        wms: microstrip width
        layer: layer of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the PPC
    '''

    # initialize Device objects
    D = pg.Device()
    R0 = pg.rectangle(size=(lin, wms), layer=layer)
    R0.add_port(name='2', midpoint=(R0.xsize,wms/2), orientation=0)
    R1 = pg.rectangle(size=((wCc-Ccgap)/2, hCc), layer=layer)
    R1.add_port(name='1', midpoint=(0, wms/2), orientation=180)
    R1.add_port(name='2', midpoint=(R1.xsize+Ccgap, 0), orientation=0)
    R2 = pg.rectangle(size=((wCc-Ccgap)/2, hCc), layer=layer)
    R2.add_port(name='1', midpoint=(0, 0), orientation=180)
    R2.add_port(name='2', midpoint=(R1.xsize, wms/2), orientation=0)
    R3 = pg.rectangle(size=(lout, wms), layer=layer)
    R3.add_port(name='1', midpoint=(0,wms/2), orientation=180)

    # create references
    rect0 = D << R0
    rect1 = D << R1
    rect2 = D << R2
    rect3 = D << R3

    # connect references together
    rect1.connect(port='1', destination=rect0.ports['2'])
    rect2.connect(port='1', destination=rect1.ports['2'])
    rect3.connect(port='1', destination=rect2.ports['2'])

    # add ports for connecting this PPC to the KID PPC and to the readout thru line
    D.add_port(name='kid', midpoint=(0, wms/2), orientation=180)
    D.add_port(name='thru', midpoint=(D.xsize, wms/2), orientation=0)

    return D