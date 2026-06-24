import numpy as np
import phidl.geometry as pg
from phidl import Path
import phidl.path as pp
import phidl.routing as pr
from scipy.constants import c
import copy

###########################
### F I L T E R B A N K ###
###########################

def make_filter(f0, eps_eff, bend_radius, h0, h1,
                w_mstrip, wcap, hcap, cgap, wgnd, hgnd,
                gnd_gap, signal_layer, gnd_layer, name=None):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0 <float>: Desired filter center frequency in Hz.
        eps_eff <float>: effective dielectric constant of the microstrip TEM mode
        bend_radius <float>: radius in microns bends in the microstrip
        h0 <float>: height of the segments connected to the feedline and
            preceding the coupling capacitors.
        h1 <float>: height of the segments leading into the first bend on either side
        w_mstrip <float>: microstrip width in microns
        wcap: coupling capacitor plate width
        hcap: coupling capacitor plate height
        cgap: gap between the two plates on the signal layer
        wgnd: width of ground cutout
        hgnd: height of ground cutout
        gnd_gap: Gap between cutout and the rest of the ground plane
        signal_layer: microstrip signal layer
        gnd_layer: layer of the ground plane
        name: name of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the filter
    '''
    wavelength = c/(f0*eps_eff**.5) * 1e6
    totlen = wavelength/2
    
    l0 = totlen - 3*np.pi*bend_radius - 2*h1
    # l1 = length of the 2 shorter segments in the horizontal part of the filter
    l1 = (l0 - 2*bend_radius)/4
    # l2 = length of the 1 longer segment in the horizontal part of the filter
    l2 = 2*(l1+bend_radius)
    
    # create Device objects
    if name is not None:
        D = pg.Device(name)
    else:
        D = pg.Device()
        
    Line0 = pg.rectangle(size=(w_mstrip, h0), layer=signal_layer)
    
    CapPlate = pg.rectangle(size=(wcap, hcap), layer=signal_layer)    
    
    GndCut = pg.rectangle(size=(wgnd+2*gnd_gap, hgnd+2*gnd_gap), layer=gnd_layer)
    Rsubtr = pg.rectangle(size=(wgnd, hgnd), layer=gnd_layer)
    Rsubtr.move(origin=(0,0), destination=(gnd_gap, gnd_gap))
    GndCut = pg.boolean(A = GndCut, B = Rsubtr, operation = 'not', layer=gnd_layer)
    
    FiltLine0 = pg.rectangle(size=(w_mstrip, h1), layer=signal_layer)
    Arc = pg.arc(radius = bend_radius, width = w_mstrip, theta = 90, layer=signal_layer)
    Arc.ports = {}
    FiltLine1 = pg.rectangle(size=(l1, w_mstrip), layer=signal_layer)
    FiltLine2 = pg.rectangle(size=(l2, w_mstrip), layer=signal_layer)

    line0 = D << Line0
    plate0 = D << CapPlate
    plate1 = D << CapPlate
    filtline0 = D << FiltLine0
    gndcut = D << GndCut
    arc0 = D << Arc
    filtline1 = D << FiltLine1
    arc1 = D << Arc
    arc2 = D << Arc
    filtline2 = D << FiltLine2
    arc3 = D << Arc
    arc4 = D << Arc
    filtline3 = D << FiltLine1
    arc5 = D << Arc
    filtline4 = D << FiltLine0
    plate2 = D << CapPlate
    plate3 = D << CapPlate
    gndcut1 = D << GndCut
    
    
    plate0.move(
        origin=(plate0.center[0],0), 
        destination=(line0.center[0], line0.ymax)
    )
    plate1.move(
        origin=(plate1.center[0],0),
        destination=(plate0.center[0], plate0.ymax+cgap)
    )
    filtline0.move(
        origin=(filtline0.center[0], 0),
        destination=(plate1.center[0], plate1.ymax)
    )
    gndcut.move(
        origin = gndcut.center,
        destination=(plate0.center[0], plate0.ymax+cgap/2)
    )
    arc0.move(
        origin=(arc0.xmax-w_mstrip/2, 0),
        destination=(filtline0.center[0], filtline0.ymax)
    )
    filtline1.move(
        origin=(filtline1.xmax, filtline1.center[1]),
        destination=(arc0.xmin, arc0.ymax-w_mstrip/2)
    )
    arc1.rotate(angle=180, center=(0,0))
    arc1.move(
        origin=(arc1.xmax, arc1.ymin+w_mstrip/2),
        destination=(filtline1.xmin, filtline1.center[1])
    )
    arc2.rotate(angle=90, center=(0,0))
    arc2.move(
        origin=(arc2.xmin, arc2.ymin),
        destination=(arc1.xmin, arc1.ymax)
    )
    filtline2.move(
        origin=(filtline2.xmin, filtline2.center[1]),
        destination=(arc2.xmax, arc2.ymax-w_mstrip/2)
    )
    arc3.rotate(angle=270, center=(0,0))
    arc3.move(
        origin=(arc3.xmin, arc3.ymin+w_mstrip/2),
        destination=(filtline2.xmax, filtline2.center[1])
    )
    arc4.move(
        origin=(arc4.xmax, arc4.ymin),
        destination=(arc3.xmax, arc3.ymax)
    )
    filtline3.move(
        origin=(filtline3.xmax, filtline3.ymax),
        destination=(arc4.xmin, arc4.ymax)
    )
    arc5.rotate(angle=180, center=(0,0))
    arc5.move(
        origin=(arc5.xmax, arc5.ymin),
        destination=(filtline3.xmin, filtline3.ymin)
    )
    filtline4.move(
        origin=(filtline4.xmin, filtline4.ymin),
        destination=(arc5.xmin, arc5.ymax)
    )
    plate2.move(
        origin=(plate2.center[0], plate2.ymin), 
        destination=(filtline4.center[0], filtline4.ymax)
    )
    plate3.move(
        origin=(plate3.center[0], plate3.ymin), 
        destination=(plate2.center[0], plate2.ymax+cgap)
    )
    gndcut1.move(
        origin = gndcut1.center,
        destination=(plate2.center[0], plate2.ymax+cgap/2)
    )
    
    # D.add_port(
    #     name='feedline', 
    #     midpoint=(w_mstrip/2, 0),
    #     orientation=270
    # )    
    
    D.add_port(
        name='output',
        midpoint=(plate3.center[0], plate3.ymax),
        orientation=90
    )

    return D

def make_broadband_coupler(l_conn, w_conn, w_cap, h_cap, ground_gap, cap_gap, 
                           signal_layer, ground_layer):
    '''
    Create a mm-wave coupling capacitor for a broadband detector.
    
    Parameters: (all lengths in microns)
        l_conn: length of line connecting the feedline to the capacitor plate
        w_conn: width of line connecting the feedline to the capacitor plate
        w_cap: width of each of the two capacitor plates
        h_cap: height of each of the two capacitor plates
        ground_gap: gap between the ground plane cutout and the rest of the ground plane
        cap_gap: gap betweem the two capacitor plates
        signal_layer: layer of the signal plane containing the capacitor plates
        ground_layer: layer of the ground plane
    Returns:
        D: A Phidl.Device object representing the broadband coupler.
    '''
    D = pg.Device()
    Rconn = pg.rectangle(size=(w_conn, l_conn), layer=signal_layer)
    Rplate = pg.rectangle(size=(w_cap, h_cap), layer=signal_layer)
    Rground = pg.rectangle(size=(w_cap+2*ground_gap, 2*h_cap+cap_gap+2*ground_gap), 
                           layer=ground_layer)
    Rsubtr = pg.rectangle(size=(w_cap, 2*h_cap+cap_gap), layer=ground_layer)
    Rsubtr.move(origin=(0,0), destination=(ground_gap, ground_gap))
    Rground = pg.boolean(A = Rground, B = Rsubtr, operation = 'not', layer=ground_layer)
    
    Rconn.add_port(name='0', 
                   midpoint=(w_conn/2, Rconn.ymin),
                   orientation = 270)
    Rconn.add_port(name='1', 
                   midpoint=(w_conn/2, Rconn.ymax),
                   orientation = 90)
    Rplate.add_port(name='0', 
                   midpoint=(w_conn/2, Rplate.ymin),
                   orientation = 270)
    Rplate.add_port(name='1', 
                   midpoint=(w_conn/2, Rplate.ymax),
                   orientation = 90)
    
    conn0 = D << Rconn
    plate0 = D << Rplate
    conn1 = D << Rconn
    plate1 = D << Rplate
    ground_cutout = D << Rground

    plate0.connect(port='0', destination=conn0.ports['1'])
    plate1.move(origin=(plate1.xmin,plate1.ymin), 
                destination=(plate0.xmin, plate0.ymax+cap_gap))
    conn1.connect(port='0', destination=plate1.ports['1'])
    ground_cutout.move(origin=(ground_cutout.xmin,ground_cutout.ymin),
                       destination=(plate0.xmin-ground_gap, plate0.ymin-ground_gap))
    
    in_port = conn0.ports['0']
    out_port = conn1.ports['1']
    D = D.flatten()
    D.ports = {'0': in_port, '1': out_port}
    
    return D


def make_filterbank(f0s, spacing, eps_eff, bend_radius, h0, h1, 
                    w_mstrip, wcaps, hcaps, cgaps, wgnd, hgnd, 
                    gnd_gap, signal_layer, gnd_layer):
    '''
    Creates a phidl Device of a filterbank.

    Parameters:
        f0s <array>: array of resonant frequencies in Hz
        spacing <float>: physical spacing between filters as a fraction of the wavelength
        Other parameters: see make_filter()
    Returns:
        D_bank <phidl.device_layout.Device>: Device object representing the filterbank
    '''
    
    ref_filts = []
    D_bank = pg.Device()
    for ii in range(len(f0s)):
        # generate a filter with resonant frequency f0
        f0 = f0s[ii]
        wcap = wcaps[ii]
        hcap = hcaps[ii]
        cgap = cgaps[ii]
        
        D_filt = make_filter(f0, eps_eff, bend_radius, h0, h1,
                w_mstrip, wcap, hcap, cgap, wgnd, hgnd,
                gnd_gap, signal_layer, gnd_layer, name=ii)
        
        if ii%2 == 1:
            # Reflect every other filter to the other side of the feedline
            D_filt = D_filt.mirror((0, -w_mstrip/2), (1, -w_mstrip/2))

        # reference the filter within the bank and keep track of the reference
        ref = D_bank << D_filt
        ref_filts.append(ref)
        
        # move the filter to the correct location in the bank
        if ii > 0:
            filt_dist = c/(f0*eps_eff**.5) * spacing * 1e6
            ref.move((ref_filts[ii-1].xmin-filt_dist, 0))

        # propagate the filter's output port up to the bank
        port = ref.ports['output']
        D_bank.add_port(f'{ii}', midpoint=port.midpoint, orientation=port.orientation)
            
    # move the filterbank to the origin
    D_bank.move((-D_bank.xmin, 0))

    # add input feedline
    R = pg.rectangle(size=(D_bank.xsize, w_mstrip), layer=signal_layer)
    R.add_port(name='feedline_in', midpoint=(R.xmax, w_mstrip/2), 
               width=w_mstrip, orientation=0)
    R.add_port(name='feedline_out', midpoint=(R.xmin, w_mstrip/2), 
               width=w_mstrip, orientation=180)
    rect = D_bank << R
    rect.move((0, -rect.ymax))
    rect.parent.name = 'Feedline'

    D_bank = D_bank.mirror((0,0), (0,1))
    return D_bank

#########################
### D E T E C T O R S ###
#########################

def make_kid(w_ind, l_ind, l0, final_h,
            hC, wC, gapC, w_coupler_connector, l0_coupler_connector,
            coupling_ground_height, coupling_ground_width,
            coupling_ground_gap, w_coupler, h_coupler, distance_to_cpw,
            Al_layer, Nb_layer, ground_layer, detector_index):
    """
    Make a single lumped element KID with parallel plate capacitors.
    All lengths are in microns.
    
    Parameters:
    w_ind: Linewidth of inductor
    l_ind: Total inductor length
    l0: Length of horizontal part of the inductor that touches the input feedline
    final_h: Distance of the inductor lines from the capacitor plates
    hC: Height of each of the two KID capacitor plates
    wC: width of each of the two KID capacitor plates
    w_coupler_connector: Width of the line connecting the KID to the coupling
        capacitor.
    l0_coupler_connector: Length of the line connecting the KID to the coupling
        capacitor.
    coupling_ground_height: Height of the ground plane cutout for the
        coupling capacitor. Should be about the same as w_coupler_connector.
    coupling_ground_width: Width of the ground plane cutout for the
        coupling capacitor.
    coupling_ground_gap: Distance between the ground plane cutout and
        the rest of the ground plane.
    w_coupler: Width of overlap between each coupler arm and the ground
        plane cutout.
    h_coupler: Height of each coupler arm. Should be about the same as
        coupling_ground_height.
    distance_to_cpw: Distance from the second coupling arm to the CPW readout line.
    Al_layer, Nb_layer, ground_layer: Layer numbers for all layers.
    
    Returns:
    D (phidl.Device): A device representing the KID.
    """
    D = pg.Device()

    R0 = pg.rectangle(size=(w_ind, l0), layer=Al_layer)
    R0.add_port(name='1', midpoint=(w_ind, w_ind/2), width=w_ind, orientation=0)
    R0.add_port(name='2', midpoint=(w_ind, l0-w_ind/2), width=w_ind, orientation=0)
    rect0 = D << R0

    startpt = np.array(rect0.ports['2'].midpoint)
    len2 = l_ind/2 - 1.5*l0 - final_h
    plate_port2 = D.add_port(name='Plate2', midpoint=(startpt[0]+len2+w_ind/2, -final_h), 
                             width=w_ind, orientation=90)
    manual_path = [
        startpt,
        (plate_port2.midpoint[0], startpt[1]),
        plate_port2.midpoint
    ]
    leg2 = D.add_ref(pr.route_sharp(rect0.ports['2'], plate_port2, path_type='manual', manual_path=manual_path, 
                                    layer=Al_layer))
    
    startpt = np.array(rect0.ports['1'].midpoint)
    len1 = plate_port2.midpoint[0] - w_ind - gapC - startpt[0]
    llip = (gapC+l0)/2
    plate_port1 = D.add_port(name='Plate1', midpoint=(startpt[0]+len1, -final_h), width=w_ind, orientation=90)
    manual_path = [
        startpt,
        (startpt[0]+len1-10, startpt[1]),
        (startpt[0]+len1-10, startpt[1]+llip),
        (startpt[0]+len1, startpt[1]+llip),
        plate_port1.midpoint
    ]
    leg1 = D.add_ref(pr.route_sharp(rect0.ports['1'], plate_port1, path_type='manual', manual_path=manual_path, 
                                    layer=Al_layer))
    

    CapPlate1 = pg.rectangle(size=(wC, hC), layer=Nb_layer)
    CapPlate1.add_port(name='1', midpoint=(w_ind/2, 0), orientation=270)
    CapPlate2 = pg.rectangle(size=(wC, hC), layer=Nb_layer)
    CapPlate2.add_port(name='1', midpoint=(wC-w_ind/2, 0), orientation=270)

    capplate1 = D << CapPlate1
    capplate2 = D << CapPlate2

    capplate1.connect(port='1', destination=leg1.ports[2])
    capplate2.connect(port='1', destination=leg2.ports[2])

    ground_overshoot = (coupling_ground_width-w_coupler)/2
    coup_port = D.add_port(name='coupler', 
                        midpoint=(capplate2.xmax-w_coupler_connector/2, capplate2.ymin),
                        orientation = 270)

    R2 = pg.rectangle(size=(l0_coupler_connector, w_coupler_connector), layer=Nb_layer)
    R2.add_port(name='1', midpoint=(0, w_coupler_connector/2), orientation=180)
    R2.add_port(name='2', 
                midpoint=(l0_coupler_connector, w_coupler_connector/2), 
                orientation=0)

    R3 = pg.rectangle(size=(w_coupler, h_coupler), layer=Nb_layer)
    R3.add_port(name='1', midpoint=(0, h_coupler-w_coupler_connector/2), orientation=180)
    R3.add_port(name='2', midpoint=(w_coupler/2, h_coupler), orientation=90)

    h_coupler1 = h_coupler + coupling_ground_gap + distance_to_cpw
    R4 = pg.rectangle(size=(h_coupler1, w_coupler), layer=Nb_layer)

    coupler_connector = D << R2
    coupler0 = D << R3
    coupler1 = D << R4

    coupler_connector.connect(port='1', destination=coup_port)
    coupler0.connect(port='1', destination=coupler_connector.ports['2'])
    pos = np.array(coupler0.ports['2'].midpoint)
    pos[0] += coupling_ground_height/2 - h_coupler
    coupler1.move(origin=(0, w_coupler/2), destination=pos)


    R5 = pg.rectangle(
        size=(coupling_ground_height+2*coupling_ground_gap,
            coupling_ground_width+2*coupling_ground_gap),
        layer=ground_layer
    )
    R6 = pg.rectangle(
        size=(coupling_ground_height,
            coupling_ground_width),
        layer=ground_layer
    )
    R6.move(origin=(0,0), destination=(coupling_ground_gap, coupling_ground_gap))
    R5 = pg.boolean(A = R5, B = R6, operation = 'not', layer=ground_layer)

    ground_gap = D << R5
    ground_gap.move(
        origin=(coupling_ground_gap+coupling_ground_height/2, 
                coupling_ground_gap+ground_overshoot),
        destination=(coupler0.xmax, coupler0.ymin)
    )

    readout_midpt = (coupler1.xmax, (coupler1.ymax+coupler1.ymin)/2)
    mmwave_midpt = (0, l0/2)

    D = D.flatten()
    D.ports = {}
    D.add_port(name=f'readout_{detector_index}', 
               midpoint=readout_midpt,
               orientation = 0)
    D.add_port(name=f'mmwave_{detector_index}', 
               midpoint=mmwave_midpt,
               orientation = 180)
    
    return D

def make_filterbank_with_kids(
    # parameters for filters
    filter_f0s, spacing, eps_eff, bend_radius, h0, h1,
    w_mstrip, filter_hcaps, filter_wcaps, filter_cgaps, 
    filter_wgnd, filter_hgnd, filter_gnd_gap,
    # parameters for KIDs
    kid_hcaps, kid_wcaps,
    w_inds, l_inds, l0, final_h,
    d_stubs, l_stubs, dist_from_mmwave_line,
    kid_cgap, kid_wcoupler_connector, kid_l0coupler_connector,
    coupling_ground_height, coupling_ground_width,
    coupling_ground_gap, kid_wcouplers, kid_hcouplers, distance_to_cpw,
    Al_layer, Nb_layer, ground_layer
    ):
    """
    Make a filterbank with KIDs.
    
    Parameters:
    d_stubs: Distances of tuning stubs from the Al line in microns
    l_stubs: Lengths of tuning stubs in microns
    dist_from_mmwave_line: Distance of the end of each coupling capacitor from the mmwave feedline.
    Line 2 and 3 - see make_fiterbank
    All other lines - see make_kid
    
    Returns:
    D (phidl.Device): A device representing the filterbank and detectors.
    """
    fb = make_filterbank(
        filter_f0s, spacing, eps_eff, bend_radius, h0, h1, 
        w_mstrip, filter_wcaps, filter_hcaps, filter_cgaps, 
        filter_wgnd, filter_hgnd, filter_gnd_gap, 
        Nb_layer, ground_layer
    )
    
    
    for ii in range(len(kid_hcaps)):
    
        w_ind = w_inds[ii]
        l_ind = l_inds[ii]
        hcap = kid_hcaps[ii]
        hcoup = kid_hcouplers[ii]
        wcap = kid_wcaps[ii]
        wcoup = kid_wcouplers[ii]
        dstub = d_stubs[ii]
        lstub = l_stubs[ii]

        Dkid = make_kid(w_ind, l_ind, l0, final_h,
            hcap, wcap, kid_cgap, kid_wcoupler_connector, kid_l0coupler_connector,
            coupling_ground_height, coupling_ground_width,
            coupling_ground_gap, wcoup, hcoup, distance_to_cpw,
            Al_layer, Nb_layer, ground_layer, detector_index=ii)
        Dkid.name = f'spectral_kid_{ii}'

        mmwave_port = Dkid.ports[f'mmwave_{ii}']
        readout_port = Dkid.ports[f'readout_{ii}']
        filter_port = fb.ports[f'{ii}']
        
        
        kid_dist = abs(readout_port.midpoint[0] - mmwave_port.midpoint[0])
        conn_len = dist_from_mmwave_line - kid_dist - abs(filter_port.midpoint[1])
        R0 = pg.rectangle(size=(w_mstrip, conn_len))
        R0.add_port(name='1', midpoint=(w_mstrip/2, 0), orientation=-90)
        R0.add_port(name='2', midpoint=(w_mstrip/2, conn_len), orientation=90)
        R0.add_port(name='stub', midpoint = (w_mstrip, conn_len-dstub-w_mstrip/2), 
                    orientation=0)
        R1 = pg.rectangle(size=(lstub, w_mstrip))
        R1.add_port(name='1', midpoint=(lstub, w_mstrip/2), orientation=0)

        rect0 = fb << R0
        rect1 = fb << R1
        kid = fb << Dkid
        rect0.connect(port='1', destination=filter_port)
        rect1.connect(port='1', destination=rect0.ports['stub'])
        kid.connect(port=f'mmwave_{ii}', destination=rect0.ports['2'])
        
    D = fb
    refs = D.references
    names = np.array([ref.parent.name for ref in refs])
    feedline_ref = np.array(refs)[names=='Feedline'][0]
    spectral_kid_refs = np.array([
        refs[ii] for ii in range(len(refs)) if 'spectral_kid' in names[ii]
    ])
    D = D.flatten()
    D.ports = feedline_ref.ports
    for ref, name in zip(spectral_kid_refs, names):
        ii = name.split('_')[-1]
        D.ports[f'spectral_kid_{ii}'] = ref.ports[f'readout_{ii}']
        D.ports[f'spectral_kid_{ii}'].name = f'spectral_kid_{ii}'
    return D


def add_dark_filters_to_filterbank(
    D, feed_side, orientation, feed_dist,
    f0, eps_eff, bend_radius, h_staple, w_mstrip, Qc_gap, layer
    ):
    """
    Add a 'dark' (no KID) filter to a filterbank.
    
    Parameters:
    D: phidl.Device object representing the filterbank.
    feed_side (int or float, array-like): The end of the filterbank to put the dark filter
        on. 0 = input, 1 = output.
    orientation (int or float, array-like): Whether to put the dark filter above or
        below the feedline. 0 = above, 1 = below.
    feed_dist (array-like): the distance from the end of the feedline to place the dark filter
        at. Distance is measured from the edge of the last spectral filter to the same edge of the 
        dark filter.
    All others: see make_filter. All these parameters should be array-like, 
        except eps_eff, w_mstrip, and layer.
        
    Returns:
    Dout: phidl.Device object that represents the filterbank + dark filters.
    """
    Dout = copy.deepcopy(D)
    old_ports = Dout.ports.values()
    Nfilts = len(feed_side)
    in_port = D.ports['feedline_in']
    out_port = D.ports['feedline_out']
    feed_xin = in_port.midpoint[0]
    feed_xout = out_port.midpoint[0]
    feed_width = in_port.width
    feed_ymid = in_port.midpoint[1]
    feed_ymax = feed_ymid+feed_width/2
    feed_ymin = feed_ymid-feed_width/2
    
    for ii in range(Nfilts):
        
        wavelength = c/(f0[ii]*eps_eff**.5) * 1e6
        totlen = wavelength/2
        l_staple = (totlen-h_staple[ii]-np.pi*bend_radius[ii])/2
        
        Dfilt = pg.Device()
        
        A = pg.arc(radius = bend_radius[ii], width = w_mstrip, theta = 90, layer=layer)

        R0 = pg.Device('rect')
        points =  [(0, 0), (l_staple, 0), (l_staple, w_mstrip), (0, w_mstrip)]
        R0.add_polygon(points, layer=layer)
        R0.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
        R0.add_port(name = '2', midpoint = [l_staple,w_mstrip/2], width = w_mstrip, orientation = 0)

        R1 = pg.Device('rect')
        points =  [(0, 0), (w_mstrip, 0), (w_mstrip, h_staple[ii]), (0, h_staple[ii])]
        R1.add_polygon(points, layer=layer)
        R1.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
        R1.add_port(name = '2', midpoint = [w_mstrip/2,h_staple[ii]], width = w_mstrip, orientation = 90)

        # create references
        arc0 = Dfilt << A
        arc1 = Dfilt << A
        rect0 = Dfilt << R0
        rect1 = Dfilt << R1
        rect2 = Dfilt << R0

        # move around the references and connect them together
        arc0.connect(port = 1, destination = rect0.ports['2'])
        rect1.connect(port = '1', destination = arc0.ports[2])
        arc1.connect(port = 1, destination = rect1.ports['2'])
        rect2.connect(port = '2', destination = arc1.ports[2])

        Dfilt = Dfilt.mirror((0,0), (0,1))

        # Add the filter to the filterbank
        dfilt = Dout << Dfilt
        
        if orientation[ii] == 0:
            y0 = dfilt.ymin
            yf = Qc_gap[ii] + feed_ymax
        elif orientation[ii] == 1:
            y0 = dfilt.ymax
            yf = -Qc_gap[ii] + feed_ymin
        
        if feed_side[ii] == 0:
            if feed_xin < feed_xout:
                x0 = dfilt.xmin
                xf = feed_xin - feed_dist[ii]
            else:
                x0 = dfilt.xmax
                xf = feed_xin + feed_dist[ii]
        elif feed_side[ii] == 1:
            if feed_xin < feed_xout:
                x0 = dfilt.xmax
                xf = feed_xout + feed_dist[ii]
            else:
                x0 = dfilt.xmin
                xf = feed_xout - feed_dist[ii]
        
        dfilt.move(origin=(x0, y0), destination=(xf, yf))
        
    # Extend the feedline to the ends of the dark filters
    new_xmin = Dout.xmin
    new_xmax = Dout.xmax
    if feed_xin < feed_xout:
        xmin = feed_xin
        xmax = feed_xout
        min_port = in_port
        max_port = out_port
        dx_in = xmin-new_xmin
        new_in_x = xmin - dx_in
        dx_out = new_xmax-xmax
        new_out_x = xmax + dx_out
        in_orientation = 180
        out_orientation = 0
    else:
        xmax = feed_xin
        xmin = feed_xout
        max_port = in_port
        min_port = out_port
        dx_out = xmin-new_xmin
        new_out_x = xmin - dx_out
        dx_in = new_xmax-xmax
        new_in_x = xmax + dx_in
        out_orientation = 180
        in_orientation = 0
        
    Rmin = pg.rectangle(size=(dx_in, feed_width))
    Rmin.add_port(name='0', midpoint=(Rmin.xmax, feed_width/2), width=feed_width, orientation=0)
    rmin = Dout << Rmin
    rmin.connect(port='0', destination=min_port)
    
    Rmax = pg.rectangle(size=(dx_out, feed_width))
    Rmax.add_port(name='0', midpoint=(Rmax.xmin, feed_width/2), width=feed_width, orientation=180)
    rmax = Dout << Rmax
    rmax.connect(port='0', destination=max_port)
    
    Dout = Dout.flatten()
    Dout.ports = {}
    Dout.add_port(name='feedline_in', midpoint=(new_in_x, feed_ymid), 
                  width=feed_width, orientation=in_orientation)
    Dout.add_port(name='feedline_out', midpoint=(new_out_x, feed_ymid), 
                  width=feed_width, orientation=out_orientation)
    for port in old_ports:
        if 'spectral_kid' in port.name:
            Dout.ports[port.name] = port

    return Dout


def add_broadbands_to_filterbank(
    D, feed_side, orientation, feed_dist, 
    # parameters for mm-wave coupling capacitor
    l_conn, w_conn, w_cap, h_cap, ground_gap, cap_gap, 
    # parameters for KID and readout coupling capacitor
    w_ind, l_ind, l0, final_h,
    hC, wC, gapC, w_coupler_connector, l0_coupler_connector,
    coupling_ground_height, coupling_ground_width,
    coupling_ground_gap, w_coupler, h_coupler, distance_to_cpw,
    # layers for inductor, top capacitor plates + feedline, and ground
    ind_layer, cap_layer, ground_layer
    ):
    """
    Add broadband-coupled KIDs to a filterbank that already contains
    spectral KIDs.
    
    Parameters: (all distances are in microns)
    D: The phidl.Device object representing the filterbank+spectral KIDs.
    feed_side (int or float, array-like): The end of the filterbank to put the broadband
        on. 0 = input, 1 = output.
    orientation (int or float, array-like): Whether to put the broadband coupler above or
        below the feedline. 0 = above, 1 = below.
    feed_dist (array-like): the distance from the end of the feedline to place the broadband
        coupler at. Distance is measured from the edge of the last spectral filter to the center of
        the line which will connect the broadband capacitor to the feedline.
    All others: See make_broadband_coupler and make_kid. All these parameters should be array-like, 
        except distance_to_cpw, ind_layer, cap_layer, and ground_layer.
        
    Returns:
    Dout: The phidl.Device object representing the filterbank+spectral KIDs+broadband KIDs.
    """
    Dout = copy.deepcopy(D)
    old_ports = Dout.ports.values()
    Nbbs = len(feed_side)
    in_port = D.ports['feedline_in']
    out_port = D.ports['feedline_out']
    feed_xin = in_port.midpoint[0]
    feed_xout = out_port.midpoint[0]
    feed_width = in_port.width
    feed_ymid = in_port.midpoint[1]
    feed_ymax = feed_ymid+feed_width/2
    feed_ymin = feed_ymid-feed_width/2
    
    broadband_kid_refs = []
    for ii in range(Nbbs):
        D_bb = make_broadband_coupler(l_conn[ii], w_conn[ii], w_cap[ii], 
                                      h_cap[ii], ground_gap[ii], cap_gap[ii], 
                                      cap_layer, ground_layer)
        
        if orientation[ii] == 0:
            this_orient = 90
            yport = feed_ymax
        elif orientation[ii] == 1:
            this_orient = 270
            yport = feed_ymin
        
        if feed_side[ii] == 0:
            xport = feed_xin
            if feed_xin > feed_xout:
                xport += feed_dist[ii]
            else:
                xport -= feed_dist[ii]
        elif feed_side[ii] == 1:
            xport = feed_xout
            if feed_xin > feed_xout:
                xport -= feed_dist[ii]
            else:
                xport += feed_dist[ii]
            
        feed_port = Dout.add_port(name=f'{ii}', midpoint=(xport, yport), 
                                   width=w_conn[ii], 
                                   orientation=this_orient)
        
        bb = Dout << D_bb
        bb.connect(port='0', destination=feed_port)
        
        D_kid = make_kid(w_ind[ii], l_ind[ii], l0[ii], final_h[ii],
            hC[ii], wC[ii], gapC[ii], w_coupler_connector[ii], l0_coupler_connector[ii],
            coupling_ground_height[ii], coupling_ground_width[ii],
            coupling_ground_gap[ii], w_coupler[ii], h_coupler[ii], distance_to_cpw,
            ind_layer, cap_layer, ground_layer, detector_index=0)
        D_kid.name = f'broadband_kid_{ii}'
        if feed_side[ii] == 0:
            D_kid = D_kid.mirror((0,0), (0,1))
        
        d_kid = Dout << D_kid
        d_kid.connect(port='mmwave_0', destination=bb.ports['1'])
        broadband_kid_refs.append(d_kid)
        
    # Extend the feedline to the ends of the dark filters
    new_xmin = Dout.xmin
    new_xmax = Dout.xmax
    if feed_xin < feed_xout:
        xmin = feed_xin
        xmax = feed_xout
        min_port = in_port
        max_port = out_port
        dx_in = xmin-new_xmin
        new_in_x = xmin - dx_in
        dx_out = new_xmax-xmax
        new_out_x = xmax + dx_out
        in_orientation = 180
        out_orientation = 0
    else:
        xmax = feed_xin
        xmin = feed_xout
        max_port = in_port
        min_port = out_port
        dx_out = xmin-new_xmin
        new_out_x = xmin - dx_out
        dx_in = new_xmax-xmax
        new_in_x = xmax + dx_in
        out_orientation = 180
        in_orientation = 0
        
    Rmin = pg.rectangle(size=(dx_in, feed_width))
    Rmin.add_port(name='0', midpoint=(Rmin.xmax, feed_width/2), width=feed_width, orientation=0)
    rmin = Dout << Rmin
    rmin.connect(port='0', destination=min_port)
    
    Rmax = pg.rectangle(size=(dx_out, feed_width))
    Rmax.add_port(name='0', midpoint=(Rmax.xmin, feed_width/2), width=feed_width, orientation=180)
    rmax = Dout << Rmax
    rmax.connect(port='0', destination=max_port)
    
    Dout = Dout.flatten()
    Dout.ports = {}
    Dout.add_port(name='feedline_in', midpoint=(new_in_x, feed_ymid), 
                  width=feed_width, orientation=in_orientation)
    Dout.add_port(name='feedline_out', midpoint=(new_out_x, feed_ymid), 
                  width=feed_width, orientation=out_orientation)
    for port in old_ports:
        if 'spectral_kid' in port.name:
            Dout.ports[port.name] = port
    for ref in broadband_kid_refs:
        name = ref.parent.name
        Dout.ports[name] = ref.ports[f'readout_0']
        Dout.ports[name].name = name
        
    return Dout

#############################
### T E R M I N A T I O N ###
#############################

def add_terminator(wtrans, ltrans, wf, meander_start_h, meander_length,
                    meander_spacing, n_meander):
    '''
    Makes a phidl Device representing a terminator.
    The terminator is composed of many weakly-coupled lossy stubs all
    coupled to the feedline, which are each designed to have much lower 
    reflection than absorption.
    '''
    D = phidl.Device()
    
    