import numpy as np
import phidl.geometry as pg
from phidl import Path
import phidl.path as pp
import phidl.routing as pr
from scipy.constants import c

###########################
### F I L T E R B A N K ###
###########################

def make_filter(f0, eps_eff, bend_radius, h_staple, l_open, w_mstrip, Qc_gap, layer, name=None):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0 <float>: Desired filter center frequency in Hz.
        eps_eff <float>: effective dielectric constant of the microstrip TEM mode
        bend_radius <float>: radius in microns of 90 degree bends
        h_staple <float>: height of the vertical (non bendy) part of the half-wave staple
        l_open <float>: number of wavelengths from the open circuit end of the output line 
            to the side of the ouput line which borders the staple
        w_mstrip <float>: microstrip width in microns
        Qc_gap <float>: coupling gap distance in microns between the staple and the output line
        layer: layer of the Device
        name: name of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the filter
    '''
    wavelength = c/(f0*eps_eff**.5) * 1e6
    totlen = wavelength/2
    l_staple = (totlen-h_staple-np.pi*bend_radius)/2
    l_open_um = wavelength * l_open

    # create Device objects
    if name is not None:
        D = pg.Device(name)
    else:
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
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, l_open_um), (0, l_open_um)]
    R3.add_polygon(points, layer=layer)
    R3.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R3.add_port(name = '2', midpoint = [w_mstrip/2,l_open_um], width = w_mstrip, orientation = 90)

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

    # add a port to connect to for the filtered mm-wave signal
    D.add_port(name='output', midpoint=(w_mstrip/2, arc3.ymax), orientation=90)

    # move the filter up along the y-axis to provide a coupling gap from the input feedline
    D.move((0, Qc_gap))

    return D

def make_filterbank(f0s, spacing, eps_eff, bend_radius, h_staple, l_open, w_mstrip, Qc_gaps, layer):
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
        Qc_gap = Qc_gaps[ii]
        D_filt = make_filter(f0, eps_eff, bend_radius, h_staple, 
                             l_open, w_mstrip, Qc_gap, layer, name=ii)

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
    R = pg.rectangle(size=(D_bank.xsize, w_mstrip), layer=layer)
    rect = D_bank << R
    rect.move((0, -rect.ymax))

    return D_bank

#########################
### D E T E C T O R S ###
#########################

def make_kid(w0, wfin, l0, h0, ltrans, lfin, final_h, llip, 
            hC, wC, w_coupler_connector, l0_coupler_connector,
            coupling_ground_height, coupling_ground_width,
            coupling_ground_gap, w_coupler, h_coupler, distance_to_cpw,
            detector_layer, ground_layer, detector_index):
    """
    Make a single lumped element KID with parallel plate capacitors.
    All lengths are in microns.
    
    Parameters:
    w0: Starting linewidth of inductor
    wfin: Final linewidth of inductor after the taper
    l0: Length of "C-shaped" part of the inductor with the starting linewidth
    h0: Height of "C-shaped" part of the inductor with the starting linewidth
    ltrans: Taper length
    lfin: Length of inductor after the taper
    final_h: Distance of the inductor lines from the capacitor plates
    llip: Height of the lip put into one of the inductor lines to 
        equalize their lengths
    hC: Height of each of the twp KID capacitor plates
    wC: width of each of the twp KID capacitor plates
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
    detector_layer, ground_layer, detector_index: Layer numbers for all layers.
    
    Returns:
    D (phidl.Device): A device representing the KID.
    """
    D = pg.Device()

    C0 = pg.C(width=w0, size=((l0-h0)/2, h0), layer=detector_layer)
    P0 = Path()
    P0.append(pp.straight(length=ltrans))
    R0 = P0.extrude([w0, wfin], layer=detector_layer)
    R0.add_port(name='1', midpoint=(0,0), orientation=180)
    R0.add_port(name='2', midpoint=(ltrans,0), orientation=0)

    startc = D << C0
    rect0 = D << R0
    rect1 = D << R0

    rect0.connect(port='1', destination=startc.ports[1])
    rect1.connect(port='1', destination=startc.ports[2])

    startpt = np.array(rect0.ports['2'].midpoint)
    len1 = lfin - final_h - h0
    plate_port1 = D.add_port(name='Plate1', midpoint=(startpt[0]+len1, startpt[1]+final_h), width=wfin, orientation=180)
    manual_path = [
        startpt,
        (startpt[0]+len1, startpt[1]),
        (startpt[0]+len1, startpt[1]+final_h/3),
        (startpt[0]+len1-llip, startpt[1]+final_h/3),
        (startpt[0]+len1-llip, startpt[1]+2*final_h/3),
        (startpt[0]+len1, startpt[1]+2*final_h/3),
        plate_port1.midpoint
    ]
    leg1 = D.add_ref(pr.route_sharp(rect0.ports['2'], plate_port1, path_type='manual', manual_path=manual_path))

    startpt = np.array(rect1.ports['2'].midpoint)
    len1 = lfin - h0
    plate_port2 = D.add_port(name='Plate2', midpoint=(startpt[0]+len1, startpt[1]+final_h+h0), width=wfin, orientation=180)
    manual_path = [
        startpt,
        (startpt[0]+len1, startpt[1]),
        plate_port2.midpoint
    ]
    leg2 = D.add_ref(pr.route_sharp(rect1.ports['2'], plate_port2, path_type='manual', manual_path=manual_path))

    CapPlate1 = pg.rectangle(size=(wC, hC), layer=detector_layer)
    CapPlate1.add_port(name='1', midpoint=(wC-wfin/2, 0), orientation=270)
    CapPlate2 = pg.rectangle(size=(wC, hC), layer=detector_layer)
    CapPlate2.add_port(name='1', midpoint=(wfin/2, 0), orientation=270)

    capplate1 = D << CapPlate1
    capplate2 = D << CapPlate2

    capplate1.connect(port='1', destination=leg1.ports[2])
    capplate2.connect(port='1', destination=leg2.ports[2])

    ground_overshoot = (coupling_ground_width-w_coupler)/2
    coup_port = D.add_port(name='coupler', 
                        midpoint=(capplate2.xmax-w_coupler_connector/2, capplate2.ymin),
                        orientation = 270)

    R2 = pg.rectangle(size=(l0_coupler_connector, w_coupler_connector), layer=detector_layer)
    R2.add_port(name='1', midpoint=(0, w_coupler_connector/2), orientation=180)
    R2.add_port(name='2', 
                midpoint=(l0_coupler_connector, w_coupler_connector/2), 
                orientation=0)

    R3 = pg.rectangle(size=(w_coupler, h_coupler), layer=detector_layer)
    R3.add_port(name='1', midpoint=(0, h_coupler-w_coupler_connector/2), orientation=180)
    R3.add_port(name='2', midpoint=(w_coupler/2, h_coupler), orientation=90)

    h_coupler1 = h_coupler + coupling_ground_gap + distance_to_cpw
    R4 = pg.rectangle(size=(h_coupler1, w_coupler), layer=detector_layer)

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

    readout_midpt = (coupler1.xmax, (coupler1.ymax-coupler1.ymin)/2)
    mmwave_midpt = (D.xmin, (startc.ymax-startc.ymin)/2)

    D = D.flatten()
    D.ports = {}
    D.add_port(name=f'readout_{detector_index}', 
               midpoint=readout_midpt,
               orientation = 0)
    D.add_port(name=f'mmwave_{detector_index}', 
               midpoint=mmwave_midpt,
               orientation = 180)
    
    return D


def make_filterbank_with_kids(filter_f0s, spacing, eps_eff, bend_radius, h_staple, 
                              l_open, w_mstrip, Qc_gaps, filter_layer,
                              w0, wfin, l0, h0, ltrans, lfin, final_h, llip, 
                            h_caps, w_caps, w_coupler_connector, l0_coupler_connector,
                            coupling_ground_height, coupling_ground_width,
                            coupling_ground_gap, w_couplers, h_couplers, distance_to_cpw,
                            detector_layer, ground_layer):
    """
    Make a filterbank with KIDs.
    
    Parameters:
    First two lines - see make_fiterbank
    All other lines - see make_kid
    
    Returns:
    D (phidl.Device): A device representing the filterbank and detectors.
    """
    fb = make_filterbank(filter_f0s, spacing, eps_eff, bend_radius, 
                         h_staple, l_open, w_mstrip, Qc_gaps, filter_layer)
    
    for ii in range(len(h_caps)):
    
        hcap = h_caps[ii]
        hcoup = h_couplers[ii]
        wcap = w_caps[ii]
        wcoup = w_couplers[ii]
        
        Dkid = make_kid(w0, wfin, l0, h0, ltrans, lfin, final_h, llip, 
                hcap, wcap, w_coupler_connector, l0_coupler_connector,
                coupling_ground_height, coupling_ground_width,
                coupling_ground_gap, wcoup, hcoup, distance_to_cpw,
                detector_layer, ground_layer, detector_index=ii)
        
        kid = fb << Dkid
        kid.connect(port=f'mmwave_{ii}', destination=fb.ports[f'{ii}'])
        
    D = fb.flatten()
    D.ports = {}
    return D
