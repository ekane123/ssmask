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

def make_filter(
    side_length, bend_radius, h_staple, l_open, 
    w_mstrip, Qc_gap1, Qc_gap2, top_layer, bottom_layer, name=None
    ):
    '''
    Creates a phidl Device of a single-pole microstrip filter.

    Parameters:
        f0 <float>: Desired filter center frequency in Hz.
        eps_eff <float>: effective dielectric constant of the microstrip TEM mode
        bend_radius <float>: radius in microns of 90 degree bends
        h_staple <float>: height of the vertical (non bendy) part of the half-wave staple
        l_open <float>: length in microns from the open circuit end of the output line 
            to the side of the ouput line which borders the staple
        w_mstrip <float>: microstrip width in microns
        Qc_gap1 <float>: coupling gap distance in microns between the input line and the staple
        Qc_gap2 <float>: coupling gap distance in microns between the staple and the output line
        top_layer: layer for the input mm-wave microstrip and staple
        bottom_layer: layer for the output mm-wave microstrip
        name: name of the Device
    Returns:
        D <phidl.device_layout.Device>: Device object representing the filter
    '''

    # create Device objects
    if name is not None:
        D = pg.Device(name)
    else:
        D = pg.Device()

    Atop = pg.arc(radius = bend_radius, width = w_mstrip, theta = 90, layer=top_layer)
    Abottom = pg.arc(radius = bend_radius, width = w_mstrip, theta = 90, layer=bottom_layer)
    
    R0 = pg.Device('rect')
    points =  [(0, 0), (side_length, 0), (side_length, w_mstrip), (0, w_mstrip)]
    R0.add_polygon(points, layer=top_layer)
    R0.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R0.add_port(name = '2', midpoint = [side_length,w_mstrip/2], width = w_mstrip, orientation = 0)

    R1 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, h_staple), (0, h_staple)]
    R1.add_polygon(points, layer=top_layer)
    R1.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R1.add_port(name = '2', midpoint = [w_mstrip/2,h_staple], width = w_mstrip, orientation = 90)

    R2 = pg.Device('rect')
    points =  [(0, 0), (side_length-Atop.xsize, 0), (side_length-Atop.xsize, w_mstrip), (0, w_mstrip)]
    R2.add_polygon(points, layer=bottom_layer)
    R2.add_port(name = '1', midpoint = [0,w_mstrip/2], width = w_mstrip, orientation = 180)
    R2.add_port(name = '2', midpoint = [side_length-Abottom.xsize,w_mstrip/2], width = w_mstrip, orientation = 0)

    R3 = pg.Device('rect')
    points =  [(0, 0), (w_mstrip, 0), (w_mstrip, l_open), (0, l_open)]
    R3.add_polygon(points, layer=bottom_layer)
    R3.add_port(name = '1', midpoint = [w_mstrip/2,0], width = w_mstrip, orientation = -90)
    R3.add_port(name = '2', midpoint = [w_mstrip/2,l_open], width = w_mstrip, orientation = 90)

    # create references
    arc0 = D << Atop
    arc1 = D << Atop
    rect0 = D << R0
    rect1 = D << R1
    rect2 = D << R0
    rect3 = D << R2
    arc2 = D << Abottom
    arc3 = D << Abottom
    rect4 = D << R3

    # move around the references and connect them together
    arc0.connect(port = 1, destination = rect0.ports['2'])
    rect1.connect(port = '1', destination = arc0.ports[2])
    arc1.connect(port = 1, destination = rect1.ports['2'])
    rect2.connect(port = '2', destination = arc1.ports[2])
    rect3.move((arc0.xsize, rect2.ymax+Qc_gap2))
    arc2.connect(port = 2, destination = rect3.ports['1'])
    arc3.connect(port = 1, destination = rect3.ports['2'])
    rect4.connect(port = '1', destination = arc3.ports[2])

    # add a port to connect to for the filtered mm-wave signal
    D.add_port(name='output', midpoint=(w_mstrip/2, arc3.ymax), orientation=90)

    # move the filter up along the y-axis to provide a coupling gap from the input feedline
    D.move((0, Qc_gap1))
    
    ports = D.ports
    D = D.flatten()
    D.ports = {}
    D.ports['output'] = ports['output']

    return D


def make_filterbank(
    side_lengths, spacings, bend_radius, h_staple, 
    l_opens, w_mstrip, Qc_gaps1, Qc_gaps2, top_layer, bottom_layer
    ):
    '''
    Creates a phidl Device of a filterbank.

    Parameters:
        f0s <array>: array of resonant frequencies in Hz
        spacings <float>: physical spacings between filters in microns
        Other parameters: see make_filter()
    Returns:
        D_bank <phidl.device_layout.Device>: Device object representing the filterbank
    '''
    
    ref_filts = []
    D_bank = pg.Device()
    for ii in range(len(side_lengths)):
        # generate a filter with resonant frequency f0
        side_length = side_lengths[ii]
        l_open = l_opens[ii]
        Qc_gap1 = Qc_gaps1[ii]
        Qc_gap2 = Qc_gaps2[ii]
        D_filt = make_filter(
            side_length, bend_radius, h_staple, l_open, 
            w_mstrip, Qc_gap1, Qc_gap2, top_layer, bottom_layer, name=None
        )

        # reference the filter within the bank and keep track of the reference
        ref = D_bank << D_filt
        ref_filts.append(ref)
        
        # move the filter to the correct location in the bank
        if ii > 0:
            filt_dist = spacings[ii]
            ref.move((ref_filts[ii-1].xmin-filt_dist, 0))

        # propagate the filter's output port up to the bank
        port = ref.ports['output']
        D_bank.add_port(f'{ii}', midpoint=port.midpoint, orientation=port.orientation)
            
    # move the filterbank to the origin
    D_bank.move((-D_bank.xmin, 0))

    # add input feedline
    R = pg.rectangle(size=(D_bank.xsize, w_mstrip), layer=top_layer)
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

def make_kid(
    w_ind, l_ind, l0, final_h,
    hC, wC, gapC, w_coupler_connector, l0_coupler_connector,
    coupling_ground_height, coupling_ground_width,
    coupling_ground_gap, h_coupler, distance_to_cpw,
    Al_layer, Nb_layer, Nb_gnd_layer, detector_index
    ):
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
    h_coupler: Overlap distance between each coupler arm and the ground
        plane cutout.
    distance_to_cpw: Distance from the second coupling arm to the CPW readout line.
    Al_layer, Nb_layer, Nb_gnd_layer: Layer numbers for all layers.
    
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

    coup_port = D.add_port(
        name='coupler', 
        midpoint=(capplate2.xmax-w_coupler_connector/2, capplate2.ymax),
        orientation = 90
    )

    # Coupler from KID capacitor to readout capacitor
    R2 = pg.rectangle(size=(l0_coupler_connector, w_coupler_connector), layer=Nb_layer)
    R2.add_port(
        name='1', 
        midpoint=(0, w_coupler_connector/2), 
        orientation=180
    )
    R2.add_port(
        name='2', 
        midpoint=(l0_coupler_connector, w_coupler_connector/2), 
        orientation=0
    )

    # First readout capacitor plate that connects to the KID capacitor
    R3 = pg.rectangle(size=(h_coupler, coupling_ground_width), layer=Nb_layer)
    R3.add_port(
        name='1', 
        midpoint=(R3.xmax-w_coupler_connector/2, R3.ymin), 
        orientation=270
    )
    # R3.add_port(name='2', midpoint=(coupling_ground_width/2, h_coupler), orientation=90)

    # Second readout capacitor plate that connects to the readout line
    h_coupler1 = h_coupler + coupling_ground_gap + distance_to_cpw
    R4 = pg.rectangle(size=(h_coupler1, coupling_ground_width), layer=Nb_layer)

    coupler_connector = D << R2
    coupler0 = D << R3
    coupler1 = D << R4

    coupler_connector.connect(port='1', destination=coup_port)
    coupler0.connect(port='1', destination=coupler_connector.ports['2'])
    coupler1.move(
        origin=(coupler1.xmin, coupler1.ymin), 
        destination=(coupler0.xmin, coupler0.ymin)
    )
    coupler1.move(
        origin=(0, 0), 
        destination=(coupling_ground_height/2, 0)
    )
    
    # Ground cutout for readout capacitor (R5)
    R5 = pg.rectangle(
        size=(coupling_ground_height+2*coupling_ground_gap,
            coupling_ground_width+2*coupling_ground_gap),
        layer=Nb_gnd_layer
    )
    R6 = pg.rectangle(
        size=(coupling_ground_height,
            coupling_ground_width),
        layer=Nb_gnd_layer
    )
    R6.move(origin=(0,0), destination=(coupling_ground_gap, coupling_ground_gap))
    R5 = pg.boolean(A = R5, B = R6, operation = 'not', layer=Nb_gnd_layer)

    ground_gap = D << R5
    ground_gap.move(
        origin=ground_gap.center,
        destination=(coupler0.xmax, coupler0.center[1])
    )

    readout_midpt = (coupler1.xmax, coupler1.center[1])
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

def make_kid_array(
    w_ind, l_ind, l0, final_h, kid_spacing, w_Nb_mstrip,
    stub_lengths, stub_bend_length, stub_dists,
    hC, wC_arr, gapC, w_coupler_connector, l0_coupler_connector,
    coupling_ground_height, coupling_ground_width,
    coupling_ground_gap, h_coupler_arr, distance_to_cpw,
    Al_layer, Nb_layer, Nb_gnd_layer
    ):
    """
    Make a linear array of KIDs, with no filters.
    The ends of the coupling capacitors that touch the
    readout feedlines are aligned with each other.
    
    Parameters (All lengths in microns):
    w_Nb_mstrip: Width of Nb microstrip line leading into the KID inductor.
    kid_spacing: Vertical distance between KIDs.
    stub_lengths: Length of each stub
    stub_bend_length: Perpendicular distance away from the post-filter Nb line
        before the stub bends to be parallel to the line.
    stub_dists: Distance of each stub away from the inductor.
    See make_kid for all other parameters.
    wC_arr and h_coupler_arr are array-like.
    
    Returns:
    Darray: phidl Device of the KID array.
    """
    Darray = pg.Device()
    nkids = len(wC_arr)
    readout_ports = []
    mmwave_ports = []
    for ii in range(nkids):
        Dkid = make_kid(
            w_ind, l_ind, l0, final_h,
            hC, wC_arr[ii], gapC, w_coupler_connector, l0_coupler_connector,
            coupling_ground_height, coupling_ground_width,
            coupling_ground_gap, h_coupler_arr[ii], distance_to_cpw,
            Al_layer, Nb_layer, Nb_gnd_layer, detector_index=ii
        )

        readout_port = Dkid.ports[f'readout_{ii}']
        Dkid.move(
            origin = readout_port.midpoint,
            destination = (0, readout_port.midpoint[1]+ii*kid_spacing)
        )

        kid = Darray << Dkid
        
        R0 = pg.rectangle(
            size=(w_Nb_mstrip + stub_dists[ii], w_Nb_mstrip),
            layer=Nb_layer
        )
        R0.add_port(
            name=f'mmwave_{ii}',
            midpoint=(R0.xmin, w_Nb_mstrip/2),
            orientation=180
        )
        R0.add_port(
            name='2',
            midpoint=(R0.xmax, w_Nb_mstrip/2),
            orientation=0
        )
        R0.add_port(
            name='3',
            midpoint=(R0.xmin+w_Nb_mstrip/2, R0.ymax),
            orientation=90
        )
        r0 = Darray << R0
        r0.connect(
            port='2',
            destination=kid.ports[f'mmwave_{ii}']
        )
        
        R1 = pg.rectangle(
            size=(w_Nb_mstrip, stub_bend_length+w_Nb_mstrip/2),
            layer=Nb_layer
        )
        R1.add_port(
            name='1',
            midpoint=(w_Nb_mstrip/2, R1.ymin),
            orientation=270
        )
        R1.add_port(
            name='2',
            midpoint=(R1.xmin, R1.ymax-w_Nb_mstrip/2),
            orientation=180
        )
        r1 = Darray << R1
        r1.connect(
            port='1',
            destination=r0.ports['3']
        )
        
        R2 = pg.rectangle(
            size=(stub_lengths[ii]-stub_bend_length-w_Nb_mstrip/2, w_Nb_mstrip),
            layer=Nb_layer
        )
        R2.add_port(
            name='1',
            midpoint=(R2.xmax, w_Nb_mstrip/2),
            orientation=0
        )
        r2 = Darray << R2
        r2.connect(
            port='1',
            destination=r1.ports['2']
        )
        
        mmwave_ports.append(r0.ports[f'mmwave_{ii}'])
        readout_ports.append(readout_port)
        
    Darray = Darray.flatten()
    Darray.ports = {}
    for ii in range(nkids):
        Darray.ports[f'mmwave_{ii}'] = mmwave_ports[ii]
        Darray.ports[f'readout_{ii}'] = readout_ports[ii]
        
    return Darray
