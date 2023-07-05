import os
import pickle
import utilities.additional_funcs as af
from collections import namedtuple
import os
import db.data_extraction as de
import matplotlib as mpl
from networks.networks import Intermediate, OrganicNetwork
import networks.network_functions as nf
from pyRDTP import geomio


def gen_organic_network(SYSTEM, pack_dict, map_dict, eth_net):
    print('Generating the Organic Network...')

    MolPack = namedtuple('MolPack',['code', 'mol','graph', 'subs'])
    
    main_path = os.getcwd()
    obj_path = SYSTEM + '/objects/'
    db_data_path = f'{SYSTEM}/db/'

    pkl_obj_path = os.path.join(main_path, obj_path)
    db_path = os.path.join(main_path,db_data_path)

    # Path to the database pickle file, properties to extract and the corresponding term
    intermediates_db = os.path.join(db_path, 'intermediates.pkl')
    intermediate_properties = ['e_zpe_solv', 'S_meV']
    term = 't_0'

    # Getting the energies and entropies from the intermediates database
    energ_entr_dict = de.read_db(pack_dict, intermediates_db, term, intermediate_properties)
 
    # Assigning the energies to intermediates

    # Making the surface and hydrogen binding intermediates
    surface = geomio.file_to_mol(f'./{SYSTEM}/dockonsurf/inputs_surface/Cu/CONTCAR', 'contcar')
    hydrogen = geomio.file_to_mol(f'./{SYSTEM}/dockonsurf/inputs_surface/H2/CONTCAR', 'contcar')['H'] # Select only H to avoid surface
    ads_h_dft = -129.49244499
    gas_h_dft = -6.76639487
    surface_dft = -125.91841662
    hydrogen_ener = (ads_h_dft - (gas_h_dft/2 + surface_dft))
    surf_inter = Intermediate.from_molecule(surface, code='000000', energy=-126.000, entropy=0, is_surface=True, phase='surface')
    surf_inter.electrons = 0
    h_inter = Intermediate.from_molecule(hydrogen, code='010101', energy=-3.25, entropy=0.55, phase='cat')
    h_inter.electrons = 1

    # Making the network again with the graphs and initializing the energies and entropies???
    lot1_att = nf.generate_dict(pack_dict, map_dict)
    network_dict = nf.generate_network_dict(map_dict, surf_inter, h_inter)
    updated_dict = nf.add_energies_to_dict(network_dict, energ_entr_dict)

    # Reaction Network Generation

    # Define the components of the network
    # eth_net = [key for key in updated_dict.keys()]

    ## Generating the C2 network

    # Generate the electronic intermediates
    electron_inter = Intermediate(code='e-')
    proton_inter = Intermediate(code='H+')

    # Generate the C2 OrganicNetwork object
    one_net = OrganicNetwork()
    for item in eth_net:
        select_net = network_dict[item]
        one_net.add_intermediates(select_net['intermediates'])
        one_net.add_intermediates({'000000': surf_inter, '010101': h_inter})
        one_net.add_ts(select_net['ts'])

    # Why?????
    for inter in one_net.intermediates.values():
        inter.molecule.connection_clear()
        inter.molecule.connectivity_search_voronoi()

    # Detect the O-H bond breaks, some problems
    # In this step, the O-H breakings are separated from the C-H breakings inherited from the previous network

    nf.oh_bond_breaks(one_net.t_states)

    # Generating the breaking TSs

    # This step search possible breakages between the C-O and C-C elements that form reactives that are already intermediates of the network. 
    # When a connection is found a transition state is created to represent this reaction in the network.
    breaking_ts = af.break_and_connect(one_net, surface=surf_inter)

    # Now we add the new TSs to the network
    one_net.add_ts(breaking_ts)


    path_dict_neb = f'{SYSTEM}/db/dict_neb.pkl'
    path_neb_db_pkl = f'{SYSTEM}/db/neb.pkl'

    neb_dict_file = os.path.join(main_path, path_dict_neb)
    neb_db_pkl = os.path.join(main_path, path_neb_db_pkl)

    with open(neb_dict_file, 'rb') as f:
        neb_dict = pickle.load(f)

    with open(neb_db_pkl, 'rb') as f:
        neb_df = pickle.load(f)

    ts_states = one_net.t_states

    nf.ts_energies(ts_states, neb_dict, neb_df, surf_inter)

    one_net.search_connections()

    ### Adding gas phase energies to the Organic Network

    # Path to the gas phase database
    path_gas_db_pkl = f'{SYSTEM}/db/gas.pkl'
    gas_db_pkl = os.path.join(main_path, path_gas_db_pkl)

    with open(gas_db_pkl, 'rb') as f:
        gas_df = pickle.load(f)

    # Generating gas dict
    gas_dict = nf.gen_gas_dict(gas_df)

    # Generating the gas inter dict
    gas_inter_dict = nf.gen_gas_inter_dict(gas_dict)

    one_net.add_gasses(gas_inter_dict)

    ##### Adding electrochemical components, how is this important? #####

    # What is this????
    # for ts in one_net.t_states:
    #     for comp in ts.full_order():
    #         for item in comp:
    #             if item not in one_net.intermediates:
    #                 print(item, ts)

    wa_inter = one_net.intermediates['021101']
    e_inter = one_net.intermediates['001101']
    proton = Intermediate(code= h_inter.code, molecule=h_inter.molecule, graph=h_inter.graph, energy=0, electrons=0, phase='gas')
    water = Intermediate(code='021101', molecule=wa_inter.molecule, graph=wa_inter.graph, energy=0, electrons=0, phase='gas')
    electron = Intermediate(code='e-', molecule=e_inter.molecule, graph=e_inter.graph, energy=0, electrons=0, phase='gas')

    # What is happening??
    ts = af.search_electro_ts(one_net, proton=proton, water=water, electron=electron)
    ts_new = []
    comp_new = []
    for item in ts:
        if item.components not in comp_new:
            comp_new.append(item.components)
            ts_new.append(item)


    vals = [inter.energy for inter in one_net.intermediates.values()]
    min_val = min(vals)
    max_val = max(vals)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val+1)
    print('Normalized within the range of {:.3f}|{:.3f} eV'.format(min_val, max_val+1))
    data_dir = os.path.join('./', f'{SYSTEM}/data/')
    os.makedirs(data_dir, exist_ok=True)
    data_ts_dir = data_dir+'ts/'
    os.makedirs(data_ts_dir, exist_ok=True)
    af.print_t_states(one_net, filename=f'{data_ts_dir}/ts_1.dat')
    return one_net