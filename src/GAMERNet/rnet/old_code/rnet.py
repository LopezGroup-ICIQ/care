import networks.gen_init_network as gin
import networks.gen_organic_network as gon
import mkm.gen_mkm_files as gmk
import pickle

# print("Working folder:")
# SYSTEM = input()
SYSTEM = "example"

# Generating the initial network
pack_dict, map_dict = gin.gen_init_network(SYSTEM)

# List of all the molecules in the network
mol_list = list(pack_dict.keys())
print(mol_list)

org_net_inputs = [mol_list[0], mol_list[1], mol_list[2], mol_list[3], mol_list[4], mol_list[5], mol_list[6], mol_list[7], mol_list[8]]

organic_network = gon.gen_organic_network(SYSTEM, pack_dict, map_dict, org_net_inputs)

#### The following lines are for exporting the organic network as a pickle file:
#### Testing the export and import of the organic network as a pickle file 

export_org_net = organic_network.to_dict()

with open(f"data/organic_network_test.obj", "wb") as outfile:
        pickle.dump(export_org_net, outfile)
with open(f"data/organic_network_test.obj", 'rb') as exp_net:
        organic_network_pkl = pickle.load(exp_net)

new_org_net = gon.OrganicNetwork()
org_2 = new_org_net.from_dict(organic_network_pkl)
