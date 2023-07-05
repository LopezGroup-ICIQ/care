from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt

import time
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

import numpy as np
import torch
from torch_geometric.data import Data

from gnn_eads.constants import ENCODER
from gnn_eads.graph_tools import convert_gpytorch_to_networkx
from gnn_eads.functions import surf
from gnn_eads.nets import PreTrainedModel

from networkx.drawing.nx_pydot import to_pydot

import json

@csrf_exempt
def index(request):
    print(request.body)
    json_object = json.loads(request.body)
    
    # 1) Load model
    MODEL_PATH = "./models/GAME-Net"
    model = PreTrainedModel(MODEL_PATH)

    elem = list(json_object["atoms"])
    source = list(json_object["bonds"]["start"])
    target = list(json_object["bonds"]["end"])
    isadsortion = (json_object.get("adsorption") != None)

    try:
        if isadsortion:
            elem_array_ads = np.array(elem).reshape(-1, 1)
            elem_enc_ads = ENCODER.transform(elem_array_ads).toarray()
            edge_index_ads = torch.tensor([source, target], dtype=torch.long)
            x_ads = torch.tensor(elem_enc_ads, dtype=torch.float)
            adsorbate_graph = Data(x=x_ads, edge_index=edge_index_ads)
            #Append provided adsorbate information
            metal = json_object["adsorption"]["metal"]
            atomnumber = json_object["adsorption"]["atomnumber"]
            source.extend(list(json_object["adsorption"]["adsorbateLinks"]["source"]))
            target.extend(list(json_object["adsorption"]["adsorbateLinks"]["target"]))
            for n in range(atomnumber):
                elem.append(metal)

            elem_array = np.array(elem).reshape(-1, 1)
            elem_enc = ENCODER.transform(elem_array).toarray()

            x = torch.tensor(elem_enc, dtype=torch.float)

            edge_index = torch.tensor([source, target], dtype=torch.long)
            ads_graph = Data(x=x, edge_index=edge_index)
            nx_ads_graph = convert_gpytorch_to_networkx(ads_graph)
            dot = to_pydot(nx_ads_graph).to_string()

            time0 = time.time()
            E_adsorbate = model.evaluate(adsorbate_graph)
            E_ensemble = model.evaluate(ads_graph)
            E_adsorption = E_ensemble - E_adsorbate
            gnn_time = time.time() - time0
            print("System: {} on {}-({})".format("Custom", metal.capitalize(), surf(metal.capitalize())))
            print("Ensemble energy = {:.2f} eV (PBE + VdW)".format(E_ensemble))
            print("Molecule energy = {:.2f} eV (PBE + VdW)".format(E_adsorbate))
            print("Adsorption energy = {:.2f} eV".format(E_adsorption))
            print("Execution time = {:.2f} ms".format(gnn_time *1000.0))

            response = {'eenergy': E_ensemble, 
                        'menergy': E_adsorbate,
                        'edsorption': E_adsorption,
                        'elapsed': gnn_time * 1000.0, 
                        'graph': dot }
            return HttpResponse(json.dumps(response, indent=4))
        else:
            elem_array = np.array(elem).reshape(-1, 1)
            elem_enc = ENCODER.transform(elem_array).toarray()
            edge_index = torch.tensor([source, target], dtype=torch.long)
            x = torch.tensor(elem_enc, dtype=torch.float)
            gas_graph = Data(x=x, edge_index=edge_index)
            nx_ads_graph = convert_gpytorch_to_networkx(gas_graph)
            dot = to_pydot(nx_ads_graph).to_string()

            time0 = time.time()
            gnn_energy = model.evaluate(gas_graph)
            gnn_time = time.time() - time0
            print("System: {} (gas phase)".format("Custom"))
            print("Molecule energy = {:.2f} eV (PBE + VdW)".format(gnn_energy))
            print("Execution time = {:.2f} ms".format(gnn_time * 1000.0))
            response = {'menergy': gnn_energy, 
                        'elapsed': gnn_time * 1000.0,
                        'graph': dot }
            return HttpResponse(json.dumps(response, indent=4))
    except Exception as error:
        print(error)
        return HttpResponse("An error occurred while processing your request")   
