from torch_geometric.data import InMemoryDataset, Data
import torch
import numpy as np

from gnn_eads.constants import ENCODER, FAMILY_DICT, METALS
from gnn_eads.graph_filters import global_filter, isomorphism_test
from gnn_eads.functions import get_graph_formula

class FGGraphDataset(InMemoryDataset):
    """
    Class for storing the graphs of a specific family of the FG-dataset (e.g., aromatic, amides, etc.)
    InMemoryDataset is the abstract class for creating custom datasets.
    Each dataset gets passed a dataset folder which indicates where the dataset should
    be stored. The dataset folder is split up into 2 folders, a raw_dir where the dataset gets downloaded to,
    and the processed_dir, where the processed dataset is being saved.
    In order to create a InMemoryDataset class, four fundamental methods must be provided:
    - raw_file_names(): a list of files in the raw_dir which needs to be found in order to skip the download
    - file_names(): a list of files in the processed_dir which needs to be found in order to skip the processing
    - download(): download raw data into raw_dir
    - process(): process raw_data and saves it into the processed_dir
    """
    def __init__(self,
                 root,
                 identifier: str):
        self.root = str(root)
        self.pre_data = str(root) + "/pre_" + identifier
        self.post_data = str(root) + "/post_" + identifier
        super().__init__(str(root))
        self.data, self.slices = torch.load(self.processed_paths[0])             

    @property
    def raw_file_names(self): 
        return self.pre_data 
    
    @property
    def processed_file_names(self): 
        return self.post_data
    
    def download(self):
        pass
    
    def process(self):  
        """
        If self.processed_file_names() does not exist, this method is run automatically to process the raw data
        starting from the path provided in self.raw_file_names() 
        """
        data_list = []
        dataset_name = self.root.split("/")[-1]
        with open(self.raw_file_names, 'r') as infile:
            lines = infile.readlines()
        split_n = lambda x, n: [x[i:i+n] for i in range(0, len(x), n)]
        splitted = split_n(lines, 5)  # Each sample =  5 text lines
        for block in splitted:        
            to_int = lambda x: [float(i) for i in x]
            _, elem, source, target, energy = block
            element_list = elem.split()
            if dataset_name[:3] != "gas":  # filter for graphs with no metal
                counter = 0
                for element in element_list:
                    if element in METALS:
                        counter += 1
                if counter == 0:
                    continue                     
            elem_array = np.array(elem.split()).reshape(-1, 1)
            elem_enc = ENCODER.transform(elem_array).toarray()
            x = torch.tensor(elem_enc, dtype=torch.float)         # Node feature matrix
            edge_index = torch.tensor([to_int(source.split()),    # Edge list COO format
                                       to_int(target.split())],
                                       dtype=torch.long)       
            y = torch.tensor([float(energy)], dtype=torch.float)  # Graph label (Edft - Eslab)
            family = FAMILY_DICT[dataset_name]                    # Chemical family of the adsorbate/molecule
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family)
            graph_formula = get_graph_formula(data, ENCODER.categories_[0])
            data = Data(x=x, edge_index=edge_index, y=y, ener=y, family=family, formula=graph_formula)
            if global_filter(data):  # To ensure correct adsorbate representation in the graph
                if isomorphism_test(data, data_list):  # To ensure absence of duplicates graphs
                     data_list.append(data)              
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_post_processed_datasets(identifier: str,
                                   paths: dict):
    """Create the graph FG-dataset. 

    Args:
        identifier (str): Graph settings identifier
        paths (dict): Data paths dictionary, each key is the family name (str)
                      and the value a dictionary of paths

    Returns:
        FG_dataset (tuple[HetGraphDataset]): FG_dataset
    """
    
    aromatics_dataset = FGGraphDataset(str(paths['aromatics']['root']), identifier)
    group2_dataset = FGGraphDataset(str(paths['group2']['root']), identifier)
    group2b_dataset = FGGraphDataset(str(paths['group2b']['root']), identifier)
    aromatics2_dataset = FGGraphDataset(str(paths['aromatics2']['root']), identifier)
    carbamate_esters_dataset = FGGraphDataset(str(paths['carbamate_esters']['root']), identifier)
    group3N_dataset = FGGraphDataset(str(paths['group3N']['root']), identifier)
    group3S_dataset = FGGraphDataset(str(paths['group3S']['root']), identifier)
    group4_dataset = FGGraphDataset(str(paths['group4']['root']), identifier)
    amides_dataset = FGGraphDataset(str(paths['amides']['root']), identifier)
    amidines_dataset = FGGraphDataset(str(paths['amidines']['root']), identifier)
    oximes_dataset = FGGraphDataset(str(paths['oximes']['root']), identifier)
    gas_amides_dataset = FGGraphDataset(str(paths['gas_amides']['root']), identifier)
    gas_amidines_dataset = FGGraphDataset(str(paths['gas_amidines']['root']), identifier)
    gas_aromatics_dataset = FGGraphDataset(str(paths['gas_aromatics']['root']), identifier)
    gas_aromatics2_dataset = FGGraphDataset(str(paths['gas_aromatics2']['root']), identifier)
    gas_group2_dataset = FGGraphDataset(str(paths['gas_group2']['root']), identifier)
    gas_group2b_dataset = FGGraphDataset(str(paths['gas_group2b']['root']), identifier)
    gas_group3N_dataset = FGGraphDataset(str(paths['gas_group3N']['root']), identifier)
    gas_group3S_dataset = FGGraphDataset(str(paths['gas_group3S']['root']), identifier)
    gas_carbamate_esters_dataset = FGGraphDataset(str(paths['gas_carbamate_esters']['root']), identifier)
    gas_oximes_dataset = FGGraphDataset(str(paths['gas_oximes']['root']), identifier)
    gas_group4_dataset = FGGraphDataset(str(paths['gas_group4']['root']), identifier)
    FG_dataset = (group2_dataset,
               group2b_dataset,
               aromatics_dataset,
               aromatics2_dataset,
               amides_dataset,
               amidines_dataset,
               oximes_dataset,
               carbamate_esters_dataset,
               group3S_dataset,
               group3N_dataset,
               group4_dataset,
               gas_amides_dataset,
               gas_amidines_dataset,
               gas_aromatics_dataset,
               gas_aromatics2_dataset,
               gas_carbamate_esters_dataset,
               gas_group2_dataset,
               gas_group2b_dataset,
               gas_group3N_dataset,
               gas_group3S_dataset,
               gas_group4_dataset,
               gas_oximes_dataset) 
    return FG_dataset
