"""Global constants for the GNN project"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.nn.functional import l1_loss, mse_loss, huber_loss
from torch_geometric.nn import SAGEConv, GATv2Conv, GraphMultisetTransformer
from torch.nn import ReLU, Tanh


CORDERO = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44, 'O' : 0.66,
           'Pd': 1.39, 'P' : 1.07, 'Pt': 1.36, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42, 'Rb': 2.20, 'Ru': 1.46,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22, 'Zr': 1.75}  # Atomic radii from Cordero 

# Atomic elements in the data and related one-hot encoder
MOL_ELEM = ['C', 'H', 'O', 'N', 'S']    
METALS = ['Ag', 'Au', 'Cd', 'Co', 'Cu',  
           'Fe', 'Ir', 'Ni', 'Os', 'Pd',
           'Pt', 'Rh', 'Ru', 'Zn']
CRYSTAL_STRUCTURES = ["fcc", "fcc", "hcp", "hcp", "fcc", 
                      "bcc", "fcc", "fcc", "hcp", "fcc", 
                      "fcc", "fcc", "hcp", "hcp"]
CRYSTAL_STRUCTURE_DICT = dict(zip(METALS, CRYSTAL_STRUCTURES))
NODE_FEATURES = len(MOL_ELEM) + len(METALS)
ENCODER = OneHotEncoder().fit(np.array(MOL_ELEM + METALS).reshape(-1, 1))  
ELEMENT_LIST = list(ENCODER.categories_[0])                                
FULL_ELEM_LIST = METALS + MOL_ELEM
FULL_ELEM_LIST.sort()

# Name of chemical families included in the dataset
FG_RAW_GROUPS = ["amides", "amidines", "group2", "group2b",
                 "group3S", "group3N", "group4", "carbamate_esters",
                 "oximes", "aromatics", "aromatics2",
                 "gas_amides", "gas_amidines", "gas_aromatics",
                 "gas_aromatics2", "gas_carbamate_esters", "gas_group2",
                 "gas_group2b", "gas_group3N", "gas_group3S",
                 "gas_group4", "gas_oximes"]  # Raw Datasets names defined during DFT data generation
FG_FAMILIES = ["Amides", "Amidines", "$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}O_{(0,1)}$",
               "$C_{x}H_{y}S$", "$C_{x}H_{y}N$", "$C_{x}H_{y}O_{(2,3)}$", "Carbamates",
               "Oximes", "Aromatics", "Aromatics", 
               "Amides", "Amidines", "Aromatics", 
               "Aromatics", "Carbamates", "$C_{x}H_{y}O_{(0,1)}$", 
               "$C_{x}H_{y}O_{(0,1)}$", "$C_{x}H_{y}N$", "$C_{x}H_{y}S$", 
               "$C_{x}H_{y}O_{(2,3)}$", "Oximes"]  # Proper chemical family name used in manuscipts
FAMILY_DICT = dict(zip(FG_RAW_GROUPS, FG_FAMILIES))  

# Dictionaries for model training features
loss_dict = {"mse": mse_loss,
             "mae": l1_loss,
             "huber": huber_loss}
pool_dict = {"GMT": GraphMultisetTransformer}
pool_seq_dict = {"1": ["GMPool_I"],
                 "2": ["GMPool_G"],
                 "3": ["GMPool_G", "GMPool_I"],
                 "4": ["GMPool_G", "SelfAtt", "GMPool_I"], 
                 "5": ["GMPool_G", "SelfAtt", "SelfAtt", "GMPool_I"]}
conv_layer = {"SAGE": SAGEConv,
              "GATv2": GATv2Conv}
sigma_dict = {"ReLU": ReLU(),
              "tanh": Tanh()}

# Others
DPI = 500

RGB_COLORS = {'Xx': (0.07, 0.5, 0.7), 'H': (0.75, 0.75, 0.75),
              'He': (0.85, 1.0, 1.0), 'Li': (0.8, 0.5, 1.0),
              'Be': (0.76, 1.0, 0.0), 'B': (1.0, 0.71, 0.71),
              'C': (0.4, 0.4, 0.4), 'N': (0.05, 0.05, 1.0),
              'O': (1.0, 0.05, 0.05), 'F': (0.5, 0.7, 1.0),
              'Ne': (0.7, 0.89, 0.96), 'Na': (0.67, 0.36, 0.95),
              'Mg': (0.54, 1.0, 0.0), 'Al': (0.75, 0.65, 0.65),
              'Si': (0.5, 0.6, 0.6), 'P': (1.0, 0.5, 0.0),
              'S': (0.7, 0.7, 0.0), 'Cl': (0.12, 0.94, 0.12),
              'Ar': (0.5, 0.82, 0.89), 'K': (0.56, 0.25, 0.83),
              'Ca': (0.24, 1.0, 0.0), 'Sc': (0.9, 0.9, 0.9),
              'Ti': (0.75, 0.76, 0.78), 'V': (0.65, 0.65, 0.67),
              'Cr': (0.54, 0.6, 0.78), 'Mn': (0.61, 0.48, 0.78),
              'Fe': (0.88, 0.4, 0.2), 'Co': (0.94, 0.56, 0.63),
              'Ni': (0.31, 0.82, 0.31), 'Cu': (0.78, 0.5, 0.2),
              'Zn': (0.49, 0.5, 0.69), 'Ga': (0.76, 0.56, 0.56),
              'Ge': (0.4, 0.56, 0.56), 'As': (0.74, 0.5, 0.89),
              'Se': (1.0, 0.63, 0.0), 'Br': (0.65, 0.16, 0.16),
              'Kr': (0.36, 0.72, 0.82), 'Rb': (0.44, 0.18, 0.69),
              'Sr': (0.0, 1.0, 0.0), 'Y': (0.58, 1.0, 1.0),
              'Zr': (0.58, 0.88, 0.88), 'Nb': (0.45, 0.76, 0.79),
              'Mo': (0.33, 0.71, 0.71), 'Tc': (0.23, 0.62, 0.62),
              'Ru': (0.14, 0.56, 0.56), 'Rh': (0.04, 0.49, 0.55),
              'Pd': (0.0, 0.41, 0.52), 'Ag': (0.88, 0.88, 1.0),
              'Cd': (1.0, 0.85, 0.56), 'In': (0.65, 0.46, 0.45),
              'Sn': (0.4, 0.5, 0.5), 'Sb': (0.62, 0.39, 0.71),
              'Te': (0.83, 0.48, 0.0), 'I': (0.58, 0.0, 0.58),
              'Xe': (0.26, 0.62, 0.69), 'Cs': (0.34, 0.09, 0.56),
              'Ba': (0.0, 0.79, 0.0), 'La': (0.44, 0.83, 1.0),
              'Ce': (1.0, 1.0, 0.78), 'Pr': (0.85, 1.0, 0.78),
              'Nd': (0.78, 1.0, 0.78), 'Pm': (0.64, 1.0, 0.78),
              'Sm': (0.56, 1.0, 0.78), 'Eu': (0.38, 1.0, 0.78),
              'Gd': (0.27, 1.0, 0.78), 'Tb': (0.19, 1.0, 0.78),
              'Dy': (0.12, 1.0, 0.78), 'Ho': (0.0, 1.0, 0.61),
              'Er': (0.0, 0.9, 0.46), 'Tm': (0.0, 0.83, 0.32),
              'Yb': (0.0, 0.75, 0.22), 'Lu': (0.0, 0.67, 0.14),
              'Hf': (0.3, 0.76, 1.0), 'Ta': (0.3, 0.65, 1.0),
              'W': (0.13, 0.58, 0.84), 'Re': (0.15, 0.49, 0.67),
              'Os': (0.15, 0.4, 0.59), 'Ir': (0.09, 0.33, 0.53),
              'Pt': (0.9, 0.85, 0.68), 'Au': (0.8, 0.82, 0.12),
              'Hg': (0.71, 0.71, 0.76), 'Tl': (0.65, 0.33, 0.3),
              'Pb': (0.34, 0.35, 0.38), 'Bi': (0.62, 0.31, 0.71),
              'Po': (0.67, 0.36, 0.0), 'At': (0.46, 0.31, 0.27),
              'Rn': (0.26, 0.51, 0.59), 'Fr': (0.26, 0.0, 0.4),
              'Ra': (0.0, 0.49, 0.0), 'Ac': (0.44, 0.67, 0.98),
              'Th': (0.0, 0.73, 1.0), 'Pa': (0.0, 0.63, 1.0),
              'U': (0.0, 0.56, 1.0), 'Np': (0.0, 0.5, 1.0),
              'Pu': (0.0, 0.42, 1.0), 'Am': (0.33, 0.36, 0.95),
              'Cm': (0.47, 0.36, 0.89), 'Bk': (0.54, 0.31, 0.89),
              'Cf': (0.63, 0.21, 0.83), 'Es': (0.7, 0.12, 0.83),
              'Fm': (0.7, 0.12, 0.73), 'Md': (0.7, 0.05, 0.65),
              'No': (0.74, 0.05, 0.53), 'Lr': (0.78, 0.0, 0.4),
              'Rf': (0.8, 0.0, 0.35), 'Db': (0.82, 0.0, 0.31),
              'Sg': (0.85, 0.0, 0.27), 'Bh': (0.88, 0.0, 0.22),
              'Hs': (0.9, 0.0, 0.18), 'Mt': (0.92, 0.0, 0.15),
              'Ds': (0.93, 0.0, 0.14), 'Rg': (0.94, 0.0, 0.13),
              'Cn': (0.95, 0.0, 0.12), 'Nh': (0.96, 0.0, 0.11),
              'Fl': (0.97, 0.0, 0.1), 'Mc': (0.98, 0.0, 0.09),
              'Lv': (0.99, 0.0, 0.08), 'Ts': (0.99, 0.0, 0.07),
              'Og': (0.99, 0.0, 0.06)}