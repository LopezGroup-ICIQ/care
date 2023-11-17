from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image
import math

def is_desired_bond(bond, atom_num1, atom_num2):
    return ((bond.GetBeginAtom().GetAtomicNum() == atom_num1 and bond.GetEndAtom().GetAtomicNum() == atom_num2) or
            (bond.GetBeginAtom().GetAtomicNum() == atom_num2 and bond.GetEndAtom().GetAtomicNum() == atom_num1))

def get_chemical_formula(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return rdMolDescriptors.CalcMolFormula(mol)

def break_bonds(molecule, bond_types, processed_fragments, depth=0, max_depth=10):
    if depth > max_depth:
        return

    for bond in molecule.GetBonds():
        for atom_num1, atom_num2 in bond_types:
            if is_desired_bond(bond, atom_num1, atom_num2):
                mol_copy = Chem.RWMol(molecule)
                mol_copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

                frags = Chem.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=False)
                for frag in frags:
                    frag_smiles = Chem.MolToSmiles(frag, isomericSmiles=True, allHsExplicit=True)
                    frag_formula = get_chemical_formula(frag_smiles)

                    if frag_smiles not in processed_fragments:
                        processed_fragments.add(frag_smiles)
                        processed_fragments_info[frag_smiles] = frag_formula

                        # Recursively break bonds in the fragments
                        frag_mol = Chem.MolFromSmiles(frag_smiles, sanitize=False)
                        break_bonds(frag_mol, bond_types, processed_fragments, depth=depth+1, max_depth=max_depth)

def process_molecule(smiles, bond_types, processed_fragments, processed_fragments_info):
    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_H = Chem.AddHs(molecule)
    break_bonds(molecule_with_H, bond_types, processed_fragments)

# List of SMILES to process
smiles_list = ['CO','C(O)O','O', 'OO', '[H][H]']  # Add your SMILES strings here

# Define bond types
bond_types = [(6, 6), (6, 1), (6, 8), (8, 8), (1, 1), (8, 1)]

# Sets to keep track of processed fragments and their information
processed_fragments = set()
processed_fragments_info = {}

# Process each molecule in the list
for smiles in smiles_list:
    process_molecule(smiles, bond_types, processed_fragments, processed_fragments_info)

# Extract the unique SMILES strings from processed_fragments_info
unique_molecules = list(processed_fragments_info.keys())

# Print the list of unique molecules
print("Unique Molecules:")
for molecule in unique_molecules:
    print(molecule)

# Convert the unique SMILES strings to RDKit Molecule objects with explicit hydrogens
molecules = [Chem.MolFromSmiles(smiles) for smiles in unique_molecules]

# Prepare drawing options to show explicit hydrogens
options = Draw.MolDrawOptions()
options.includeAtomNumbers = False
options.explicitMethyl = True
options.explicitOnly = True
options.explicitHs = True

# Generate drawings for each unique molecule
drawings = [Draw.MolToImage(mol, size=(200, 200), options=options) for mol in molecules]

# Determine grid size for the plot
grid_size = math.ceil(math.sqrt(len(drawings)))

# Create a blank image for the grid
grid_image = Image.new('RGB', (200 * grid_size, 200 * grid_size), 'white')

# Place each molecule drawing in the grid
for i, img in enumerate(drawings):
    row = i // grid_size
    col = i % grid_size
    grid_image.paste(img, (col * 200, row * 200))

# Save or display the grid image
grid_image.save("unique_molecules_grid.png")
grid_image.show()