import pubchempy as pcp
import os
import json as js
# Loading parameter file
with open('data/parameters.json') as f:
    params = js.load(f)

ADD_H_LABEL = params['ADD_H_LABEL']

# Function that explores the PubChem database for a given compound and returns the XYZ file
# Old explore_db function
def get_xyz_pubchem(data_dict: dict, TMP_DIR) -> None:
    """Get the structure (XYZ file) for the updated systems of the dictionary.
    Updaed systems: Those where hydrogens where added to the graph.

    Parameters
    ----------
    data_dict : dict
        Dictionary containing the molecular formulas and graphs for all the systems.
    TMP_DIR : _type_
        Temporary path where the XYZ files are stored.
    """

    # Making the temporary directory
    os.makedirs(TMP_DIR, exist_ok=True)

    updated_molecules = [molec for molec in data_dict.keys() if molec.endswith(ADD_H_LABEL)]
    
    # Iterating through updated molecules in the dictionary
    for molec in updated_molecules:
        formula = data_dict[molec]['formula']
        pubchem_compounds = pcp.get_compounds(formula, 'formula', record_type='3d', listkey_count=20)
        # Iterating through the PubChem compounds
        for compound in pubchem_compounds:
            pubchem_cid = str(compound.cid)
            pubchem_atoms = compound.atoms
            n_atoms = len(pubchem_atoms)
            # Generating the XYZ file
            with open(TMP_DIR + "/" + str(pubchem_cid) + ".xyz", "w") as f:
                    f.write(f"{n_atoms}\n\n")
                    for atom in pubchem_atoms:
                        f.write(f"{atom.element}\t{atom.x} {atom.y} {atom.z}\n")