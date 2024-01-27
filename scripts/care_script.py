import os
import toml

from ase.db import connect

from care import MODEL_PATH, DB_PATH, Surface
from care.constants import METAL_STRUCT_DICT
from care.crn.utilities.chemspace import gen_chemical_space
from care.gnn.interface import GameNetUQ, ReactionEnergy


def main():
    # Load configuration file
    with open("../notebooks/config.toml", "r") as f:
        config = toml.load(f)

    care_logo = """
              _____                    _____                    _____                    _____          
             /\    \                  /\    \                  /\    \                  /\    \         
            /::\    \                /::\    \                /::\    \                /::\    \        
           /::::\    \              /::::\    \              /::::\    \              /::::\    \       
          /::::::\    \            /::::::\    \            /::::::\    \            /::::::\    \      
         /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \          /:::/\:::\    \     
        /:::/  \:::\    \        /:::/__\:::\    \        /:::/__\:::\    \        /:::/__\:::\    \    
       /:::/    \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \      /::::\   \:::\    \   
      /:::/    / \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \    /::::::\   \:::\    \  
     /:::/    /   \:::\    \  /:::/\:::\   \:::\    \  /:::/\:::\   \:::\____\  /:::/\:::\   \:::\    \ 
    /:::/____/     \:::\____\/:::/  \:::\   \:::\____\/:::/  \:::\   \:::|    |/:::/__\:::\   \:::\____\\
    \:::\    \      \::/    /\::/    \:::\  /:::/    /\::/   |::::\  /:::|____|\:::\   \:::\   \::/    /
     \:::\    \      \/____/  \/____/ \:::\/:::/    /  \/____|:::::\/:::/    /  \:::\   \:::\   \/____/ 
      \:::\    \                       \::::::/    /         |:::::::::/    /    \:::\   \:::\    \     
       \:::\    \                       \::::/    /          |::|\::::/    /      \:::\   \:::\____\    
        \:::\    \                      /:::/    /           |::| \::/____/        \:::\   \::/    /    
         \:::\    \                    /:::/    /            |::|  ~|               \:::\   \/____/     
          \:::\    \                  /:::/    /             |::|   |                \:::\    \         
           \:::\____\                /:::/    /              \::|   |                 \:::\____\        
            \::/    /                \::/    /                \:|   |                  \::/    /        
             \/____/                  \/____/                  \|___|                   \/____/ """

    print(care_logo)
    print("\nInitializing CARE (Catalytic Automatic Reaction Estimator)...\n")

    # Loading parameters
    ncc = config['chemspace']['ncc']
    noc = config['chemspace']['noc']
    # If noc is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    noc = noc if noc > 0 else ncc * 2 + 2

    # If noc is > ncc * 2 + 2, raise an error
    if noc > ncc * 2 + 2:
        raise ValueError("The noc value cannot be greater than ncc * 2 + 2.")

    metal = config['surface']['metal']
    hkl = config['surface']['hkl']

    # Loading surface from database
    metal_db = connect(os.path.abspath(DB_PATH))
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    surface_ase = metal_db.get_atoms(
        calc_type="surface", metal=metal, facet=metal_structure)
    surface = Surface(surface_ase, str(hkl))

    # Create output directory
    output_dir = f"C{ncc}O{noc}/{metal}{hkl}"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Generate the chemical space (chemical spieces and reactions)
    print("Generating the chemical space...")

    intermediates, reactions = gen_chemical_space(ncc, noc)

    print("Chemical space generated.\n")

    # 2. Evaluation of the chemical space
    print("Evaluating the chemical space...")

    # 2.1. Adsorbate placement and energy estimation
    print("Placing the adsorbates and estimating its energies...")

    # Loading the model
    model = GameNetUQ(MODEL_PATH, surface)

    # Estimate the energy of each intermediate
    for intermediate in intermediates.values():
        model.estimate_energy(intermediate)

    print("Adsorbates placed and energies estimated.\n")

if __name__ == "__main__":
    main()
