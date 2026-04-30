import random
import pathlib

from ase.io import read

from care import Surface, gen_blueprint, Intermediate
from care.io import load_network

TEST_DIR = pathlib.Path(__file__).parent
crn = gen_blueprint(1, 1, False, False, False)
intermediates = crn.intermediates
adsorbed_inters = {k:v for k,v in intermediates.items() if v.phase == "ads"}
rxns = crn.reactions
surface = Surface.from_metal_db(metal="Co", hkl="0001")
surface_from_bulk = Surface.from_bulk_poscar(str(TEST_DIR) + "/files/Ni_fcc.poscar", hkl="111", num_layers=3, xy_repeat=2)
surface_from_slab = Surface.from_poscar(str(TEST_DIR) + "/files/Os0001.poscar")
co2_from_poscar = Intermediate.from_molecule(str(TEST_DIR) + "/files/CO2.poscar", code="CO2g", phase="gas")
ammonia_from_poscar = Intermediate.from_molecule(str(TEST_DIR) + "/files/NH3.poscar", code="NH3g", phase="gas")
ase_adsorbate_linear = read(str(TEST_DIR) + "/files/C3H6O3_Cu111.poscar", format="vasp")
ase_adsorbate_ring = read(str(TEST_DIR) + "/files/aromatic_Ag111.poscar", format="vasp")
ase_adsorbate_fragmented = read(str(TEST_DIR) + "/files/C3H6O3_fragmented_Cu111.poscar", format="vasp")
ase_adsorbate_oxide = read(str(TEST_DIR) + "/files/carbonic_acid_TiO2.poscar", format="vasp")
test_inters = random.sample(list(adsorbed_inters.values()), 4)
evaluated_network = load_network(str(TEST_DIR) + "/files/c1o2_Ru0001.json.gz")

__all__ = [
    "intermediates",
    "rxns",
    "surface",
    "surface_from_bulk", 
    "surface_from_slab",
    "test_inters", 
    "TEST_DIR",
    "evaluated_network",
]