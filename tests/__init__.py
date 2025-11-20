import random
import pathlib
    
from care import Surface, gen_blueprint, Intermediate
from care.evaluators import NEBReactionEnergyEstimator, MACEIntermediateEvaluator

TEST_DIR = pathlib.Path(__file__).parent
intermediates, rxns = gen_blueprint(1, 1, False, False, False)
surface = Surface.from_metal_db(metal="Co", hkl="0001")
surface_from_bulk = Surface.from_bulk_poscar(str(TEST_DIR) + "/files/Ni_fcc.poscar", hkl="111", num_layers=3, xy_repeat=2)
surface_from_slab = Surface.from_poscar(str(TEST_DIR) + "/files/Os0001.poscar")
inter_from_poscar = Intermediate.from_molecule(str(TEST_DIR) + "/files/CO2.poscar", code="CO2g", phase="gas")
test_inters = random.sample(list(intermediates.values()), 4)
mlp = MACEIntermediateEvaluator(surface=surface, size="small", max_steps=5)
neb = NEBReactionEnergyEstimator(mlp=mlp)

__all__ = [
    "intermediates",
    "rxns",
    "surface",
    "surface_from_bulk", 
    "surface_from_slab",
    "test_inters",
    "mlp",
    "neb", 
    "TEST_DIR",
]