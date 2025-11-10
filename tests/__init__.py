import random
    
from care import Surface, gen_blueprint
from care.evaluators import NEBReactionEnergyEstimator, MACEIntermediateEvaluator

intermediates, rxns = gen_blueprint(1, 1, False, False, False)
surface = Surface.from_metal_db(metal="Co", hkl="0001")
test_inters = random.sample(list(intermediates.values()), 4)
mlp = MACEIntermediateEvaluator(surface=surface, size="small", max_steps=5)
neb = NEBReactionEnergyEstimator(mlp=mlp)

__all__ = [
    "intermediates",
    "rxns",
    "surface",
    "test_inters",
    "mlp",
    "neb"
]