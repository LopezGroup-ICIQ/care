import random
    
from care import Surface, gen_blueprint

intermediates, rxns = gen_blueprint(1, 1, False, False, False)
surface = Surface.from_metal_db(metal="Co", hkl="0001")
test_inters = random.sample(list(intermediates.values()), 4)

__all__ = [
    "intermediates",
    "rxns",
    "surface",
    "test_inters"
]