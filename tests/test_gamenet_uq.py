import unittest

from care.crn.utils.blueprint import gen_blueprint
from care.evaluators.gamenet_uq import load_surface, GameNetUQInter, METALS, METAL_STRUCT_DICT, FACET_DICT

intermediates, _ = gen_blueprint(1, 1, False, False, False)
surface = load_surface("Pt", "111")
interface = GameNetUQInter(surface)

class TestEvaluator(unittest.TestCase):

    def test_surface(self):
        """Check that all surfaces have active sites
        """
        for metal in METALS:
            for facet in FACET_DICT[METAL_STRUCT_DICT[metal]]:
                surface = load_surface(metal, facet)
                assert surface.num_atoms != 0
                assert len(surface.active_sites) != 0

    def test_model(self):
        assert interface.model.parameters() != None

    def test_serial_eval(self):
        for inter in intermediates.values():
            if inter.phase == 'surf':
                continue
            interface.eval(inter)
            assert len(inter.ads_configs) != 0
