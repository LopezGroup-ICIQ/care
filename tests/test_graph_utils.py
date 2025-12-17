import unittest

from networkx import Graph
from networkx.utils import graphs_equal

from care.evaluators.utils import atoms_to_data, extract_adsorbate, get_connectivity_dict, connectivity_signature, is_adsorbate_fragmented, is_ring
from tests import ase_adsorbate_linear as x1
from tests import ase_adsorbate_ring as x2
from tests import ase_adsorbate_fragmented as x3

x11 = [0] * 48 + [1] * 12
x22 = [0] * 48 + [1] * 15
x33 = x11

g1o1 = atoms_to_data(x1, x11, 1)
g2o1 = atoms_to_data(x2, x22, 1)
g1o2 = atoms_to_data(x1, x11, 2)
g2o2 = atoms_to_data(x2, x22, 2)
g1om1 = atoms_to_data(x1, x11, -1)
g2om1 = atoms_to_data(x2, x22, -1)
ads1_o1 = extract_adsorbate(g1o1)
ads2_o1 = extract_adsorbate(g2o1)
ads1_o2 = extract_adsorbate(g1o2)
ads2_o2 = extract_adsorbate(g2o2)
ads1_om1 = extract_adsorbate(g1om1)
ads2_om1 = extract_adsorbate(g2om1)

sig1_o1 = connectivity_signature(g1o1)
sig2_o1 = connectivity_signature(g2o1)
condict_o1 = get_connectivity_dict(x1, x11)
condict_o2 = get_connectivity_dict(x2, x22)

g_fragmented_nofilter = atoms_to_data(x3, x33, 1, filter=False)
g_fragmented = atoms_to_data(x3, x33, 1, filter=True)

class TestGraphUtils(unittest.TestCase):
    def test_graph_gen(self):
        self.assertIsInstance(g1o1, Graph)
        self.assertIsInstance(g2o1, Graph)
        self.assertIsInstance(g1o2, Graph)
        self.assertIsInstance(g2o2, Graph)
        self.assertIsInstance(g1om1, Graph)
        self.assertIsInstance(g2om1, Graph)
        self.assertIsInstance(ads1_o1, Graph)
        self.assertIsInstance(ads2_o1, Graph)
        self.assertIsInstance(ads1_o2, Graph)
        self.assertIsInstance(ads2_o2, Graph)
        self.assertIsInstance(ads1_om1, Graph)
        self.assertIsInstance(ads2_om1, Graph)
        self.assertIsNone(g_fragmented)
        graphs_equal(ads1_o1, ads1_o2)
        graphs_equal(ads2_o1, ads2_o2)
        graphs_equal(ads1_o1, ads1_om1)
        graphs_equal(ads2_o1, ads2_om1)
        self.assertEqual(len(g1om1), 60)
        self.assertEqual(len(g2om1), 63)
        self.assertEqual(len(ads1_o1), 12)
        self.assertEqual(len(ads2_o1), 15)
        self.assertGreater(len(g1o2), len(g1o1))
        self.assertGreater(len(g2o2), len(g2o1))
        self.assertGreater(len(g1om1), len(g1o2))
        self.assertGreater(len(g2om1), len(g2o2))
        self.assertFalse(is_adsorbate_fragmented(g1o1))
        self.assertFalse(is_adsorbate_fragmented(g2o1))
        self.assertTrue(is_adsorbate_fragmented(g_fragmented_nofilter))
        self.assertFalse(is_ring(ads1_o1))
        self.assertTrue(is_ring(ads2_o1))

    def test_connectivity_dict(self):
        self.assertIsInstance(sig1_o1, list)
        self.assertIsInstance(sig2_o1, list)
        self.assertIsInstance(condict_o1, dict)
        self.assertIsInstance(condict_o2, dict)
