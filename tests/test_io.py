import unittest

from care.io import save_network, load_network
from tests.shared_data import TEST_DIR

class TestIO(unittest.TestCase):
    def test_reconstructed_network(self):
        crn = load_network(str(TEST_DIR) + "/files/c1o2_Ru0001_evaluated.json.gz")
        save_network(crn, str(TEST_DIR) + "/files/c1o2_Ru0001_new.json")
        crn_reconstructed = load_network(str(TEST_DIR) + "/files/c1o2_Ru0001_new.json.gz")
        self.assertEqual(crn, crn_reconstructed)

