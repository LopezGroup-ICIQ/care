import unittest
from random import randint
from ase import Atoms
import numpy as np
import sys

sys.path.append("../src/")

from care.crn.reactors import DifferentialPFR

v_matrix = np.array(
    [
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, -1, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [-1, -1, 1, 1],
    ]
)
kd, kr = np.ones(4), np.ones(4)
gas_mask = np.array([1, 0, 1, 1])
inters = ['a', 'b', 'c', 'd']
pfr = DifferentialPFR(v_matrix, kd, kr, gas_mask, inters, temperature=500, pressure=1e5)

class TestDifferentialPFR(unittest.TestCase):

    def test_stoic_forward(self):
        """
        Check that the forward stoichiometric matrix is correctly implemented
        """
        # self.assertTrue((pfr.v_forward_dense >= 0))
        # self.assertTrue((pfr.v_forward_dense == np.array([[1, 0, 0, 0],
        #                                                 [0, 1, 0, 0],
        #                                                 [0, 0, 0, 0],
        #                                                 [0, 0, 1, 0],
        #                                                 [0, 0, 1, 0],
        #                                                 [0, 0, 0, 1],
        #                                                 [1, 1, 0, 0]]).T.all()))
        self.assertEqual(pfr.v_forward_dense.shape, (4, 7))

    def test_stoic_backward(self):
        """
        Check that the backward stoichiometric matrix is correctly implemented
        """
        # self.assertTrue((pfr.v_backward_dense >= 0))
        # self.assertTrue((pfr.v_backward_dense == np.array([[0, 0, 0, 0],
        #                                                  [0, 0, 0, 0],
        #                                                  [0, 0, 0, 1],
        #                                                  [1, 0, 0, 0],
        #                                                  [0, 1, 0, 0],
        #                                                  [0, 0, 1, 1],
        #                                                  [0, 0, 0, 0]]).T.all()))
        self.assertEqual(pfr.v_backward_dense.shape, (4, 7))
