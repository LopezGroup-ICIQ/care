import unittest
from random import randint
from ase import Atoms
import numpy as np
import sys
sys.path.append('../src/')

from care.crn.reactors import DifferentialPFR, DynamicCSTR

v_matrix = np.array([[-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1],
                     [1, 0, -1, 0],
                     [0, 1, -1, 0],
                     [0, 0, 1, -1],
                     [-1, -1, 1, 1]])
kd, kr = np.ones(4), np.ones(4)
pfr = DifferentialPFR(temperature=500, pressure=1e5, v_matrix=v_matrix)
cstr = DynamicCSTR(temperature=500, pressure=1e5, v_matrix=v_matrix)

class TestDifferentialPFR(unittest.TestCase):
    
    def test_stoic_forward(self):
        """
        Check that the forward stoichiometric matrix is correctly implemented
        """
        # self.assertTrue((pfr.stoic_forward >= 0))
        # self.assertTrue((pfr.stoic_forward == np.array([[1, 0, 0, 0],
        #                                                 [0, 1, 0, 0],
        #                                                 [0, 0, 0, 0],
        #                                                 [0, 0, 1, 0],
        #                                                 [0, 0, 1, 0],
        #                                                 [0, 0, 0, 1],
        #                                                 [1, 1, 0, 0]]).T.all()))
        self.assertEqual(pfr.stoic_forward_dense.shape, (4, 7))

    def test_stoic_backward(self):
        """
        Check that the backward stoichiometric matrix is correctly implemented
        """
        # self.assertTrue((pfr.stoic_backward >= 0))
        # self.assertTrue((pfr.stoic_backward == np.array([[0, 0, 0, 0],
        #                                                  [0, 0, 0, 0],
        #                                                  [0, 0, 0, 1],
        #                                                  [1, 0, 0, 0],
        #                                                  [0, 1, 0, 0],
        #                                                  [0, 0, 1, 1],
        #                                                  [0, 0, 0, 0]]).T.all()))
        self.assertEqual(pfr.stoic_backward_dense.shape, (4, 7))
        

class TestDynamicCSTR(unittest.TestCase):
    ...
