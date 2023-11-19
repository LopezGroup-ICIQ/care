import sys, os

import unittest
from random import randint
from care.rnet.networks.reaction_network import ReactionNetwork
from care.rnet.networks.elementary_reaction import ElementaryReaction 
from care.rnet.networks.intermediate import Intermediate
from care.rnet.intermediates_funcs import generate_intermediates
from care.rnet.gen_rxn_net import generate_reactions

inters = generate_intermediates(2,-1)
steps = generate_reactions(inters)
net = ReactionNetwork()
class TestElementaryReaction(unittest.TestCase):
	def test_stoichiometry(self):
		wrong_stoich = 0
		for step in steps:
			for element in Intermediate.elements:
				element_balance = sum([step.stoic[inter]*inter[element] for inter in list(step.reactants)+list(step.products)])
				if element_balance != 0:
					print(step, step.r_type)
					wrong_stoich += 1
					continue
		self.assertEqual(wrong_stoich, 0)
		
# class TestIntermediate(unittest.TestCase):
# 	pass

class TestReactionNetwork(unittest.TestCase):
	def test_reaction_network(self):
		self.assertEqual(len(net), 0)

		

	
