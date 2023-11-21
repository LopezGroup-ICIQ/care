import sys, os

import unittest
from random import randint
from care.rnet.networks.reaction_network import ReactionNetwork
from care.rnet.networks.elementary_reaction import ElementaryReaction 
from care.rnet.networks.intermediate import Intermediate
from care.rnet.netgen_fns import generate_inters_and_rxns

inters, steps = generate_inters_and_rxns(4,1)
net = ReactionNetwork()

class TestElementaryReaction(unittest.TestCase):
	def test_stoichiometry(self):
		"""
		Check correctness of steps by checking material balance for each element
		"""
		wrong_stoich = 0
		for step in steps:
			for element in Intermediate.elements:
				element_balance = sum([step.stoic[inter]*inter[element] for inter in list(step.reactants)+list(step.products)])
				if element_balance != 0:
					print(step, step.r_type)
					wrong_stoich += 1
					continue
		self.assertEqual(wrong_stoich, 0)
	
	def test_uniqueness(self):
		"""
		Check that no duplicated steps are present in the network
		"""
		self.assertEqual(len(steps), len(set(steps)))

	def test_adsorption(self):
		"""
		Check that adsorption/desorption steps are correctly defined
		"""
		adsorption_steps = [step for step in steps if step.r_type in ('adsorption', 'desorption', 'eley_rideal')]
		good_adsorption = 0
		for step in adsorption_steps:
			gas_phase = [inter for inter in list(step.reactants)+list(step.products) if inter.phase == 'gas']
			if len(gas_phase) == 1:
				good_adsorption += 1
		self.assertEqual(good_adsorption, len(adsorption_steps))

	def test_rearrengement(self):
		"""
		Check that rearrangement steps are correctly defined
		"""
		rearrangement_steps = [step for step in steps if step.r_type == 'rearrangement']
		good_rearrangement = 0
		for step in rearrangement_steps:
			if len(step.reactants) == 1 and len(step.products) == 1:
				good_rearrangement += 1
		self.assertEqual(good_rearrangement, len(rearrangement_steps))

	def test_energy_barrier(self):
		"""
		Check that all the reaction energy barriers are greater than zero.
		Performed only if ElementaryReaction.e_act is not None.
		"""
		for reaction in steps:
			if reaction.e_act != None:
				self.assertGreaterEqual(reaction.e_act, 0)			


class TestIntermediate(unittest.TestCase):
	def test_uniqueness(self):
		"""
		Check that no duplicated intermediates are present in the network
		"""
		self.assertEqual(len(inters), len(set(inters)))
		
# class TestIntermediate(unittest.TestCase):
# 	pass

class TestReactionNetwork(unittest.TestCase):
	def test_reaction_network(self):
		self.assertEqual(len(net), 0)

		

	
