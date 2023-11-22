import sys, os

import unittest
from random import randint
from care.rnet.networks.reaction_network import ReactionNetwork
from care.rnet.networks.elementary_reaction import ElementaryReaction 
from care.rnet.networks.intermediate import Intermediate
from care.rnet.netgen_fns import generate_inters_and_rxns

inters, steps = generate_inters_and_rxns(3, 1)
net = ReactionNetwork()

class TestElementaryReaction(unittest.TestCase):

	def test_type(self):
		"""
		Check that all the steps are of type ElementaryReaction
		"""
		for step in steps:
			self.assertIsInstance(step, ElementaryReaction)

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

	def test_addition(self):
		"""
		Check that addition steps are correctly implemented
		"""
		step1 = steps[randint(0, len(steps)-1)]
		step2 = steps[randint(0, len(steps)-1)]
		addition_step = step1 + step2
		total = 0
		for element in Intermediate.elements:
			element_balance = sum([addition_step.stoic[inter]*inter[element] for inter in list(addition_step.reactants)+list(addition_step.products)])
			total += element_balance
		self.assertEqual(total, 0)

	def test_multiplication(self):
		"""
		Check that multiplication steps are correctly implemented
		"""
		step = steps[randint(0, len(steps)-1)]
		multiplication_step = step * randint(1, 5)
		total = 0
		for element in Intermediate.elements:
			element_balance = sum([multiplication_step.stoic[inter]*inter[element] for inter in list(multiplication_step.reactants)+list(multiplication_step.products)])
			total += element_balance
		self.assertEqual(total, 0)	

	def test_reverse(self):
		"""
		Check that reverse steps are correctly implemented
		"""
		step = steps[randint(0, len(steps)-1)]
		reactants, products = step.reactants, step.products
		step.energy = -1.0, 0.1  # mean, std
		step.e_act = 0.5, 0.1  # mean, std
		step.reverse()
		self.assertEqual(products, step.reactants)
		self.assertEqual(reactants, step.products)
		self.assertEqual(step.energy[0], 1.0)
		self.assertEqual(step.e_act[0], 1.5)



class TestIntermediate(unittest.TestCase):
	def test_uniqueness(self):
		"""
		Check that no duplicated intermediates are present in the network
		"""
		self.assertEqual(len(inters), len(set(inters)))

	def test_type(self):
		"""
		Check that all the intermediates is a dict[str, Intermediate]
		and that the length of all keys is exactly 28
		"""
		for key, inter in inters.items():
			self.assertEqual(len(key), 28) # InChI key (27) + id for adsorbed ("*") or gas ("g") phase
			self.assertIsInstance(key, str)
			self.assertIsInstance(inter, Intermediate)

	def test_getitem(self):
		"""
		Check that the __getitem__ method works correctly
		"""
		for key, inter in inters.items():
			for element in Intermediate.elements:
				self.assertIsInstance(inter[element], int)
			

class TestReactionNetwork(unittest.TestCase):
	def test_reaction_network(self):
		self.assertEqual(len(net), 0)

		

	
