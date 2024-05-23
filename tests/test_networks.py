import unittest
from random import randint
from ase import Atoms
import sys
sys.path.append('../src/')

from care import Intermediate, ElementaryReaction, ReactionNetwork
from care.crn.utils.chemspace import gen_chemical_space
from care.constants import INTER_ELEMS

inters, steps = gen_chemical_space(3, 1, True)
net = ReactionNetwork()
net.add_intermediates(inters)
net.add_reactions(steps)

class TestElementaryReaction(unittest.TestCase):

	def test_type(self):
		"""
		Check that all the steps are of type ElementaryReaction
		"""
		for step in steps:
			self.assertIsInstance(step, ElementaryReaction)
			self.assertIn(step.r_type, ElementaryReaction.r_types)

	def test_stoichiometry(self):
		"""
		Check correctness of steps by checking material balance for each element
		"""
		wrong = 0
		for step in steps:
			for element in INTER_ELEMS:
				element_balance = sum([step.stoic[inter]*inter[element] for inter in list(step.reactants)+list(step.products)])
				if element_balance != 0:
					print(step, step.r_type)
					wrong += 1
					continue
		self.assertEqual(wrong, 0)
	
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
		good = 0
		for step in adsorption_steps:
			gas_phase = [inter for inter in list(step.reactants)+list(step.products) if inter.phase == 'gas']
			if len(gas_phase) == 1:
				good += 1
		self.assertEqual(good, len(adsorption_steps))

	def test_rearrengement(self):
		"""
		Check that rearrangement steps are correctly defined
		"""
		rearrangement_steps = [step for step in steps if step.r_type == 'rearrangement']
		good = 0
		for step in rearrangement_steps:
			if len(step.reactants) == 1 and len(step.products) == 1:
				good += 1
		self.assertEqual(good, len(rearrangement_steps))

	def test_energy_barrier(self):
		"""
		Check that all the reaction energy barriers are greater than zero.
		Performed only if ElementaryReaction.e_act is not None.
		"""
		for reaction in steps:
			if reaction.e_act != None and reaction.e_rxn != None:
				self.assertGreaterEqual(reaction.e_act, 0)
				if reaction.e_rxn[0] > 0:
					self.assertGreaterEqual(reaction.e_act, reaction.e_rxn[0])
				else:
					self.assertGreaterEqual(reaction.e_act, 0)

	def test_addition(self):
		"""
		Check that addition steps are correctly implemented
		"""
		step1 = steps[randint(0, len(steps)-1)]
		step2 = steps[randint(0, len(steps)-1)]
		step1.e_rxn = -1.0, 0.1
		step2.e_rxn = -0.3, 0.2
		addition_step = step1 + step2
		total = 0
		for element in INTER_ELEMS:
			element_balance = sum([addition_step.stoic[inter]*inter[element] for inter in list(addition_step.reactants)+list(addition_step.products)])
			total += element_balance
		self.assertEqual(total, 0)
		self.assertEqual(addition_step.e_rxn[0], -1.3)
		self.assertEqual(addition_step.e_rxn[1], (0.1**2+0.2**2)**0.5)
		self.assertEqual(addition_step.r_type, 'pseudo')

	def test_multiplication(self):
		"""
		Check that multiplication steps are correctly implemented
		"""
		step = steps[randint(0, len(steps)-1)]
		step.e_rxn = -1.0, 0.1
		random_num = randint(1, 5)
		mul_step = step * random_num
		total = 0
		for element in INTER_ELEMS:
			element_balance = sum([mul_step.stoic[inter]*inter[element] for inter in list(mul_step.reactants)+list(mul_step.products)])
			total += element_balance
		self.assertEqual(total, 0)	
		self.assertEqual(mul_step.e_rxn[0], step.e_rxn[0]*random_num)
		self.assertEqual(mul_step.e_rxn[1], abs(random_num)*step.e_rxn[1])
		self.assertEqual(mul_step.r_type, 'pseudo')

	def test_reverse(self):
		"""
		Check that reverse steps are correctly implemented
		"""
		step = steps[randint(0, len(steps)-1)]
		step.r_type = 'C-C'
		reactants, products = step.reactants, step.products
		step.e_is, step.e_fs, step.e_ts = (10, 0.1), (9, 0.1), (10.5, 0.1)
		step.e_rxn, step.e_act = (-1.0, 0.1), (1.5, 0.1)
		step.reverse()
		self.assertEqual(products, step.reactants)
		self.assertEqual(reactants, step.products)
		self.assertEqual(step.e_rxn[0], 1.0)
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
			self.assertIsInstance(inter.ads_configs, (dict, None))
			self.assertIsInstance(inter.phase, str)
			self.assertIsInstance(inter.closed_shell, (bool, None))	
			self.assertIsInstance(inter.molecule, (Atoms, None))

	def test_getitem(self):
		"""
		Check that the __getitem__ method works correctly
		"""
		for _, inter in inters.items():
			for element in INTER_ELEMS:
				self.assertIsInstance(inter[element], int)
			

class TestReactionNetwork(unittest.TestCase):
	def test_reaction_network(self):
		self.assertEqual(len(net), len(steps))
		self.assertEqual(len(net.intermediates), len(inters))

	def test_getitem(self):
		"""
		Check that the __getitem__ method works correctly
		"""
		random_step = net[randint(0, len(steps)-1)]
		self.assertIsInstance(random_step, ElementaryReaction)
		random_inter_key = list(inters.keys())[randint(0, len(inters)-1)]
		random_inter = net[random_inter_key]
		self.assertIsInstance(random_inter, Intermediate)

	

		

	
