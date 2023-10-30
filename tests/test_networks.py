# elementary_reaction.py

import unittest
import pickle
from random import randint
from GAMERNet.rnet.networks.reaction_network import ReactionNetwork
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction 
from GAMERNet.rnet.networks.intermediate import Intermediate

# content = pickle.load(open('./C1_O1_Pd111/rxn_net_bp.pkl', 'rb'))
# net = ReactionNetwork().from_dict(content)
# step = net[randint(0, len(net)-1)]
# inter = step.reactants[0]

net = ReactionNetwork()
# surf_inter = Intermediate.from_molecule(slab_ase_obj, code='0000000000*', is_surface=True, phase='surf')


# class TestElementaryReaction(unittest.TestCase):
# 	def test_elementary_reaction(self):
# 		# test elementary reaction
# 		pass

# class TestIntermediate(unittest.TestCase):
# 	pass

class TestReactionNetwork(unittest.TestCase):
	def test_reaction_network(self):
		# test reaction network
		self.assertEqual(len(net), 0)

		

	
