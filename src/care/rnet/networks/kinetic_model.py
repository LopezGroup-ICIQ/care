from care.rnet.networks.reaction_network import ReactionNetwork

class KineticModel:
    def __init__(self, rxn_net: ReactionNetwork):
        self.rxn_net = rxn_net
        self.rxn_net.generate_kinetic_model()
        self.kinetic_model = self.rxn_net.kinetic_model