from care import Intermediate, Surface, ElementaryReaction
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator

class ExampleIntermediate(IntermediateEnergyEstimator):
    def __init__(
        self, surface: Surface
    ):
        """Example Interface
        """

        self.surface = surface

    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N', 'S']

    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return ['Pd']

    def eval(
        self,
        intermediate: Intermediate,
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """
        pass


class ExampleReaction(ReactionEnergyEstimator):
    def __init__(
        self, surface: Surface
    ):
        """Example Interface
        """

        self.surface = surface

    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N', 'S']

    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return ['Pd']

    def eval(
        self,
        reaction: ElementaryReaction,
    ):

        """
        Given the reaction, return the properties of the reaction as attributes of the reaction object.
        """
        pass
