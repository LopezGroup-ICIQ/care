from care import Intermediate
from care.crn.templates import Adsorption, Desorption

class AssociativeAdsorption(Adsorption):
    """
    Eley-Rideal (associative adsorption) elementary reaction:
    A* + B(g) -> C*
    """
    def __init__(self, components, r_type):
        super().__init__(components, r_type)

    def reverse(self):
        self.__class__ = DissociativeDesorption
        self.r_type = "desorption"
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.e_rxn != None:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )


class DissociativeDesorption(Desorption):
    """
    Dissociative desorption elementary reaction of the type:
    A* -> B(g) * C*
    """
    def __init__(self, components, r_type):
        super().__init__(components, r_type)

    def reverse(self):
        self.__class__ = AssociativeAdsorption
        self.r_type = "eley_rideal"
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.e_rxn != None:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )


def gen_eleyrideal(gas_inters: list[Intermediate],
                   ads_inters: list[Intermediate]) -> list[AssociativeAdsorption]:
    """
    Generate Eley-Rideal reactions from a list of gas-phase intermediates and
    adsorbed intermediates.

    Args:
        gas_inters (list[Intermediate]): List of gas-phase intermediates.
        ads_inters (list[Intermediate]): List of adsorbed intermediates.

    Returns:
        list[AssociativeAdsorption]: List of Eley-Rideal reactions.
    """
    eleyrideal_reactions = []
    for gas in gas_inters:
        for ads in ads_inters:
            products = check_eleyrideal(gas, ads)
            if products:
                for product in products:
                    eleyrideal_reactions.append(AssociativeAdsorption(
                        components=[gas, ads, product], r_type="eley_rideal"))
    return eleyrideal_reactions

def check_eleyrideal(gas: Intermediate,
                     ads: Intermediate) -> list[Intermediate]:
    """
    Check if a Eley-Rideal reaction is possible between a gas-phase intermediate
    and an adsorbed intermediate.
    """
    # TODO: Implement this function
    pass
