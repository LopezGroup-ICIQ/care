import unittest
from random import randint

from ase import Atoms
from scipy.sparse import csr_matrix
import numpy as np
import pytest
from networkx import DiGraph

from care import Intermediate, ElementaryReaction, ReactionMechanism, ReactionNetwork
from care.crn.templates import PCET, Rearrangement, Adsorption, Desorption, BondBreaking, BondFormation
from care.constants import INTER_ELEMS
from care.crn.utils.blueprint import gen_blueprint
from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite

from tests.shared_data import co2_from_poscar, ammonia_from_poscar


net = gen_blueprint(ncc=1, 
                    noc=2, 
                    additional_rxns=True, 
                    electro=True)
inters = net.intermediates
steps = net.reactions

S_dict = {
    "DMS_hydrogenation": {
        "reactants": ["CSC", "[H][H]"],
        "products": ["C", "S"], 
        "elements": ["C", "H", "S"]          # Methane, Hydrogen Sulfide
    },
    "Methanethiol_HDS": {
        "reactants": ["CS", "[H][H]"],
        "products": ["C", "S"], 
        "elements": ["C", "H", "S"]          # Methane, Hydrogen Sulfide
    },
    "Claus_catalytic_step": {
        "reactants": ["S", "O=S=O"],    # Hydrogen Sulfide, Sulfur Dioxide
        "products": ["[S]", "O"], 
        "elements": ["H", "O", "S"]        # Elemental Sulfur, Water
    },
    "Mercaptan_sweetening": {
        "reactants": ["CS", "O=O"],     # Methanethiol, Oxygen
        "products": ["CSSC", "O"], 
        "elements": ["C", "H", "O", "S"]       # Dimethyl disulfide, Water
    }
}

N_dict = {
    "Haber_Bosch": {
        "reactants": ["N#N", "[H][H]"],
        "products": ["N"], 
        "elements": ["H", "N"]   # N2 + 3H2 -> NH3
    },
    "Ostwald_oxidation": {
        "reactants": ["N", "O=O"],      # 4NH3 + 5O2 -> 4NO + 6H2O
        "products": ["N=O", "O"], 
        "elements": ["H", "N", "O"]
    },
    "Methylamine_synthesis": {
        "reactants": ["CO", "N"],       # CH4O + NH3 -> CH5N + H2O
        "products": ["CN", "O"], 
        "elements": ["C", "H", "N", "O"]
    },
    "SCR_NOx_reduction": {
        "reactants": ["N=O", "N", "O=O"],  # 4NO + 4NH3 + O2 -> 4N2 + 6H2O
        "products": ["N#N", "O"], 
        "elements": ["H", "N", "O"]
    }
}

halogen_dict = {"methane chlorination": {
        "reactants": ["C", "ClCl"],
        "products": ["CCl", "Cl"], 
        "elements": ["C", "Cl", "H"]
    },
    "bromine addition": {
        "reactants": ["C=C", "BrBr"],
        "products": ["BrCCBr"], 
        "elements": ["Br", "C", "H"]
    },
    "methane fluorination": {
        "reactants": ["C", "FF"],
        "products": ["CF", "F"], 
        "elements": ["C", "F", "H"]
    }}




class TestElementaryReaction(unittest.TestCase):
    def test_repr(self):
        for x in net.reactions:
            self.assertIn("\u27F9", str(x))
            self.assertGreaterEqual(len(str(x)), 71)
            if not isinstance(x, Rearrangement):
                self.assertIn("+", str(x))

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
                element_balance = sum(
                    [
                        step.stoic[inter] * inter[element]
                        for inter in list(step.reactants) + list(step.products)
                    ]
                )
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
        adsorption_steps = [
            step
            for step in steps
            if step.r_type == 'adsorption'
        ]
        good = 0
        for step in adsorption_steps:
            self.assertIsInstance(step, Adsorption)
            gas_phase = [
                inter
                for inter in list(step.reactants) + list(step.products)
                if inter.phase == "gas"
            ]
            if len(gas_phase) == 1:
                good += 1
            assert hasattr(step, "adsorbate")
            assert hasattr(step, "adsorbate_mass")
        self.assertEqual(good, len(adsorption_steps))
        

    def test_rearrengement(self):
        """
        Check that rearrangement steps are correctly defined
        """
        rearrangement_steps = [step for step in steps if step.r_type == "rearrangement"]
        good = 0
        for step in rearrangement_steps:
            self.assertIsInstance(step, Rearrangement)
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
                if reaction.e_rxn > 0:
                    self.assertGreaterEqual(reaction.e_act, reaction.e_rxn)
                else:
                    self.assertGreaterEqual(reaction.e_act, 0)

    def test_crn_size(self):
        """
        Check that the number of steps and intermediates in the CRN are as expected
        """
        expected_gas_species = 10
        expected_surface_species = 28
        expected_adsorptions = 12
        expected_desorptions = 0
        expected_bond_breakings = 50
        expected_bond_formations = 0
        expected_rearrangements = 7
        expected_pcets = 39
        self.assertEqual(
            len(inters),
            expected_gas_species + expected_surface_species,
        )
        self.assertEqual(len(steps), sum([
            expected_adsorptions,
            expected_bond_breakings,
            expected_bond_formations,
            expected_rearrangements,
            expected_pcets,
        ]))
        self.assertEqual(expected_adsorptions, len([step for step in steps if isinstance(step, (Adsorption))]))
        self.assertEqual(expected_pcets, len([step for step in steps if isinstance(step, PCET)]))
        self.assertEqual(expected_rearrangements, len([step for step in steps if isinstance(step, Rearrangement)]))
        self.assertEqual(expected_bond_breakings, len([step for step in steps if isinstance(step, BondBreaking)]))
        self.assertEqual(expected_bond_formations, len([step for step in steps if isinstance(step, BondFormation)]))
        self.assertEqual(expected_desorptions, len([step for step in steps if isinstance(step, Desorption)]))

    def test_addition_substraction(self):
        """
        Check that addition and subtraction steps are correctly implemented
        """
        step1 = steps[randint(0, len(steps) - 1)]
        step2 = steps[randint(0, len(steps) - 1)]
        step1.e_rxn = -1.0
        step2.e_rxn = -0.3
        addition_step = step1 + step2
        subtraction_step = step1 - step2
        total_add, total_sub = 0, 0
        for element in INTER_ELEMS:
            element_balance = sum(
                [
                    addition_step.stoic[inter] * inter[element]
                    for inter in list(addition_step.reactants)
                    + list(addition_step.products)
                ]
            )
            total_add += element_balance
            element_balance = sum(
                [
                    subtraction_step.stoic[inter] * inter[element]
                    for inter in list(subtraction_step.reactants)
                    + list(subtraction_step.products)
                ]
            )
            total_sub += element_balance
        self.assertIsInstance(addition_step, ReactionMechanism)
        self.assertIsInstance(subtraction_step, ReactionMechanism)
        self.assertEqual(total_add, 0)
        self.assertEqual(total_sub, 0)
        self.assertEqual(addition_step.e_rxn, -1.3)
        self.assertEqual(subtraction_step.e_rxn, -0.7)
        self.assertEqual(addition_step.r_type, "pseudo")
        self.assertEqual(subtraction_step.r_type, "pseudo")

    def test_multiplication(self):
        """
        Check that multiplication steps are correctly implemented
        """
        step = steps[randint(0, len(steps) - 1)]
        step.e_rxn = -1.0
        random_num = randint(1, 5)
        mul_step = step * random_num
        total = 0
        for element in INTER_ELEMS:
            element_balance = sum(
                [
                    mul_step.stoic[inter] * inter[element]
                    for inter in list(mul_step.reactants) + list(mul_step.products)
                ]
            )
            total += element_balance
        self.assertIsInstance(mul_step, ReactionMechanism)
        self.assertEqual(total, 0)
        self.assertEqual(mul_step.e_rxn, step.e_rxn * random_num)
        self.assertEqual(mul_step.r_type, "pseudo")
        # test multiplication by negative number reverses the step
        neg_mul_step = step * -1
        self.assertEqual(neg_mul_step.reactants, step.products)
        self.assertEqual(neg_mul_step.products, step.reactants)
        self.assertEqual(neg_mul_step.e_rxn, -step.e_rxn)

    def test_reverse(self):
        """
        Check that reverse steps are correctly implemented
        """
        for step in steps:
            step_class = step.__class__
            reactants, products = step.reactants, step.products
            step.e_is, step.e_fs, step.e_ts = 10.0, 9.0, 11.0
            step.reverse()
            if step.__class__ == BondFormation:
                self.assertEqual(step_class, BondBreaking)
            if step.__class__ == BondBreaking:
                self.assertEqual(step_class, BondFormation)
            if step.__class__ == Adsorption:
                self.assertEqual(step_class, Desorption)
            if step.__class__ == Desorption:
                self.assertEqual(step_class, Adsorption)
            if step.__class__ == Rearrangement:
                self.assertEqual(step_class, Rearrangement)
            if step.__class__ == PCET:
                self.assertEqual(step_class, PCET)
            self.assertEqual(products, step.reactants)
            self.assertEqual(reactants, step.products)
            self.assertEqual(step.e_rxn, 1.0, msg=f"e_rxn for step {step}  ({type(step)}) is not 1.0 after reversal")
            self.assertEqual(step.e_act, 2.0)

    def test_energy_setters(self):
        """
        Check that energy setters correctly invalidate dependent states and
        raise ValueErrors when conflicting overrides are attempted.
        """
        A = GasSpecies("Ag", Atoms("CHO"))
        B = GasSpecies("Bg", Atoms("CHO"))

        A.E = None 
        B.E = None
        
        step = ElementaryReaction(components=([A], [B]), stoic={A.code: -1, B.code: 1})

        step.e_is = 0.0
        step.e_fs = -1.0
        step.e_ts = 1.0
        
        self.assertEqual(step.e_rxn, -1.0)
        self.assertEqual(step.e_act, 1.0)
        
        step.e_fs = -2.0
        self.assertEqual(step.e_rxn, -2.0)
        
        step.e_is = 0.5
        self.assertEqual(step.e_rxn, -2.5)
        self.assertEqual(step.e_act, 0.5)

        with self.assertRaises(ValueError):
            step.e_rxn = -3.0  # Both e_is and e_fs are defined

        with self.assertRaises(ValueError):
            step.e_act = 0.8   # Both e_is and e_ts are defined

        step2 = ElementaryReaction(components=([A], [B]), stoic={A.code: -1, B.code: 1})
        step2.e_is = 0.0 
        
        # Setting e_rxn is valid and clears e_is.
        step2.e_rxn = -1.5
        self.assertIsNone(step2.e_is)
        self.assertIsNone(step2.e_fs)
        self.assertEqual(step2.e_rxn, -1.5)

        # Restore e_is to test e_act
        step2.e_is = 0.0
        step2.e_act = 1.2
        self.assertIsNone(step2.e_ts)
        self.assertEqual(step2.e_act, 1.2)


class TestIntermediate(unittest.TestCase):
    def test_repr(self):
        for k, v in inters.items():
            self.assertTrue(len(str(v)) >= 31)  # IncHiKey (27) + phase id (1) + "(x)" (>=3)

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
            self.assertEqual(
                len(key), 28
            )  # InChI key (27) + id for adsorbed ("*") or gas ("g") phase
            self.assertIsInstance(key, str)
            self.assertIsInstance(inter, Intermediate)
            if isinstance(inter, AdsorbedSpecies):
                self.assertIsInstance(inter.ads_configs, (dict, None))
            self.assertIsInstance(inter.phase, str)
            self.assertIsInstance(inter.closed_shell, (bool, None))
            self.assertIsInstance(inter.molecule, (Atoms, None))
            self.assertFalse(inter.is_evaluated)

    def test_getitem(self):
        """
        Check that the __getitem__ method works correctly
        """
        for _, inter in inters.items():
            for element in INTER_ELEMS:
                self.assertIsInstance(inter[element], int)
                
    def test_from_molecule(self):
        """
        Check that the from_molecule method works correctly
        """
        self.assertIsInstance(co2_from_poscar, Intermediate)
        self.assertEqual(co2_from_poscar.code, "CO2g")
        self.assertEqual(co2_from_poscar.phase, "gas")
        self.assertIsInstance(co2_from_poscar.molecule, Atoms)

    def test_electrons(self):
        self.assertEqual(co2_from_poscar.electrons, 8)
        self.assertEqual(ammonia_from_poscar.electrons, 6)

class TestReactionNetwork(unittest.TestCase):
    def test_repr(self):
        keys = ["ReactionNetwork(", 
                "surface species,", 
                "molecules,", 
                "elementary reactions)", 
                "Elements:", 
                "Catalyst:", 
                "Type:", 
                "Evaluated: Thermodynamics (", 
                "| Kinetics ("]
        for k in keys:
            self.assertIn(k, str(net))

    def test_creation_from_species(self):
        crn = ReactionNetwork.from_species(reactants=["O=C=O", "[H][H]"], products=["CO", "O"])
        self.assertIsInstance(crn, ReactionNetwork)
        num_desorptions = len([x for x in crn.reactions if isinstance(x, Desorption)])
        self.assertGreaterEqual(num_desorptions, 2)
        # test mismatch: oxygen missing in products
        with pytest.raises(ValueError) as exc_info:
            gen_blueprint(reactants=["C=O", "[H][H]"], products=["CCC"])
        error_msg = str(exc_info.value)
        self.assertIn("Element mismatch between reactants and products!", error_msg)
        self.assertIn("Elements in reactants but missing in products:", error_msg)
        self.assertIn("'O'", error_msg)
        self.assertNotIn("missing in reactants", error_msg)
        # test mismatch: oxygen missing in reactants
        with pytest.raises(ValueError) as exc_info:
            gen_blueprint(reactants=["C"], products=["CO"])
        error_msg = str(exc_info.value)
        self.assertIn("Element mismatch between reactants and products!", error_msg)
        self.assertIn( "Elements in products but missing in reactants:", error_msg)
        self.assertIn("'O'", error_msg)
        self.assertNotIn("missing in products", error_msg)

    def test_invalid_element(self):
        invalid_cs = ["C[Hg]C"] 
        expected_error_pattern = "unsupported elements: .*'Hg'.*"
        with self.assertRaisesRegex(ValueError, expected_error_pattern):
            ReactionNetwork.from_chemical_space(invalid_cs)

    def test_invalid_smiles_in_cs_issues_warning(self):
        with pytest.warns(UserWarning, match="Some SMILES in 'cs' were invalid and removed."):
            gen_blueprint(cs=["C", "Ciao"])

    def test_insufficient_parameters_raises_error(self):
        error_pattern = r"Insufficient parameters\. Provide \(reactants/products\), \(cs\), or \(ncc/noc\)\."        
        with pytest.raises(ValueError, match=error_pattern):
            gen_blueprint() 
        with pytest.raises(ValueError, match=error_pattern):
            gen_blueprint(cyclic=True, electro=True, additional_rxns=True)

    def test_creation_from_cs(self):
        crn = ReactionNetwork.from_chemical_space(cs=["CCO"])
        self.assertIsInstance(crn, ReactionNetwork)
        num_desorptions = len([x for x in crn.reactions if isinstance(x, Desorption)])
        self.assertEqual(num_desorptions, 0)

    def test_creation_from_cutoffs(self):
        crn = ReactionNetwork.from_cutoffs(ncc=2, noc=1)
        self.assertIsInstance(crn, ReactionNetwork)
        num_desorptions = len([x for x in crn.reactions if isinstance(x, Desorption)])
        self.assertEqual(num_desorptions, 0)

    def test_crn_with_S(self):
        for inputs in S_dict.values():
            reactants = inputs["reactants"]
            products = inputs["products"]
            elements = inputs["elements"]
            crn = ReactionNetwork.from_species(reactants, products)
            self.assertIsInstance(crn, ReactionNetwork)
            self.assertEqual(crn.elements, elements)

    def test_crn_with_N(self):
        for inputs in N_dict.values():
            reactants = inputs["reactants"]
            products = inputs["products"]
            elements = inputs["elements"]
            crn = ReactionNetwork.from_species(reactants, products)
            self.assertIsInstance(crn, ReactionNetwork)
            self.assertEqual(crn.elements, elements)

    def test_crn_with_halogens(self):
        for inputs in halogen_dict.values():
            reactants = inputs["reactants"]
            products = inputs["products"]
            elements = inputs["elements"]
            crn = ReactionNetwork.from_species(reactants, products)
            self.assertIsInstance(crn, ReactionNetwork)
            self.assertEqual(crn.elements, elements)
        
    def test_reaction_network(self):
        self.assertIsInstance(net, DiGraph)
        self.assertGreater(net.number_of_edges(), 2 * len(net))
        self.assertEqual(net.number_of_nodes(), len(steps)+len(inters)+1+3)
        self.assertEqual(len(net), len(steps))
        self.assertEqual(len(net.intermediates), len(inters))
        self.assertEqual(net.ncc, 1)
        self.assertEqual(net.noc, 2)
        self.assertTrue("*" in net)
        self.assertIsNone(net.get_reaction_table())

    def test_getitem(self):
        random_step = net[randint(0, len(steps) - 1)]
        self.assertIsInstance(random_step, ElementaryReaction)
        random_inter_key = list(inters.keys())[randint(0, len(inters) - 1)]
        random_inter = net[random_inter_key]
        self.assertIsInstance(random_inter, Intermediate)

    def test_stoichiometry(self):
        self.assertIsInstance(net.v, csr_matrix)
        self.assertEqual(net.v.shape, (len(net.intermediates)+1, len(steps)))

    def test_element_species_matrix(self):
        self.assertIsInstance(net.es, np.ndarray)
        self.assertEqual(net.es.shape, (len(INTER_ELEMS)-1, len(net.intermediates)+1))

    def test_reverse(self):
        i = randint(0, len(steps) - 1)
        reactants, products = net[i].reactants, net[i].products
        net.reverse_reaction(i)
        self.assertEqual(net[i].reactants, products)
        self.assertEqual(net[i].products, reactants)
        net.reverse_reaction(i)
        self.assertEqual(net[i].reactants, reactants)
        self.assertEqual(net[i].products, products)

    def test_hubs(self):
        hubs = net.get_hubs(6)
        self.assertIsInstance(hubs, dict)
        self.assertEqual(list(hubs.values())[0], max(list(hubs.values())))

    def test_reaction_removal(self):
        """
        Test that removing a reaction correctly orphans and cascades 
        the removal of dead-end species and dependent reactions.
        """
        
        star = SurfaceSite("*")

        CO_atoms = Atoms("CO")
        O_atoms = Atoms("O")
        H2_atoms = Atoms("H2")
        CO2_atoms = Atoms("CO2")
        H2O_atoms = Atoms("H2O")

        COg = GasSpecies("COg", CO_atoms)
        Og = GasSpecies("Og", O_atoms)
        H2g = GasSpecies("H2g", H2_atoms)
        CO2g = GasSpecies("CO2g", CO2_atoms)
        H2Og = GasSpecies("H2Og", H2O_atoms)

        COs = AdsorbedSpecies("COs", CO_atoms)
        Os = AdsorbedSpecies("Os", O_atoms)
        H2s = AdsorbedSpecies("H2s", H2_atoms)
        CO2s = AdsorbedSpecies("CO2s", CO2_atoms)
        H2Os = AdsorbedSpecies("H2Os", H2O_atoms)

        reactions = [
            Adsorption(components=([COg, star], [COs])),
            Adsorption(components=([Og, star], [Os])),
            Adsorption(components=([H2g, star], [H2s])),
            Desorption(components=([CO2s], [CO2g, star])),
            Desorption(components=([H2Os], [H2Og, star])),
            ElementaryReaction(components=([COs, Os], [CO2s, star])),
            ElementaryReaction(components=([H2s, Os], [H2Os, star])),
        ]

        crn = ReactionNetwork(reactions)
        
        self.assertEqual(len(crn.reactions), 7)
        self.assertIn(H2Og.code, crn.intermediates)
        
        crn.remove_reaction(crn.reactions[-1])
        self.assertEqual(len(crn.reactions), 4)
        
        self.assertNotIn(H2g.code, crn.intermediates)
        self.assertNotIn(H2s.code, crn.intermediates)
        self.assertNotIn(H2Og.code, crn.intermediates)
        self.assertNotIn(H2Os.code, crn.intermediates)
        
        self.assertIn(CO2g.code, crn.intermediates)
        self.assertIn(Os.code, crn.intermediates)