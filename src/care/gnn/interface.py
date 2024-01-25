from typing import Optional

from care import Intermediate, Surface, EnergyEstimator

class GameNetUQ(EnergyEstimator):
    def __init__(self, 
                 model_path: str):
        self.path = model_path
        pass

    def estimate_energy(self, inter: Intermediate, surface: Optional[Surface] = None):
        # Check
        if inter.phase == 'gas' and surface is not None:
            raise ValueError('Surface must be None for gas phase intermediates.')
        elif inter.phase == 'ads' and surface is None:
            raise ValueError('Surface must be provided for adsorbed intermediates.')
        elif inter.phase == 'ads' and isinstance(surface, Surface):
            adsorption = True
        else:
            adsorption = False

        # Adsorbate placement (if necessary)
        if adsorption:
            pass # TODO

        # Graph conversion
        graph = None

        # Model prediction
        pass