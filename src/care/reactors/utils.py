import numpy as np
from numba import njit
from sklearn.linear_model import LinearRegression

from care.constants import R


@njit
def net_rate(y, kd, kr, sf, sb):
    rates = np.empty_like(kd)
    for i in range(kd.shape[0]):  # Assuming kd and kr have the same shape
        forward_product = 1.0
        backward_product = 1.0
        for j in range(
            sf.shape[1]
        ):  # Assuming sf and sb have the same shape [reactions, species]
            forward_product *= y[j] ** sf[i, j]
            backward_product *= y[j] ** sb[i, j]
        rates[i] = kd[i] * forward_product - kr[i] * backward_product
    return rates


def calc_eapp(t, r, gas_mask):
    """
    Evaluates the apparent activation energy for all the species whose formation rate is higher than zero.
    Args:
        temperature_vector(ndarray): Array containing the studied temperature range in Kelvin
        reaction_rate_vector(ndarray): Array containing the reaction rate at different temperatures
    Returns:
        Apparent reaction energy in kJ/mol at the specified temperature.
    """
    x = 1 / t
    eapp = np.zeros(len(gas_mask[:-1]))
    for i, inter in enumerate(gas_mask[:-1]):
        Eapp = -(R / 1000.0)
        if inter and np.all(r[:, i] > 0):
            lm = LinearRegression()
            reg = lm.fit(x.reshape(-1, 1), np.log(r[:, i]).reshape(-1, 1))
            Eapp *= reg.coef_[0, 0]  # kJ/mol
            eapp[i] = Eapp
        else:
            eapp[i] = None
    return eapp