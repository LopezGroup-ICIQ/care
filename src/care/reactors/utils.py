import numpy as np
from numba import njit
from sklearn.linear_model import LinearRegression

from care.constants import R


@njit
def net_rate(y, kd, kr,
             sf_data, sf_indices, sf_indptr,
             sb_data, sb_indices, sb_indptr):
    rates = np.empty_like(kd)
    n_reactions = kd.shape[0]

    for i in range(n_reactions):  # loop over reactions
        forward_product = 1.0
        backward_product = 1.0

        # forward exponents (row i of sf)
        for idx in range(sf_indptr[i], sf_indptr[i+1]):
            j = sf_indices[idx]        # species index
            exp = sf_data[idx]         # exponent
            forward_product *= y[j] ** exp

        # backward exponents (row i of sb)
        for idx in range(sb_indptr[i], sb_indptr[i+1]):
            j = sb_indices[idx]
            exp = sb_data[idx]
            backward_product *= y[j] ** exp

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