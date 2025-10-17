import numpy as np
from numpy import typing as npt
from scipy.optimize import curve_fit


def eyring_viscosity(gamma_dot, eta_N, sigma_E):
    tau = eta_N / sigma_E
    return (eta_N / (tau * gamma_dot)) * np.log(tau * gamma_dot + np.sqrt((tau * gamma_dot)**2 + 1))

def fit_viscosity_eyring(
        shear_rates: npt.NDArray,
        viscosities: npt.NDArray,
    ) -> tuple[bool, float, float]:
    """Fit viscosity data to the Eyring model."""
    fit_successful = True
    eta_N, sigma_E = None, None

    eta_N_guess = np.median(viscosities)
    sigma_E_guess = 1e8
    p0 = [eta_N_guess, sigma_E_guess]

    try:
        popt, _ = curve_fit(
            eyring_viscosity,
            shear_rates,
            viscosities,
            p0=p0,
            maxfev=10000  # Increase max number of function evaluations just in case
        )
        eta_N, sigma_E = popt
    except Exception as e:
        fit_successful = False

    return fit_successful, eta_N, sigma_E
