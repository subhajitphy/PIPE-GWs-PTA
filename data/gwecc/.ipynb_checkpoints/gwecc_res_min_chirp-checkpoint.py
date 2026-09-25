#!/usr/bin/env python
# coding: utf-8
"""
gwecc_res.py  (simple module)

Exports ONLY:
  - add_ecc_cgw(...)   # exactly your function (unchanged)
  - get_phi(tarr, M, z0, eta)  # helper you can call as:
        z0 = [n0, e0, gamma_0]
        phi = get_phi(tarr, M, z0, eta)

Assumes these are importable from your PKG_PATH:
  constants.py: c, tsun
  mikkola.py: get_u
  antenna_pattern.py: antenna_pattern
"""

import numpy as np,  sys
from constants import c, tsun
from mikkola import get_u
from antenna_pattern import antenna_pattern


def get_phi(tarr, M, z0, eta):
    """
    Compute orbital phase phi(t) using the SAME ingredients as add_ecc_cgw's earth_term.

    Parameters
    ----------
    tarr : array_like
        Time array (seconds) - should already be relative to tref if you want that convention.
    M : float
        Total mass in solar masses (NOT log10).
    z0 : list/tuple/array
        [n0, e0, gamma_0] where:
          n0      : orbital frequency (NOT log10)
          e0      : eccentricity
          gamma_0 : initial periastron angle
    eta : float
        Symmetric mass ratio = q/(1+q)^2

    Returns
    -------
    phi : ndarray
        Orbital phase array (radians), same definition used in add_ecc_cgw.
    """
    n0, e0, gamma_0 = float(z0[0]), float(z0[1]), float(z0[2])

    x0 = (tsun * M * n0) ** (2.0 / 3.0)

    l = n0 * np.asarray(tarr)
    u = get_u(l, e0)

    OTS = np.sqrt(1.0 - e0 * e0)
    k0 = 3.0 * x0 / OTS**2
    beta = (1.0 - OTS) / e0 * (1.0 + (8.0 - 2.0 * eta + (4.0 - eta) / OTS) * x0)

    vmu = 2.0 * np.arctan(beta * np.sin(u) / (1.0 - beta * np.cos(u)))

    W0 = e0 * np.sin(u) + vmu
    W1 = 3.0 * (vmu + e0 * np.sin(u)) * x0 / OTS**2

    phi = gamma_0 + (1.0 + k0) * l + (W0 + W1)
    return phi


def add_ecc_cgw(
    toas,
    theta,
    phi,
    cos_gwtheta,
    gwphi,
    psi,
    cos_inc,
    log10_n,
    q,
    log10_A,
    e0,
    log10_Mc,
    tref,
    gamma_0,
    l_0,
    pdist,
    res="Both",
):
    # Geometry / angles
    inc = np.arccos(cos_inc)

    gwra  = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra  = phi
    psrdec = np.pi / 2.0 - theta

    cosmu, Fp, Fx = antenna_pattern(gwra, gwdec, psrra, psrdec)

    # Relative time
    tarr = np.asarray(toas) - tref

    # Pulsar term delay: pdist/c * (1 - cosmu)
    tP_arr = tarr - (pdist / c) * (1.0 - cosmu)

    def earth_term(t):
        n0  = 10.0 ** log10_n
        Mc   = 10.0 ** log10_Mc
        eta = q / (1.0 + q) ** 2
        M=Mc/eta ** (3.0 / 5.0)
        x0 = (tsun * M * n0) ** (2.0 / 3.0)

        # mean anomaly / eccentric anomaly
        l = n0 * t + l_0
        u = get_u(l, e0)

        OTS  = np.sqrt(1.0 - e0 * e0)
        k0   = 3.0 * x0 / OTS**2
        beta = (1.0 - OTS) / e0 * (1.0 + (8.0 - 2.0 * eta + (4.0 - eta) / OTS) * x0)

        vmu = 2.0 * np.arctan(beta * np.sin(u) / (1.0 - beta * np.cos(u)))

        W0  = e0 * np.sin(u) + vmu
        W1  = 3.0 * (vmu + e0 * np.sin(u)) * x0 / OTS**2
        Phi = gamma_0 + (1.0 + k0) * l + (W0 + W1)

        v   = u + vmu
        omg = Phi - v

        w = 1.0 - e0 * np.cos(u)
        P = np.sqrt(1.0 - e0 * e0) * (np.cos(2.0 * u) - e0 * np.cos(u)) / w
        Q = ((e0 * e0 - 2.0) * np.cos(u) + e0) * np.sin(u) / w
        R = e0 * np.sin(u)

        A = 10.0 ** log10_A
        spA = A * (
            (np.cos(inc)**2 + 1.0) * (-P * np.sin(2.0 * omg) + Q * np.cos(2.0 * omg))
            + np.sin(inc)**2 * R
        )
        sxA = 2.0 * A * np.cos(inc) * (P * np.cos(2.0 * omg) + Q * np.sin(2.0 * omg))

        c2 = np.cos(2.0 * psi)
        s2 = np.sin(2.0 * psi)

        return Fp * (c2 * spA - s2 * sxA) + Fx * (s2 * spA + c2 * sxA)

    sE = earth_term(tarr)
    sP = earth_term(tP_arr)

    if res == "Earth":
        s = sE
    elif res == "Pulsar":
        s = -sP
    elif res == "Both":
        s = sE - sP
    else:
        raise ValueError("res must be one of: 'Earth', 'Pulsar', 'Both'")

    return s - s.mean()
