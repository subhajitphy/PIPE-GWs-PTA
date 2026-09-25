import numpy as np

def get_u(l, e):
    """
    Solve Kepler's equation l = u - e*sin(u) (elliptic case) using a modified Mikkola method.

    Parameters
    ----------
    l : float or array_like
        Mean anomaly (radians). Can be scalar or array.
    e : float or array_like
        Eccentricity, must satisfy 0 <= e < 1 for valid elliptic orbits.
        Can be scalar or array. Will broadcast with `l`.

    Returns
    -------
    u : float or np.ndarray
        Eccentric anomaly (radians), same broadcasted shape as (l, e). If both inputs
        are scalars, a scalar float is returned.

    Notes
    -----
    - Supports (l_arr, e), (l, e_arr), and (l_arr, e_arr) via NumPy broadcasting.
    - For e == 0, Kepler’s equation reduces to u = l (returned exactly, no folding).
    - For invalid eccentricities (e < 0 or e >= 1), returns np.nan at those entries.
    """

    # Convert to arrays and broadcast
    l_in = np.asarray(l, dtype=float)
    e_in = np.asarray(e, dtype=float)
    l_b, e_b = np.broadcast_arrays(l_in, e_in)

    # Track scalar return
    return_scalar = (np.ndim(l_in) == 0) and (np.ndim(e_in) == 0)

    Pi = np.pi
    TwoPi = 2.0 * Pi

    # Output array
    u_out = np.empty_like(l_b, dtype=float)

    # Masks
    valid_mask = (e_b >= 0.0) & (e_b < 1.0)
    circ_mask  = valid_mask & (e_b == 0.0)             # circular: u = l exactly
    work_mask  = valid_mask & (e_b != 0.0)             # elliptic, non-zero e
    invalid_mask = ~valid_mask

    # 1) Invalid e -> NaN
    u_out[invalid_mask] = np.nan

    # 2) Circular e == 0 -> u = l (no folding)
    u_out[circ_mask] = l_b[circ_mask]

    # 3) Modified Mikkola for 0 < e < 1
    if np.any(work_mask):
        l_w = l_b[work_mask]
        e_w = e_b[work_mask]

        # Keep original signs and cycles to restore at the end
        sgn = np.sign(l_w)
        l_abs = sgn * l_w                           # make l non-negative
        ncycles = np.floor(l_abs / TwoPi)           # number of full 2π cycles
        l_mod = l_abs - ncycles * TwoPi             # 0 <= l_mod < 2π

        # Fold to [0, π] using symmetry
        flag = (l_mod > Pi)
        l_fold = l_mod.copy()
        l_fold[flag] = TwoPi - l_fold[flag]         # now 0 <= l_fold <= π

        # Mikkola’s auxiliary quantities
        alpha  = (1.0 - e_w) / (4.0 * e_w + 0.5)
        alpha3 = alpha**3
        beta   = (l_fold / 2.0) / (4.0 * e_w + 0.5)
        beta2  = beta**2

        # z via cubic roots, branch by sign(beta)
        z = np.empty_like(l_fold)
        root_term = np.sqrt(alpha3 + beta2)
        pos = beta > 0.0
        z[pos]  = np.cbrt(beta[pos]  + root_term[pos])
        z[~pos] = np.cbrt(beta[~pos] - root_term[~pos])

        s  = z - alpha / z
        s5 = s**5
        w  = s - (0.078 * s5) / (1.0 + e_w)
        w3 = w**3
        E0 = l_fold + e_w * (3.0 * w - 4.0 * w3)     # initial guess
        u  = E0

        # Newton-like 4th-order correction (vectorized)
        esu = e_w * np.sin(u)
        ecu = e_w * np.cos(u)

        fu  = (u - esu - l_fold)
        f1u = (1.0 - ecu)
        f2u = (esu)
        f3u = (ecu)
        f4u = -(esu)

        u1 = -fu / f1u
        u2 = -fu / (f1u + 0.5 * f2u * u1)
        u3 = -fu / (f1u + 0.5 * f2u * u2 + (f3u * (u2**2)) / 6.0)
        u4 = -fu / (f1u + 0.5 * f2u * u3 + (f3u * (u3**2)) / 6.0 + (f4u * (u3**3)) / 24.0)
        xi = E0 + u4

        # Unfold from [0, π] back to [0, 2π)
        sol = xi.copy()
        sol[flag] = TwoPi - xi[flag]

        # Restore cycles and original sign
        u_w = sgn * (sol + ncycles * TwoPi)

        # Write back
        u_out[work_mask] = u_w

    # Return scalar if both inputs were scalars
    if return_scalar:
        return float(u_out.item())
    return u_out


# # ------------------ quick sanity checks ------------------
# if __name__ == "__main__":
#     # 1) Scalars
#     print("scalar:", get_u(1.0, 0.1))

#     # 2) (l_arr, e) single e
#     l_test = np.linspace(-10, 10, 5)
#     print("l_arr, e:", get_u(l_test, 0.3))

#     # 3) (l, e_arr) single l
#     e_test = np.linspace(0.0, 0.9, 5)
#     print("l, e_arr:", get_u(1.0, e_test))

#     # 4) (l_arr, e_arr) same shape
#     l_arr = np.linspace(0, 2*np.pi, 6)
#     e_arr = np.linspace(0.1, 0.6, 6)
#     print("l_arr, e_arr:", get_u(l_arr, e_arr))
