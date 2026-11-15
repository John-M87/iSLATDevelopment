"""Float64 NumPy references for the overlapping-line (blend) integral.

Test helper (not a test module): the oracle of the per-line overlap
kernel.  Independent of ``iSLAT.Modules.DataTypes._overlap_perline``: no
neighbour tables, no padding, no buckets, no float32, no threads.

Conventions (matching the slab model):

* ``x`` is velocity in units of the Doppler parameter
  ``b = dv / (2 sqrt(ln 2))`` (``dv`` = intrinsic FWHM, km/s);
* line ``j`` sits at ``d_ij = c (nu_j - nu_i) / nu_i / b`` from line ``i``;
* ``fint_i = integral exp(-x^2) tau_i (1 - exp(-tau_tot)) / tau_tot dx``
  with ``tau_tot(x) = sum_j tau_j exp(-(x - d_ij)^2)``.
"""
import numpy as np

C_KMS = 299792.458
TWO_SQRT_LN2 = 2.0 * np.sqrt(np.log(2.0))


def gauss_legendre_nodes(n=20, half_width=6.0):
    """The slab model's quadrature: ``n``-point Gauss-Legendre on ``[-hw, hw]``."""
    x, w = np.polynomial.legendre.leggauss(n)
    return half_width * x, half_width * w


def escape_factor(tau):
    """``(1 - exp(-tau)) / tau`` in float64, exact limit 1 at ``tau = 0``."""
    tau = np.asarray(tau, dtype=np.float64)
    out = np.ones_like(tau)
    pos = tau > 1e-300
    out[pos] = -np.expm1(-tau[pos]) / tau[pos]
    return out


def perline_fint_reference(tau, freq, dv, x_quad=None, w_quad=None,
                           radius_x=18.0, lines_per_block=1024):
    """Per-line overlap integral, float64, on the given quadrature nodes.

    Every line is integrated on its own window with ALL lines within
    ``radius_x`` Doppler widths as neighbours (18 is "everything": at
    the outermost node, |x| < 6, a neighbour at 18 contributes
    exp(-144)).  Neighbours are searched per condition at that
    condition's own ``dv``.

    Parameters
    ----------
    tau  : (n_cond, n_lines) line-centre opacities
    freq : (n_lines,) rest frequencies (Hz), any order
    dv   : (n_cond,) intrinsic FWHM (km/s)

    Returns
    -------
    (n_cond, n_lines) float64, in the input line order.
    """
    if x_quad is None:
        x_quad, w_quad = gauss_legendre_nodes()
    tau = np.atleast_2d(np.asarray(tau, dtype=np.float64))
    freq = np.asarray(freq, dtype=np.float64)
    dv = np.atleast_1d(np.asarray(dv, dtype=np.float64))
    n_cond, n_lines = tau.shape

    order = np.argsort(freq, kind="stable")
    fs = freq[order]
    out = np.zeros_like(tau)

    for c in range(n_cond):
        b = dv[c] / TWO_SQRT_LN2
        tau_s = tau[c][order]
        acc = np.zeros(n_lines)
        for start in range(0, n_lines, lines_per_block):
            stop = min(start + lines_per_block, n_lines)
            f_i = fs[start:stop]
            f_lo = f_i * (1.0 - radius_x * b / C_KMS)
            f_hi = f_i * (1.0 + radius_x * b / C_KMS)
            lo = np.searchsorted(fs, f_lo, side="left")
            hi = np.searchsorted(fs, f_hi, side="right")
            width = int((hi - lo).max())
            pos = lo[:, None] + np.arange(width)[None, :]
            valid = pos < hi[:, None]
            pos = np.where(valid, pos, 0)
            d = C_KMS * (fs[pos] - f_i[:, None]) / f_i[:, None] / b
            t_nb = np.where(valid, tau_s[pos], 0.0)
            blk = np.zeros(stop - start)
            for xq, wq in zip(x_quad, w_quad):
                tau_tot = np.sum(t_nb * np.exp(-(xq - d) ** 2), axis=1)
                blk += wq * np.exp(-xq * xq) * escape_factor(tau_tot)
            acc[start:stop] = blk
        out[c, order] = acc * tau_s
    return out


def isolated_fint_reference(tau, x_quad=None, w_quad=None):
    """No-overlap curve of growth on the same nodes: ``sum_q w_q (1 - exp(-tau exp(-x_q^2)))``."""
    if x_quad is None:
        x_quad, w_quad = gauss_legendre_nodes()
    tau = np.asarray(tau, dtype=np.float64)
    out = np.zeros_like(tau)
    for xq, wq in zip(x_quad, w_quad):
        out += wq * -np.expm1(-tau * np.exp(-xq * xq))
    return out


def blend_equivalent_width(tau, freq, dv, pad_x=12.0, step_x=0.01):
    """Exact total emission of a blend, ``integral (1 - exp(-tau_tot(x))) dx``.

    Dense trapezoid over the whole blend, in Doppler-width units of the
    single condition ``dv`` - no quadrature rule, no per-line split.
    By construction ``sum_i fint_i`` must equal this (each line gets
    the share ``tau_i phi_i / tau_tot`` of the blend's emission).

    ``tau`` is (n_lines,), ``dv`` a scalar.
    """
    tau = np.asarray(tau, dtype=np.float64)
    freq = np.asarray(freq, dtype=np.float64)
    b = float(dv) / TWO_SQRT_LN2
    pos = C_KMS * (freq - freq.min()) / freq.min() / b      # line centres, units of b
    grid = np.arange(-pad_x, pos.max() + pad_x + step_x, step_x)
    tau_tot = np.zeros_like(grid)
    for p, t in zip(pos, tau):
        near = np.abs(grid - p) < 9.0          # exp(-81) beyond
        tau_tot[near] += t * np.exp(-(grid[near] - p) ** 2)
    return float(np.sum(-np.expm1(-tau_tot)) * step_x)


def overlap_intensity_reference(intensity_no_overlap, tau, freq, dv, **kwargs):
    """Overlap-aware line intensities from no-overlap ones (float64).

    ``intensity = prefactor * fint`` with the same prefactor with or
    without overlap, so the no-overlap intensity of each line is scaled
    by the reference's ratio of overlapping to isolated ``fint``.

    Parameters
    ----------
    intensity_no_overlap, tau : (n_lines,)
        From ``Intensity`` with ``method="curve_growth_no_overlap"``
        (float64 tau and Planck).
    freq : (n_lines,) rest frequencies (Hz)
    dv : float, intrinsic FWHM (km/s)
    """
    intensity_no_overlap = np.asarray(intensity_no_overlap, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    fint_ov = perline_fint_reference(tau[None, :], freq, np.array([float(dv)]),
                                     **kwargs)[0]
    fint_iso = isolated_fint_reference(tau)
    out = np.zeros_like(intensity_no_overlap)
    live = fint_iso > 0
    out[live] = intensity_no_overlap[live] * fint_ov[live] / fint_iso[live]
    return out
