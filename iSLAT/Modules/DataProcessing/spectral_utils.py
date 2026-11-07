"""Shared spectral resampling utilities.

Contains the flux-conserving ``spectres``-style resampling function used
by both :class:`~iSLAT.Modules.DataTypes.Molecule.Molecule` and
:class:`~iSLAT.Modules.DataTypes.Spectrum.Spectrum`.  Having a single
canonical implementation avoids duplication and makes updates easier.
"""

from __future__ import annotations

from typing import Tuple
import warnings

import numpy as np

import iSLAT.Constants as c

def make_bins(wavs: np.ndarray):
    """Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.

    Parameters
    ----------
    wavs : np.ndarray
        1-D array of wavelength centers (must be sorted, length ≥ 2).

    Returns
    -------
    edges : np.ndarray
        Bin edges (length ``len(wavs) + 1``).
    widths : np.ndarray
        Bin widths (same length as *wavs*).
    """
    edges = np.zeros(wavs.shape[0] + 1)
    widths = np.zeros(wavs.shape[0])
    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2
    widths[-1] = (wavs[-1] - wavs[-2])
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2]) / 2
    edges[1:-1] = (wavs[1:] + wavs[:-1]) / 2
    widths[:-1] = edges[1:-1] - edges[:-2]
    return edges, widths


def spectres(
    new_wavs: np.ndarray,
    spec_wavs: np.ndarray,
    spec_fluxes: np.ndarray,
    fill: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """Flux-conserving spectral resampling onto a new wavelength basis.

    Vectorized implementation using ``np.searchsorted`` to map new bin edges
    onto the old wavelength grid in *O(n log m)* time.

    Parameters
    ----------
    new_wavs : numpy.ndarray
        Array containing the new wavelength sampling desired.
    spec_wavs : numpy.ndarray
        1-D array containing the current wavelength sampling.
    spec_fluxes : numpy.ndarray
        Array containing spectral fluxes at the wavelengths in *spec_wavs*.
    fill : float, optional
        Value to use where *new_wavs* extends outside *spec_wavs* range.
    verbose : bool, optional
        If *True*, warn when fill values are used.

    Returns
    -------
    new_fluxes : numpy.ndarray
        Array of resampled flux values with same length as *new_wavs*.
    """
    old_edges, old_widths = make_bins(spec_wavs)
    new_edges, _ = make_bins(new_wavs)

    n_new = new_wavs.shape[0]
    new_fluxes = np.full(n_new, fill, dtype=np.float64)

    # Identify new bins that fall entirely within the old grid
    valid = (new_edges[:-1] >= old_edges[0]) & (new_edges[1:] <= old_edges[-1])
    if not np.any(valid):
        if verbose:
            warnings.warn(
                "spectres: new_wavs contains values outside the range "
                "in spec_wavs, new_fluxes will be filled with fill value.",
                category=RuntimeWarning,
            )
        return new_fluxes

    valid_idx = np.where(valid)[0]

    # Map new bin edges onto old bin edges via searchsorted.
    left_edges = new_edges[valid_idx]
    right_edges = new_edges[valid_idx + 1]

    start_idx = np.searchsorted(old_edges, left_edges, side="right") - 1
    stop_idx = np.searchsorted(old_edges, right_edges, side="left") - 1

    # Clip to valid bin range
    n_old = spec_wavs.shape[0]
    np.clip(start_idx, 0, n_old - 1, out=start_idx)
    np.clip(stop_idx, 0, n_old - 1, out=stop_idx)

    # Fast path: bins where start == stop (new bin is entirely inside one old bin)
    same = start_idx == stop_idx
    if np.any(same):
        new_fluxes[valid_idx[same]] = spec_fluxes[start_idx[same]]

    # Handle bins that span multiple old bins
    diff_mask = ~same
    if np.any(diff_mask):
        diff_idx = valid_idx[diff_mask]
        d_start = start_idx[diff_mask]
        d_stop = stop_idx[diff_mask]
        d_left = left_edges[diff_mask]
        d_right = right_edges[diff_mask]

        start_factor = (
            (old_edges[d_start + 1] - d_left)
            / (old_edges[d_start + 1] - old_edges[d_start])
        )
        end_factor = (
            (d_right - old_edges[d_stop])
            / (old_edges[d_stop + 1] - old_edges[d_stop])
        )

        for k in range(len(diff_idx)):
            s = d_start[k]
            e = d_stop[k]
            sl = slice(s, e + 1)
            w = old_widths[sl].copy()
            w[0] *= start_factor[k]
            w[-1] *= end_factor[k]
            fw = w * spec_fluxes[sl]
            new_fluxes[diff_idx[k]] = fw.sum() / w.sum()

    return new_fluxes

def extend_grid_to_range(grid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Pad a sorted wavelength *grid* so it covers ``[lo, hi]``.

    Extra pixels are added below and/or above at the grid's median spacing,
    so the extension keeps the sampling of the observed data.  The input is
    returned unchanged when it already covers the range, has fewer than two
    points, or has no usable spacing.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        return grid
    finite = grid[np.isfinite(grid)]
    if finite.size < 2:
        return grid
    spacing = float(np.median(np.diff(np.sort(finite))))
    if not np.isfinite(spacing) or spacing <= 0:
        return grid
    gmin, gmax = float(finite.min()), float(finite.max())
    parts = []
    if lo is not None and np.isfinite(lo) and lo < gmin - spacing:
        n_lo = int(np.ceil((gmin - lo) / spacing))
        parts.append(gmin - spacing * np.arange(n_lo, 0, -1))
    parts.append(grid)
    if hi is not None and np.isfinite(hi) and hi > gmax + spacing:
        n_hi = int(np.ceil((hi - gmax) / spacing))
        parts.append(gmax + spacing * np.arange(1, n_hi + 1))
    if len(parts) == 1:
        return grid
    return np.concatenate(parts)


def flux_integral(lam, flux, err, lam_min, lam_max) -> Tuple[float, float]:
    """Integrate a flux density over ``[lam_min, lam_max]``.

    Parameters
    ----------
    lam : np.ndarray
        Wavelengths in um.
    flux : np.ndarray
        Flux density F_nu in Jy at *lam*.
    err : np.ndarray or None
        Flux uncertainties in Jy (integrated the same way as *flux*).
    lam_min, lam_max : float
        Wavelength range in um.

    Returns
    -------
    (float, float)
        The flux and the uncertainty integrated over frequency, in
        erg/s/cm^2, or ``(0.0, 0.0)`` with fewer than two points in range.
    """
    lam = np.asarray(lam, dtype=float)
    wavelength_mask = (lam >= lam_min) & (lam <= lam_max)

    if not np.any (wavelength_mask):
        return 0.0, 0.0

    lam_range = lam[wavelength_mask]
    flux_range = np.asarray(flux, dtype=float)[wavelength_mask]

    if len (lam_range) < 2:
        return 0.0, 0.0

    # Integrate in frequency space.  Every flux value must stay paired with
    # its own frequency, so sort both by frequency together (ascending
    # frequency, i.e. descending wavelength) rather than reversing only one of them.
    freq_range = c.SPEED_OF_LIGHT_MICRONS / lam_range
    order = np.argsort(freq_range)
    freq_sorted = freq_range[order]

    line_flux_meas = np.trapezoid(flux_range[order], x=freq_sorted)
    line_flux_meas = line_flux_meas * 1e-23  # Convert Jy*Hz to erg/s/cm^2

    # Calculate error propagation if error data provided
    if err is not None:
        err_range = np.asarray(err, dtype=float)[wavelength_mask]
        line_err_meas = np.trapezoid(err_range[order], x=freq_sorted) * 1e-23
    else:
        line_err_meas = 0.0

    return line_flux_meas, line_err_meas


def opacity_integral(lam, tau, lam_min, lam_max) -> float:
    """Integrate a model optical-depth profile over ``[lam_min, lam_max]``.

    *tau* is the convolved profile of :meth:`Molecule.get_tau` /
    :attr:`Spectrum.tau_profile`, in which every line's line-center optical
    depth is spread over an area-normalised Gaussian in wavelength (units of
    1/um).  Its integral over wavelength therefore gives the **summed
    line-center optical depth of the lines in the range** (dimensionless;
    lines near the range edges count only with the part of their profile
    inside the range).

    Unlike :func:`flux_integral`, no frequency conversion or Jy unit factor
    is applied - the optical depth is not a flux density.

    Returns
    -------
    float
        The optical-depth profile integrated over wavelength in the range,
        or ``0.0`` with fewer than two points in range.
    """
    lam = np.asarray(lam, dtype=float)
    mask = (lam >= lam_min) & (lam <= lam_max)
    if np.count_nonzero(mask) < 2:
        return 0.0
    lam_range = lam[mask]
    order = np.argsort(lam_range)
    return float(np.trapezoid(np.asarray(tau, dtype=float)[mask][order], x=lam_range[order]))