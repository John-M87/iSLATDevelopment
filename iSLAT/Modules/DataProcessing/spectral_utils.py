"""Shared spectral resampling utilities.

Contains the flux-conserving ``spectres``-style resampling function used
by both :class:`~iSLAT.Modules.DataTypes.Molecule.Molecule` and
:class:`~iSLAT.Modules.DataTypes.Spectrum.Spectrum`.  Having a single
canonical implementation avoids duplication and makes updates easier.
"""
from __future__ import annotations
from typing import Optional, Tuple
import warnings
import numpy as np
from numpy.typing import ArrayLike, NDArray
import iSLAT.Constants as c

def make_bins(wavs: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Given a series of wavelength points, find the edges and widths
    of corresponding wavelength bins.

    Inner edges are the midpoints between neighbouring wavelengths; the two
    outer edges lie half a spacing beyond the first and last point.

    Parameters
    ----------
    wavs : array_like
        1-D wavelength centers, sorted ascending, at least two points.

    Returns
    -------
    edges : numpy.ndarray
        Bin edges (length ``len(wavs) + 1``).
    widths : numpy.ndarray
        Bin widths, ``numpy.diff(edges)`` (same length as *wavs*).
    """
    wavs = np.asarray(wavs, dtype=np.float64)
    edges = np.empty(wavs.shape[0] + 1)
    edges[1:-1] = (wavs[1:] + wavs[:-1]) / 2
    edges[0] = wavs[0] - (wavs[1] - wavs[0]) / 2
    edges[-1] = wavs[-1] + (wavs[-1] - wavs[-2]) / 2
    return edges, np.diff(edges)

def spectres(
    new_wavs: ArrayLike,
    spec_wavs: ArrayLike,
    spec_fluxes: ArrayLike,
    fill: float = 0.0,
    verbose: bool = False,
) -> NDArray[np.float64]:
    """Flux-conserving spectral resampling onto a new wavelength basis.

    Each new bin gets the mean of the old spectrum over the bin, with the old
    spectrum taken as constant across each old bin (the SpectRes algorithm,
    Carnall 2017).  Old bins only partly inside a new bin count with the
    fraction of their width that is inside it.

    Fully vectorized: the new bin edges are located on the old grid with
    ``numpy.searchsorted`` and the old bins between them are summed per new
    bin with ``numpy.add.reduceat``.  Each sum covers only that bin's own old
    pixels, so a NaN affects only the new bins that contain it and faint bins
    keep full precision next to bright lines (a running cumulative sum would
    spread a NaN to every later bin and lose precision after a bright line).

    Parameters
    ----------
    new_wavs : array_like
        The new wavelength sampling, sorted ascending.
    spec_wavs : array_like
        1-D current wavelength sampling, sorted ascending.
    spec_fluxes : array_like
        Spectral fluxes at the wavelengths in *spec_wavs*.
    fill : float, optional
        Value for new bins that are not entirely inside the range of
        *spec_wavs*.
    verbose : bool, optional
        If *True*, warn when no new bin is inside the range of *spec_wavs*.

    Returns
    -------
    new_fluxes : numpy.ndarray
        Resampled float64 fluxes, same length as *new_wavs*.
    """
    new_wavs = np.asarray(new_wavs, dtype=np.float64)
    spec_wavs = np.asarray(spec_wavs, dtype=np.float64)
    spec_fluxes = np.asarray(spec_fluxes, dtype=np.float64)
    old_edges, old_widths = make_bins(spec_wavs)
    new_edges, _ = make_bins(new_wavs)
    new_fluxes = np.full(new_wavs.shape[0], fill, dtype=np.float64)

    # Only new bins entirely inside the old grid are resampled; the rest keep *fill*.
    valid = (new_edges[:-1] >= old_edges[0]) & (new_edges[1:] <= old_edges[-1])
    if not np.any(valid):
        if verbose:
            warnings.warn(
                "spectres: new_wavs contains values outside the range "
                "in spec_wavs, new_fluxes will be filled with fill value.",
                category=RuntimeWarning,
            )
        return new_fluxes
    idx = np.flatnonzero(valid)
    left = new_edges[idx]
    right = new_edges[idx + 1]

    # Old bins holding each new bin's left and right edge.
    last = spec_wavs.shape[0] - 1
    start = np.clip(np.searchsorted(old_edges, left, side="right") - 1, 0, last)
    stop = np.clip(np.searchsorted(old_edges, right, side="left") - 1, 0, last)

    # New bin inside a single old bin: that bin's flux.
    single = start == stop
    new_fluxes[idx[single]] = spec_fluxes[start[single]]

    # New bin across several old bins: width-weighted mean flux, with the
    # first and last old bin weighted by the fraction of them inside it.
    multi = ~single
    if np.any(multi):
        idx, left, right = idx[multi], left[multi], right[multi]
        start, stop = start[multi], stop[multi]
        start_factor = (old_edges[start + 1] - left) / (old_edges[start + 1] - old_edges[start])
        end_factor = (right - old_edges[stop]) / (old_edges[stop + 1] - old_edges[stop])
        w_start = old_widths[start] * start_factor
        w_stop = old_widths[stop] * end_factor
        # Old bins strictly between the first and last one (none when they are neighbours).
        inner_w = _segment_sums(old_widths, start + 1, stop)
        inner_fw = _segment_sums(old_widths * spec_fluxes, start + 1, stop)
        numerator = w_start * spec_fluxes[start] + inner_fw + w_stop * spec_fluxes[stop]
        new_fluxes[idx] = numerator / (w_start + inner_w + w_stop)

    return new_fluxes

def _segment_sums(values: NDArray[np.float64], begin: NDArray[np.intp], end: NDArray[np.intp]) -> NDArray[np.float64]:
    """``values[begin[i]:end[i]].sum()`` for every *i*, 0 for empty segments.

    Requires ``begin <= end < len(values)`` wherever ``begin < end``.
    """
    sums = np.zeros(begin.shape[0])
    nonempty = end > begin
    if np.any(nonempty):
        # reduceat sums values[i0:i1], values[i1:i2], ...; with (begin, end)
        # pairs interleaved, every other result is one segment's sum.
        bounds = np.column_stack((begin[nonempty], end[nonempty])).ravel()
        sums[nonempty] = np.add.reduceat(values, bounds)[::2]
    return sums

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

def flux_integral(
    lam: ArrayLike,
    flux: ArrayLike,
    err: Optional[ArrayLike],
    lam_min: float,
    lam_max: float,
) -> Tuple[float, float]:
    """Integrate a flux density over ``[lam_min, lam_max]``.

    Parameters
    ----------
    lam : array_like
        Wavelengths in um, in any order.
    flux : array_like
        Flux density F_nu in Jy at *lam*.
    err : array_like or None
        Flux uncertainties in Jy at *lam* (integrated the same way as *flux*).
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

def opacity_integral(lam: ArrayLike, tau: ArrayLike, lam_min: float, lam_max: float) -> float:
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

    Parameters
    ----------
    lam : array_like
        Wavelengths in um, in any order.
    tau : array_like
        Optical-depth profile at *lam*, in 1/um.
    lam_min, lam_max : float
        Wavelength range in um.

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