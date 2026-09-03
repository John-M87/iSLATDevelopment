# -*- coding: utf-8 -*-
"""
Instrumental line-spread-function (LSF) profile classes.

Each profile exposes a single method ``get_R(wavelength_um)`` that returns
the resolving power *R = λ / δλ* at the given wavelength(s).  Profiles are
registered in :data:`PROFILE_REGISTRY` so that new instruments can be added
without touching :class:`~iSLAT.Modules.DataTypes.Molecule.Molecule` or
:class:`~iSLAT.Modules.DataTypes.Spectrum.Spectrum`.

Adding a new profile
--------------------
1. Subclass :class:`InstrumentalProfile` and implement ``get_R``.
2. Add it to :data:`PROFILE_REGISTRY` with a unique string key.
3. The key will automatically appear in the GUI drop-down.
"""
from __future__ import annotations

import numpy as np
import iSLAT.Constants as c

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class InstrumentalProfile:
    """Abstract base class for instrumental LSF profiles.

    Subclasses must implement :meth:`get_R`.
    """

    #: Human-readable name shown in the GUI drop-down.
    display_name: str = "Unknown profile"

    def get_R(self, wavelength_um: "np.ndarray | float") -> "np.ndarray | float":
        """Return resolving power *R(λ)* at *wavelength_um* (µm).

        Parameters
        ----------
        wavelength_um:
            One or more wavelengths in microns.

        Returns
        -------
        float or np.ndarray
            Resolving power *R = λ / δλ* (dimensionless).
        """
        raise NotImplementedError

    def get_fwhm_kms(self, wavelength_um: "np.ndarray | float") -> "np.ndarray | float":
        """Return FWHM in km/s:  ``c / R(λ)``."""
        return c.SPEED_OF_LIGHT_KMS / self.get_R(wavelength_um)

# ---------------------------------------------------------------------------
# Constant (instrument-independent) profile
# ---------------------------------------------------------------------------
class ConstantProfile(InstrumentalProfile):
    """Wavelength-independent Gaussian LSF defined by a constant FWHM in km/s.

    This is the classical single-value ``fwhm`` behaviour and is the default.

    Parameters
    ----------
    fwhm_kms:
        Instrumental FWHM in km/s.  Must be > 0.
    """

    display_name = "Constant FWHM"

    def __init__(self, fwhm_kms: float = c.DEFAULT_FWHM):
        if fwhm_kms <= 0:
            raise ValueError(f"fwhm_kms must be positive, got {fwhm_kms}")
        self._fwhm_kms = float(fwhm_kms)
        # Pre-compute R so get_R is as cheap as possible
        self._R = c.SPEED_OF_LIGHT_KMS / self._fwhm_kms

    def get_R(self, wavelength_um: "np.ndarray | float") -> "np.ndarray | float":  # noqa: ARG002
        """Return the constant resolving power (wavelength is ignored)."""
        return self._R

    def get_fwhm_kms(self, wavelength_um: "np.ndarray | float") -> "np.ndarray | float":  # noqa: ARG002
        return self._fwhm_kms

# ---------------------------------------------------------------------------
# JWST MIRI MRS profile
# ---------------------------------------------------------------------------
class MiriMrsProfile(InstrumentalProfile):
    """Wavelength-dependent LSF for the JWST MIRI Medium-Resolution Spectrometer.

    Uses a piecewise-linear fit  ``R = a + b·λ``  for each of the 12 MRS
    sub-bands.  Wavelengths outside all band boundaries (i.e. outside the
    4.9 - 28.1 µm coverage) return ``np.nan``.

    No constructor arguments are required.
    """

    display_name = "JWST MIRI MRS"

    # (a, b, λ_min, λ_max) — R = a + b·λ, bands are half-open (λ_min, λ_max]
    _BANDS: list[tuple[float, float, float, float]] = [
        ( -3451, 216, 24.19, 28.10),
        ( -1076, 150, 20.69, 24.48),
        ( -2066, 225, 17.70, 20.95),
        ( -2240, 312, 15.41, 17.98),
        ( -1871, 317, 13.34, 15.57),
        ( -5120, 633, 11.55, 13.47),
        (   430, 264, 10.02, 11.70),
        (  -331, 400,  8.67, 10.13),
        (   332, 400,  7.51,  8.77),
        (  -543, 601,  6.53,  7.65),
        (  2742, 150,  5.66,  6.63),
        (   -20, 650,  4.90,  5.74),
    ]

    def get_R(self, wavelength_um: "np.ndarray | float") -> "np.ndarray | float":
        """Return the MIRI MRS resolving power *R(λ)* at *wavelength_um* (µm).

        Parameters
        ----------
        wavelength_um:
            Wavelength(s) in microns.

        Returns
        -------
        float or np.ndarray
            Resolving power *R*.  ``np.nan`` for wavelengths outside the
            MIRI MRS coverage (4.9 - 28.1 µm).
        """
        scalar_input = np.ndim(wavelength_um) == 0
        wave = np.atleast_1d(np.asarray(wavelength_um, dtype=float))

        R = np.full(wave.shape, np.nan, dtype=float)
        for a, b, lo, hi in self._BANDS:
            mask = (wave > lo) & (wave <= hi)
            R[mask] = a + b * wave[mask]

        if scalar_input:
            return float(R[0])
        return R

# ---------------------------------------------------------------------------
# Registry — maps string key → profile class
# ---------------------------------------------------------------------------

#: Maps string keys to :class:`InstrumentalProfile` subclasses.
#:
#: Keys are stored per-molecule in ``Molecule.instrumental_profile_key`` and
#: persisted in save files.  The key ``"constant"`` is the default fallback
#: and **must** always be present.
PROFILE_REGISTRY: dict[str, type[InstrumentalProfile]] = {
    "constant": ConstantProfile,
    "miri_mrs": MiriMrsProfile,
}

#: Human-readable labels for GUI drop-downs, keyed by registry key.
PROFILE_DISPLAY_NAMES: dict[str, str] = {
    key: cls.display_name for key, cls in PROFILE_REGISTRY.items()
}