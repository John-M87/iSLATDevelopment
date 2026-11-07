"""Strategy pattern for intensity calculation methods.

Defines a :class:`IntensityStrategy` Protocol and concrete implementations for each calculation method supported by :class:`Intensity`.
New methods can be added by implementing the protocol and registering via ``Intensity.register_strategy()``.

The strategy is called by :meth:`Intensity._calc_intensity_core` during the final intensity-from-tau step.
Everything upstream (Boltzmann factors, optical-depth computation, partition function interpolation) is method-agnostic and stays in the core.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .Intensity import Intensity

# ===================================================================
#  Protocol
# ===================================================================
@runtime_checkable
class IntensityStrategy(Protocol):
    """Protocol for intensity calculation strategies.

    A strategy converts center-of-line optical depths into line
    intensities.  It receives pre-computed physical scalars and arrays
    from the ``Intensity`` object.

    Implementations must be **stateless** (no mutable instance data) —
    all information is passed via the call arguments.
    """

    def calculate(
        self,
        intensity_obj: "Intensity",
        center_tau: np.ndarray,
        dv_vals: np.ndarray,
        bb_vals: np.ndarray,
        freq_ratio: np.ndarray,
        sqrt_ln2_inv: float,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        """Compute line intensities from optical depths.

        Parameters
        ----------
        intensity_obj : Intensity
            The calling ``Intensity`` instance (provides quadrature
            data, overlap caches, etc.).
        center_tau : np.ndarray, shape ``(n_cond, n_lines)``
            Line-center optical depths.
        dv_vals : np.ndarray, shape ``(n_cond,)``
            Doppler broadening (FWHM) in km/s per condition.
        bb_vals : np.ndarray, shape ``(n_cond, n_lines)``
            Planck function values.
        freq_ratio : np.ndarray, shape ``(1, n_lines)``
            ``nu / c`` conversion factors.
        sqrt_ln2_inv : float
            ``1 / (2 √(ln 2))`` normalisation constant.
        frequencies : np.ndarray, shape ``(n_lines,)``
            Line rest frequencies in Hz.

        Returns
        -------
        np.ndarray, shape ``(n_cond, n_lines)``
            Line intensities.
        """
        ...

# ===================================================================
#  Concrete strategies
# ===================================================================
class RadexStrategy:
    """Simple ``1 - e^{−τ}`` formula (van der Tak et al. 2007).

    Saturates at τ ~= a few; identical to curve-of-growth for τ < 1.
    """

    __slots__ = ()

    def calculate(
        self,
        intensity_obj: "Intensity",
        center_tau: np.ndarray,
        dv_vals: np.ndarray,
        bb_vals: np.ndarray,
        freq_ratio: np.ndarray,
        sqrt_ln2_inv: float,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        import iSLAT.Constants as c
        return (
            c.FGAUSS_PREFACTOR * 1e5
            * dv_vals[:, np.newaxis]
            * freq_ratio
            * bb_vals
            * (-np.expm1(-center_tau))
        )

class CurveOfGrowthStrategy:
    """Full curve-of-growth with overlap treatment (McCaskill et al. 2026).

    Delegates to ``Intensity._fint_multi`` which handles isolated-line
    fast paths, tau-regime shortcuts, and multi-line Gauss-Legendre
    quadrature for blended groups.
    """

    __slots__ = ()

    def calculate(
        self,
        intensity_obj: "Intensity",
        center_tau: np.ndarray,
        dv_vals: np.ndarray,
        bb_vals: np.ndarray,
        freq_ratio: np.ndarray,
        sqrt_ln2_inv: float,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        return intensity_obj._fint_multi(
            center_tau, dv_vals, bb_vals, freq_ratio, sqrt_ln2_inv, frequencies
        )

class CurveOfGrowthNoOverlapStrategy:
    """Curve-of-growth without overlap grouping.

    Uses the same Gauss-Legendre quadrature as :class:`CurveOfGrowthStrategy`
    but treats every line independently (Banzatti et al. 2012).
    """

    __slots__ = ()

    def calculate(
        self,
        intensity_obj: "Intensity",
        center_tau: np.ndarray,
        dv_vals: np.ndarray,
        bb_vals: np.ndarray,
        freq_ratio: np.ndarray,
        sqrt_ln2_inv: float,
        frequencies: np.ndarray,
    ) -> np.ndarray:
        cls = intensity_obj.__class__
        if not cls._GAUSS_QUAD_INITIALIZED:
            cls._initialize_gauss_quad()

        original_shape = center_tau.shape
        tau_flat = center_tau.ravel()
        integrand = 1.0 - np.exp(
            -tau_flat[:, np.newaxis] * cls._GAUSS_QUAD_EXP[np.newaxis, :]
        )
        fint_vals = np.dot(integrand, cls._GAUSS_QUAD_W).reshape(original_shape)

        return (
            sqrt_ln2_inv * 1e5
            * dv_vals[:, np.newaxis]
            * freq_ratio
            * bb_vals
            * fint_vals
        )

# ===================================================================
#  Default registry
# ===================================================================
#: Mapping of method name → strategy instance.
#: ``Intensity._get_strategy()`` consults this when dispatching.
DEFAULT_STRATEGIES: dict[str, IntensityStrategy] = {
    "radex": RadexStrategy(),
    "curve_growth": CurveOfGrowthStrategy(),
    "curve_growth_no_overlap": CurveOfGrowthNoOverlapStrategy(),
}

#: Display name and short description of each built-in method, in the order
#: they are offered to the user (the default, ``"curve_growth"``, first).
#: See :meth:`Intensity.available_strategies` / :meth:`Intensity.describe_strategy`.
STRATEGY_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "curve_growth": (
        "Curve of growth (with line overlap)",
        "Full curve-of-growth integral (McCaskill et al. 2026). Lines that "
        "overlap in velocity are integrated together, so blended optically "
        "thick lines are not over-counted. Default.",
    ),
    "curve_growth_no_overlap": (
        "Curve of growth (no overlap)",
        "The same curve-of-growth integral, but every line is treated "
        "independently (Banzatti et al. 2012).",
    ),
    "radex": (
        "RADEX (1 - e^−τ)",
        "Simple expression used e.g. in RADEX (van der Tak et al. 2007). "
        "Saturates at τ ≈ a few; identical to the curve of growth for τ < 1.",
    ),
}