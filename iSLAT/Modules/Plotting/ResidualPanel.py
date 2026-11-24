"""
ResidualPanel -- Concrete :class:`SpectralPanel` for residual display.

Renders the ``(observed - model)`` residual with error bars, a zero reference line, optional error-envelope bands, and excluded-range shading.
This is the lower sub-panel in each :class:`ResidualSpectrumPlot` row.
"""

from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .ErrorBars import fast_errorbar
from .SpectralPanel import SpectralPanel, GapMode

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

def residual_ylim(flux: np.ndarray, model: np.ndarray, error: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Symmetric ``(-pad, pad)`` limits: the largest finite |flux - model| or error, plus 30%."""
    values = [np.abs(flux - model)] + ([error] if error is not None else [])
    finite = [v[np.isfinite(v)] for v in values]
    top = max((float(v.max()) for v in finite if v.size), default=0.0)
    pad = top * 1.3 if top > 0 else 0.01
    return (-pad, pad)


class ResidualPanel(SpectralPanel):
    """
    Single-axes panel showing ``(data - model)`` residuals.

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array.
    xmin, xmax : float
        Wavelength bounds for this panel.
    model_flux_adj : np.ndarray
        Adjusted model flux array (same grid as *wave_data*).
    error_data : np.ndarray, optional
        Original pipeline flux uncertainties.
    error_adj : np.ndarray, optional
        Effective errors (e.g. with noise floor added in quadrature).
    has_noise_floor : bool
        When *True*, both original and expanded error envelopes are drawn.
    excluded_ranges : list[tuple[float, float]], optional
        Wavelength ranges to shade as excluded.
    atomic_lines : pd.DataFrame, optional
        Atomic-line DataFrame (for exclusion-window shading).
    exclude_lines_half_width : float, optional
        Half-width of exclusion windows around atomic lines.
    is_first_row : bool
        When *True*, error-envelope labels are added for the legend.
    ax : Axes, optional
        Pre-existing axes to draw into.
    **kwargs
        Forwarded to :class:`SpectralPanel` / :class:`BasePlot`.
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        model_flux_adj: np.ndarray,
        error_data: Optional[np.ndarray] = None,
        error_adj: Optional[np.ndarray] = None,
        has_noise_floor: bool = False,
        excluded_ranges: Optional[List[Tuple[float, float]]] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        exclude_lines_half_width: Optional[float] = None,
        is_first_row: bool = False,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        super().__init__(
            wave_data=wave_data,
            flux_data=flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=error_data,
            ax=ax,
            **kwargs,
        )
        self.model_flux_adj = np.asarray(model_flux_adj)
        self.error_adj = (
            np.asarray(error_adj) if error_adj is not None else None
        )
        self.has_noise_floor = has_noise_floor
        self.excluded_ranges: List[Tuple[float, float]] = excluded_ranges or []
        self.atomic_lines = atomic_lines
        self.exclude_lines_half_width = exclude_lines_half_width
        self.is_first_row = is_first_row

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Render the residual panel."""
        ax = self._resolve_axes()
        xr = self.xlim
        fg = self._get_theme_value("foreground", "black")

        mask = self.get_panel_mask()
        panel_wave = self.wave_data[mask]
        panel_flux = self.flux_data[mask]
        panel_model = self.model_flux_adj[mask]
        panel_err_raw = (
            self.error_data[mask] if self.error_data is not None else None
        )
        panel_err = (
            self.error_adj[mask] if self.error_adj is not None else None
        )

        residuals = panel_flux - panel_model

        # When gap_mode is SKIP, filter out NaN residuals so the
        # scatter / errorbar calls receive only finite values.
        if self.gap_mode is GapMode.SKIP:
            finite = np.isfinite(residuals)
            panel_wave = panel_wave[finite]
            residuals = residuals[finite]
            panel_flux = panel_flux[finite]
            panel_model = panel_model[finite]
            if panel_err_raw is not None:
                panel_err_raw = panel_err_raw[finite]
            if panel_err is not None:
                panel_err = panel_err[finite]

        # -- Residual data points ---------------------------------------
        if panel_err is not None and len(panel_err) == len(residuals):
            # Same artists as ax.errorbar, with the bars in one path (see ErrorBars).
            fast_errorbar(
                ax,
                panel_wave,
                residuals,
                panel_err,
                fmt=".",
                markersize=2,
                linewidth=None,
                color=fg,
                ecolor="lightgray",
                elinewidth=0.5,
                zorder=2,
            )
        else:
            ax.plot(
                panel_wave,
                residuals,
                ".",
                ms=2,
                color=fg,
                zorder=2,
            )
        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)

        # -- Error-envelope bands ---------------------------------------
        if (
            self.has_noise_floor
            and panel_err is not None
            and len(panel_err) > 0
        ):
            ax.fill_between(
                panel_wave,
                -panel_err_raw,
                panel_err_raw,
                color="salmon",
                alpha=0.13,
                zorder=1,
                label=(
                    r"$\pm 1\,\sigma_{\rm pipe}$"
                    if self.is_first_row
                    else ""
                ),
            )
            ax.fill_between(
                panel_wave,
                -panel_err,
                panel_err,
                color="dodgerblue",
                alpha=0.13,
                zorder=1,
                label=(
                    r"$\pm 1\,\sigma_{\rm eff}$"
                    if self.is_first_row
                    else ""
                ),
            )

        ax.set_xlim(*xr)

        # -- Shade excluded ranges --------------------------------------
        for exc_lo, exc_hi in self.excluded_ranges:
            if exc_hi >= xr[0] and exc_lo <= xr[1]:
                ax.axvspan(
                    max(exc_lo, xr[0]),
                    min(exc_hi, xr[1]),
                    color="lightcoral",
                    alpha=0.15,
                )

        # -- Shade atomic-line exclusion windows ------------------------
        if (
            self.exclude_lines_half_width is not None
            and self.atomic_lines is not None
            and len(self.atomic_lines) > 0
        ):
            hw = self.exclude_lines_half_width
            for _, arow in self.atomic_lines.iterrows():
                wl = arow["wave"]
                a_lo, a_hi = wl - hw, wl + hw
                if a_hi >= xr[0] and a_lo <= xr[1]:
                    ax.axvspan(
                        max(a_lo, xr[0]),
                        min(a_hi, xr[1]),
                        color="lightsalmon",
                        alpha=0.12,
                    )

        # -- Gap indicators ---------------------------------------------
        # NOTE: Gap indicators are drawn later by _post_render_cell()
        # after y-limits are finalised, so that the break-mark lines
        # and text annotations are positioned correctly.

    # ------------------------------------------------------------------
    def compute_ylim(self, ymax_factor: float = 0.3) -> Tuple[float, float]:
        """Symmetric y-limits around zero (see :func:`residual_ylim`)."""
        mask = self.get_panel_mask()
        error = self.error_adj[mask] if self.error_adj is not None else None
        return residual_ylim(self.flux_data[mask], self.model_flux_adj[mask], error)
