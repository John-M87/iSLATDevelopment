"""
ResidualSpectrumPlot -- Multi-panel spectrum overview with per-panel residuals.

Extends :class:`FullSpectrumPlot` by adding a residual sub-panel below each wavelength-range row, showing ``(observed - model)`` with error bars.
Optionally annotates each row with its reduced chi^2 and displays global goodness-of-fit statistics (chi^2, degrees of freedom, reduced chi^2) at the bottom of the figure.

Each cell in the stacked layout contains two sub-panels produced by :meth:`_create_cell`: a :class:`SpectrumPanel` (top) and a :class:`ResidualPanel` (bottom), arranged via a nested ``GridSpecFromSubplotSpec``.

Designed for best-fit comparison plots, works with any model flux array.
"""
from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from .FullSpectrumPlot import FullSpectrumPlot
from .SpectralPanel import SpectralPanel, GapMode
from .SpectrumPanel import SpectrumPanel
from .ResidualPanel import ResidualPanel
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class ResidualSpectrumPlot(FullSpectrumPlot):
    """
    Multi-panel spectrum + residual plot.

    Inherits the observed-spectrum, molecule-overlay, summed-model, and
    annotation rendering from :class:`FullSpectrumPlot` and adds:

    * A **residual sub-panel** below each spectrum panel showing ``(data - model)`` with error bars.
    * **Per-panel** and **global** reduced chi^2 statistics.
    * Support for **nuisance parameters** (noise floor, linear continuum).
    * **Excluded wavelength ranges** and **atomic-line exclusion windows**.
    * Explicit **model component** overlays (as an alternative to ``MoleculeDict``).

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array.
    flux_data : np.ndarray
        Observed flux array.
    model_flux : np.ndarray
        Model flux array on the same grid as *wave_data* / *flux_data*.
    error_data : np.ndarray, optional
        Flux uncertainties (used for residual error bars and chi^2).
    model_components : list[dict], optional
        Individual model components for overlay.
        Each dict should contain ``"wave"``, ``"flux"``, ``"color"``, and ``"label"`` keys.
    molecules : MoleculeDict, optional
        If provided (instead of *model_components*), visible molecules are plotted as in :class:`FullSpectrumPlot`.
    model_wave : np.ndarray, optional
        Wavelength grid of the combined model (for the filled-area "summed" display).
        Defaults to *wave_data*.
    model_flux_hires : np.ndarray, optional
        High-resolution combined model flux corresponding to *model_wave*.
        When given, the filled area uses this array while residuals still use *model_flux* (which is on the data grid).
    match_pixel_sampling : bool
        When *True*, render the summed-model fill and each entry of
        *model_components* on the observed *wave_data* grid instead of
        their native (typically higher-resolution) grids.
        The fill uses *model_flux* directly (matching the residual /
        chi^2 arrays) and components are flux-conservingly resampled
        with :func:`spectres` once at construction time.
        Default *False* (native-grid rendering, backward compatible).
    line_list : pd.DataFrame, optional
        Saved-line annotations.
    atomic_lines : pd.DataFrame, optional
        Atomic-line annotations.
    n_panels : int
        Target number of wavelength rows.  Default 5.
    step : float, optional
        Fixed wavelength width per row (overrides *n_panels*).
    xlim_range : tuple[float, float], optional
        Wavelength sub-range to display.
    ymax_factor : float
        Fractional headroom above the flux peak per panel (0.2 = 20 %).
    residual_height_ratio : float
        Height of the residual sub-panel relative to the spectrum sub-panel.
        Default 0.3 (residual is 30 % of spectrum height).
    uniform_ylim_spectra : bool
        When *True* all spectrum sub-panels share the same vertical scale (global min/max across panels).
        Default *False*.
    uniform_ylim_residuals : bool
        When *True* all residual sub-panels share the same vertical scale (the worst-case symmetric range across panels).
        Default *True*.
    show_chi2_per_panel : bool
        Annotate each row with its per-panel reduced chi^2.
        Default True.
    show_total_chi2 : bool
        Show global chi^2, dof, and reduced chi^2 at the bottom of the figure.
        Default True.
    n_free_params : int
        Number of free model parameters (used for dof calculation).
        Default 0.
    excluded_ranges : list[tuple[float, float]], optional
        Wavelength ranges excluded from the fit. 
        These are shaded in the plot and excluded from the chi^2 calculation.
    exclude_lines_half_width : float, optional
        Half-width (in the same units as *wave_data*) of exclusion
        windows centered on each line in *atomic_lines*.
        When set together with *atomic_lines*, data points within +/- this distance of any atomic line are excluded from chi^2 statistics (and shaded on the residual panels).
        Default ``None`` (no extra exclusion beyond *excluded_ranges*).
    noise_floor : float, optional
        Systematic noise floor *s* (Jy).
        When given, the effective per-pixel uncertainty becomes ``sigma_eff = sqrt(sigma^2 + s^2)`` for residual error bars and adjusted chi^2 statistics.
        Both the raw and adjusted chi^2 are shown.
        Default ``None``.
    continuum_c0 : float, optional
        Continuum offset constant (Jy).  When given (together with
        *continuum_c1* and *lam_ref*), the additive continuum
        ``c_0 + c_1*(lam - lam_ref)`` is overlaid on spectrum panels and
        added to the model for residual / chi^2 computation.
    continuum_c1 : float, optional
        Continuum slope (Jy / um).  Default ``None``.
    lam_ref : float, optional
        Reference wavelength for the continuum model (um).
    figsize : tuple, optional
        Figure size.  Auto-scaled if *None*.
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        model_flux: np.ndarray,
        error_data: Optional[np.ndarray] = None,
        model_components: Optional[List[Dict[str, Any]]] = None,
        molecules: Optional["MoleculeDict"] = None,
        model_wave: Optional[np.ndarray] = None,
        model_flux_hires: Optional[np.ndarray] = None,
        match_pixel_sampling: bool = False,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        n_panels: int = 5,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        residual_height_ratio: float = 0.3,
        uniform_ylim_spectra: bool = False,
        uniform_ylim_residuals: bool = True,
        show_chi2_per_panel: bool = True,
        show_total_chi2: bool = True,
        n_free_params: int = 0,
        excluded_ranges: Optional[List[Tuple[float, float]]] = None,
        exclude_lines_half_width: Optional[float] = None,
        noise_floor: Optional[float] = None,
        continuum_c0: Optional[float] = None,
        continuum_c1: Optional[float] = None,
        lam_ref: Optional[float] = None,
        **kwargs,
    ):
        # -- Initialise the FullSpectrumPlot base ---------------------
        # Pass figsize=None so the parent stores None; we override
        # self._figsize below to account for the residual sub-panels.
        caller_figsize = kwargs.pop("figsize", None)
        super().__init__(
            wave_data=wave_data,
            flux_data=flux_data,
            molecules=molecules,
            error_data=error_data,
            line_list=line_list,
            atomic_lines=atomic_lines,
            n_panels=n_panels,
            step=step,
            xlim_range=xlim_range,
            ymax_factor=ymax_factor,
            uniform_ylim=uniform_ylim_spectra,
            figsize=caller_figsize,
            **kwargs,
        )

        # Uniform y-limit flags (spectra flag stored on parent as
        # self.uniform_ylim; residuals flag is RSP-specific)
        self.uniform_ylim_residuals = uniform_ylim_residuals

        # -- Model arrays (unique to ResidualSpectrumPlot) ------------
        self.model_flux = np.asarray(model_flux)
        self.match_pixel_sampling = match_pixel_sampling
        self.model_components = self._resample_components_to_data_grid(
            model_components
        )
        self.model_wave = (
            np.asarray(model_wave) if model_wave is not None else self.wave_data
        )
        self.model_flux_hires = (
            np.asarray(model_flux_hires)
            if model_flux_hires is not None
            else None
        )

        # -- Residual / chi^2 settings --------------------------------
        self.residual_height_ratio = residual_height_ratio
        self.show_chi2_per_panel = show_chi2_per_panel
        self.show_total_chi2 = show_total_chi2
        self.n_free_params = n_free_params
        self.excluded_ranges = excluded_ranges or []
        self.exclude_lines_half_width = exclude_lines_half_width

        # -- Nuisance parameters --------------------------------------
        self.noise_floor = noise_floor
        self.continuum_c0 = continuum_c0
        self.continuum_c1 = continuum_c1
        self.lam_ref = lam_ref
        self._has_continuum = (
            continuum_c0 is not None
            and continuum_c1 is not None
            and lam_ref is not None
        )
        self._has_noise_floor = noise_floor is not None and noise_floor > 0.0

        # -- Pre-compute adjusted arrays ------------------------------
        if self._has_continuum:
            cont = continuum_c0 + continuum_c1 * (self.wave_data - lam_ref)
            self._model_flux_adj = self.model_flux + cont
            if self.model_flux_hires is not None:
                cont_hr = continuum_c0 + continuum_c1 * (self.model_wave - lam_ref)
                self._model_flux_hires_adj = self.model_flux_hires + cont_hr
            else:
                self._model_flux_hires_adj = None
        else:
            self._model_flux_adj = self.model_flux
            self._model_flux_hires_adj = self.model_flux_hires

        if self._has_noise_floor and self.error_data is not None:
            self._error_adj = np.sqrt(self.error_data ** 2 + noise_floor ** 2)
        else:
            self._error_adj = self.error_data

        # -- Override figsize for the taller residual layout ----------
        if caller_figsize is None:
            n = len(self._panel_edges)
            row_height = 2.0 + 2.0 * self.residual_height_ratio
            self._figsize = (
                14,
                row_height * n + (1.0 if self.show_total_chi2 else 0.0),
            )

        # Subplot storage override: {panel_idx: (spectrum_ax, residual_ax)}
        self.subplots: Dict[int, Tuple[Axes, Axes]] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_excluded(self, wave_val: float) -> bool:
        """Return True if a wavelength falls in an excluded range."""
        for lo, hi in self.excluded_ranges:
            if lo <= wave_val <= hi:
                return True
        return False

    def _get_fit_mask(self, wave_arr: np.ndarray) -> np.ndarray:
        """Boolean mask: True for data points NOT in any excluded range
        and NOT within *exclude_lines_half_width* of any atomic line."""
        mask = np.ones(len(wave_arr), dtype=bool)
        for lo, hi in self.excluded_ranges:
            mask &= ~((wave_arr >= lo) & (wave_arr <= hi))
        # Also exclude points near atomic / ionic lines
        if (
            self.exclude_lines_half_width is not None
            and self.atomic_lines is not None
            and len(self.atomic_lines) > 0
        ):
            hw = self.exclude_lines_half_width
            for wl in self.atomic_lines["wave"].values:
                mask &= ~(np.abs(wave_arr - wl) <= hw)
        return mask

    def _compute_chi2(
        self,
        flux: np.ndarray,
        model_raw: np.ndarray,
        model_adj: np.ndarray,
        error_raw: Optional[np.ndarray],
        error_adj: Optional[np.ndarray],
        fit_mask: np.ndarray,
    ) -> Tuple[float, float, int]:
        """Compute raw and adjusted chi^2 over the given fit mask.

        Parameters
        ----------
        flux : np.ndarray
            Observed flux.
        model_raw : np.ndarray
            Raw model flux (before nuisance continuum).
        model_adj : np.ndarray
            Adjusted model flux (with continuum added).
        error_raw : np.ndarray or None
            Original pipeline errors.
        error_adj : np.ndarray or None
            Effective errors (with noise floor added in quadrature).
        fit_mask : np.ndarray[bool]
            Boolean mask selecting points to include.

        Returns
        -------
        chi2_raw : float
            chi^2 computed with *model_raw* and *error_raw*.
        chi2_adj : float
            chi^2 computed with *model_adj* and *error_adj*.
        n_fit : int
            Number of True entries in *fit_mask*.
        """
        n_fit = int(np.sum(fit_mask))
        if n_fit == 0 or error_adj is None:
            return 0.0, 0.0, n_fit

        chi2_raw = (
            float(np.sum(
                ((flux[fit_mask] - model_raw[fit_mask])
                 / error_raw[fit_mask]) ** 2
            ))
            if error_raw is not None
            else 0.0
        )
        chi2_adj = float(np.sum(
            ((flux[fit_mask] - model_adj[fit_mask])
             / error_adj[fit_mask]) ** 2
        ))
        return chi2_raw, chi2_adj, n_fit

    def _format_chi2_annotation(
        self,
        chi2_raw: float,
        chi2_adj: float,
        dof: int,
        has_nuisance: bool,
    ) -> str:
        """Build a LaTeX annotation string for reduced chi^2 values."""
        if has_nuisance:
            r_raw = chi2_raw / dof
            r_adj = chi2_adj / dof
            return (
                f"$\\chi^2_\\nu = {r_raw:.2f}$\n"
                f"$\\chi^2_{{\\nu,\\mathrm{{adj}}}} = {r_adj:.2f}$"
            )
        return f"$\\chi^2_\\nu = {chi2_adj / dof:.2f}$"

    # ------------------------------------------------------------------
    def _residual_ylim_fn(self, mask: np.ndarray) -> Tuple[float, float]:
        """Per-panel y-limit function for residual sub-panels.

        Computes a symmetric ``(-pad, +pad)`` range based on the maximum absolute residual and (if available) the maximum adjusted error within the panel.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask into :attr:`wave_data` selecting the points that fall within the current panel.

        Returns
        -------
        tuple[float, float]
            ``(-pad, +pad)`` for the residual panel.
        """
        panel_flux = self.flux_data[mask]
        panel_model = self._model_flux_adj[mask]
        panel_err = (
            self._error_adj[mask] if self._error_adj is not None else None
        )
        if len(panel_flux) > 0:
            residuals = panel_flux - panel_model
            finite = np.isfinite(residuals)
            if np.any(finite):
                res_abs_max = float(np.nanmax(np.abs(residuals[finite])))
            else:
                res_abs_max = 0.0
            if panel_err is not None and len(panel_err) > 0:
                finite_err = np.isfinite(panel_err)
                if np.any(finite_err):
                    err_max = float(np.nanmax(panel_err[finite_err]))
                    res_abs_max = max(res_abs_max, err_max)
            res_pad = res_abs_max * 1.3 if res_abs_max > 0 else 0.01
            return (-res_pad, res_pad)
        return (-0.01, 0.01)

    # ------------------------------------------------------------------
    # StackedSpectralPanel factory (overrides FSP)
    # ------------------------------------------------------------------
    def _create_cell(
        self,
        idx: int,
        xmin: float,
        xmax: float,
        gs_slot: "SubplotSpec",
        **kwargs,
    ) -> List[SpectralPanel]:
        """Create a spectrum + residual panel pair for the given row.

        Uses a nested ``GridSpecFromSubplotSpec`` to split the allocated slot into two sub-rows with configurable height ratios.

        The molecule cache, summed spectrum, and model-related data are passed through *kwargs* from :meth:`generate_plot`.
        """
        gs_inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=gs_slot,
            height_ratios=[1.0, self.residual_height_ratio],
            hspace=0.0,
        )
        ax_spec = self.fig.add_subplot(gs_inner[0, 0])
        ax_res = self.fig.add_subplot(gs_inner[1, 0], sharex=ax_spec)

        # Populate backward-compatible subplots dict
        self.subplots[idx] = (ax_spec, ax_res)

        # --- Build the summed spectrum for this row --------------------
        # The model fill uses the adjusted hires model if available, otherwise falls back to the data-grid model.
        # With match_pixel_sampling the data-grid model is always used so the fill matches the residual / chi^2 arrays.
        _hires_adj = self._model_flux_hires_adj
        if _hires_adj is not None and not self.match_pixel_sampling:
            summed_w = self.model_wave
            summed_f = _hires_adj
        else:
            summed_w = self.wave_data
            summed_f = self._model_flux_adj

        # --- Spectrum panel (top) --------------------------------------
        spectrum_panel = SpectrumPanel(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=self.error_data,
            molecules=self.molecules,
            mol_cache=kwargs.get("mol_cache", []),
            summed_wave=summed_w,
            summed_flux=summed_f,
            line_list=self.line_list,
            atomic_lines=self.atomic_lines,
            wave_data_obs=getattr(self, "wave_data_obs", None),
            ax=ax_spec,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        )
        # Attach model_components for rendering in _post_render_cell
        spectrum_panel._model_components = self.model_components
        spectrum_panel._has_continuum = self._has_continuum
        spectrum_panel._continuum_params = (
            (self.continuum_c0, self.continuum_c1, self.lam_ref)
            if self._has_continuum else None
        )

        # --- Residual panel (bottom) -----------------------------------
        residual_panel = ResidualPanel(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            model_flux_adj=self._model_flux_adj,
            error_data=self.error_data,
            error_adj=self._error_adj,
            has_noise_floor=self._has_noise_floor,
            excluded_ranges=self.excluded_ranges,
            atomic_lines=self.atomic_lines,
            exclude_lines_half_width=self.exclude_lines_half_width,
            is_first_row=(idx == 0),
            ax=ax_res,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        )

        return [spectrum_panel, residual_panel]

    # ------------------------------------------------------------------
    # Chi^2 annotation helpers (shared by _post_render_cell,
    # update_panels_inplace, and generate_plot)
    # ------------------------------------------------------------------
    def _render_per_panel_chi2(
        self,
        ax_spec: "Axes",
        xmin: float,
        xmax: float,
        fg: str,
    ) -> None:
        """Compute and annotate the per-panel reduced chi^2 on *ax_spec*.

        Accumulates into :attr:`_total_chi2_raw`, :attr:`_total_chi2_adj`, and :attr:`_total_n_points` for later use by :meth:`_render_global_chi2_annotation`.

        No-ops when :attr:`show_chi2_per_panel` is *False* or when no error data is available.
        """
        if not self.show_chi2_per_panel or self.error_data is None:
            return

        _has_nuisance = self._has_continuum or self._has_noise_floor
        mask = (self.wave_data >= xmin) & (self.wave_data <= xmax)
        panel_wave = self.wave_data[mask]
        panel_flux = self.flux_data[mask]
        panel_model_raw = self.model_flux[mask]
        panel_model = self._model_flux_adj[mask]
        panel_err_raw = self.error_data[mask]
        panel_err = (
            self._error_adj[mask] if self._error_adj is not None else None
        )

        if panel_err is None or len(panel_err) == 0:
            return

        fit_mask = self._get_fit_mask(panel_wave)
        p_raw, p_adj, n_fit = self._compute_chi2(
            panel_flux, panel_model_raw, panel_model,
            panel_err_raw, panel_err, fit_mask,
        )
        if n_fit == 0:
            return

        panel_dof = max(n_fit - 1, 1)
        self._total_chi2_raw += p_raw
        self._total_chi2_adj += p_adj
        self._total_n_points += n_fit
        ann = self._format_chi2_annotation(p_raw, p_adj, panel_dof, _has_nuisance)
        ax_spec.text(
            1.01, 0.5, ann,
            transform=ax_spec.transAxes,
            fontsize=7, va="center", ha="left",
            color=fg,
        )

    def _render_global_chi2_annotation(self, fg: str) -> None:
        """Build and place the global chi^2 summary text at the bottom of the figure.

        No-ops when :attr:`show_total_chi2` is *False* or when no error
        data is available.
        """
        if not self.show_total_chi2 or self.error_data is None:
            return

        _has_nuisance = self._has_continuum or self._has_noise_floor
        full_fit_mask = self._get_fit_mask(self.wave_data)
        g_raw, g_adj, g_n = self._compute_chi2(
            self.flux_data, self.model_flux, self._model_flux_adj,
            self.error_data, self._error_adj, full_fit_mask,
        )
        global_dof = max(g_n - self.n_free_params, 1)

        if _has_nuisance:
            chi2_text = (
                f"$\\chi^2 = {g_raw:.1f}$"
                f"    $\\mathrm{{dof}} = {global_dof}$"
                f"    $\\chi^2_\\nu = {g_raw / global_dof:.2f}$"
                f"\n"
                f"$\\chi^2_{{\\mathrm{{adj}}}} = {g_adj:.1f}$"
                f"    $\\chi^2_{{\\nu,\\mathrm{{adj}}}} = {g_adj / global_dof:.2f}$"
            )
            extras = []
            if self._has_noise_floor:
                extras.append(f"$s = {self.noise_floor:.4f}$ Jy")
            if self._has_continuum:
                extras.append(
                    f"$c_0 = {self.continuum_c0:.4f}$ Jy"
                    f"  $c_1 = {self.continuum_c1:.5f}$ Jy/\u03bcm"
                )
            if extras:
                chi2_text += "\n" + "    ".join(extras)
        else:
            chi2_text = (
                f"$\\chi^2 = {g_raw:.1f}$"
                f"    $\\mathrm{{dof}} = {global_dof}$"
                f"    $\\chi^2_\\nu = {g_raw / global_dof:.2f}$"
            )

        chi2_box = dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
            linewidth=0.8,
        )
        self.fig.text(
            0.52, 0.01, chi2_text,
            ha="center", va="bottom",
            fontsize=10, color=fg,
            fontweight="bold", bbox=chi2_box,
        )

    # ------------------------------------------------------------------
    def _post_render_cell(
        self,
        idx: int,
        cell_panels: List[SpectralPanel],
        is_last: bool,
    ) -> None:
        """Apply y-limits, tick formatting, model components, chi^2, and
        excluded-range shading to the spectrum + residual pair."""
        fg = self._get_theme_value("foreground", "black")
        n = len(self._panel_edges)
        xmin = self._panel_edges[idx]
        xmax = self._panel_ends[idx]
        xr = (xmin, xmax)

        spectrum_panel = cell_panels[0]
        residual_panel = cell_panels[1]
        ax_spec = spectrum_panel.ax
        ax_res = residual_panel.ax

        # --- Spectrum y-limits (pre-computed) --------------------------
        spec_ylims = getattr(self, "_spec_ylims", None)
        if spec_ylims is not None and idx < len(spec_ylims):
            ax_spec.set_ylim(*spec_ylims[idx])

        # --- Residual y-limits (pre-computed) --------------------------
        res_ylims = getattr(self, "_res_ylims", None)
        if res_ylims is not None and idx < len(res_ylims):
            ax_res.set_ylim(*res_ylims[idx])

        # --- Gap indicators (BEFORE annotations and shading so that xlim tightening is not undone by artists drawn in gap regions)
        if self.gap_mode is GapMode.SKIP:
            shared_gaps = spectrum_panel.detect_gaps()
            for panel in cell_panels:
                panel.draw_gap_indicators(gaps=shared_gaps)

        # Re-read the (potentially tightened) x-limits for subsequent annotation and shading operations.
        xr = tuple(ax_spec.get_xlim())

        # --- Draw annotations AFTER gap indicators (use the tightened
        #     xlim so annotations are clipped to the visible range) ------
        for panel in cell_panels:
            if hasattr(panel, "atomic_lines") and panel.atomic_lines is not None and len(panel.atomic_lines) > 0:
                panel.plot_atomic_lines(panel.atomic_lines)
            if hasattr(panel, "line_list") and panel.line_list is not None and len(panel.line_list) > 0:
                panel.plot_saved_lines(panel.line_list, show_labels=self._saved_line_labels_shown())

        # --- Explicit model components on spectrum panel ---------------
        model_components = getattr(spectrum_panel, "_model_components", None)
        if model_components:
            for comp in model_components:
                c_wave = np.asarray(comp["wave"])
                c_flux = np.asarray(comp["flux"])
                c_mask = (c_wave >= xr[0]) & (c_wave <= xr[1])
                if np.any(c_mask):
                    ax_spec.plot(
                        c_wave[c_mask],
                        c_flux[c_mask],
                        color=comp.get("color", "blue"),
                        linewidth=comp.get("linewidth", 0.7),
                        alpha=comp.get("alpha", 0.6),
                        label=comp.get("label", ""),
                        zorder=self._get_theme_value("zorder_model", 3),
                    )

        # --- Continuum offset overlay ---------------------------------
        continuum_params = getattr(spectrum_panel, "_continuum_params", None)
        if continuum_params is not None:
            c0, c1, lam_ref = continuum_params
            mask = spectrum_panel.get_panel_mask()
            panel_wave = self.wave_data[mask]
            if len(panel_wave) > 0:
                cont_panel = c0 + c1 * (panel_wave - lam_ref)
                ax_spec.plot(
                    panel_wave, cont_panel,
                    color="dimgray", ls="-.", lw=1.0, alpha=0.7,
                    label="Continuum offset" if idx == 0 else "",
                    zorder=self._get_theme_value("zorder_model", 3) - 1,
                )

        # --- Tick formatting -------------------------------------------
        ax_spec.tick_params(axis="x", labelbottom=False, labelsize=7)
        ax_spec.tick_params(axis="y", labelsize=7)
        ax_spec.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
        ax_spec.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

        ax_res.tick_params(axis="x", labelbottom=True, labelsize=7)
        ax_res.tick_params(axis="y", labelsize=7)
        ax_res.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
        ax_res.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

        if is_last:
            ax_res.set_xlabel("Wavelength (\u03bcm)", fontsize=8, color=fg)
        else:
            ax_res.set_xlabel("")

        # --- Shade excluded ranges on BOTH sub-panels ------------------
        for exc_lo, exc_hi in self.excluded_ranges:
            if exc_hi >= xr[0] and exc_lo <= xr[1]:
                for _ax in (ax_spec, ax_res):
                    _ax.axvspan(
                        max(exc_lo, xr[0]),
                        min(exc_hi, xr[1]),
                        color="lightcoral",
                        alpha=0.15,
                    )

        # Shade atomic-line exclusion windows on spectrum panel too
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
                    ax_spec.axvspan(
                        max(a_lo, xr[0]),
                        min(a_hi, xr[1]),
                        color="lightsalmon",
                        alpha=0.12,
                    )

        # --- Per-panel chi-squared -------------------------------------
        self._render_per_panel_chi2(ax_spec, xr[0], xr[1], fg)

    # ------------------------------------------------------------------
    def update_panels_inplace(self) -> None:
        """Fast in-place update for ResidualSpectrumPlot.

        Handles the tuple-valued ``subplots`` dict
        ``{idx: (ax_spec, ax_res)}`` that the base-class implementation
        cannot process directly (it assumes plain ``Axes`` values and
        would raise ``'tuple' object has no attribute 'lines'``).

        Steps
        -----
        1. Recompute ``model_flux`` / ``_model_flux_adj`` from the current molecule state so residuals reflect any parameter changes.
        2. Delegate spectrum sub-panel updates (molecule overlays, summed fill, observed spectrum) 
           to the parent by temporarily substituting plain-``Axes`` entries into ``self.subplots``.
        3. Clear and re-render each residual sub-panel with the updated residuals and recomputed y-limits.

        Falls back to a full :meth:`generate_plot` if the subplot dict is empty or structurally mismatched.
        """
        n = len(self._panel_edges)
        if not self.subplots or len(self.subplots) != n:
            self.generate_plot()
            return

        # --- 1. Recompute model flux and derived arrays from the current state --
        # This handles both molecule parameter changes and spectrum reloads
        # (update_data() may have replaced wave_data / flux_data / error_data
        # with a new spectrum's arrays; we must keep all derived arrays in sync).

        # 1a. Recompute _error_adj from the (potentially updated) error_data.
        if self._has_noise_floor and self.error_data is not None:
            self._error_adj = np.sqrt(self.error_data ** 2 + self.noise_floor ** 2)
        else:
            self._error_adj = self.error_data

        # 1b. Recompute model flux from the current visible molecules.
        if self.molecules is not None:
            try:
                _, new_model = self.molecules.get_summed_flux_resampled(
                    self.wave_data, visible_only=True,
                )
                if len(new_model) == len(self.wave_data):
                    self.model_flux = new_model
                    if self._has_continuum:
                        cont = (
                            self.continuum_c0
                            + self.continuum_c1 * (self.wave_data - self.lam_ref)
                        )
                        self._model_flux_adj = self.model_flux + cont
                    else:
                        self._model_flux_adj = self.model_flux
            except Exception:
                pass

        # --- 2. Update spectrum sub-panels via parent -----------------
        # Temporarily remap subplots to single Axes so the parent loop can call ax.lines / ax.collections without seeing tuples.
        _orig_subplots = self.subplots
        self.subplots = {idx: pair[0] for idx, pair in _orig_subplots.items()}
        try:
            super().update_panels_inplace()
        finally:
            self.subplots = _orig_subplots

        # --- 3. Re-render residual sub-panels ------------------------
        res_ylims = self._compute_panel_ylims(
            uniform=self.uniform_ylim_residuals,
            ylim_fn=self._residual_ylim_fn,
        )
        fg = self._get_theme_value("foreground", "black")

        for idx in range(n):
            ax_spec, ax_res = self.subplots[idx]
            xmin = self._panel_edges[idx]
            xmax = self._panel_ends[idx]

            mask = (self.wave_data >= xmin) & (self.wave_data <= xmax)
            panel_wave = self.wave_data[mask]
            panel_flux = self.flux_data[mask]
            panel_model = self._model_flux_adj[mask]
            panel_err = (
                self._error_adj[mask] if self._error_adj is not None else None
            )
            residuals = panel_flux - panel_model

            # Clear existing residual artists; sharex linkage is
            # preserved across cla() in modern matplotlib.
            ax_res.cla()

            if panel_err is not None and len(panel_err) == len(residuals):
                ax_res.errorbar(
                    panel_wave, residuals, yerr=panel_err,
                    fmt=".", ms=2, color=fg, ecolor="lightgray",
                    elinewidth=0.5, zorder=2,
                )
            else:
                ax_res.plot(panel_wave, residuals, ".", ms=2, color=fg, zorder=2)

            ax_res.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
            ax_res.set_ylim(*res_ylims[idx])

            # Shade excluded ranges
            for exc_lo, exc_hi in self.excluded_ranges:
                if exc_hi >= xmin and exc_lo <= xmax:
                    ax_res.axvspan(
                        max(exc_lo, xmin), min(exc_hi, xmax),
                        color="lightcoral", alpha=0.15,
                    )

            # Restore tick formatting (cla() resets these)
            is_last = idx == n - 1
            ax_res.tick_params(axis="x", labelbottom=True, labelsize=7)
            ax_res.tick_params(axis="y", labelsize=7)
            ax_res.xaxis.set_major_locator(MaxNLocator(nbins=8, prune="both"))
            ax_res.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            if is_last:
                ax_res.set_xlabel("Wavelength (\u03bcm)", fontsize=8, color=fg)

        # --- 4. Update per-panel chi^2 annotations -------------------
        # Reset accumulators; remove stale text on each spectrum axes
        # then delegate to the shared helper.
        self._total_chi2_raw = 0.0
        self._total_chi2_adj = 0.0
        self._total_n_points = 0

        for idx in range(n):
            ax_spec, _ = self.subplots[idx]
            for txt in ax_spec.texts[:]:
                txt.remove()
            xmin = self._panel_edges[idx]
            xmax = self._panel_ends[idx]
            self._render_per_panel_chi2(ax_spec, xmin, xmax, fg)

        # --- 5. Update global chi^2 annotation -----------------------
        # Clear the stale figure-level text then delegate to the helper.
        for txt in self.fig.texts[:]:
            txt.remove()
        self._render_global_chi2_annotation(fg)

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the multi-panel figure with residual rows.

        Pre-computes y-limits for both spectrum and residual sub-panels,
        then delegates to the parent chain (FSP -> SSP) for stacking.
        """
        # Reset backward-compat dict
        self.subplots.clear()

        # Recompute _error_adj from current error_data so it stays in
        # sync with wave_data after a spectrum reload.
        if self._has_noise_floor and self.error_data is not None:
            self._error_adj = np.sqrt(self.error_data ** 2 + self.noise_floor ** 2)
        else:
            self._error_adj = self.error_data

        # Pre-compute y-limits for both sub-panel types
        self._spec_ylims = self._compute_panel_ylims(
            ylim_fn=self._spectrum_ylim_fn,
        )
        self._res_ylims = self._compute_panel_ylims(
            uniform=self.uniform_ylim_residuals,
            ylim_fn=self._residual_ylim_fn,
        )

        # Initialise global chi-squared accumulators
        self._total_chi2_raw = 0.0
        self._total_chi2_adj = 0.0
        self._total_n_points = 0

        # The parent FSP.generate_plot() handles:
        #   - mol_cache, summed spectrum computation
        #   - calling SSP.generate_plot() which iterates _create_cell
        #   - _post_render_cell for each row
        #   - legend on first panel
        super().generate_plot(**kwargs)

        # --- Override the legend to use model_components if needed ------
        if 0 in self.subplots:
            first_ax = self.subplots[0][0]  # spectrum axes of first row
            mol_cache, mol_labels, mol_colors = self._build_mol_cache()
            if mol_labels:
                self.legend_strategy.build_legend(
                    first_ax, self.fig, mol_labels, mol_colors,
                    fontsize=7,
                )
            elif self.model_components:
                comp_labels = [c.get("label", "") for c in self.model_components if c.get("label")]
                comp_colors = [c.get("color", "blue") for c in self.model_components if c.get("label")]
                if comp_labels:
                    self.legend_strategy.build_legend(
                        first_ax, self.fig, comp_labels, comp_colors,
                        fontsize=7,
                    )

        # --- Total chi-squared annotation at the bottom ----------------
        fg = self._get_theme_value("foreground", "black")
        self._render_global_chi2_annotation(fg)

    # ------------------------------------------------------------------
    # Convenience setters
    # ------------------------------------------------------------------
    def set_model_flux(self, model_flux: np.ndarray) -> None:
        """Replace the model flux array."""
        self.model_flux = np.asarray(model_flux)

    def set_model_components(
        self, components: List[Dict[str, Any]]
    ) -> None:
        """Replace the model component list.

        When :attr:`match_pixel_sampling` is enabled, components are
        flux-conservingly resampled onto the observed wavelength grid.
        """
        self.model_components = self._resample_components_to_data_grid(
            components
        )

    def _resample_components_to_data_grid(
        self, components: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Resample model components onto :attr:`wave_data`.

        Returns *components* unchanged unless
        :attr:`match_pixel_sampling` is enabled.  Otherwise returns
        shallow copies of each component dict with ``"wave"`` /
        ``"flux"`` replaced by the observed grid and the
        flux-conservingly resampled flux (``spectres``, NaN fill
        outside the component's native range).  The caller's dicts are
        never mutated.
        """
        if not components or not self.match_pixel_sampling:
            return components

        from iSLAT.Modules.DataProcessing.spectral_utils import spectres

        resampled: List[Dict[str, Any]] = []
        for comp in components:
            c_wave = np.asarray(comp["wave"], dtype=float)
            c_flux = np.asarray(comp["flux"], dtype=float)
            new_comp = dict(comp)
            if not (
                len(c_wave) == len(self.wave_data)
                and np.array_equal(c_wave, self.wave_data)
            ):
                new_comp["wave"] = self.wave_data
                new_comp["flux"] = spectres(
                    self.wave_data, c_wave, c_flux, fill=np.nan
                )
            resampled.append(new_comp)
        return resampled