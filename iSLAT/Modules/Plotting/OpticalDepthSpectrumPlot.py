"""
OpticalDepthSpectrumPlot -- Multi-panel stacked optical-depth overview.

Generates a vertically stacked series of panels, each showing the convolved τ(λ) profile (or line-center stems, or each molecule's fractional share of the total τ per pixel) for visible molecules with color-coded stacked fills.

Inherits the stacking layout from :class:`StackedSpectralPanel` and implements :meth:`_create_cell` to produce one :class:`OpticalDepthPanel` per row.
With ``show_flux=True`` each row also gets a :class:`SpectrumPanel` directly beneath its τ panel (sharing its wavelength axis, like the residual sub-panels of :class:`ResidualSpectrumPlot`) showing the observed flux, the molecule models and the summed model.

Can also be combined with :class:`FullSpectrumPlot` using the ``+`` operator or :meth:`stack_with` to produce a composite flux + τ overview.
"""
from typing import Optional, Tuple, List, Dict, Any, Callable, TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from .StackedSpectralPanel import StackedSpectralPanel
from .SpectralPanel import SpectralPanel, GapMode
from .SpectrumPanel import SpectrumPanel
from .OpticalDepthPanel import (
    OpticalDepthPanel,
    FRACTION_YLIM,
    FRACTION_YLABEL,
    FRACTION_YLABEL_SHORT,
)
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

#: Height (inches) of one τ panel when the figure size is computed automatically.
_TAU_ROW_HEIGHT = 1.6


class OpticalDepthSpectrumPlot(StackedSpectralPanel):
    """
    Multi-panel optical-depth plot.

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array.
    flux_data : np.ndarray
        Observed flux array (plotted only in the flux sub-panels, see *show_flux*).
    molecules : MoleculeDict, optional
        Collection of molecules -- visible ones are plotted.
    error_data : np.ndarray, optional
        Flux uncertainties (error bars in the flux sub-panels).
    n_panels : int, optional
        Target number of panels.  Default 10.
    step : float, optional
        Wavelength width per panel (overrides *n_panels*).
    xlim_range : tuple[float, float], optional
        Wavelength range.  Defaults to full data range.
    ymax_factor : float, optional
        Fractional padding above the peak flux in the flux sub-panels (0.2 = 20 %).
    uniform_ylim : bool, optional
        When True every panel shares the same τ scale.  Default False.
    display_mode : str, optional
        ``"profile"`` (convolved τ), ``"stem"`` (per-line τ) or
        ``"fraction"`` (each molecule's share of the total τ per pixel,
        y axis 0 to 1).  Default ``"profile"``.
    log_scale : bool, optional
        Logarithmic y-axis.  Default False.  Not applied in ``"fraction"`` mode.
    show_total : bool, optional
        Show combined τ line.  Default True.  Not drawn in ``"fraction"`` mode.
    stacked_fill : bool, optional
        Cumulative fill per molecule.  Default True.
    tau_floor : float, optional
        Minimum τ for log-scale.  Default ``1e-10``.
    fraction_min_tau : float, optional
        ``"fraction"`` mode leaves pixels whose total τ is at or below this
        blank.  Default ``1e-6``.
    show_flux : bool, optional
        Add a flux sub-panel (observed spectrum, molecule models and summed
        model, as in :class:`FullSpectrumPlot`) below each τ panel.  Default False.
    flux_height_ratio : float, optional
        Height of each flux sub-panel relative to its τ panel.  Default 1.0.
    figsize : tuple, optional
        Figure size.  Height is auto-scaled if None.
    wave_data_obs : np.ndarray, optional
        Observer-frame wavelengths for matched spectral sampling.
    **kwargs
        Forwarded to :class:`StackedSpectralPanel`.
    """

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        n_panels: int = 10,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        uniform_ylim: bool = False,
        display_mode: str = "profile",
        log_scale: bool = False,
        show_total: bool = True,
        stacked_fill: bool = True,
        tau_floor: float = 1e-10,
        fraction_min_tau: float = 1e-6,
        show_flux: bool = False,
        flux_height_ratio: float = 1.0,
        figsize: Optional[Tuple[float, float]] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        flux_height_ratio = float(flux_height_ratio)
        if not np.isfinite(flux_height_ratio) or flux_height_ratio <= 0:
            raise ValueError(f"flux_height_ratio must be a positive number, got {flux_height_ratio!r}")
        kwargs.setdefault(
            "row_height",
            _TAU_ROW_HEIGHT * (1.0 + flux_height_ratio) if show_flux else _TAU_ROW_HEIGHT,
        )

        super().__init__(
            wave_data=wave_data,
            flux_data=flux_data,
            molecules=molecules,
            error_data=error_data,
            n_panels=n_panels,
            step=step,
            xlim_range=xlim_range,
            ymax_factor=ymax_factor,
            uniform_ylim=uniform_ylim,
            figsize=figsize,
            **kwargs,
        )

        self.wave_data_obs: np.ndarray = (
            np.asarray(wave_data_obs) if wave_data_obs is not None
            else self.wave_data
        )
        self.display_mode = display_mode
        self.log_scale = log_scale
        self.show_total = show_total
        self.stacked_fill = stacked_fill
        self.tau_floor = tau_floor
        self.fraction_min_tau = fraction_min_tau
        self.show_flux = bool(show_flux)
        self.flux_height_ratio = flux_height_ratio

        if figsize is None:
            self._figsize = (12, self.row_height * len(self._panel_edges))

        # Per-row axes: {idx: tau_ax} or, with show_flux, {idx: (tau_ax, flux_ax)}
        self.subplots: Dict[int, Any] = {}

    @property
    def y_axis_label(self) -> str:
        """Figure-wide y label.

        Blank with flux sub-panels: each sub-panel then carries its own short
        label ("τ" / "Jy"), and a figure-wide label would collide with them.
        """
        if self.show_flux:
            return ""
        return FRACTION_YLABEL if self.display_mode == "fraction" else "Optical Depth (τ)"

    # ------------------------------------------------------------------
    # Axes accessors
    # ------------------------------------------------------------------
    def get_tau_axes(self) -> Dict[int, Axes]:
        """Return ``{row_index: Axes}`` for the τ panels."""
        return {
            idx: (val[0] if isinstance(val, tuple) else val)
            for idx, val in self.subplots.items()
        }

    def get_flux_axes(self) -> Dict[int, Axes]:
        """Return ``{row_index: Axes}`` for the flux sub-panels (empty unless *show_flux*)."""
        return {
            idx: val[1]
            for idx, val in self.subplots.items()
            if isinstance(val, tuple) and len(val) > 1
        }

    # ------------------------------------------------------------------
    # Public data mutator (for interactive reuse)
    # ------------------------------------------------------------------
    def update_data(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        wave_data_obs: Optional[np.ndarray] = None,
    ) -> bool:
        """Replace the data arrays and recompute the panel layout.

        The wavelength range (``_xlim_start`` / ``_xlim_end``) is kept, as in
        :meth:`FullSpectrumPlot.update_data`.  Call :meth:`generate_plot`
        afterwards to re-render.

        Returns *True* if the panel edges changed.
        """
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.wave_data_obs = (
            np.asarray(wave_data_obs) if wave_data_obs is not None
            else self.wave_data
        )
        if molecules is not None:
            self.molecules = molecules
        self.error_data = np.asarray(error_data) if error_data is not None else None

        old_edges = self._panel_edges.copy()
        if len(self._panel_edges) == 0:
            self._xlim_start = float(np.nanmin(self.wave_data))
            self._xlim_end = float(np.nanmax(self.wave_data))
        self._compute_panel_layout()

        return (
            len(old_edges) != len(self._panel_edges)
            or not np.allclose(old_edges, self._panel_edges, atol=1e-6)
        )

    # ------------------------------------------------------------------
    # Molecule cache helpers
    # ------------------------------------------------------------------
    def _build_mol_data_cache(
        self,
        getter: Callable[..., Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
    ) -> Tuple[List[tuple], List[str], List[str]]:
        """Pre-compute per-molecule ``(wavelengths, values)`` with *getter*.

        *getter* is :meth:`BasePlot.get_molecule_tau_data` or
        :meth:`BasePlot.get_molecule_spectrum_data`.  Returns
        ``(cache, mol_labels, mol_colors)`` where each *cache* entry is
        ``(wavelengths, values, color, label, name)``.
        """
        cache: List[tuple] = []
        labels: List[str] = []
        colors: List[str] = []
        if self.molecules is None:
            return cache, labels, colors

        visible = self.molecules.get_visible_molecules(return_objects=True)
        labels = [self.get_molecule_display_name(m) for m in visible]
        colors = [self.get_molecule_color(m) for m in visible]

        use_interp = False
        target_wave = None
        ref_wave = getattr(self, "wave_data_obs", self.wave_data)
        if ref_wave is not None and hasattr(
            self.molecules, "get_matched_sampling_wavelengths"
        ):
            use_interp, target_wave = (
                self.molecules.get_matched_sampling_wavelengths(ref_wave)
            )
            if not use_interp:
                target_wave = None

        for mol in visible:
            lam, values = getter(
                mol, self.wave_data,
                interpolate_to_input=use_interp,
                target_wavelengths=target_wave,
                extend_to_range=True,
            )
            if lam is not None and values is not None and len(values) > 0:
                cache.append((
                    lam, values,
                    self.get_molecule_color(mol),
                    self.get_molecule_display_name(mol),
                    getattr(mol, "name", "unknown"),
                ))
        return cache, labels, colors

    def _build_mol_tau_cache(
        self,
    ) -> Tuple[List[tuple], List[str], List[str]]:
        """Pre-compute molecule tau data, labels, and colors.

        Returns ``(mol_tau_cache, mol_labels, mol_colors)`` where each
        entry in *mol_tau_cache* is
        ``(wavelengths, tau, color, label, name)``.
        """
        return self._build_mol_data_cache(self.get_molecule_tau_data)

    def _build_mol_flux_cache(self) -> List[tuple]:
        """Pre-compute the molecule model fluxes for the flux sub-panels.

        Entries are ``(wavelengths, flux, color, label, name)``, the format
        :class:`SpectrumPanel` expects (same as :meth:`FullSpectrumPlot._build_mol_cache`).
        """
        cache, _labels, _colors = self._build_mol_data_cache(self.get_molecule_spectrum_data)
        return cache

    def _summed_model_flux(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return the visible molecules' summed model ``(wave, flux)``, or ``(None, None)``."""
        if self.molecules is None:
            return None, None
        try:
            return self.molecules.get_summed_flux(
                self.wave_data_obs, visible_only=True, extend_to_global_range=True,
            )
        except Exception:
            return None, None

    # ------------------------------------------------------------------
    # Y-limit functions
    # ------------------------------------------------------------------
    def _tau_ylim_fn(self, mask: np.ndarray) -> Tuple[float, float]:
        """Per-panel y-limit function driven by pre-computed tau cache.

        Falls back to a simple default if the cache is not yet available.
        Shares (``"fraction"`` mode) always span 0 to 1.
        """
        if self.display_mode == "fraction":
            return FRACTION_YLIM
        cache = getattr(self, "_mol_tau_cache", [])
        if not cache:
            return (self.tau_floor, 1.0) if self.log_scale else (0.0, 1.0)

        xmin = float(np.nanmin(self.wave_data[mask])) if np.any(mask) else 0.0
        xmax = float(np.nanmax(self.wave_data[mask])) if np.any(mask) else 1.0

        max_tau = 0.0
        for m_lam, m_tau, *_ in cache:
            m_mask = (m_lam >= xmin) & (m_lam <= xmax)
            if np.any(m_mask):
                finite = np.isfinite(m_tau[m_mask])
                if np.any(finite):
                    # Use sum if stacked to get total height
                    max_tau = max(max_tau, float(np.nanmax(m_tau[m_mask][finite])))

        # For stacked fill, sum contributions at peak
        if self.stacked_fill and len(cache) > 1:
            sum_tau = None
            for m_lam, m_tau, *_ in cache:
                m_mask = (m_lam >= xmin) & (m_lam <= xmax)
                if np.any(m_mask):
                    seg = m_tau[m_mask]
                    if sum_tau is None:
                        sum_tau = np.zeros_like(seg)
                    if len(seg) == len(sum_tau):
                        sum_tau = sum_tau + seg
            if sum_tau is not None and len(sum_tau) > 0:
                finite = np.isfinite(sum_tau)
                if np.any(finite):
                    max_tau = float(np.nanmax(sum_tau[finite]))

        if max_tau <= 0:
            max_tau = 1.0

        if self.log_scale:
            return (self.tau_floor, max_tau * 2.0)
        return (0.0, max_tau * 1.1)

    def _flux_ylim_fn(self, mask: np.ndarray) -> Tuple[float, float]:
        """Per-panel flux sub-panel limits: the observed flux, as in :class:`FullSpectrumPlot`."""
        return self._default_ylim_fn(self.wave_data, self.flux_data, mask, self.ymax_factor)

    def compute_flux_ylims(self) -> List[Tuple[float, float]]:
        """Return the automatic ``(ymin, ymax)`` of every flux sub-panel (each scaled independently)."""
        return self._compute_panel_ylims(uniform=False, ylim_fn=self._flux_ylim_fn)

    def _apply_stem_ylims(self) -> None:
        """Fit each panel's y-limits to its tallest stem (``"stem"`` mode).

        Honours :attr:`uniform_ylim` and :attr:`log_scale` like the profile limits.
        """
        peaks = {
            idx: max((p.stem_tau_max() for p in cell if isinstance(p, OpticalDepthPanel)), default=0.0)
            for idx, cell in self.panels.items()
        }
        if not peaks:
            return
        if self.uniform_ylim:
            top = max(peaks.values())
            peaks = {idx: top for idx in peaks}
        self._panel_ylims = list(getattr(self, "_panel_ylims", []) or [])
        for idx, peak in peaks.items():
            if peak <= 0:
                peak = 1.0
            ylim = (self.tau_floor, peak * 2.0) if self.log_scale else (0.0, peak * 1.1)
            self.panels[idx][0].ax.set_ylim(*ylim)
            if idx < len(self._panel_ylims):
                self._panel_ylims[idx] = ylim

    # ------------------------------------------------------------------
    # StackedSpectralPanel factory
    # ------------------------------------------------------------------
    def _create_cell(
        self,
        idx: int,
        xmin: float,
        xmax: float,
        gs_slot: "SubplotSpec",
        **kwargs,
    ) -> List[SpectralPanel]:
        """Create the τ panel for a row, plus its flux sub-panel when *show_flux* is set.

        With flux the slot is split by a nested ``GridSpecFromSubplotSpec``
        (τ on top, flux below, no gap, shared wavelength axis).
        """
        if self.show_flux:
            gs_inner = GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=gs_slot,
                height_ratios=[1.0, self.flux_height_ratio],
                hspace=0.0,
            )
            ax_tau = self.fig.add_subplot(gs_inner[0, 0])
            ax_flux = self.fig.add_subplot(gs_inner[1, 0], sharex=ax_tau)
        else:
            ax_tau = self.fig.add_subplot(gs_slot)
            ax_flux = None

        panels: List[SpectralPanel] = [OpticalDepthPanel(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=self.error_data,
            molecules=self.molecules,
            mol_tau_cache=kwargs.get("mol_tau_cache", []),
            display_mode=self.display_mode,
            log_scale=self.log_scale,
            show_total=self.show_total,
            stacked_fill=self.stacked_fill,
            tau_floor=self.tau_floor,
            fraction_min_tau=self.fraction_min_tau,
            ax=ax_tau,
            theme=self.theme,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        )]

        if ax_flux is None:
            self.subplots[idx] = ax_tau
            return panels

        panels.append(SpectrumPanel(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=self.error_data,
            molecules=self.molecules,
            mol_cache=kwargs.get("mol_flux_cache", []),
            summed_wave=kwargs.get("summed_wave"),
            summed_flux=kwargs.get("summed_flux"),
            wave_data_obs=self.wave_data_obs,
            ax=ax_flux,
            theme=self.theme,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        ))
        self.subplots[idx] = (ax_tau, ax_flux)
        return panels

    # ------------------------------------------------------------------
    def _post_render_cell(
        self,
        idx: int,
        cell_panels: List[SpectralPanel],
        is_last: bool,
    ) -> None:
        fg = self._get_theme_value("foreground", "black")

        ylims = getattr(self, "_panel_ylims", None)
        if ylims is not None and idx < len(ylims):
            cell_panels[0].ax.set_ylim(*ylims[idx])

        ax = cell_panels[0].ax
        flux_ax = cell_panels[1].ax if len(cell_panels) > 1 else None

        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

        if flux_ax is None:
            # The figure-wide y label (y_axis_label) replaces the per-panel one.
            ax.set_ylabel("")
            if is_last:
                ax.set_xlabel("Wavelength (μm)", color=fg)
        else:
            # Short labels tell the two sub-panels apart; the wavelength
            # ticks sit under the flux sub-panel only.
            tau_label = FRACTION_YLABEL_SHORT if self.display_mode == "fraction" else "τ"
            ax.set_ylabel(tau_label, fontsize=8, color=fg)
            ax.tick_params(axis="x", labelbottom=False)
            # Drop τ's bottom tick label so it does not collide with the
            # flux panel's top one at the shared edge.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=self._y_tick_bins(ax), prune="lower"))
            flux_ylims = getattr(self, "_flux_ylims", None)
            if flux_ylims is not None and idx < len(flux_ylims):
                flux_ax.set_ylim(*flux_ylims[idx])
            flux_ax.set_ylabel("Jy", fontsize=8, color=fg)
            flux_ax.tick_params(axis="x", labelsize=7, labelbottom=True)
            flux_ax.tick_params(axis="y", labelsize=7)
            flux_ax.yaxis.set_major_locator(MaxNLocator(nbins=self._y_tick_bins(flux_ax), prune="both"))
            if is_last:
                flux_ax.set_xlabel("Wavelength (μm)", color=fg)

        if self.gap_mode is GapMode.SKIP:
            for panel in cell_panels:
                panel.draw_gap_indicators()

    def _y_tick_bins(self, ax: Axes) -> int:
        """2-4 y-tick intervals depending on *ax*'s height, so short sub-panels stay legible."""
        height_in = ax.get_position().height * self.fig.get_figheight()
        return int(np.clip(height_in / 0.15, 2, 4))

    # ------------------------------------------------------------------
    @property
    def _legend_axes(self):
        """Return the axes for the molecule color legend."""
        val = self.subplots.get(0)
        if val is None:
            return None
        if isinstance(val, tuple):
            return val[0]
        return val

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the multi-panel optical-depth figure (with flux sub-panels when *show_flux*)."""
        self.subplots.clear()

        # Pre-compute molecule tau data
        mol_tau_cache, mol_labels, mol_colors = self._build_mol_tau_cache()
        self._mol_tau_cache = mol_tau_cache  # store for _tau_ylim_fn

        # Pre-compute per-panel y-limits
        self._panel_ylims = self._compute_panel_ylims(
            ylim_fn=self._tau_ylim_fn,
        )

        cell_kwargs: Dict[str, Any] = {"mol_tau_cache": mol_tau_cache}
        if self.show_flux:
            summed_wave, summed_flux = self._summed_model_flux()
            self._flux_ylims = self.compute_flux_ylims()
            cell_kwargs.update(
                mol_flux_cache=self._build_mol_flux_cache(),
                summed_wave=summed_wave,
                summed_flux=summed_flux,
                # Flux sub-panels use the Full Spectrum model styling.
                model_linewidth=self._get_theme_value("full_spectrum_model_linewidth", 1.0),
                model_alpha=self._get_theme_value("full_spectrum_model_alpha", 0.8),
            )
        else:
            self._flux_ylims = None
        cell_kwargs.update(kwargs)

        # Delegate stacking to the parent class
        super().generate_plot(**cell_kwargs)

        # Stems show the per-line τ, which the profile-based limits do not cover.
        if self.display_mode == "stem":
            self._apply_stem_ylims()

        # color legend on the first panel
        legend_ax = self._legend_axes
        if legend_ax is not None:
            self.legend_strategy.build_legend(
                legend_ax, self.fig, mol_labels, mol_colors,
            )
