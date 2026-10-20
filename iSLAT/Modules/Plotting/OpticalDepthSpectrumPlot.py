"""
OpticalDepthSpectrumPlot -- Multi-panel stacked optical-depth overview.

Generates a vertically stacked series of panels, each showing the convolved τ(λ) profile (or line-center stems) for visible molecules with color-coded stacked fills.

Inherits the stacking layout from :class:`StackedSpectralPanel` and implements :meth:`_create_cell` to produce a single :class:`OpticalDepthPanel` per row.

Can be combined with :class:`FullSpectrumPlot` using the ``+`` operator or :meth:`stack_with` to produce a composite flux + τ overview.
"""
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING

import numpy as np
from matplotlib.ticker import MaxNLocator

from .StackedSpectralPanel import StackedSpectralPanel
from .SpectralPanel import SpectralPanel, GapMode
from .OpticalDepthPanel import OpticalDepthPanel
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class OpticalDepthSpectrumPlot(StackedSpectralPanel):
    """
    Multi-panel optical-depth plot.

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array.
    flux_data : np.ndarray
        Observed flux array (used only for layout / gap detection).
    molecules : MoleculeDict, optional
        Collection of molecules -- visible ones are plotted.
    error_data : np.ndarray, optional
        Flux uncertainties (forwarded to base; not plotted directly).
    n_panels : int, optional
        Target number of panels.  Default 10.
    step : float, optional
        Wavelength width per panel (overrides *n_panels*).
    xlim_range : tuple[float, float], optional
        Wavelength range.  Defaults to full data range.
    ymax_factor : float, optional
        Fractional padding above peak τ (0.2 = 20 %).
    uniform_ylim : bool, optional
        When True every panel shares the same τ scale.  Default False.
    display_mode : str, optional
        ``"profile"`` or ``"stem"``.  Default ``"profile"``.
    log_scale : bool, optional
        Logarithmic y-axis.  Default False.
    show_total : bool, optional
        Show combined τ line.  Default True.
    stacked_fill : bool, optional
        Cumulative fill per molecule.  Default True.
    tau_floor : float, optional
        Minimum τ for log-scale.  Default ``1e-10``.
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
        figsize: Optional[Tuple[float, float]] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        kwargs.setdefault("row_height", 1.6)

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

        if figsize is None:
            self._figsize = (12, 1.6 * len(self._panel_edges))

        # Backward-compatible per-row axes storage
        self.subplots: Dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Molecule tau-cache helper
    # ------------------------------------------------------------------
    def _build_mol_tau_cache(
        self,
    ) -> Tuple[List[tuple], List[str], List[str]]:
        """Pre-compute molecule tau data, labels, and colors.

        Returns ``(mol_tau_cache, mol_labels, mol_colors)`` where each
        entry in *mol_tau_cache* is
        ``(wavelengths, tau, color, label, name)``.
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
            lam, tau = self.get_molecule_tau_data(
                mol, self.wave_data,
                interpolate_to_input=use_interp,
                target_wavelengths=target_wave,
                extend_to_range=True,
            )
            if lam is not None and tau is not None and len(tau) > 0:
                cache.append((
                    lam, tau,
                    self.get_molecule_color(mol),
                    self.get_molecule_display_name(mol),
                    getattr(mol, "name", "unknown"),
                ))
        return cache, labels, colors

    # ------------------------------------------------------------------
    # Y-limit function for tau
    # ------------------------------------------------------------------
    def _tau_ylim_fn(self, mask: np.ndarray) -> Tuple[float, float]:
        """Per-panel y-limit function driven by pre-computed tau cache.

        Falls back to a simple default if the cache is not yet available.
        """
        cache = getattr(self, "_mol_tau_cache", [])
        if not cache:
            return (0.0, 1.0)

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
        ax = self.fig.add_subplot(gs_slot)

        panel = OpticalDepthPanel(
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
            ax=ax,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        )
        self.subplots[idx] = ax
        return [panel]

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
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        if is_last:
            ax.set_xlabel("Wavelength (\u03bcm)", color=fg)

        if self.gap_mode is GapMode.SKIP:
            for panel in cell_panels:
                panel.draw_gap_indicators()

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
        """Build the multi-panel optical-depth figure."""
        self.subplots.clear()

        # Pre-compute molecule tau data
        mol_tau_cache, mol_labels, mol_colors = self._build_mol_tau_cache()
        self._mol_tau_cache = mol_tau_cache  # store for _tau_ylim_fn

        # Pre-compute per-panel y-limits
        self._panel_ylims = self._compute_panel_ylims(
            ylim_fn=self._tau_ylim_fn,
        )

        # Delegate stacking to the parent class
        super().generate_plot(
            mol_tau_cache=mol_tau_cache,
            **kwargs,
        )

        # color legend on the first panel
        legend_ax = self._legend_axes
        if legend_ax is not None:
            self.legend_strategy.build_legend(
                legend_ax, self.fig, mol_labels, mol_colors,
            )