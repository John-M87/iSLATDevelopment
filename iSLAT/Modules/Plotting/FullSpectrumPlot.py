"""
FullSpectrumPlot -- Multi-panel overview of an entire observed spectrum.

Generates a vertically stacked series of wavelength-range panels, each
showing the observed data, individual molecule models, summed model
spectrum, and optionally line-list annotations and atomic lines.

Inherits the stacking layout from :class:`StackedSpectralPanel` and implements :meth:`_create_cell` to produce a single :class:`SpectrumPanel` per row.

Can be used standalone (notebook / script) or embedded in a GUI layout.
The interactive :class:`FullSpectrumView` composes an instance of this class for rendering, adding span-selectors and toggle sync on top.
"""
from typing import Callable, Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .StackedSpectralPanel import StackedSpectralPanel
from .SpectralPanel import SpectralPanel, GapMode
from .SpectrumPanel import SpectrumPanel
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class FullSpectrumPlot(StackedSpectralPanel):
    """
    Multi-panel full-spectrum plot.

    Inherits from :class:`StackedSpectralPanel` which manages the
    vertical stacking, GridSpec layout, and panel-edge computation.
    Each row (cell) contains a single :class:`SpectrumPanel`.

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array (already RV-corrected if needed).
    flux_data : np.ndarray
        Observed flux array.
    molecules : MoleculeDict, optional
        Collection of molecules -- visible ones are plotted.
    error_data : np.ndarray, optional
        Flux uncertainties.
    line_list : pd.DataFrame, optional
        Saved line annotations (columns: ``wave`` / ``lam``, ``species``,
        ``line``).
    atomic_lines : pd.DataFrame, optional
        Atomic line annotations (columns: ``wave``, ``species``, ``line``).
    n_panels : int, optional
        Target number of panels. Defaults to 10.
    step : float, optional
        Wavelength width of each panel. Overrides *n_panels*.
    xlim_range : tuple[float, float], optional
        ``(start, end)`` wavelength range. Defaults to full data range.
    ymax_factor : float, optional
        Fractional padding above the peak flux in each panel (0.2 = 20 %).
    uniform_ylim : bool, optional
        When *True* every panel shares the same vertical scale,
        determined by the global flux minimum and maximum across all panels.
        Default *False* (each panel auto-scales independently).
    figsize : tuple, optional
        Figure size.  Height is scaled automatically if *None*.
    wave_data_obs : np.ndarray, optional
        Observer-frame wavelengths, used by MoleculeDict methods that apply the stellar RV correction internally.
    show_observed : bool, optional
        Draw the observed spectrum visibly.  When *False* the observed
        artists are still created (tagged ``_islat_observed``) but hidden,
        matching the GUI's *Show Observed Data* toggle.  Default *True*.
    show_summed : bool, optional
        Show the summed-model fill.  Default *True*.
    y_limits : tuple[float, float], optional
        Manual ``(ymin, ymax)`` applied to every panel, overriding the
        automatic per-panel limits (the GUI's *Set Y Limits* setting).
    y_autoscale_source : {"data", "model"}, optional
        What the automatic per-panel limits fit to when *y_limits* is not
        given: the observed data (default, honouring *ymax_factor* and
        *uniform_ylim*) or the drawn molecule models and summed fill.
    y_data_sync : {"global", "view", "panel"}, optional
        Set every panel to the exact min / max of the observed flux (the
        GUI's *Sync Y to Data Min/Max* setting): over the whole spectrum,
        over all panels' wavelength ranges together, or over each panel's
        own range.  Takes precedence over *y_autoscale_source*; manual
        *y_limits* still win.  Default ``None`` (off).
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        n_panels: int = 10,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        uniform_ylim: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        show_observed: bool = True,
        show_summed: bool = True,
        y_limits: Optional[Tuple[float, float]] = None,
        y_autoscale_source: str = "data",
        y_data_sync: Optional[str] = None,
        **kwargs,
    ):
        # Allow callers to pass row_height via kwargs; default to
        # a compact 1.6 inches per cell for the full-spectrum layout.
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

        # Observer-frame wavelengths for model computation
        # (get_summed_flux, get_matched_sampling_wavelengths).
        # Falls back to wave_data when no observer-frame array is
        # provided (e.g. notebook usage).
        self.wave_data_obs: np.ndarray = (
            np.asarray(wave_data_obs) if wave_data_obs is not None
            else self.wave_data
        )
        self.line_list = line_list
        self.atomic_lines = atomic_lines

        # Display options mirroring the GUI toggle state.
        self.show_observed: bool = bool(show_observed)
        self.show_summed: bool = bool(show_summed)
        self.y_limits: Optional[Tuple[float, float]] = (
            (float(y_limits[0]), float(y_limits[1])) if y_limits is not None else None
        )
        self.y_autoscale_source: str = (
            "model" if str(y_autoscale_source).lower() == "model" else "data"
        )
        from .PlotView import normalize_y_data_sync
        self.y_data_sync: Optional[str] = normalize_y_data_sync(y_data_sync)

        # Override default figsize (SSP uses 14-wide; FSP uses 12)
        if figsize is None:
            self._figsize = (12, self.row_height * len(self._panel_edges))

        # Backward-compatible storage for single-Axes per row.
        # Populated in _create_cell() from the cell panels.
        self.subplots: Dict[int, Axes] = {}

    # ------------------------------------------------------------------
    # Public data mutators (for interactive reuse)
    # ------------------------------------------------------------------
    def update_data(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        wave_data_obs: Optional[np.ndarray] = None,
    ) -> bool:
        """Replace the data arrays and recompute panel layout.

        Returns *True* if the panel edges changed (caller should rebuild
        subplots); *False* if only the data values changed.
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
        if line_list is not None:
            self.line_list = line_list
        if atomic_lines is not None:
            self.atomic_lines = atomic_lines

        old_edges = self._panel_edges.copy()

        # _xlim_start/_xlim_end define the *plot display range* and are owned by the view
        # (set via _set_display_range / plot-start / plot-range fields in the GUI).
        # update_data must not touch them.
        # On the very first render (no layout yet) initialise from the data bounds as a safe fallback; the view will override them once active.
        if len(self._panel_edges) == 0:
            self._xlim_start = float(np.nanmin(self.wave_data))
            self._xlim_end = float(np.nanmax(self.wave_data))
        self._compute_panel_layout()

        return (
            len(old_edges) != len(self._panel_edges)
            or not np.allclose(old_edges, self._panel_edges, atol=1e-6)
        )

    # ------------------------------------------------------------------
    # Molecule-cache helper (shared with ResidualSpectrumPlot)
    # ------------------------------------------------------------------
    def _build_mol_cache(self) -> Tuple[List[tuple], List[str], List[str]]:
        """Pre-compute molecule spectrum data, labels, and colors.

        Returns ``(mol_cache, mol_labels, mol_colors)`` where each entry in *mol_cache* is ``(wavelengths, flux, color, label, name)``.
        Sub-classes override or reuse this to avoid duplicating the molecule-caching logic.
        """
        mol_cache: List[tuple] = []
        mol_labels: List[str] = []
        mol_colors: List[str] = []
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
            mol_labels = [self.get_molecule_display_name(m) for m in visible]
            mol_colors = [self.get_molecule_color(m) for m in visible]

            use_interp = False
            target_wave = None
            ref_wave = getattr(self, 'wave_data_obs', self.wave_data)
            if ref_wave is not None and hasattr(self.molecules, 'get_matched_sampling_wavelengths'):
                use_interp, target_wave = (
                    self.molecules.get_matched_sampling_wavelengths(ref_wave)
                )
                if not use_interp:
                    target_wave = None

            for mol in visible:
                lam, flux = self.get_molecule_spectrum_data(
                    mol, self.wave_data,
                    interpolate_to_input=use_interp,
                    target_wavelengths=target_wave,
                    extend_to_range=True,
                )
                if lam is not None and flux is not None and len(flux) > 0:
                    mol_cache.append((
                        lam, flux,
                        self.get_molecule_color(mol),
                        self.get_molecule_display_name(mol),
                        getattr(mol, "name", "unknown"),
                    ))
        return mol_cache, mol_labels, mol_colors

    # ------------------------------------------------------------------
    def _spectrum_ylim_fn(self, mask: np.ndarray) -> Tuple[float, float]:
        """Default per-panel y-limit function (observed spectrum).

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask into :attr:`wave_data` / :attr:`flux_data`
            selecting the points that fall within the current panel.

        Returns
        -------
        tuple[float, float]
            ``(ymin, ymax)`` for the panel.
        """
        if np.any(mask):
            finite = np.isfinite(self.flux_data[mask])
            if np.any(finite):
                _ymax = float(np.nanmax(self.flux_data[mask][finite]))
                _ymax += _ymax * self.ymax_factor
                return (-0.005, _ymax)
        return (-0.005, 0.1)

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
        """Create a single :class:`SpectrumPanel` for the given row.

        The summed spectrum and molecule cache are computed once in :meth:`generate_plot` and forwarded via *kwargs*.
        """
        ax = self.fig.add_subplot(gs_slot)

        panel = SpectrumPanel(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=self.error_data,
            molecules=self.molecules,
            mol_cache=kwargs.get("mol_cache", []),
            summed_wave=kwargs.get("summed_wave"),
            summed_flux=kwargs.get("summed_flux"),
            line_list=self.line_list,
            atomic_lines=self.atomic_lines,
            wave_data_obs=self.wave_data_obs,
            ax=ax,
            gap_mode=self.gap_mode,
            gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling,
        )
        # Populate backward-compatible subplots dict
        self.subplots[idx] = ax
        return [panel]

    # ------------------------------------------------------------------
    # Display options (observed / summed visibility, Y-limit policy)
    # ------------------------------------------------------------------
    def _apply_display_visibility(self, ax: Axes) -> None:
        """Hide the observed / summed artists on *ax* when disabled.

        Only ever hides, so the defaults leave artists exactly as drawn.
        """
        if self.show_observed and self.show_summed:
            return
        for art in list(ax.lines) + list(ax.collections):
            if not self.show_observed and hasattr(art, "_islat_observed"):
                art.set_visible(False)
            elif not self.show_summed and hasattr(art, "_islat_summed"):
                art.set_visible(False)

    def _resolve_panel_ylim(
        self,
        idx: int,
        ax: Axes,
        panel_ylims: Optional[List[Tuple[float, float]]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return the Y limits for panel *idx* under the current policy.

        Manual :attr:`y_limits` win.  Next, :attr:`y_data_sync` sets the
        panel to the observed data's exact min / max for its scope.
        Otherwise the ``"model"`` source fits the visible models drawn on
        *ax* (falling back to the observed data in range, as the GUI does),
        and the ``"data"`` source uses the pre-computed layout limits
        *panel_ylims* (default :attr:`_panel_ylims`).
        """
        if self.y_limits is not None:
            return self.y_limits
        if self.y_data_sync is not None:
            lims = self._data_synced_panel_ylim(ax)
            if lims is not None:
                return lims
        if self.y_autoscale_source == "model":
            from .PlotView import fit_y_limits_to_data, fit_y_limits_to_model
            lims = fit_y_limits_to_model(ax)
            if lims is None:
                lims = fit_y_limits_to_data(ax, self.wave_data, self.flux_data)
            if lims is not None:
                return lims
        if panel_ylims is None:
            panel_ylims = getattr(self, "_panel_ylims", None)
        if panel_ylims is not None and idx < len(panel_ylims):
            return panel_ylims[idx]
        return None

    def _data_synced_panel_ylim(self, ax: Axes) -> Optional[Tuple[float, float]]:
        """Exact observed-flux ``(min, max)`` for *ax* under :attr:`y_data_sync`.

        ``"global"`` uses all the data, ``"view"`` the data inside any
        panel's wavelength range, and ``"panel"`` the data inside *ax*'s
        own x-range.  ``None`` when there is no data there.
        """
        from .PlotView import data_y_range
        if self.y_data_sync == "global":
            return data_y_range(self.wave_data, self.flux_data)
        if self.y_data_sync == "panel":
            return data_y_range(self.wave_data, self.flux_data, [ax.get_xlim()])
        ranges = list(zip(getattr(self, "_panel_edges", []), getattr(self, "_panel_ends", [])))
        if not ranges:
            return data_y_range(self.wave_data, self.flux_data, [ax.get_xlim()])
        return data_y_range(self.wave_data, self.flux_data, ranges)

    # ------------------------------------------------------------------
    def _post_render_cell(
        self,
        idx: int,
        cell_panels: List[SpectralPanel],
        is_last: bool,
    ) -> None:
        """Apply per-panel y-limits, tick formatting, and annotations."""
        fg = self._get_theme_value("foreground", "black")

        ax = cell_panels[0].ax

        # Visibility first: a hidden summed fill must not count towards a
        # model-based Y fit.  Limits are then set before the annotations,
        # whose labels are positioned from the final Y range.
        self._apply_display_visibility(ax)
        ylim = self._resolve_panel_ylim(idx, ax)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        if is_last:
            ax.set_xlabel("Wavelength (\u03bcm)", color=fg)

        # --- Gap indicators (BEFORE annotations so that xlim tightening
        #     is not undone by annotation artists outside the visible range)
        if self.gap_mode is GapMode.SKIP:
            for panel in cell_panels:
                panel.draw_gap_indicators()

        # --- Draw annotations AFTER gap indicators (use the tightened
        #     xlim so annotations are clipped to the visible range) ------
        for panel in cell_panels:
            if hasattr(panel, "atomic_lines") and panel.atomic_lines is not None and len(panel.atomic_lines) > 0:
                panel.plot_atomic_lines(panel.atomic_lines)
            if hasattr(panel, "line_list") and panel.line_list is not None and len(panel.line_list) > 0:
                panel.plot_saved_lines(panel.line_list, show_labels=self._saved_line_labels_shown())

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the multi-panel figure.

        Pre-computes shared data (mol cache, summed spectrum, y-limits)
        and passes them through *kwargs* to :meth:`_create_cell` via
        the parent's :meth:`~StackedSpectralPanel.generate_plot`.
        """
        # Reset backward-compat dict before the parent clears panels
        self.subplots.clear()

        # Compute summed flux once (if molecules are available)
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data_obs, visible_only=True,
                    extend_to_global_range=True,
                )
            except Exception:
                pass

        # Pre-compute molecule data via shared helper
        mol_cache, mol_labels, mol_colors = self._build_mol_cache()

        # Pre-compute per-panel y-limits (pass 1)
        self._panel_ylims = self._compute_panel_ylims(
            ylim_fn=self._spectrum_ylim_fn,
        )

        # Delegate stacking to the parent class
        super().generate_plot(
            mol_cache=mol_cache,
            summed_wave=summed_wave,
            summed_flux=summed_flux,
            model_linewidth=kwargs.get("model_linewidth", self._get_theme_value("full_spectrum_model_linewidth", 1.0)),
            model_alpha=kwargs.get("model_alpha", self._get_theme_value("full_spectrum_model_alpha", 0.8)),
            **kwargs,
        )

        # --- Global labels and legend ----------------------------------
        # color-legend on the first panel (handles removal when empty).
        legend_ax = self._legend_axes
        if legend_ax is not None:
            self.legend_strategy.build_legend(
                legend_ax, self.fig, mol_labels, mol_colors,
            )

    # ------------------------------------------------------------------
    def _spectrum_panel_for_row(self, idx: int) -> Optional[SpectrumPanel]:
        """Return the :class:`SpectrumPanel` rendered in row *idx*, or ``None``."""
        for panel in self.panels.get(idx, ()):
            if isinstance(panel, SpectrumPanel):
                return panel
        return None

    def _update_observed_inplace(
        self,
        idx: int,
        ax: Axes,
        xr: Tuple[float, float],
    ) -> None:
        """Replace the observed-spectrum artists on *ax* with the current data.

        ``ax.errorbar`` creates a data ``Line2D`` *and* ``LineCollection`` error bars, all tagged ``_islat_observed``.
        Calling ``set_data`` on the line alone (the previous behaviour) left the error bars of the previous spectrum on screen whenever a new spectrum was loaded into an unchanged layout, e.g. when cycling through a sample.
        Every tagged artist is therefore removed and the spectrum is re-drawn through the row's :class:`SpectrumPanel`, which also keeps that panel's data arrays in sync and honours its gap mode.
        The visibility of the replaced artists (observed-data toggle) is carried over to the new ones.
        """
        was_visible = True
        for art in list(ax.lines) + list(ax.collections):
            if hasattr(art, "_islat_observed"):
                was_visible = bool(art.get_visible())
                break

        BasePlot._clear_tagged_artists(ax, "_islat_observed")

        panel = self._spectrum_panel_for_row(idx)
        if panel is not None:
            panel.wave_data = self.wave_data
            panel.flux_data = self.flux_data
            panel.error_data = self.error_data
            panel.wave_data_obs = self.wave_data_obs
            panel.xmin, panel.xmax = xr
            panel_wave, panel_flux, panel_err = panel.get_panel_data_with_gaps()
            panel._plot_observed_spectrum(ax, panel_wave, panel_flux, panel_err)
        else:
            # No panel object for this row (axes supplied externally):
            # draw the data line directly and tag it.
            obs_mask = (self.wave_data >= xr[0]) & (self.wave_data <= xr[1])
            line, = ax.plot(
                self.wave_data[obs_mask], self.flux_data[obs_mask],
                color=self._get_theme_value("observed_spectrum_color", "black"),
                linewidth=self._get_theme_value("observed_spectrum_linewidth", 0.8),
                zorder=self._get_theme_value("zorder_observed", 2),
                label="Observed",
            )
            line._islat_observed = True

        if not was_visible:
            for art in list(ax.lines) + list(ax.collections):
                if hasattr(art, "_islat_observed"):
                    art.set_visible(False)

    def _refresh_gap_indicators(
        self,
        idx: int,
        ax: Axes,
        xr: Tuple[float, float],
    ) -> None:
        """Redraw the gap markers of row *idx* for the current data (``GapMode.SKIP``).

        The old markers are removed and the x-range reset to the panel's
        full range before re-detecting, since the previous data may have
        tightened it.
        """
        panel = self._spectrum_panel_for_row(idx)
        if panel is None:
            return
        panel.remove_gap_indicators()
        ax.set_xlim(*xr)
        panel.draw_gap_indicators()

    def update_panels_inplace(self) -> None:
        """Fast in-place update of existing subplot data without fig.clf().

        This is the fast-path used by :class:`FullSpectrumView` when the panel layout (edges/count) hasn't changed.
        Instead of destroying and re-creating every axes object, we update the data on existing ``Line2D`` artists and ``PolyCollection`` fills.

        Falls back to a full :meth:`generate_plot` if the subplot dict is empty (first render) or structurally mismatched,
        including when the set of panels with data changed in ``GapMode.SKIP``.
        In that mode each panel's gap markers are redrawn for the new data.
        """
        active = list(self._active_panel_edges()[0])
        if not self.subplots or sorted(self.subplots) != active:
            # Structural mismatch -- fall back to full rebuild
            self.generate_plot()
            return

        # --- Pre-compute molecule data via shared helper -----------------
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data_obs, visible_only=True,
                    extend_to_global_range=True,
                )
            except Exception:
                pass

        mol_cache, _labels, _colors = self._build_mol_cache()

        visible = []
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
        visible_names = {getattr(m, "name", None) for m in visible}

        # --- Pre-compute per-panel y-limits ----------------------------
        panel_ylims = self._compute_panel_ylims(
            ylim_fn=self._spectrum_ylim_fn,
        )

        # --- Update each panel in place ---------------------------------
        for idx in active:
            xr = (self._panel_edges[idx], self._panel_ends[idx])
            ax = self.subplots[idx]

            # Update observed spectrum (tagged with _islat_observed).
            # Re-drawn rather than mutated so the error bars follow the
            # data -- see _update_observed_inplace.
            self._update_observed_inplace(idx, ax, xr)
            if self.gap_mode is GapMode.SKIP:
                self._refresh_gap_indicators(idx, ax, xr)

            # Update molecule lines in place
            existing_mol_lines = {}
            for art in ax.lines[:]:
                if hasattr(art, "_molecule_name"):
                    existing_mol_lines[art._molecule_name] = art

            rendered_names = set()
            base_model_linewidth = self._get_theme_value("full_spectrum_model_linewidth", 0.8)
            for m_lam, m_flux, m_color, m_label, m_name in mol_cache:
                m_mask = (m_lam >= xr[0]) & (m_lam <= xr[1])
                rendered_names.add(m_name)
                m_linewidth = self.get_molecule_linewidth_by_name(m_name, base_model_linewidth)
                if m_name in existing_mol_lines:
                    line = existing_mol_lines[m_name]
                    if np.any(m_mask):
                        line.set_data(m_lam[m_mask], m_flux[m_mask])
                        line.set_color(m_color)
                        line.set_linewidth(m_linewidth)
                        line.set_visible(True)
                    else:
                        line.set_data([], [])
                else:
                    if np.any(m_mask):
                        new_line, = ax.plot(
                            m_lam[m_mask], m_flux[m_mask],
                            linestyle="--", color=m_color,
                            alpha=self._get_theme_value("full_spectrum_model_alpha", 0.8),
                            linewidth=m_linewidth,
                            label=m_label,
                            zorder=self._get_theme_value("zorder_model", 3),
                        )
                        new_line._molecule_name = m_name

            # Remove stale molecule lines (molecule deleted or hidden)
            for m_name, art in existing_mol_lines.items():
                if m_name not in rendered_names:
                    art.remove()

            # Update summed spectrum fill
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.remove()
            if summed_wave is not None and summed_flux is not None:
                s_mask = (summed_wave >= xr[0]) & (summed_wave <= xr[1])
                if np.any(s_mask):
                    self._plot_summed_spectrum(ax, summed_wave[s_mask], summed_flux[s_mask])

            # Limits last, so a model-based fit sees the updated artists.
            self._apply_display_visibility(ax)
            ylim = self._resolve_panel_ylim(idx, ax, panel_ylims)
            if ylim is not None:
                ax.set_ylim(*ylim)

        # --- Update legend on first panel -------------------------------
        legend_ax = self._legend_axes
        if legend_ax is not None:
            mol_labels = [self.get_molecule_display_name(m) for m in visible] if visible else []
            mol_colors = [self.get_molecule_color(m) for m in visible] if visible else []
            self.legend_strategy.build_legend(
                legend_ax, self.fig, mol_labels, mol_colors,
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def _plot_summed_spectrum(
        self,
        ax: "Axes",
        wave_data: "np.ndarray",
        summed_flux: "np.ndarray",
        color: Optional[str] = None,
        label: str = "Sum",
        deduplicate: bool = False,
    ) -> None:
        """Plot the summed model spectrum as a filled area on *ax*."""
        from .BasePlot import BasePlot
        if summed_flux is None or len(summed_flux) == 0:
            return
        if not np.any(summed_flux > 0):
            return
        if deduplicate:
            BasePlot._clear_tagged_artists(ax, "_islat_summed", lines=False)
        fill_color = color or self._get_theme_value("summed_spectra_color", "lightgray")
        fill = ax.fill_between(
            wave_data, 0, summed_flux,
            color=fill_color, alpha=1.0, label=label,
            zorder=self._get_theme_value("zorder_summed", 1),
        )
        fill._islat_summed = True

    @property
    def _legend_axes(self) -> Optional["Axes"]:
        """Return the axes that should receive the molecule color legend.

        For single-axes cells this is the first rendered panel (panel 0
        unless it was skipped for having no data).  For multi-axes cells
        (e.g. :class:`ResidualSpectrumPlot` where each value is a tuple) it
        returns the first element of that cell.
        """
        if not self.subplots:
            return None
        val = self.subplots[min(self.subplots)]
        if isinstance(val, tuple):
            return val[0]
        return val

    def set_line_list(self, df: pd.DataFrame) -> None:
        """Attach or replace the line-list DataFrame."""
        self.line_list = df

    def set_atomic_lines(self, df: pd.DataFrame) -> None:
        """Attach or replace the atomic-lines DataFrame."""
        self.atomic_lines = df