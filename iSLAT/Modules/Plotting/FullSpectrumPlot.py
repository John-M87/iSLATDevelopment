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

from .StackedSpectralPanel import StackedSpectralPanel
from .SpectralPanel import SpectralPanel, GapMode, peak_ylim
from .SpectrumPanel import SpectrumPanel
from .ArtistTags import (
    MOLECULE_TAG, OBSERVED_TAG, SUMMED_TAG,
    iter_tagged, remove_tagged, set_tagged_visible,
)
from .ModelOverlays import (
    build_molecule_traces, draw_summed_fill, plot_molecule_trace,
)

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
    default_fig_width = 12.0

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
            wave_data_obs=wave_data_obs,
            **kwargs,
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
        """Replace the data arrays (and, when given, the line annotations) and recompute the panel layout.

        See :meth:`StackedSpectralPanel.update_data`; *line_list* and
        *atomic_lines* are kept when ``None``.
        """
        if line_list is not None:
            self.line_list = line_list
        if atomic_lines is not None:
            self.atomic_lines = atomic_lines
        return super().update_data(
            wave_data, flux_data,
            molecules=molecules, error_data=error_data, wave_data_obs=wave_data_obs,
        )

    # ------------------------------------------------------------------
    # Molecule-cache helper (shared with ResidualSpectrumPlot)
    # ------------------------------------------------------------------
    def _build_mol_cache(self) -> Tuple[List[tuple], List[str], List[str]]:
        """Pre-compute molecule spectrum data, labels, and colors.

        Returns ``(mol_cache, mol_labels, mol_colors)`` where each entry in *mol_cache* is ``(wavelengths, flux, color, label, name)``.
        Sub-classes override or reuse this to avoid duplicating the molecule-caching logic.
        """
        return self.molecule_cache(self.molecules, self.wave_data, self.wave_data_obs)

    @classmethod
    def molecule_cache(
        cls,
        molecules: Optional["MoleculeDict"],
        wave_data: Optional[np.ndarray],
        wave_data_obs: Optional[np.ndarray] = None,
    ) -> Tuple[List[tuple], List[str], List[str]]:
        """The :meth:`_build_mol_cache` result for any molecule set and spectrum.

        *wave_data* are the display (rest-frame) wavelengths and
        *wave_data_obs* the observer-frame ones used for matched spectral
        sampling (default: *wave_data*).  The cache entries are
        :class:`~iSLAT.Modules.Plotting.ModelOverlays.MoleculeTrace` tuples.
        """
        return build_molecule_traces(
            molecules, wave_data, cls.get_molecule_spectrum_data, wave_data_obs,
        )

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
        return peak_ylim(self.flux_data[mask], self.ymax_factor)

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

        panel = self._make_spectrum_panel(
            xmin, xmax, ax,
            mol_cache=kwargs.get("mol_cache", []),
            summed_wave=kwargs.get("summed_wave"),
            summed_flux=kwargs.get("summed_flux"),
        )
        # Populate backward-compatible subplots dict
        self.subplots[idx] = ax
        return [panel]

    def _make_spectrum_panel(
        self,
        xmin: float,
        xmax: float,
        ax: Axes,
        **panel_kw: Any,
    ) -> SpectrumPanel:
        """As :meth:`StackedSpectralPanel._make_spectrum_panel`, also passing the saved and atomic lines."""
        panel_kw.setdefault("line_list", self.line_list)
        panel_kw.setdefault("atomic_lines", self.atomic_lines)
        return super()._make_spectrum_panel(xmin, xmax, ax, **panel_kw)

    # ------------------------------------------------------------------
    # Display options (observed / summed visibility, Y-limit policy)
    # ------------------------------------------------------------------
    def _apply_display_visibility(self, ax: Axes) -> None:
        """Hide the observed / summed artists on *ax* when disabled.

        Only ever hides, so the defaults leave artists exactly as drawn.
        """
        if not self.show_observed:
            set_tagged_visible(ax, OBSERVED_TAG, False)
        if not self.show_summed:
            set_tagged_visible(ax, SUMMED_TAG, False)

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
        ax = cell_panels[0].ax

        # Visibility first: a hidden summed fill must not count towards a
        # model-based Y fit.  Limits are then set before the annotations,
        # whose labels are positioned from the final Y range.
        self._apply_display_visibility(ax)
        ylim = self._resolve_panel_ylim(idx, ax)
        if ylim is not None:
            ax.set_ylim(*ylim)

        self._style_cell_axes(ax, is_last=is_last, x_nbins=6, y_nbins=None)

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
    # Render hooks
    # ------------------------------------------------------------------
    def _prepare_render(self) -> Dict[str, Any]:
        """Compute the summed model, the molecule cache and the per-panel y-limits once per render."""
        summed_wave, summed_flux = self._summed_model_flux()
        mol_cache, mol_labels, mol_colors = self._build_mol_cache()
        self._mol_legend = (mol_labels, mol_colors)
        self._panel_ylims = self._compute_panel_ylims(
            ylim_fn=self._spectrum_ylim_fn,
        )
        return dict(
            mol_cache=mol_cache,
            summed_wave=summed_wave,
            summed_flux=summed_flux,
            **self._model_style_kwargs(),
        )

    def _legend_entries(self) -> Optional[Tuple[List[str], List[str]]]:
        """The molecules in the cache built by :meth:`_prepare_render`."""
        return self._mol_legend

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
        first_observed = next(iter_tagged(ax, OBSERVED_TAG), None)
        was_visible = True if first_observed is None else bool(first_observed.get_visible())

        remove_tagged(ax, OBSERVED_TAG)

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
            set_tagged_visible(ax, OBSERVED_TAG, False)

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
        summed_wave, summed_flux = self._summed_model_flux()

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
            existing_mol_lines = {
                art._molecule_name: art
                for art in iter_tagged(ax, MOLECULE_TAG, collections=False)
            }

            rendered_names = set()
            base_model_linewidth = self._get_theme_value("full_spectrum_model_linewidth", 0.8)
            for trace in mol_cache:
                m_lam, m_flux, m_color, m_label, m_name = trace[:5]
                rendered_names.add(m_name)
                m_linewidth = self.get_molecule_linewidth_by_name(m_name, base_model_linewidth)
                if m_name in existing_mol_lines:
                    m_mask = (m_lam >= xr[0]) & (m_lam <= xr[1])
                    line = existing_mol_lines[m_name]
                    if np.any(m_mask):
                        line.set_data(m_lam[m_mask], m_flux[m_mask])
                        line.set_color(m_color)
                        line.set_linewidth(m_linewidth)
                        line.set_visible(True)
                    else:
                        line.set_data([], [])
                else:
                    plot_molecule_trace(
                        ax, trace, xr,
                        linewidth=m_linewidth,
                        alpha=self._get_theme_value("full_spectrum_model_alpha", 0.8),
                        zorder=self._get_theme_value("zorder_model", 3),
                    )

            # Remove stale molecule lines (molecule deleted or hidden)
            for m_name, art in existing_mol_lines.items():
                if m_name not in rendered_names:
                    art.remove()

            # Update summed spectrum fill
            remove_tagged(ax, SUMMED_TAG, lines=False)
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
        self._build_legend(
            [self.get_molecule_display_name(m) for m in visible],
            [self.get_molecule_color(m) for m in visible],
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
        draw_summed_fill(
            ax, wave_data, summed_flux,
            color=color or self._get_theme_value("summed_spectra_color", "lightgray"),
            zorder=self._get_theme_value("zorder_summed", 1),
            label=label,
            replace=deduplicate,
        )

    def set_line_list(self, df: pd.DataFrame) -> None:
        """Attach or replace the line-list DataFrame."""
        self.line_list = df

    def set_atomic_lines(self, df: pd.DataFrame) -> None:
        """Attach or replace the atomic-lines DataFrame."""
        self.atomic_lines = df