"""
OpticalDepthView -- :class:`PlotView` for the multi-panel optical depth τ(λ) overview.

Shows an :class:`OpticalDepthSpectrumPlot` (the convolved τ of every visible molecule, stacked per molecule, over the whole wavelength range) in the main plotting area.

The view builds on :class:`FullSpectrumView`, which already drives any :class:`StackedSpectralPanel`, so it shares the canvas lifecycle, the *N Panels* and *Plot Start / Plot Range* controls, saved / atomic line overlays, wavelength markers, the legend toggle and the right-click *Open in Three Panel view* item.
What differs from the full spectrum view:

* The *Total Model* toggle shows / hides the combined "Total τ" line (and the summed-model fill of the flux sub-panels).
* View-specific toggles (in the view-controls section of the Views menu) switch between a linear and a logarithmic τ axis, from the convolved profile to per-line stems at the line centres or to each molecule's fractional share of the total τ per pixel (*τ Fraction*), between stacked and overlaid molecules, and add a flux sub-panel below each τ panel (*Flux Panels*), plus the data-density x scaling shared with the full spectrum view.
* The flux Y-axis settings (auto-scale, *Set Y Limits…*, in Jy) and the observed-data toggle apply to the flux sub-panels only; each τ panel is scaled to its own τ.
* Dragging a range on a panel opens it in the Three Panel view for line inspection.
* *Save Figure…* exports a fresh standalone optical depth figure.

The τ display options live in the controller's ``toggle_state`` (see :data:`TAU_OPTION_DEFAULTS`).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from .ArtistTags import OBSERVED_TAG, SUMMED_TAG, iter_tagged, set_tagged_visible
from .FullSpectrumView import FullSpectrumView, draw_toggle_state_markers
from .PlotView import (
    PlotView, data_synced_y_limits, fit_y_limits_to_data, fit_y_limits_to_model,
    normalize_y_data_sync,
)
from .OpticalDepthPanel import TOTAL_TAU_TAG
from .OpticalDepthSpectrumPlot import OpticalDepthSpectrumPlot
from .SpectralPanel import XScaling
from ..GUI.KeyBindings import shortcut_text

from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
from iSLAT.Modules.FileHandling import absolute_data_files_path

if TYPE_CHECKING:
    import numpy as np
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

#: ``toggle_state`` keys (and defaults) for the optical depth display options.
TAU_OPTION_DEFAULTS: Dict[str, Any] = {
    "tau_log_scale": False,         # logarithmic τ axis
    "tau_display_mode": "profile",  # "profile" (convolved τ), "stem" (per-line τ) or "fraction" (share of total τ)
    "tau_stacked_fill": True,       # stack molecules (False: overlay one line each)
    "tau_show_flux": False,         # flux sub-panel below each τ panel
}


class OpticalDepthView(FullSpectrumView):
    """Multi-panel optical depth view backed by an :class:`OpticalDepthSpectrumPlot`.

    Parameters
    ----------
    plot_manager : iSLATPlot
        The main controller (``iSLAT.Modules.Plotting.MainPlot.iSLATPlot``).
    """

    VIEW_NAME: str = "Optical Depth"

    #: Stacking the sample's spectra is a Full Spectrum view option.
    SUPPORTS_SAMPLE_STACK: bool = False

    # ==================================================================
    # Display options (stored in the controller's toggle_state)
    # ==================================================================
    def _tau_option(self, key: str) -> Any:
        return self._pm.toggle_state.get(key, TAU_OPTION_DEFAULTS[key])

    def _tau_plot_kwargs(self) -> Dict[str, Any]:
        """The :class:`OpticalDepthSpectrumPlot` keywords for the current options."""
        mode = self._tau_option("tau_display_mode")
        return {
            "display_mode": mode if mode in ("stem", "fraction") else "profile",
            "log_scale": bool(self._tau_option("tau_log_scale")),
            "stacked_fill": bool(self._tau_option("tau_stacked_fill")),
            "show_flux": bool(self._tau_option("tau_show_flux")),
        }

    def _x_scaling(self) -> XScaling:
        if self._pm.toggle_state.get("data_density", False):
            return XScaling.DATA_DENSITY
        return XScaling.WAVELENGTH

    def _set_tau_option(self, key: str, value: Any) -> None:
        """Store *value* for *key* and re-render when it changed."""
        if self._tau_option(key) == value:
            return
        self._pm.toggle_state[key] = value
        if self._plot is None:
            return
        for attr, attr_value in self._tau_plot_kwargs().items():
            setattr(self._plot, attr, attr_value)
        self._regenerate_layout()

    def _get_log_scale(self) -> bool:
        return bool(self._tau_option("tau_log_scale"))

    def _set_log_scale(self, value: bool) -> None:
        self._set_tau_option("tau_log_scale", bool(value))

    def _set_display_mode(self, mode: str, on: bool) -> None:
        """Switch to *mode* (``"stem"`` / ``"fraction"``), or back to the profile when turning the active one off.

        The modes are exclusive, so turning one on replaces the other;
        turning off a mode that is not active changes nothing.
        """
        if on:
            self._set_tau_option("tau_display_mode", mode)
        elif self._tau_option("tau_display_mode") == mode:
            self._set_tau_option("tau_display_mode", "profile")

    def _get_stem_mode(self) -> bool:
        return self._tau_option("tau_display_mode") == "stem"

    def _set_stem_mode(self, value: bool) -> None:
        self._set_display_mode("stem", bool(value))

    def _get_fraction_mode(self) -> bool:
        return self._tau_option("tau_display_mode") == "fraction"

    def _set_fraction_mode(self, value: bool) -> None:
        self._set_display_mode("fraction", bool(value))

    def _get_stacked_fill(self) -> bool:
        return bool(self._tau_option("tau_stacked_fill"))

    def _set_stacked_fill(self, value: bool) -> None:
        self._set_tau_option("tau_stacked_fill", bool(value))

    def _get_show_flux(self) -> bool:
        return bool(self._tau_option("tau_show_flux"))

    def _set_show_flux(self, value: bool) -> None:
        self._set_tau_option("tau_show_flux", bool(value))

    # ==================================================================
    # Plot creation / refresh
    # ==================================================================
    def _create_plot(self) -> OpticalDepthSpectrumPlot:
        """Build a fresh :class:`OpticalDepthSpectrumPlot` from the current iSLAT state."""
        wave, flux, wave_obs = self._load_spectrum_data()
        self.line_data = self._load_line_data()
        return OpticalDepthSpectrumPlot(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
            error_data=getattr(self._islat, "err_data", None),
            wave_data_obs=wave_obs,
            figsize=self._compute_interactive_figsize(wave),
            theme=self._pm.theme,
            x_scaling=self._x_scaling(),
            gap_mode=self._gap_mode(),
            **self._tau_plot_kwargs(),
        )

    def _rebuild_plot(self) -> None:
        """Refresh the data (new spectrum, molecules, parameters) and re-render.

        The panel layout (count and wavelength range) is kept.  τ fills are
        stacked over all visible molecules, so the panels are always fully
        re-rendered rather than updated in place.
        """
        if self._plot is None:
            self._plot = self._create_plot()
        else:
            wave, flux, wave_obs = self._load_spectrum_data()
            self.line_data = self._load_line_data()
            self._plot.update_data(
                wave_data=wave,
                flux_data=flux,
                molecules=self._islat.molecules_dict,
                error_data=getattr(self._islat, "err_data", None),
                wave_data_obs=wave_obs,
            )
            for attr, value in self._tau_plot_kwargs().items():
                setattr(self._plot, attr, value)
        self.span_selectors.clear()
        self._plot.generate_plot()
        self._install_span_selectors()
        self._notify_canvas_changed()

    def _regenerate_layout(self) -> None:
        """Re-render and restore the toggle state; the axes are new, so reset the toolbar history."""
        super()._regenerate_layout()
        self._notify_canvas_changed()

    # ==================================================================
    # Y-axis policy: the flux settings (Jy) apply to the flux sub-panels only
    # ==================================================================
    def _flux_axes(self):
        """The flux sub-panels' axes (none unless *Flux Panels* is on)."""
        if self._plot is None:
            return []
        return list(self._plot.get_flux_axes().values())

    def _iter_observed_axes(self):
        """The observed spectrum is drawn in the flux sub-panels only."""
        return iter(self._flux_axes())

    def _iter_y_axes(self):
        """The main plot's Y limits are flux, so they govern the flux sub-panels; τ panels keep their own scale."""
        return iter(self._flux_axes())

    def _autoscale_y_axes(self) -> bool:
        """Re-fit the flux sub-panels to the observed data (per panel) or to the drawn model."""
        flux_axes = self._plot.get_flux_axes() if self._plot is not None else {}
        if not flux_axes:
            return False
        if self._y_autoscale_source() == "model":
            return PlotView._autoscale_y_axes(self)
        ylims = self._plot.compute_flux_ylims()
        for idx, ax in flux_axes.items():
            if idx < len(ylims):
                ax.set_ylim(*ylims[idx])
        return True

    # ==================================================================
    # Toggles
    # ==================================================================
    def _set_summed_visibility(self, visible: bool) -> None:
        """*Total Model* shows / hides the "Total τ" line and the flux sub-panels' summed fill."""
        super()._set_summed_visibility(visible)
        for ax in self._iter_toggle_axes():
            set_tagged_visible(ax, TOTAL_TAU_TAG, visible, collections=False)

    def toggle_residuals(self, show: bool) -> None:
        """No-op - residuals belong to the flux spectrum."""

    # ==================================================================
    # Controls
    # ==================================================================
    def _register_control_fields(self) -> None:
        """Register the panel layout entries and the τ display toggles on the :class:`ControlBus`."""
        bus = getattr(self._pm, "control_bus", None)
        if bus is None:
            return

        from iSLAT.Modules.GUI.ControlField import EntryField, DisplayRangeField, ToggleField

        bus.unregister_owner(self)

        bus.register_many([
            EntryField(
                "n_panels", "N Panels:", self._get_n_panels, self._set_n_panels,
                datatype="int", width=5, owner=self,
                tip="Number of panels in the\noptical depth stacked layout",
            ),
            DisplayRangeField(
                "_odv_display_range",
                getter=self._get_display_range,
                setter=self._set_display_range,
                owner=self,
            ),
        ], "control_panel")

        bus.register_many([
            ToggleField(
                "tau_log_scale", "Log τ",
                getter=self._get_log_scale,
                setter=self._set_log_scale,
                owner=self,
                tip="Toggle a logarithmic optical depth axis",
            ),
            ToggleField(
                "tau_stem_mode", "τ Stems",
                getter=self._get_stem_mode,
                setter=self._set_stem_mode,
                owner=self,
                tip=(
                    "Toggle the display mode.\n"
                    "OFF: convolved τ(λ) profile (default).\n"
                    "ON:  one stem per line at its centre,\n"
                    "     as tall as the line's τ.\n"
                    "Replaces 'τ Fraction' when turned on."
                ),
            ),
            ToggleField(
                "tau_fraction_mode", "τ Fraction",
                getter=self._get_fraction_mode,
                setter=self._set_fraction_mode,
                owner=self,
                tip=(
                    "Toggle the fractional share of each\n"
                    "molecule in the total optical depth\n"
                    "per pixel (molecule τ / total τ, 0 to 1).\n"
                    "Stacked: the fills add up to 1.\n"
                    "Pixels with negligible total τ are left blank.\n"
                    "The axis is linear ('Log τ' does not apply).\n"
                    "Replaces 'τ Stems' when turned on."
                ),
            ),
            ToggleField(
                "tau_stacked_fill", "Stacked τ",
                getter=self._get_stacked_fill,
                setter=self._set_stacked_fill,
                owner=self,
                tip=(
                    "Toggle how molecules are combined.\n"
                    "ON:  stacked fills (default).\n"
                    "OFF: one line per molecule.\n"
                    "The total τ line follows 'Total Model'\n"
                    f"(Keybind: {shortcut_text('toggle_summed_spectrum')})."
                ),
            ),
            ToggleField(
                "tau_show_flux", "Flux Panels",
                getter=self._get_show_flux,
                setter=self._set_show_flux,
                owner=self,
                tip=(
                    "Toggle a flux panel below each\n"
                    "optical depth panel: observed data,\n"
                    "molecule models and total model.\n"
                    "The Y-axis options in the Views menu\n"
                    "apply to these flux panels."
                ),
            ),
            ToggleField(
                "data_density", "Data Density X-Scale",
                getter=self._get_data_density,
                setter=self._set_data_density,
                owner=self,
                tip=(
                    "Toggle X-axis scaling mode.\n"
                    "OFF: equal wavelength width per panel (default).\n"
                    "ON:  equal data-point count per panel (DATA_DENSITY)."
                ),
            ),
            #self._skip_gaps_field(ToggleField),
        ], "views_menu")

    # ==================================================================
    # Selection: inspect lines in the Three Panel view
    # ==================================================================
    def _span_selector_axes(self):
        """Every axes of a panel (τ and flux) can be dragged to select a range."""
        raw = getattr(self._plot, "subplots", {}) if self._plot is not None else {}
        for idx, val in raw.items():
            for k, ax in enumerate(val if isinstance(val, tuple) else (val,)):
                yield (idx, k), idx, ax

    def _on_span_select(self, xmin: float, xmax: float, subplot_index: int) -> None:
        """Open the dragged range in the Three Panel view (line inspection lives there)."""
        if abs(xmax - xmin) < 0.001:
            return
        pm = self._pm
        if getattr(pm, "active_view", None) is self:
            # Mirror toggle_three_panel so the back-toggle returns here.
            pm._pre_threepanel_view_name = self.VIEW_NAME
            pm.switch_view("Three Panel")
            root = getattr(self._islat, "root", None)
            if root is not None:
                root.after(100, lambda: self._apply_selection(xmin, xmax))
                return
        self._apply_selection(xmin, xmax)

    # ==================================================================
    # Molecule callbacks: stacked τ depends on every visible molecule
    # ==================================================================
    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Re-render when a visible molecule's parameter changes."""
        if parameter_name == "is_visible":
            return
        if not self._initialised or self._plot is None:
            self._needs_refresh = True
            return
        mol_dict = getattr(self._islat, "molecules_dict", None)
        molecule = mol_dict.get(molecule_name) if mol_dict is not None else None
        if molecule is None or not getattr(molecule, "is_visible", True):
            # Hidden molecules are not drawn; showing one re-renders anyway.
            return
        self._regenerate_layout()

    def on_molecule_visibility_changed(
        self,
        molecule_name: str,
        is_visible: bool,
        molecules_dict: "MoleculeDict",
        wave_data: Any,
        active_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
        force_rerender: bool = False,
    ) -> None:
        """Re-render: the stacked fills and total τ change with every molecule shown or hidden."""
        if not self._initialised or self._plot is None:
            return
        self._regenerate_layout()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Re-render without the deleted molecule (it is already gone from the dict)."""
        if not self._initialised or self._plot is None:
            self._needs_refresh = True
            return
        self._regenerate_layout()

    # ==================================================================
    # Theme
    # ==================================================================
    def apply_theme(self, theme: dict) -> None:
        """Apply *theme*, including the foreground-colored "Total τ" line."""
        super().apply_theme(theme)
        if self._plot is None:
            return
        fg = theme.get("foreground", "black")
        for ax in self._iter_all_axes():
            for line in iter_tagged(ax, TOTAL_TAU_TAG, collections=False):
                line.set_color(fg)

    # ==================================================================
    # File output
    # ==================================================================
    def _create_standalone_plot(
        self,
        wave: "np.ndarray",
        flux: "np.ndarray",
        wave_obs: "np.ndarray",
        *,
        xlim_range: Optional[Tuple[float, float]] = None,
        n_panels: Optional[int] = None,
    ) -> OpticalDepthSpectrumPlot:
        """Build a fresh (not yet rendered) :class:`OpticalDepthSpectrumPlot` for export."""
        kw: Dict[str, Any] = dict(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
            error_data=getattr(self._islat, "err_data", None),
            wave_data_obs=wave_obs,
            theme=self._pm.theme,
            xlim_range=xlim_range,
            x_scaling=self._x_scaling(),
            gap_mode=self._gap_mode(),
            **self._tau_plot_kwargs(),
        )
        if n_panels is not None:
            kw["n_panels"] = n_panels
        return OpticalDepthSpectrumPlot(**kw)

    @staticmethod
    def _apply_flux_display_state(plot: OpticalDepthSpectrumPlot, ts: dict) -> None:
        """Give *plot*'s flux sub-panels the GUI's observed-data visibility and Y-axis setting.

        Mirrors the live view: hidden observed data, manual Y limits while
        auto-scale is off, otherwise the observed data's min / max when
        ``y_data_sync`` is set, or a fit to the drawn model when the
        auto-scale source is "model" (the plot's own per-panel data limits
        are already set).
        """
        flux_axes = list(plot.get_flux_axes().values())
        if not flux_axes:
            return
        if not ts.get("observed", True):
            for ax in flux_axes:
                set_tagged_visible(ax, OBSERVED_TAG, False)
        manual = ts.get("y_limits")
        sync = normalize_y_data_sync(ts.get("y_data_sync"))
        if not ts.get("y_autoscale", True):
            if manual is not None:
                for ax in flux_axes:
                    ax.set_ylim(float(manual[0]), float(manual[1]))
        elif sync is not None:
            synced = data_synced_y_limits(flux_axes, plot.wave_data, plot.flux_data, sync)
            for ax, lims in zip(flux_axes, synced):
                if lims is not None:
                    ax.set_ylim(*lims)
        elif ts.get("y_autoscale_source") == "model":
            for ax in flux_axes:
                lims = fit_y_limits_to_model(ax) or fit_y_limits_to_data(ax, plot.wave_data, plot.flux_data)
                if lims is not None:
                    ax.set_ylim(*lims)

    def save_figure(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        dpi: int | None = None,
        rasterized: bool = False,
        **kwargs,
    ) -> str | None:
        """Export the optical depth plot as shown, using a fresh standalone figure.

        The figure uses the live panel count and wavelength range and follows
        the toggle state: flux sub-panels (with their observed-data and Y-axis
        settings), total τ line / summed model, legend, saved / atomic lines
        and wavelength markers.
        """
        if save_path is None:
            from tkinter import filedialog
            try:
                default_name = Path(self._islat.loaded_spectrum_file).stem + f"_optical_depth.{file_format}"
            except Exception:
                default_name = f"optical_depth.{file_format}"
            save_path = filedialog.asksaveasfilename(
                title="Save Optical Depth Figure",
                defaultextension=f".{file_format}",
                initialfile=default_name,
                initialdir=absolute_data_files_path,
                filetypes=[(f"{file_format.upper()} files", f"*.{file_format}")],
            )
        if not save_path:
            return None

        ts = self._pm.toggle_state
        xlim_range: Optional[Tuple[float, float]] = None
        n_panels: Optional[int] = None
        if self._plot is not None:
            xlim_range = (float(self._plot._xlim_start), float(self._plot._xlim_end))
            n_panels = self._plot.n_panels

        wave, flux, wave_obs = self._load_spectrum_data()
        standalone = self._create_standalone_plot(
            wave, flux, wave_obs, xlim_range=xlim_range, n_panels=n_panels,
        )
        try:
            standalone.generate_plot()

            if not ts.get("summed", True):
                for ax in standalone.fig.axes:
                    set_tagged_visible(ax, (TOTAL_TAU_TAG, SUMMED_TAG), False)
            self._apply_flux_display_state(standalone, ts)
            if not ts.get("legend", True) and standalone._legend_axes is not None:
                standalone.legend_strategy.update_visibility(standalone._legend_axes, False)
            if ts.get("atomic_lines", False):
                standalone.plot_atomic_lines(load_atomic_lines())
            if ts.get("saved_lines", False):
                line_data = self.line_data if self.line_data is not None else self._load_line_data()
                if line_data is not None:
                    if "line" not in line_data.columns:
                        line_data["line"] = [""] * len(line_data)
                    standalone.plot_saved_lines(line_data, show_labels=self._saved_line_labels_visible())
            draw_toggle_state_markers(standalone.fig, ts)

            if rasterized:
                for ax in standalone.fig.axes:
                    ax.set_rasterized(True)

            save_kw: Dict[str, Any] = {"bbox_inches": "tight", "format": file_format, "dpi": dpi or 300}
            save_kw.update(kwargs)
            standalone.fig.savefig(save_path, **save_kw)
        finally:
            standalone.close()

        if hasattr(self._islat, "GUI") and hasattr(self._islat.GUI, "data_field"):
            self._islat.GUI.data_field.insert_text(f"Optical depth figure saved to: {save_path}")

        return save_path
