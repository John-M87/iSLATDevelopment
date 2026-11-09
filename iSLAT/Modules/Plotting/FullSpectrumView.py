"""
FullSpectrumView -- :class:`PlotView` implementation for multi-panel stacked spectrum layouts.

This view **composes** a :class:`StackedSpectralPanel` subclass (by default a :class:`FullSpectrumPlot`) for all rendering, then adds interactive features on top:

* Span selectors on every panel (click-to-inspect)
* Dynamic overlay toggles (atomic / saved lines, summed spectrum)
* *Stack Sample Spectra*: every spectrum of the sample, each with its own parameter set (:class:`SampleStackPlot`)
* Canvas lifecycle management for the Tk GUI

The view is generic: it can drive **any** :class:`StackedSpectralPanel` implementation that exposes the standard stacked-panel interface
(``fig``, ``subplots``, ``_panel_edges``, ``_step``, ``_xlim_start``, ``_xlim_end``, ``wave_data``, ``generate_plot()``, etc.).

The concrete plot class is determined by :meth:`_create_plot`, which subclasses can override to swap in a different stacked-panel implementation.
"""
from __future__ import annotations

import os
import warnings
from os.path import dirname
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.figure import Figure as MplFigure
from matplotlib.widgets import SpanSelector

from .FigureExport import save_figure_file
from .OwnedCanvasMixin import OwnedCanvasMixin
from .PlotDebug import debug_config
from .PlotView import (
    PlotView, ask_save_path, default_save_name, draw_wavelength_markers,
    normalize_y_data_sync, notify_user,
)
from ..GUI.KeyBindings import shortcut_text
from .StackedSpectralPanel import StackedSpectralPanel
from .FullSpectrumPlot import FullSpectrumPlot
from .ResidualSpectrumPlot import ResidualSpectrumPlot
from .SampleStackPlot import (
    SampleStackPlot, normalize_stack_layout, normalize_stack_model_mode,
)
from .ArtistTags import (
    ATOMIC_LINE_TAG, OBSERVED_TAG, SAVED_LINE_TAG, SUMMED_TAG, iter_tagged,
    remove_molecule_lines, remove_tagged, set_molecule_visible, set_tagged_visible,
)
from .SavedLineLabels import get_saved_line_label_settings
from .BasePlot import BasePlot
from .ModelOverlays import draw_summed_fill, plot_molecule_trace
from .Theming import apply_canvas_theme
from .ToggleMixin import ToggleMixin
from .SpectralPanel import GapMode

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
from iSLAT.Modules.FileHandling import absolute_data_files_path
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh

# Suppress constrained_layout warnings triggered by adding/removing overlay
# artists after the layout engine has already run.
warnings.filterwarnings(
    "ignore",
    message=".*constrained_layout.*",
    category=UserWarning,
    module="matplotlib",
)

#: Exported figure size (inches) when no override is set.
DEFAULT_EXPORT_WIDTH: float = 12.0
DEFAULT_EXPORT_HEIGHT: float = 16.0
#: Upper bounds accepted for the export size overrides (inches).
MAX_EXPORT_WIDTH: float = 200.0
MAX_EXPORT_PANEL_HEIGHT: float = 20.0

#: ``toggle_state`` keys (and defaults) for stacking the sample's spectra in the Full Spectrum view.
SAMPLE_STACK_DEFAULTS: Dict[str, Any] = {
    "sample_stack": False,              # draw every sample spectrum (SampleStackPlot)
    "sample_stack_layout": "offset",    # "offset" (same panel, offset) or "rows" (one row each)
    "sample_stack_models": "total",     # "data", "total" or "molecules"
}

#: Views-menu labels of the stack layouts and model modes (value -> label).
SAMPLE_STACK_LAYOUT_LABELS: Dict[str, str] = {
    "offset": "Offset in Each Panel",
    "rows": "One Row per Spectrum",
}
SAMPLE_STACK_MODEL_LABELS: Dict[str, str] = {
    "data": "Data Only",
    "total": "Total Model",
    "molecules": "Each Molecule + Total",
}

#: Tooltip of the *Stack Sample Spectra* toggle.
SAMPLE_STACK_TIP: str = (
    "Show every spectrum of the sample (Manage Sample)\n"
    "in the panels, each with the models of its own\n"
    "parameter set: its parameter file and overrides,\n"
    "or the current parameters (plus its overrides)\n"
    "when it has no parameter file.  The active\n"
    "spectrum uses the current parameters.\n"
    "Choose the layout and models below.\n"
    "Turns 'Residuals' off (and vice versa)."
)

#: Tooltip of the *Skip Data Gaps* toggle (Full Spectrum and Optical Depth views).
SKIP_GAPS_TIP: str = (
    "Toggle how gaps in the observed data are drawn.\n"
    "OFF: one connected line across gaps (default).\n"
    "ON:  the line breaks at gaps (NaN flux or a\n"
    "     wavelength jump); gaps inside a panel are\n"
    "     blanked and labelled, empty stretches at\n"
    "     panel edges are cropped, and panels with\n"
    "     no data are left out."
)

class FullSpectrumView(ToggleMixin, OwnedCanvasMixin, PlotView):
    """
    Multi-panel full spectrum view backed by a :class:`StackedSpectralPanel`.

    Rendering is delegated to the composed stacked-panel plot (by default a :class:`FullSpectrumPlot`)
    which owns the figure, axes, and all the standard rendering helpers.
    This view adds:

    - Tk canvas management (pack / unpack, from :class:`OwnedCanvasMixin`)
    - Span selectors for click-to-inspect
    - Interactive overlay toggles (atomic lines, saved lines, summed)
    - PDF export (creates a fresh standalone plot)

    Toggle-state management (atomic lines, saved lines, summed spectrum, legend) is provided by :class:`ToggleMixin`.

    The view is **generic**: override :meth:`_create_plot` and
    :meth:`_create_standalone_plot` (or :meth:`_create_export_plot`) to
    back it with any :class:`StackedSpectralPanel` subclass.
    """

    #: Name this view is registered under in the Views menu (``iSLATPlot.views``).
    VIEW_NAME: str = "Full Spectrum"

    #: Whether the view offers *Stack Sample Spectra* (subclasses with another plot turn it off).
    SUPPORTS_SAMPLE_STACK: bool = True

    # Save dialog and message (see PlotView.save_figure).
    SAVE_DIALOG_TITLE: str = "Save Spectrum Output"
    SAVE_FILE_SUFFIX: Optional[str] = "full_output"
    SAVE_INITIAL_DIR: Optional[str] = absolute_data_files_path
    SAVE_MESSAGE: Optional[str] = "Spectrum output saved to: {path}"

    def __init__(self, plot_manager: Any) -> None:
        # Shared view state (_pm, _islat, _parent_frame, _initialised,
        # _needs_refresh, and the lazily built _canvas) comes from PlotView.
        super().__init__(plot_manager)

        # The composed plot -- created on first activation.
        # Typed as the abstract base so the view can drive any subclass.
        self._plot: Optional[StackedSpectralPanel] = None

        # Span selectors for interactive inspection
        self.span_selectors: Dict[int, SpanSelector] = {}

        # Line data (for saved-line annotations in interactive mode)
        self.line_data: Optional[pd.DataFrame] = None

        # What the last drawing of each overlay ("atomic_lines" /
        # "saved_lines") made and depended on: (artists, key).  Lets
        # sync_toggle_state keep overlays that are still current
        # (see _overlay_drawn).
        self._overlay_records: Dict[str, Tuple[Tuple[Any, ...], Any]] = {}

        # Stack Sample Spectra: cached spectra / parameter sets and the
        # sample configuration they were built from.
        self._sample_stack_builder: Any = None
        self._sample_stack_spectra: List[Any] = []
        self._sample_stack_signature: Any = None

    # --- helpers used by _register_control_fields ---
    def _get_n_panels(self):
        if self._plot is not None:
            return self._plot.n_panels
        return 10

    def _set_n_panels(self, value):
        if self._plot is None:
            return
        n = max(1, int(value))
        if n == self._plot.n_panels:
            return
        self._plot.n_panels = n
        self._plot._compute_panel_layout()
        # Full rebuild required because subplot grid changes
        self._regenerate_layout()

    def _regenerate_layout(self) -> None:
        """Rebuild the panels after a layout change and restore the toggle state.

        ``generate_plot`` recreates every artist, so the overlays and the
        observed-data / Y-limit settings must be reconciled again exactly
        as after a model update; ``sync_toggle_state`` does that and draws.
        """
        self.span_selectors.clear()
        self._plot.generate_plot()
        self._install_span_selectors()
        self.sync_toggle_state(self._pm.toggle_state)

    def _get_display_range(self):
        if self._plot is not None:
            return (self._plot._xlim_start, self._plot._xlim_end)
        return (0.0, 0.0)

    def _set_display_range(self, start, end):
        if self._plot is None:
            return
        self._plot._xlim_start = float(start)
        self._plot._xlim_end = float(end)
        self._plot._compute_panel_layout()
        self._regenerate_layout()

    def _get_data_density(self) -> bool:
        return self._pm.toggle_state.get("data_density", False)

    def _set_data_density(self, value: bool) -> None:
        from .SpectralPanel import XScaling
        value = bool(value)
        if value == self._pm.toggle_state.get("data_density", False):
            return
        self._pm.toggle_state["data_density"] = value
        if self._plot is None:
            return
        self._plot.x_scaling = XScaling.DATA_DENSITY if value else XScaling.WAVELENGTH
        self._plot._compute_panel_layout()
        self._regenerate_layout()

    # --- Stack Sample Spectra -------------------------------------------
    def _sample_stack_on(self, toggle_state: Optional[dict] = None) -> bool:
        """Whether this view draws the whole sample (*Stack Sample Spectra* on, residuals off)."""
        ts = toggle_state if isinstance(toggle_state, dict) else self._pm.toggle_state
        return (
            self.SUPPORTS_SAMPLE_STACK
            and bool(ts.get("sample_stack", False))
            and not ts.get("show_residuals", False)
        )

    def _stack_layout(self) -> str:
        return normalize_stack_layout(self._pm.toggle_state.get("sample_stack_layout", "offset"))

    def _stack_model_mode(self) -> str:
        return normalize_stack_model_mode(self._pm.toggle_state.get("sample_stack_models", "total"))

    def _get_sample_stack(self) -> bool:
        return bool(self._pm.toggle_state.get("sample_stack", False))

    def _set_sample_stack(self, value: bool) -> None:
        """Turn *Stack Sample Spectra* on or off (turning it on turns *Residuals* off)."""
        value = bool(value)
        ts = self._pm.toggle_state
        if value == self._get_sample_stack():
            return
        ts["sample_stack"] = value
        if value and ts.get("show_residuals", False):
            ts["show_residuals"] = False
        if not value:
            self._release_sample_stack()
        if self._initialised:
            self._replace_plot()
        if value and self._initialised and len(self._sample_stack_spectra) < 2:
            self._notify_user(
                "The sample has no other spectra to stack yet: add them with "
                "Spectral Functions > Manage Sample...\n",
                clear_after=False,
            )

    def _get_stack_layout_label(self) -> str:
        return SAMPLE_STACK_LAYOUT_LABELS[self._stack_layout()]

    def _set_stack_layout_label(self, label: str) -> None:
        value = next((k for k, v in SAMPLE_STACK_LAYOUT_LABELS.items() if v == label), None)
        if value is None or value == self._stack_layout():
            return
        self._pm.toggle_state["sample_stack_layout"] = value
        if isinstance(self._plot, SampleStackPlot):
            self._plot.set_options(layout=value)
            self._regenerate_layout()

    def _get_stack_models_label(self) -> str:
        return SAMPLE_STACK_MODEL_LABELS[self._stack_model_mode()]

    def _set_stack_models_label(self, label: str) -> None:
        value = next((k for k, v in SAMPLE_STACK_MODEL_LABELS.items() if v == label), None)
        if value is None or value == self._stack_model_mode():
            return
        self._pm.toggle_state["sample_stack_models"] = value
        if isinstance(self._plot, SampleStackPlot):
            self._plot.set_options(model_mode=value)
            self._regenerate_layout()

    def _build_sample_spectra(self) -> List[Any]:
        """(Re)build the sample's spectra and parameter sets, reporting entries that were skipped."""
        from iSLAT.Modules.DataProcessing.SampleStack import SampleStackBuilder, sample_signature
        if self._sample_stack_builder is None:
            self._sample_stack_builder = SampleStackBuilder()
        spectra = self._sample_stack_builder.build(self._islat)
        self._sample_stack_spectra = spectra
        self._sample_stack_signature = sample_signature(self._islat)
        problems = self._sample_stack_builder.warnings
        if problems:
            msg = "Could not stack some sample spectra:\n  " + "\n  ".join(problems)
            debug_config.warning("full_spectrum_view", msg)
            self._notify_user(msg + "\n", clear_after=False)
        return spectra

    def _release_sample_stack(self) -> None:
        """Free the stacked spectra and their parameter sets."""
        if self._sample_stack_builder is not None:
            self._sample_stack_builder.release()
        self._sample_stack_spectra = []
        self._sample_stack_signature = None

    def _refresh_sample_stack(self) -> None:
        """Rebuild the stacked spectra (their parameter sets follow the live molecules) and redraw."""
        if not isinstance(self._plot, SampleStackPlot):
            return
        spectra = self._build_sample_spectra()
        if spectra:
            self._plot.set_spectra(spectra)
        self._regenerate_layout()

    def on_sample_changed(self) -> None:
        """React to a change of the sample (spectra, parameter files, overrides)."""
        if not self._sample_stack_on():
            return
        from iSLAT.Modules.DataProcessing.SampleStack import sample_signature
        if sample_signature(self._islat) == self._sample_stack_signature:
            return
        if not self._initialised or getattr(self._pm, "active_view", None) is not self:
            self._needs_refresh = True
            return
        if isinstance(self._plot, SampleStackPlot):
            self._refresh_sample_stack()
        else:
            self._replace_plot()

    def _gap_mode(self) -> GapMode:
        """``GapMode.SKIP`` while *Skip Data Gaps* (``toggle_state["skip_gaps"]``) is on, else ``CONNECT``."""
        return GapMode.SKIP if self._pm.toggle_state.get("skip_gaps", False) else GapMode.CONNECT

    def _get_skip_gaps(self) -> bool:
        return bool(self._pm.toggle_state.get("skip_gaps", False))

    def _set_skip_gaps(self, value: bool) -> None:
        """Turn gap skipping on or off and re-render the panels."""
        value = bool(value)
        if value == self._get_skip_gaps():
            return
        self._pm.toggle_state["skip_gaps"] = value
        if self._plot is None:
            return
        self._plot.gap_mode = self._gap_mode()
        self._regenerate_layout()

    # ==================================================================
    # Convenience accessors (delegate to composed plot)
    # ==================================================================
    @property
    def fig(self) -> Optional["Figure"]:
        return self._plot.fig if self._plot is not None else None

    @property
    def subplots(self) -> Dict[int, "Axes"]:
        """Return a ``{idx: Axes}`` mapping for every panel in the plot.

        For single-axes-per-cell layouts (e.g. :class:`FullSpectrumPlot`) this returns the dict directly.
        For multi-axes cells (e.g. :class:`ResidualSpectrumPlot` where each cell is a tuple)
        the first axes of each cell is returned so that overlay operations have a consistent interface.
        """
        if self._plot is None:
            return {}
        raw = getattr(self._plot, "subplots", {})
        if not raw:
            return {}
        # Peek at the first value to decide shape
        first = next(iter(raw.values()))
        if isinstance(first, tuple):
            # Multi-axes cell -- return the primary (spectrum) axes
            return {k: v[0] for k, v in raw.items()}
        return raw

    def _iter_all_axes(self) -> List["Axes"]:
        """Yield every axes object across all cells (flat)."""
        if self._plot is None:
            return []
        raw = getattr(self._plot, "subplots", {})
        axes: List["Axes"] = []
        for val in raw.values():
            if isinstance(val, tuple):
                axes.extend(val)
            else:
                axes.append(val)
        return axes

    # ------------------------------------------------------------------
    # ToggleMixin hooks
    # ------------------------------------------------------------------
    def _toggle_ready(self) -> bool:
        return self._initialised

    def _iter_toggle_axes(self):
        return self._iter_all_axes()

    def _iter_marker_axes(self):
        """Wavelength markers span every panel, including residual sub-panels."""
        return self._iter_all_axes()

    def _iter_observed_axes(self):
        """The spectrum axes of every panel (residual panels carry no observed data).

        A sample stack draws observed data on every axes.
        """
        if isinstance(self._plot, SampleStackPlot):
            return iter(self._iter_all_axes())
        return iter(self.subplots.values())

    def _iter_y_axes(self):
        """The spectrum axes of every panel share the Y policy.

        In a sample stack every row follows it (with its own spectrum's
        data); offset stacks keep the limits that fit the stack.
        """
        if isinstance(self._plot, SampleStackPlot):
            if self._plot.layout == "rows":
                return iter(self._iter_all_axes())
            return iter(())
        return iter(self.subplots.values())

    def _sync_stack_y_options(self) -> None:
        """Give the sample stack the controller's auto-scale source and data-sync scope."""
        plot = self._plot
        ts = self._pm.toggle_state
        plot.y_limits = None    # manual limits are applied by apply_y_limits
        plot.y_data_sync = self._y_data_sync_scope(ts) if ts.get("y_autoscale", True) else None
        plot.y_autoscale_source = self._y_autoscale_source()

    def _data_synced_y_limits_by_axes(self, scope: str):
        """Sample-stack rows sync to their own spectrum's data min / max."""
        if not isinstance(self._plot, SampleStackPlot):
            return super()._data_synced_y_limits_by_axes(scope)
        from .PlotView import data_synced_y_limits
        if self._plot.layout != "rows":
            return []
        pairs = []
        for spectrum, axes in self._plot.axes_by_spectrum():
            limits = data_synced_y_limits(axes, spectrum.wave, spectrum.flux, scope)
            pairs.extend(zip(axes, limits))
        return pairs

    def _autoscale_y_axes(self) -> bool:
        """Re-fit every panel with the composed plot's own per-panel rule.

        Uses the same ``_compute_panel_ylims`` the plot applies when it
        renders, so turning auto-scale back on restores exactly the
        layout's limits (including the uniform-ylim option).  A sample
        stack re-fits each axes with its own rule.
        """
        plot = self._plot
        if isinstance(plot, SampleStackPlot):
            self._sync_stack_y_options()
            changed = False
            for ax, lims in plot.stack_ylims():
                if lims is not None:
                    ax.set_ylim(*lims)
                    changed = True
            return changed
        if plot is None or not hasattr(plot, "_compute_panel_ylims"):
            return super()._autoscale_y_axes()
        if self._y_autoscale_source() == "model":
            # Fit each panel to the model it draws rather than the layout's data rule.
            return super()._autoscale_y_axes()
        try:
            ylims = plot._compute_panel_ylims(
                ylim_fn=getattr(plot, "_spectrum_ylim_fn", None),
            )
        except Exception:
            return super()._autoscale_y_axes()
        changed = False
        for idx, ax in self.subplots.items():
            if idx < len(ylims):
                ax.set_ylim(*ylims[idx])
                changed = True
        return changed

    def _reapply_display_state(self, toggle_state: dict) -> None:
        """Also refit to the model after a re-render when auto-scaling to it.

        Every render path sets each panel to the layout's data-based limits,
        which the base leaves alone while auto-scaling.  That matches the
        ``"data"`` source but would silently drop a ``"model"`` fit on every
        model update, panel-count or range change.
        """
        super()._reapply_display_state(toggle_state)
        if (isinstance(toggle_state, dict)
                and toggle_state.get("y_autoscale", True)
                and self._y_data_sync_scope(toggle_state) is None
                and self._y_autoscale_source() == "model"):
            self._autoscale_y_axes()

    def _y_data_arrays(self):
        """The plot's own (RV-corrected) arrays, which match the panels' wavelength axes."""
        plot = self._plot
        if plot is not None and getattr(plot, "flux_data", None) is not None:
            return getattr(plot, "wave_data", None), plot.flux_data
        return super()._y_data_arrays()

    def _load_saved_line_data(self):
        return self._load_line_data()

    # ==================================================================
    # Data loading
    # ==================================================================
    def _load_spectrum_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return observed spectrum arrays from in-memory iSLAT data.

        Returns
        -------
        wave : np.ndarray
            RV-corrected (rest-frame) wavelengths for display.
        flux : np.ndarray
            Observed flux array.
        wave_obs : np.ndarray
            Observer-frame wavelengths, used by MoleculeDict methods that apply the stellar RV correction internally.
        """
        wave_obs = np.array(self._islat.wave_data_original, copy=True)
        wave = self._islat.molecules_dict.apply_stellar_rv(wave_obs)
        flux = np.array(self._islat.flux_data, copy=True)
        return wave, flux, wave_obs

    def _load_line_data(self) -> Optional[pd.DataFrame]:
        """Load the saved line list (for annotations)."""
        try:
            return pd.read_csv(self._islat.input_line_list, sep=",")
        except Exception:
            return None

    # ==================================================================
    # Plot creation / refresh (delegates to StackedSpectralPanel)
    # ==================================================================
    def _create_plot(self) -> StackedSpectralPanel:
        """Build a fresh stacked-panel plot from the current iSLAT state.

        When ``show_residuals`` is active in the toggle state a :class:`ResidualSpectrumPlot` (spectrum + residual sub-panels) is returned;
        with *Stack Sample Spectra* on a :class:`SampleStackPlot` of the whole sample; otherwise a plain :class:`FullSpectrumPlot`.

        Subclasses can override this to return a different :class:`StackedSpectralPanel` implementation.
        """
        wave, flux, wave_obs = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        # Compute a figsize appropriate for the interactive GUI view.
        # Without an explicit figsize the auto-calculation in the
        # stacked panel produces dimensions suited for PDF export
        # (e.g. 12 x 16 in) which are far too tall for an embedded
        # canvas -- especially on Windows where the oversized figure
        # causes rendering/layout problems.
        # Cap the height to a screen-proportional value.
        figsize = self._compute_interactive_figsize(wave)

        show_residuals = self._pm.toggle_state.get("show_residuals", False)
        from .SpectralPanel import XScaling
        x_scaling = (
            XScaling.DATA_DENSITY
            if self._pm.toggle_state.get("data_density", False)
            else XScaling.WAVELENGTH
        )

        if self._sample_stack_on():
            spectra = self._build_sample_spectra()
            if spectra:
                return SampleStackPlot(
                    spectra,
                    layout=self._stack_layout(),
                    model_mode=self._stack_model_mode(),
                    figsize=figsize,
                    theme=self._pm.theme,
                    x_scaling=x_scaling,
                    gap_mode=self._gap_mode(),
                )

        if show_residuals:
            model_flux = self._compute_model_flux(wave_obs, wave)
            error_data = getattr(self._islat, "err_data", None)
            plot = ResidualSpectrumPlot(
                wave_data=wave,
                flux_data=flux,
                model_flux=model_flux,
                error_data=error_data,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                figsize=figsize,
                theme=self._pm.theme,
                x_scaling=x_scaling,
                gap_mode=self._gap_mode(),
            )
        else:
            # The composed plot handles all rendering via BasePlot helpers.
            # We do NOT pass line_list / atomic_lines here -- those are applied
            # dynamically by sync_toggle_state() so they can be toggled.
            plot = FullSpectrumPlot(
                wave_data=wave,
                flux_data=flux,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                figsize=figsize,
                theme=self._pm.theme,
                x_scaling=x_scaling,
                gap_mode=self._gap_mode(),
            )
        return plot

    # ------------------------------------------------------------------
    # Model-flux helpers
    # ------------------------------------------------------------------
    def _compute_model_flux(
        self,
        wave_obs: np.ndarray,
        wave_rest: np.ndarray,
    ) -> np.ndarray:
        """Return the summed model flux on the rest-frame wavelength grid.

        The model is constructed from all molecules via
        :meth:`MoleculeDict.get_summed_flux_resampled`, which uses
        flux-conserving resampling (spectres) to map the model onto
        *wave_rest*.  *wave_rest* must already be in the rest frame;
        the observer-frame array *wave_obs* is accepted for API
        compatibility but is no longer used inside this method.
        """
        try:
            _, model_flux = self._islat.molecules_dict.get_summed_flux_resampled(
                wave_rest, visible_only=True,
            )
        except Exception as exc:
            debug_config.warning(
                "full_spectrum_view",
                f"Could not compute model flux for residuals: {exc}",
            )
            return np.zeros_like(wave_rest)

        if len(model_flux) == 0:
            return np.zeros_like(wave_rest)

        return model_flux

    def _compute_interactive_figsize(
        self, wave: np.ndarray
    ) -> Tuple[float, float]:
        """Return a ``(width, height)`` suitable for the embedded GUI canvas.

        Uses the Tk screen height (if available) to cap the figure so it fits within the application window.
        Falls back to sensible defaults when no Tk root is reachable.
        """
        # Try to read screen dimensions from the Tk root
        screen_h_px: Optional[int] = None
        try:
            root = getattr(self._islat, "root", None)
            if root is not None:
                screen_h_px = root.winfo_screenheight()
        except Exception:
            pass

        if screen_h_px is None:
            # Fallback: assume a modest screen height
            screen_h_px = 900

        # Convert screen pixels to approximate matplotlib inches at the
        # logical DPI (100). Leave room for toolbars / window chrome.
        dpi = 100
        max_fig_h = min((screen_h_px - 160) / dpi, 14.0)
        # Ensure at least a reasonable minimum height
        max_fig_h = max(max_fig_h, 4.0)

        return (12, max_fig_h)

    @staticmethod
    def _plot_kind(plot: Any) -> str:
        """``"stack"``, ``"residual"`` or ``"plain"`` for a composed plot."""
        if isinstance(plot, SampleStackPlot):
            return "stack"
        if isinstance(plot, ResidualSpectrumPlot):
            return "residual"
        return "plain"

    def _wanted_plot_kind(self) -> str:
        """The plot kind the toggle state asks for (see :meth:`_create_plot`)."""
        if self._sample_stack_on():
            return "stack"
        if self._pm.toggle_state.get("show_residuals", False):
            return "residual"
        return "plain"

    def _replace_plot(self) -> None:
        """Replace the composed plot with a new one from :meth:`_create_plot`, keeping the panel layout.

        Used when the axes layout changes (residual sub-panels, sample
        stack).  The canvas is rebuilt for the new figure and the overlays
        are re-applied.
        """
        if not self._initialised:
            return

        # Preserve panel layout from the current plot so the user
        # sees the same wavelength ranges after the switch.
        old_edges = getattr(self._plot, "_panel_edges", None)
        old_ends = getattr(self._plot, "_panel_ends", None)
        old_step = getattr(self._plot, "_step", None)
        old_xlim = (
            getattr(self._plot, "_xlim_start", None),
            getattr(self._plot, "_xlim_end", None),
        )

        # Force a full teardown so the new figure replaces the old one
        old_fig = self.fig
        self.span_selectors.clear()
        self._plot = self._create_plot()

        # Restore the panel layout from the previous plot.
        if old_edges is not None and old_ends is not None:
            self._plot._panel_edges = old_edges
            self._plot._panel_ends = old_ends
            self._plot.n_panels = len(old_edges)
            self._plot._step = old_step
            if old_xlim[0] is not None:
                self._plot._xlim_start = old_xlim[0]
            if old_xlim[1] is not None:
                self._plot._xlim_end = old_xlim[1]

        self._plot.generate_plot()
        self._install_span_selectors()

        # The figure object changed (it always does): rebuild the canvas.
        self._show_canvas_if_replaced(old_fig)

        # Re-apply overlays
        self.sync_toggle_state(self._pm.toggle_state)
        self.draw()

    def _rebuild_plot(self) -> None:
        """Refresh data and regenerate the composed plot.

        If the panel layout changed, the figure is rebuilt from scratch.
        If only data/molecules changed and the composed plot supports
        fast in-place updates, existing axes are updated via
        ``update_panels_inplace()`` for a significant speed-up (avoids
        ``fig.clf()`` and re-creating all subplot objects).

        When the plot type needs to change (e.g. the ``show_residuals``
        toggle was flipped while the view was inactive), the plot is
        rebuilt from scratch via :meth:`_create_plot`.
        """
        wave, flux, wave_obs = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        if self._plot is None:
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            return

        # Detect a plot-type mismatch (e.g. FSP is active but show_residuals is now True,
        # or Stack Sample Spectra was turned on).  Force a full rebuild.
        type_mismatch = self._plot_kind(self._plot) != self._wanted_plot_kind()
        if not type_mismatch and isinstance(self._plot, SampleStackPlot):
            # Same kind: refresh the sample (spectra, parameter sets) in place.
            spectra = self._build_sample_spectra()
            if spectra:
                self._plot.set_spectra(spectra)
            self._plot.set_options(layout=self._stack_layout(), model_mode=self._stack_model_mode())
            self.span_selectors.clear()
            self._plot.generate_plot()
            self._install_span_selectors()
            return
        if type_mismatch:
            # Preserve the panel wavelength ranges so the user sees the same layout after the toggle.
            old_edges = getattr(self._plot, "_panel_edges", None)
            old_ends = getattr(self._plot, "_panel_ends", None)
            old_step = getattr(self._plot, "_step", None)
            old_xlim = (
                getattr(self._plot, "_xlim_start", None),
                getattr(self._plot, "_xlim_end", None),
            )

            self.span_selectors.clear()
            self._plot = self._create_plot()

            # Restore the panel layout from the previous plot so switching between FSP and RSP keeps the same wavelength ranges.
            if old_edges is not None and old_ends is not None:
                self._plot._panel_edges = old_edges
                self._plot._panel_ends = old_ends
                self._plot.n_panels = len(old_edges)
                self._plot._step = old_step
                if old_xlim[0] is not None:
                    self._plot._xlim_start = old_xlim[0]
                if old_xlim[1] is not None:
                    self._plot._xlim_end = old_xlim[1]

            self._plot.generate_plot()
            self._install_span_selectors()
            return

        # If the active plot is an RSP, keep its model_flux in sync.
        if isinstance(self._plot, ResidualSpectrumPlot):
            model_flux = self._compute_model_flux(wave_obs, wave)
            self._plot.model_flux = model_flux
            self._plot._model_flux_adj = model_flux

        # Try the fast-path: update_data + update_panels_inplace
        updater = getattr(self._plot, "update_data", None)
        if updater is not None:
            error_data = getattr(self._islat, "err_data", None)
            layout_changed = updater(
                wave_data=wave,
                flux_data=flux,
                molecules=self._islat.molecules_dict,
                wave_data_obs=wave_obs,
                error_data=error_data,
            )
        else:
            layout_changed = True

        if layout_changed:
            # Panel edges changed -- full rebuild
            self.span_selectors.clear()
            self._plot.generate_plot()
            self._install_span_selectors()
        else:
            # Layout unchanged -- fast in-place update of existing axes.
            inplace = getattr(self._plot, "update_panels_inplace", None)
            if inplace is not None:
                inplace()
            else:
                self._plot.generate_plot()

    # ==================================================================
    # Span selector (interactive-only feature)
    # ==================================================================
    def _span_selector_axes(self):
        """Yield ``(key, panel_idx, ax)`` for every axes that gets a span selector.

        Defaults to the primary axes of each panel, keyed by panel index;
        a sample stack in rows gets one per row, keyed ``(panel, row)``.
        """
        if isinstance(self._plot, SampleStackPlot) and self._plot.layout == "rows":
            for idx, cell in self._plot.subplots.items():
                for k, ax in enumerate(cell if isinstance(cell, tuple) else (cell,)):
                    yield (idx, k), idx, ax
            return
        for idx, ax in self.subplots.items():
            yield idx, idx, ax

    def _install_span_selectors(self) -> None:
        """Add span selectors (see :meth:`_span_selector_axes`) for click-to-inspect.

        :class:`SingleDrawSpanSelector`: with stock selectors a span left
        visible in one panel made every selector re-render the figure on
        each draw (one extra render per panel).
        """
        # Imported here: importing the GUI package while the Plotting package
        # is still initialising is circular (GUI -> FullSpectrumWindow -> this module).
        from ..GUI.InteractionHandler import SingleDrawSpanSelector

        self.span_selectors.clear()
        for key, idx, ax in self._span_selector_axes():
            span = SingleDrawSpanSelector(
                ax,
                lambda xmin, xmax, i=idx: self._on_span_select(xmin, xmax, i),
                direction="horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="lime"),
                interactive=True,
                drag_from_anywhere=True,
            )
            self.span_selectors[key] = span

    def _panel_index_of(self, ax: Any) -> Optional[int]:
        """Return the panel index whose cell contains *ax* (any of its sub-axes), or ``None``."""
        if ax is None or self._plot is None:
            return None
        for idx, val in getattr(self._plot, "subplots", {}).items():
            if ax is val or (isinstance(val, tuple) and any(ax is sub for sub in val)):
                return idx
        return None

    def _on_span_select(self, xmin: float, xmax: float, subplot_index: int) -> None:
        if abs(xmax - xmin) < 0.001:
            return
        main_plot = self._islat.GUI.plot
        if hasattr(main_plot, "is_full_spectrum") and main_plot.is_full_spectrum:
            main_plot.toggle_full_spectrum()
            if hasattr(self._islat, "root"):
                self._islat.root.after(100, lambda: self._apply_selection(xmin, xmax))
            else:
                self._apply_selection(xmin, xmax)
        else:
            self._apply_selection(xmin, xmax)

    def _apply_selection(self, xmin: float, xmax: float) -> None:
        main_plot = self._islat.GUI.plot

        main_plot.current_selection = (xmin, xmax)

        selection_center = (xmin + xmax) / 2
        selection_width = xmax - xmin
        xlim_start = self._plot._xlim_start
        xlim_end = self._plot._xlim_end
        total_range = xlim_end - xlim_start
        min_padding = total_range * 0.025
        view_padding = max(selection_width * 2, min_padding)

        main_plot.ax1.set_xlim(
            selection_center - view_padding,
            selection_center + view_padding,
        )

        # show_span keeps the main plot's span selector consistent with the
        # range just set (no blit of the background from before the move).
        ih = getattr(main_plot, "interaction_handler", None)
        if ih is not None:
            try:
                ih.show_span(xmin, xmax)
            except Exception as exc:
                debug_config.warning("full_spectrum_view", f"Could not show the selection span: {exc}")

        main_plot.onselect(xmin, xmax)
        main_plot.canvas.draw_idle()

    # ==================================================================
    # Context menu (the canvas and right-click routing come from OwnedCanvasMixin)
    # ==================================================================
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Return a ``tk.Menu`` for the right-clicked panel, or ``None``."""
        try:
            import tkinter as tk
        except ImportError:
            return None

        pm = self._pm
        plot = self._plot

        # Identify which panel was clicked (any sub-axes of its cell)
        panel_idx = self._panel_index_of(event.inaxes)

        menu = tk.Menu(canvas_widget, tearoff=0)

        if panel_idx is not None:
            try:
                xmin = float(plot._panel_edges[panel_idx])
                xmax = float(plot._panel_ends[panel_idx])
            except Exception:
                panel_idx = None

        if panel_idx is not None:
            def _open_in_three_panel():
                pm.islat.display_range = (xmin, xmax)
                # Mirror toggle_three_panel so the back-button works.
                pm._pre_threepanel_view_name = self.VIEW_NAME
                pm.switch_view("Three Panel")
                try:
                    pm.ax1.set_xlim(xmin, xmax)
                    pm.match_display_range(match_y=True)
                except Exception as exc:
                    debug_config.warning(
                        "full_spectrum_view",
                        f"build_context_menu: range update failed: {exc}",
                    )
                '''# Fire the span-selector so line inspection populates.
                try:
                    ih = getattr(pm, "interaction_handler", None)
                    if ih is not None:
                        ih.show_span(xmin, xmax)
                    pm.onselect(xmin, xmax)
                except Exception as exc:
                    debug_config.warning(
                        "full_spectrum_view",
                        f"build_context_menu: onselect failed: {exc}",
                    )'''

            menu.add_command(
                label="Open in Three Panel view",
                command=_open_in_three_panel,
            )

        # Sample stack: switch to the spectrum that was clicked.
        if isinstance(plot, SampleStackPlot):
            spectrum = plot.spectrum_at(event.inaxes, getattr(event, "ydata", None))
            if spectrum is not None and not getattr(spectrum, "is_active", False) and spectrum.path:
                menu.add_command(
                    label=f"Make {spectrum.name} the Active Spectrum",
                    command=lambda s=spectrum: self._activate_sample_spectrum(s),
                )

        self._append_save_figure_item(menu)
        return menu

    def _activate_sample_spectrum(self, spectrum: Any) -> None:
        """Make *spectrum* (a sample entry) the loaded spectrum, as the Sample Manager does."""
        from iSLAT.Modules.DataProcessing.SampleStack import same_path
        sample = list(getattr(self._islat, "sample_spectra", None) or [])
        index = next((i for i, p in enumerate(sample) if same_path(p, spectrum.path)), None)
        if index is None:
            return
        self._islat.switch_to_spectrum(index)
        pane = getattr(getattr(self._islat, "GUI", None), "file_interaction_pane", None)
        if pane is not None and hasattr(pane, "update_sample_spectra_label"):
            try:
                pane.update_sample_spectra_label()
            except Exception:
                pass

    # ==================================================================
    # PlotView lifecycle
    # ==================================================================
    #: The current selection is only handed on from here, never drawn.
    _DISPLAY_IGNORED_STATE = frozenset({"current_selection"})

    def _display_inputs(self):
        """Adds the saved-line file and label settings and the atomic-line table (the overlays)."""
        values, objects = super()._display_inputs()
        values = values + (
            self._saved_line_file_signature(),
            get_saved_line_label_settings(),
            self._saved_line_labels_visible(),
        )
        return values, objects + (load_atomic_lines(),)

    def deactivate(self) -> None:
        """Remember what the panels show, then unregister the controls and unpack the canvas."""
        self._remember_shown_frame()
        super().deactivate()

    def activate(self, parent_frame: Any) -> None:
        self._parent_frame = parent_frame

        if self._initialised and self._shown_frame_is_current():
            # Nothing it draws changed while another view was shown: re-pack
            # the canvas, which still holds its last frame, without
            # re-syncing or redrawing.
            self._show_canvas()
            self._repaint_shown_frame()
            self._register_control_fields()
            return

        if not self._initialised:
            # First activation - build the composed plot
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            self._initialised = True
            self._needs_refresh = False
        elif self._needs_refresh:
            # Data changed while we were inactive - full refresh
            self._rebuild_plot()
            self._needs_refresh = False
        # else: simple view toggle - just repack the existing canvas

        # (Re)builds the canvas when the figure was replaced, then packs it.
        self._show_canvas()

        # Sync theme in case it changed while the other view was active.
        self.apply_theme(self._pm.theme)

        # Reconcile overlays with the controller's toggle dict
        self.sync_toggle_state(self._pm.toggle_state)

        # Register view-specific controls with the ControlBus
        self._register_control_fields()

        if self._canvas is not None:
            self._canvas.draw_idle()

    def _register_control_fields(self) -> None:
        """Register this view's :class:`ControlField` objects on the :class:`ControlBus`.

        Called from :meth:`activate`.  Safe to call even when the bus is not
        yet wired (e.g. in lightweight test environments) - missing surfaces
        are silently ignored.
        """
        bus = getattr(self._pm, 'control_bus', None)
        if bus is None:
            return

        from iSLAT.Modules.GUI.ControlField import (
            DisplayRangeField, DropdownField, EntryField, ToggleField,
        )

        # First unregister any stale fields from a previous activation
        bus.unregister_owner(self)

        # --- ControlPanel fields ---
        bus.register_many([
            EntryField(
                "n_panels", "N Panels:", self._get_n_panels, self._set_n_panels,
                datatype="int", width=5, owner=self,
                tip="Number of panels in the\nfull-spectrum stacked layout",
            ),
            DisplayRangeField(
                "_fsv_display_range",
                getter=self._get_display_range,
                setter=self._set_display_range,
                owner=self,
            ),
        ], "control_panel")

        # --- Views menu toggles (view controls section) ---
        pm = self._pm

        def _residuals_getter() -> bool:
            return pm.toggle_state.get("show_residuals", False)

        def _residuals_setter(value: bool) -> None:
            if value != pm.toggle_state.get("show_residuals", False):
                pm.toggle_residuals()

        bus.register(
            ToggleField(
                "show_residuals", "Residuals",
                getter=_residuals_getter,
                setter=_residuals_setter,
                owner=self,
                tip=f"Toggle residual sub-panels on/off\nin full spectrum mode\nKeybind: {shortcut_text('toggle_residuals')}",
            ),
            "views_menu",
        )

        bus.register(
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
            "views_menu",
        )

        #bus.register(self._skip_gaps_field(ToggleField), "views_menu")

        if self.SUPPORTS_SAMPLE_STACK:
            bus.register_many([
                ToggleField(
                    "sample_stack", "Stack Sample Spectra",
                    getter=self._get_sample_stack,
                    setter=self._set_sample_stack,
                    owner=self,
                    tip=SAMPLE_STACK_TIP,
                ),
                DropdownField(
                    "sample_stack_layout", "Stack Layout",
                    getter=self._get_stack_layout_label,
                    setter=self._set_stack_layout_label,
                    options=list(SAMPLE_STACK_LAYOUT_LABELS.values()),
                    owner=self,
                    tip=(
                        "How the sample's spectra are stacked:\n"
                        "Offset in Each Panel: all in the same panel,\n"
                        "  shifted up so they do not overlap (the Y-axis\n"
                        "  options do not apply).\n"
                        "One Row per Spectrum: each panel split into\n"
                        "  one row per spectrum, each with its own\n"
                        "  Y scale (fewer N Panels give taller rows)."
                    ),
                ),
                DropdownField(
                    "sample_stack_models", "Stack Models",
                    getter=self._get_stack_models_label,
                    setter=self._set_stack_models_label,
                    options=list(SAMPLE_STACK_MODEL_LABELS.values()),
                    owner=self,
                    tip=(
                        "Models drawn with each stacked spectrum,\n"
                        "from that spectrum's parameter set:\n"
                        "Data Only, the Total Model (filled), or\n"
                        "Each Molecule + Total.  The molecules shown\n"
                        "follow the control panel's visibility."
                    ),
                ),
            ], "views_menu")

    def _skip_gaps_field(self, toggle_field_cls: Any) -> Any:
        """The *Skip Data Gaps* Views-menu toggle (shared with :class:`OpticalDepthView`)."""
        return toggle_field_cls(
            "skip_gaps", "Skip Data Gaps",
            getter=self._get_skip_gaps,
            setter=self._set_skip_gaps,
            owner=self,
            tip=SKIP_GAPS_TIP,
        )

    # deactivate (unregister the ControlBus fields, unpack the canvas) comes
    # from OwnedCanvasMixin.

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self, theme: dict) -> None:
        """Apply *theme* to the composed plot, figure, and canvas widget.

        Propagates the theme to:
        - the composed :class:`StackedSpectralPanel` (``self._plot.theme``)
        - every axes in the figure via :meth:`BasePlot.apply_theme_to_figure`
        - the Tk canvas widget background
        """
        if self._plot is not None:
            self._plot.theme = theme
            self._plot.apply_theme_to_figure()
        if self._canvas is not None:
            apply_canvas_theme(self._canvas, theme)
            self._canvas.draw_idle()

    # ==================================================================
    # Core rendering
    # ==================================================================
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not self._initialised:
            self._needs_refresh = True
            return

        old_fig = self.fig
        self._rebuild_plot()
        self._needs_refresh = False
        self._show_canvas_if_replaced(old_fig)

        # Apply toggle state overlays after regeneration
        self.sync_toggle_state(self._pm.toggle_state)
        self.draw()

    # on_selection, clear_selection and clear_active_lines keep PlotView's
    # no-op defaults: this view has no line-inspection panel and selects
    # through its own span selectors (see _on_span_select).

    # ------------------------------------------------------------------
    # Molecule lifecycle callbacks
    # ------------------------------------------------------------------
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Mark stale - FSV doesn't have per-molecule panels to update."""
        self._needs_refresh = True

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Re-render immediately if this view is active, otherwise mark stale."""
        if parameter_name == 'is_visible':
            return

        if not self._initialised or self._plot is None:
            # Not yet rendered - just mark stale for the next activate().
            self._needs_refresh = True
            return

        mol_dict = getattr(self._islat, 'molecules_dict', None)
        if mol_dict is None:
            self._needs_refresh = True
            return

        molecule = mol_dict.get(molecule_name)
        if molecule is None or not getattr(molecule, 'is_visible', True):
            # Hidden molecules don't need an immediate repaint; mark stale
            # so they re-render correctly when made visible again.
            self._needs_refresh = True
            return

        if isinstance(self._plot, SampleStackPlot):
            # Copies of the current parameters must follow the change, and
            # the stack offsets depend on every model: rebuild the stack.
            self._refresh_sample_stack()
            return

        # Active view with a visible molecule - update in-place now.
        self._plot.update_panels_inplace()
        # update_panels_inplace always rebuilds the legend; re-hide it when
        # the user has toggled the legend off.
        if not self._pm.legend_toggle:
            self.toggle_legend(False)
        # ...and recreates observed artists / panel Y limits.
        self._reapply_display_state(self._pm.toggle_state)
        if self._canvas is not None:
            self._canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """Mark stale so the next activate() does a full rebuild."""
        self._needs_refresh = True

    # ------------------------------------------------------------------
    # Artist-manipulation helpers (replace removed PlotRenderer calls)
    # ------------------------------------------------------------------
    def _set_molecule_visibility(
        self, molecule_name: str, is_visible: bool, ax: "Axes"
    ) -> bool:
        """Toggle visibility of every Line2D tagged with *molecule_name* on *ax*.

        Returns ``True`` if at least one artist was found and toggled.
        """
        return set_molecule_visible(ax, molecule_name, is_visible)

    def _remove_molecule_lines(self, molecule_name: str, ax: "Axes") -> None:
        """Remove all Line2D artists tagged with *molecule_name* from *ax*."""
        remove_molecule_lines(ax, molecule_name)

    def _render_molecule_spectrum(
        self, molecule: "Molecule", wave_data: "np.ndarray", ax: "Axes"
    ) -> None:
        """Plot *molecule*'s spectrum onto *ax* using the composed plot's styling.

        Falls back to a full ``update_panels_inplace()`` rebuild when the composed plot object is not yet available.
        """
        if self._plot is None:
            return
        # Delegate to FullSpectrumPlot's mol-cache helper so color/style
        # stays consistent with the rest of the composed plot.
        # _build_mol_cache returns (lam, flux, color, label, name) traces;
        # key them by name for O(1) lookup.
        mol_cache_list, _labels, _colors = self._plot._build_mol_cache()
        mol_cache = {trace[4]: trace for trace in mol_cache_list}
        trace = mol_cache.get(getattr(molecule, "name", None))
        if trace is None:
            # Molecule not in cache - trigger a full inplace refresh instead.
            self._plot.update_panels_inplace()
            self._reapply_display_state(self._pm.toggle_state)
            if self._canvas is not None:
                self._canvas.draw_idle()
            return
        plot_molecule_trace(
            ax, trace, ax.get_xlim(),
            linewidth=self._plot.get_molecule_linewidth(
                molecule,
                self._plot._get_theme_value("full_spectrum_model_linewidth", 0.8),
            ),
            alpha=self._plot._get_theme_value("full_spectrum_model_alpha", 0.8),
            zorder=self._plot._get_theme_value("zorder_model", 3),
        )

    # ------------------------------------------------------------------
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
        if not self._initialised or not self.subplots:
            return

        if isinstance(self._plot, SampleStackPlot):
            # Visibility applies to every spectrum of the stack.
            self._refresh_sample_stack()
            return

        # Use the same RV-corrected wavelengths as the rendered panels
        # for display positioning, and observer-frame wavelengths for
        # model computation (get_summed_flux expects observer frame).
        rv_wave = self._plot.wave_data if self._plot is not None else wave_data
        wave_obs = (
            getattr(self._plot, "wave_data_obs", self._plot.wave_data)
            if self._plot is not None
            else wave_data
        )

        # For molecule artist operations use the primary (spectrum) axes only.
        # For FSP these are the same as all axes; for RSP this avoids placing molecule overlays on residual sub-panels.
        is_rsp = isinstance(self._plot, ResidualSpectrumPlot)
        primary_axes = list(self.subplots.values())

        # 1. Toggle molecule line visibility per subplot.
        #    When force_rerender is True (parameters changed while hidden),
        #    destroy the stale artists and re-create them from fresh data.
        if force_rerender and is_visible:
            molecule = molecules_dict.get(molecule_name)
            for ax in primary_axes:
                self._remove_molecule_lines(molecule_name, ax)
                if molecule is not None:
                    self._render_molecule_spectrum(molecule, rv_wave, ax)
        else:
            # Try fast artist-toggle first.
            toggled = False
            for ax in primary_axes:
                if self._set_molecule_visibility(molecule_name, is_visible, ax):
                    toggled = True

            # If turning ON but no artists exist (e.g. after a full rebuild while the molecule was hidden), create them from scratch.
            if is_visible and not toggled:
                molecule = molecules_dict.get(molecule_name)
                if molecule is not None:
                    for ax in primary_axes:
                        self._render_molecule_spectrum(molecule, rv_wave, ax)

        # 2. Recompute summed spectrum / model and update residuals + chi^2.
        if is_rsp:
            # RSP's update_panels_inplace recomputes the visible-only model
            # flux, redraws the summed fill on spectrum panels, redraws all
            # residual panels, and refreshes the per-panel and global chi^2
            # annotations - all in one call.
            self._plot.update_panels_inplace()
            self._reapply_display_state(self._pm.toggle_state)
        else:
            try:
                summed_wavelengths, summed_flux = molecules_dict.get_summed_flux(
                    wave_obs, visible_only=True, extend_to_global_range=True,
                )
            except Exception as exc:
                debug_config.warning("full_spectrum_view", f"Could not compute summed flux: {exc}")
                summed_wavelengths = rv_wave if rv_wave is not None else np.array([])
                summed_flux = np.zeros_like(summed_wavelengths)

            summed_visible = self._pm.summed_toggle and bool(
                molecules_dict.get_visible_molecules(return_objects=True)
            )

            for ax in primary_axes:
                remove_tagged(ax, SUMMED_TAG, lines=False)
                fill = draw_summed_fill(
                    ax, summed_wavelengths, summed_flux,
                    color=self._plot._get_theme_value("summed_spectra_color", "lightgray"),
                    zorder=self._plot._get_theme_value("zorder_summed", 1),
                    xr=ax.get_xlim(),
                )
                if fill is not None:
                    fill.set_visible(summed_visible)

        # 3. Rebuild legend
        self._update_full_spectrum_legend(molecules_dict)
        self.draw()

    # ==================================================================
    # Toggle helpers - overrides and specializations of ToggleMixin
    # ==================================================================
    def _set_legend_visibility(self, visible: bool) -> None:
        """Override ToggleMixin hook to use the composed plot's legend strategy."""
        legend_ax = self.subplots.get(0)
        if legend_ax is not None and self._plot is not None:
            self._plot.legend_strategy.update_visibility(legend_ax, visible)

    # sync_toggle_state, toggle_legend, toggle_summed_spectrum, toggle_saved_lines
    # and toggle_atomic_lines come from ToggleMixin (which keeps overlays that
    # _overlay_drawn reports as current, re-reads the saved lines from disk
    # before drawing them, and shows / hides the legend through
    # _set_legend_visibility above).

    def toggle_residuals(self, show: bool) -> None:
        """Switch between :class:`FullSpectrumPlot` and :class:`ResidualSpectrumPlot`.

        Triggers a full plot rebuild because the axes layout changes (RSP has paired spectrum + residual sub-panels).
        The canvas is destroyed and re-created so the Tk widget reflects the new figure dimensions.
        Turning residuals on turns *Stack Sample Spectra* off (the two are exclusive).
        """
        if show and self._pm.toggle_state.get("sample_stack", False):
            self._pm.toggle_state["sample_stack"] = False
            self._release_sample_stack()
        self._replace_plot()

    # ==================================================================
    # Overlay artist helpers (interactive-only)
    # ==================================================================
    def _add_saved_line_artists(self) -> None:
        """Add saved-line annotations to every panel in the composed plot.

        Delegates to :meth:`StackedSpectralPanel.plot_saved_lines` which
        iterates over every :class:`SpectralPanel` so each panel places labels relative to its own y-limits.
        """
        if self._plot is None:
            return

        # Reload line data from disk so we always have the latest
        if self.line_data is None:
            self.line_data = self._load_line_data()
        if self.line_data is None:
            return

        # Ensure the 'line' column exists (BasePlot expects it)
        if "line" not in self.line_data.columns:
            self.line_data["line"] = [""] * len(self.line_data)

        self._plot.plot_saved_lines(self.line_data, show_labels=self._saved_line_labels_visible())
        self._record_overlay("saved_lines")

    def _saved_line_labels_visible(self) -> bool:
        """Whether this view labels its saved lines (its :attr:`VIEW_NAME` toggle)."""
        from .SavedLineLabels import labels_visible
        return labels_visible(self.VIEW_NAME)

    def _remove_saved_line_artists(self) -> None:
        """Remove all ``_islat_saved_line`` artists."""
        self._overlay_records.pop("saved_lines", None)
        if self._plot is not None:
            self._plot.remove_saved_lines()

    def _add_atomic_line_artists(self) -> None:
        """Add atomic-line annotations to every panel in the composed plot.

        Delegates to :meth:`StackedSpectralPanel.plot_atomic_lines` which iterates over
        every :class:`SpectralPanel` so each panel places labels relative to its own y-limits.
        """
        if self._plot is None:
            return

        atomic_data = load_atomic_lines()
        self._plot.plot_atomic_lines(atomic_data)
        self._record_overlay("atomic_lines")

    def _remove_atomic_line_artists(self) -> None:
        """Remove all ``_islat_atomic_line`` artists."""
        self._overlay_records.pop("atomic_lines", None)
        if self._plot is not None:
            self._plot.remove_atomic_lines()

    # ------------------------------------------------------------------
    # Overlay bookkeeping: sync_toggle_state keeps overlays that are current
    # ------------------------------------------------------------------
    _OVERLAY_TAGS = {"atomic_lines": ATOMIC_LINE_TAG, "saved_lines": SAVED_LINE_TAG}

    def _overlay_artists(self, kind: str) -> Tuple[Any, ...]:
        """The *kind* overlay's lines, collections and texts on every panel axes."""
        tag = self._OVERLAY_TAGS[kind]
        return tuple(
            artist
            for ax in self._iter_toggle_axes()
            for artist in iter_tagged(ax, tag, lines=True, collections=True, texts=True)
        )

    def _overlay_key(self, kind: str) -> Any:
        """Everything besides its artists that the *kind* overlay was drawn from.

        The panels' x / y limits (the markers are drawn for the panel range
        and the labels at the top of the y range they were drawn at), plus
        the atomic-line table, or the saved-line file and label settings.
        """
        limits = tuple(
            (tuple(ax.get_xlim()), tuple(ax.get_ylim())) for ax in self._iter_toggle_axes()
        )
        if kind == "atomic_lines":
            return limits, id(load_atomic_lines())
        return (
            limits, self._saved_line_file_signature(),
            get_saved_line_label_settings(), self._saved_line_labels_visible(),
        )

    def _saved_line_file_signature(self) -> Any:
        """``(path, mtime, size)`` of the input line list, so an edited or replaced list is redrawn."""
        path = getattr(self._islat, "input_line_list", None)
        if not path:
            return None
        try:
            stat = os.stat(path)
        except (OSError, TypeError, ValueError):
            return (str(path), None, None)
        return (str(path), stat.st_mtime_ns, stat.st_size)

    def _record_overlay(self, kind: str) -> None:
        """Remember the artists the *kind* overlay drew and what they were drawn from."""
        self._overlay_records[kind] = (self._overlay_artists(kind), self._overlay_key(kind))

    def _overlay_drawn(self, kind: str) -> Optional[bool]:
        """Whether the *kind* overlay is on the panels and still current (ToggleMixin hook).

        ``True`` when exactly the artists of its last drawing are on the axes
        and nothing they were drawn from changed, so :meth:`sync_toggle_state`
        leaves them alone instead of re-creating them (and re-reading the
        saved-line file).  ``False`` when none are on the axes.  ``None``
        (remove, then redraw) otherwise: the line list was edited, the
        panels' limits moved, a plot update removed some of them, ...
        """
        current = self._overlay_artists(kind)
        record = self._overlay_records.get(kind)
        if record is not None:
            artists, key = record
            if ({id(a) for a in current} == {id(a) for a in artists}
                    and key == self._overlay_key(kind)):
                return True
            # Stale: drop it, so it does not keep a replaced figure's artists alive.
            del self._overlay_records[kind]
        return None if current else False

    # ==================================================================
    # File output: PlotView.save_figure saves what _build_export_figure builds
    # ==================================================================
    def _create_standalone_plot(
        self,
        wave: np.ndarray,
        flux: np.ndarray,
        wave_obs: np.ndarray,
        *,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        figsize: Tuple[float, float] = (DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT),
        xlim_range: Optional[Tuple[float, float]] = None,
        n_panels: Optional[int] = None,
        display_options: Optional[Dict[str, Any]] = None,
    ) -> StackedSpectralPanel:
        """Build a fresh standalone :class:`StackedSpectralPanel` for export.

        Subclasses can override this to return a different stacked-panel implementation (e.g. :class:`ResidualSpectrumPlot`) when the user saves the figure from a residual-backed view.

        *display_options* are the :class:`FullSpectrumPlot` display keywords
        (see :func:`toggle_state_display_kwargs`).

        The returned plot is **not yet rendered**; the caller is responsible for calling :meth:`generate_plot`.
        """
        kw: dict = dict(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
            line_list=line_list,
            atomic_lines=atomic_lines,
            figsize=figsize,
            wave_data_obs=wave_obs,
            theme=self._pm.theme,
            xlim_range=xlim_range,
        )
        if n_panels is not None:
            kw["n_panels"] = n_panels
        if display_options:
            kw.update(display_options)
        if self._sample_stack_on():
            spectra = self._sample_stack_spectra or self._build_sample_spectra()
            if spectra:
                for key in ("wave_data", "flux_data", "molecules", "wave_data_obs"):
                    kw.pop(key, None)
                return SampleStackPlot(
                    spectra, layout=self._stack_layout(), model_mode=self._stack_model_mode(), **kw,
                )
        return FullSpectrumPlot(**kw)

    def _can_save_figure(self) -> bool:
        """Exports build a figure of their own, so saving works even before the view was first shown."""
        return True

    def _build_export_figure(self):
        """A fresh standalone figure with the toggle state baked in, so saving never touches the GUI figure."""
        plot = self._create_export_plot()
        return plot.fig, plot.close

    def _export_range(self) -> Tuple[Optional[Tuple[float, float]], Optional[int]]:
        """``(xlim_range, n_panels)`` of the live plot, so the saved figure matches what the user sees.

        Before the first render (no live plot) the molecules' global
        wavelength range and the default panel count are used.
        """
        xlim_range: Optional[Tuple[float, float]] = None
        n_panels: Optional[int] = None
        if self._plot is not None:
            x_start = getattr(self._plot, '_xlim_start', None)
            x_end   = getattr(self._plot, '_xlim_end',   None)
            if x_start is not None and x_end is not None:
                xlim_range = (float(x_start), float(x_end))
            n_panels = getattr(self._plot, 'n_panels', None)
        if xlim_range is None:
            mol_dict = getattr(self._islat, 'molecules_dict', None)
            mol_range = getattr(mol_dict, '_global_wavelength_range', None) if mol_dict else None
            if mol_range is not None:
                xlim_range = (float(mol_range[0]), float(mol_range[1]))
        return xlim_range, n_panels

    def _create_export_plot(self) -> StackedSpectralPanel:
        """Build and render the standalone plot that :meth:`save_figure` writes.

        It follows the toggle state (saved / atomic lines, observed data,
        summed model, Y-axis policy, wavelength marks), the live wavelength
        range and panel count, and the session's export size.  The caller
        closes it after saving.
        """
        ts = self._pm.toggle_state
        wave, flux, wave_obs = self._load_spectrum_data()
        xlim_range, n_panels = self._export_range()
        plot = self._create_standalone_plot(
            wave, flux, wave_obs,
            line_list=self.line_data if ts.get("saved_lines", False) else None,
            atomic_lines=load_atomic_lines() if ts.get("atomic_lines", False) else None,
            figsize=(DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT),
            xlim_range=xlim_range,
            n_panels=n_panels,
            display_options=toggle_state_display_kwargs(ts),
        )
        try:
            apply_export_size(plot, get_export_settings(self._pm))
            plot.generate_plot()
            draw_toggle_state_markers(plot.fig, ts)

            # Safety net for factory overrides whose plot ignores display_options:
            # hide toggled-off artists on every axes (idempotent for FullSpectrumPlot).
            hidden_tags = [
                tag for tag, key in ((SUMMED_TAG, "summed"), (OBSERVED_TAG, "observed"))
                if not ts.get(key, True)
            ]
            if hidden_tags:
                for ax in plot.fig.axes:
                    set_tagged_visible(ax, hidden_tags, False)
        except Exception:
            plot.close()
            raise
        return plot

    # ==================================================================
    # Figure (draw and get_canvas come from OwnedCanvasMixin)
    # ==================================================================
    def get_figure(self) -> "Figure":
        return self.fig  # type: ignore[return-value]

    # ==================================================================
    # Internal legend helper
    # ==================================================================
    def _get_legend(self) -> Optional["Legend"]:
        if self.subplots and 0 in self.subplots:
            return self.subplots[0].legend_
        return None

    def _update_full_spectrum_legend(self, molecules_dict: "MoleculeDict") -> None:
        visible_mols = molecules_dict.get_visible_molecules(return_objects=True)
        mol_labels = [BasePlot.get_molecule_display_name(mol) for mol in visible_mols]
        mol_colors = [BasePlot.get_molecule_color(mol) for mol in visible_mols]

        legend_ax = self.subplots.get(0)
        if legend_ax is None:
            return

        self._plot.legend_strategy.build_legend(
            legend_ax, self._plot.fig, mol_labels, mol_colors,
        )

        # Respect the legend toggle state from the controller
        if not self._pm.legend_toggle:
            self._plot.legend_strategy.update_visibility(legend_ax, False)

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy(self) -> None:
        """Permanently dispose of the full spectrum plot and canvas."""
        self._destroy_canvas()
        if self._plot is not None:
            self._plot.close()
            self._plot = None
        self._release_sample_stack()
        self._initialised = False
        self._needs_refresh = True

# ======================================================================
# Export helpers (shared by output_full_spectrum and FullSpectrumView.save_figure)
# ======================================================================
_EXPORT_SETTINGS_ATTR = "full_spectrum_export_settings"


def toggle_state_display_kwargs(toggle_state: Optional[dict]) -> Dict[str, Any]:
    """Translate the GUI *toggle_state* into :class:`FullSpectrumPlot` display options.

    Mirrors what the interactive views show: observed-data and summed-model
    visibility, manual Y limits while auto-scaling is off, and otherwise the
    data min / max sync scope (``y_data_sync``, only when set) or the
    ``"data"`` / ``"model"`` auto-scale source.  *Skip Data Gaps*
    (``skip_gaps``) adds ``gap_mode="skip"``.
    """
    ts = toggle_state if isinstance(toggle_state, dict) else {}
    kw: Dict[str, Any] = {
        "show_observed": bool(ts.get("observed", True)),
        "show_summed": bool(ts.get("summed", True)),
        "y_autoscale_source": ts.get("y_autoscale_source", "data"),
    }
    if ts.get("skip_gaps", False):
        kw["gap_mode"] = GapMode.SKIP.value
    sync = normalize_y_data_sync(ts.get("y_data_sync"))
    if ts.get("y_autoscale", True) and sync is not None:
        kw["y_data_sync"] = sync
    if not ts.get("y_autoscale", True):
        manual = ts.get("y_limits")
        if manual is not None:
            kw["y_limits"] = (float(manual[0]), float(manual[1]))
        else:
            # Auto-scale off without stored limits leaves the layout's own limits in the GUI.
            kw["y_autoscale_source"] = "data"
    return kw


def draw_toggle_state_markers(fig: Any, toggle_state: Optional[dict]) -> None:
    """Draw the GUI's user wavelength markers on every axes of *fig* (exports / standalone plots)."""
    if fig is None or not isinstance(toggle_state, dict):
        return
    markers = toggle_state.get("wavelength_markers") or []
    if not markers:
        return
    for ax in fig.axes:
        draw_wavelength_markers(ax, markers)


def _validate_export_dim(name: str, value: Any, maximum: float) -> Optional[float]:
    """Return *value* as a float in ``(0, maximum]``, or ``None`` for "use the default"."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number.") from None
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    if value > maximum:
        raise ValueError(f"{name} must be at most {maximum:g} in.")
    return value


def get_export_settings(plot_manager: Any) -> Dict[str, Optional[float]]:
    """Return the session's full-spectrum export size overrides.

    ``{"width": ..., "panel_height": ...}`` in inches, where ``None`` means
    the default (:data:`DEFAULT_EXPORT_WIDTH` wide, :data:`DEFAULT_EXPORT_HEIGHT`
    tall in total).  Invalid stored values fall back to the defaults.
    """
    settings: Dict[str, Optional[float]] = {"width": None, "panel_height": None}
    raw = getattr(plot_manager, _EXPORT_SETTINGS_ATTR, None)
    if isinstance(raw, dict):
        for key, maximum in (("width", MAX_EXPORT_WIDTH), ("panel_height", MAX_EXPORT_PANEL_HEIGHT)):
            try:
                settings[key] = _validate_export_dim(key, raw.get(key), maximum)
            except ValueError:
                pass
    return settings


def set_export_settings(
    plot_manager: Any,
    width: Any = None,
    panel_height: Any = None,
) -> Dict[str, Optional[float]]:
    """Validate and remember the full-spectrum export size for the session.

    *width* is the figure width and *panel_height* the height of each panel,
    both in inches; ``None`` or a blank string restores the default.

    Raises
    ------
    ValueError
        If a value is not a positive number within the accepted range.
    """
    settings = {
        "width": _validate_export_dim("Figure width", width, MAX_EXPORT_WIDTH),
        "panel_height": _validate_export_dim("Panel height", panel_height, MAX_EXPORT_PANEL_HEIGHT),
    }
    setattr(plot_manager, _EXPORT_SETTINGS_ATTR, dict(settings))
    return settings


def apply_export_size(
    plot: StackedSpectralPanel,
    settings: Optional[Dict[str, Optional[float]]] = None,
) -> Tuple[float, float]:
    """Size *plot* for export and return the resulting ``(width, height)``.

    Defaults to :data:`DEFAULT_EXPORT_WIDTH` x :data:`DEFAULT_EXPORT_HEIGHT`.
    A ``width`` override widens the figure (longer panels); a
    ``panel_height`` override makes the figure ``panel_height * n_panels`` tall.
    """
    settings = settings or {}
    panel_height = settings.get("panel_height") or None
    return plot.set_figure_size(
        width=settings.get("width") or DEFAULT_EXPORT_WIDTH,
        row_height=panel_height,
        height=None if panel_height else DEFAULT_EXPORT_HEIGHT,
    )


# ======================================================================
# Backward-compatible top-level function
# ======================================================================
def output_full_spectrum(
    islat_ref: Any,
    rasterized: bool = False,
    figure_width: Optional[float] = None,
    panel_height: Optional[float] = None,
) -> str | None:
    """
    Create a fresh full-spectrum PDF and a companion parameters CSV.

    The figure follows the GUI's toggle state (saved / atomic lines, summed
    model, observed data, Y auto-scale / manual limits) and is sized from the
    session's export settings (see :func:`set_export_settings`).

    Parameters
    ----------
    islat_ref : iSLAT
        The running iSLAT instance.
    rasterized : bool
        Rasterize the axes (smaller PDFs for very dense data).
    figure_width, panel_height : float, optional
        Override the figure width / per-panel height (inches) for this export
        only, instead of the session's export settings.

    This is the backward-compatible replacement for the function that previously lived in ``OutputFullSpectrum.py``.
    """
    # Use in-memory data instead of re-reading the CSV from disk.
    # Delegate the stellar RV correction to MoleculeDict so the formula lives in one place.
    wave_obs = np.array(islat_ref.wave_data_original, copy=True)
    wave = islat_ref.molecules_dict.apply_stellar_rv(wave_obs)
    flux = np.array(islat_ref.flux_data, copy=True)

    # Read toggle state and export settings if available
    ts: dict = {}
    theme: Optional[dict] = None
    gui_plot: Any = None
    if hasattr(islat_ref, "GUI") and hasattr(islat_ref.GUI, "plot"):
        gui_plot = islat_ref.GUI.plot
        ts = getattr(gui_plot, "toggle_state", {})
        theme = getattr(gui_plot, "theme", None)
    if not isinstance(ts, dict):
        ts = {}

    export_settings = get_export_settings(gui_plot)
    if figure_width is not None:
        export_settings["width"] = _validate_export_dim("Figure width", figure_width, MAX_EXPORT_WIDTH)
    if panel_height is not None:
        export_settings["panel_height"] = _validate_export_dim("Panel height", panel_height, MAX_EXPORT_PANEL_HEIGHT)

    line_list_df: Optional[pd.DataFrame] = None
    if ts.get("saved_lines", False):
        try:
            line_list_df = pd.read_csv(islat_ref.input_line_list, sep=",")
        except Exception:
            pass

    atomic_lines_df: Optional[pd.DataFrame] = None
    if ts.get("atomic_lines", False):
        atomic_lines_df = load_atomic_lines()

    # Create standalone plot - uses BasePlot._ensure_figure (non-pyplot MplFigure)
    standalone_kwargs = dict(
        wave_data=wave,
        flux_data=flux,
        molecules=islat_ref.molecules_dict,
        line_list=line_list_df,
        atomic_lines=atomic_lines_df,
        figsize=(DEFAULT_EXPORT_WIDTH, DEFAULT_EXPORT_HEIGHT),
        wave_data_obs=wave_obs,
        **toggle_state_display_kwargs(ts),
    )
    if theme is not None:
        standalone_kwargs["theme"] = theme
    # Respect the GUI's wavelength range so the saved figure matches
    # what is shown on screen rather than always spanning all data.
    mol_range = getattr(islat_ref.molecules_dict, '_global_wavelength_range', None)
    if mol_range is not None:
        standalone_kwargs["xlim_range"] = (float(mol_range[0]), float(mol_range[1]))
    # Respect the GUI's panel count.
    try:
        gui_plot = islat_ref.GUI.plot
        live_fsv = getattr(gui_plot, '_full_spectrum_view', None) or getattr(gui_plot, 'current_view', None)
        live_plot = getattr(live_fsv, '_plot', None)
        if live_plot is None:
            # ThreePanelView / direct attribute
            live_plot = getattr(gui_plot, '_plot', None)
        if live_plot is not None:
            n_panels = getattr(live_plot, 'n_panels', None)
            if n_panels is not None:
                standalone_kwargs["n_panels"] = n_panels
    except Exception:
        pass
    standalone = FullSpectrumPlot(**standalone_kwargs)
    apply_export_size(standalone, export_settings)
    standalone.generate_plot()
    draw_toggle_state_markers(standalone.fig, ts)

    save_path = ask_save_path(
        FullSpectrumView.SAVE_DIALOG_TITLE, "pdf",
        initialfile=default_save_name(islat_ref, FullSpectrumView.SAVE_FILE_SUFFIX, "pdf"),
        initialdir=absolute_data_files_path,
    )
    if not save_path:
        standalone.close()
        return None

    try:
        save_figure_file(standalone.fig, save_path, file_format="pdf", rasterized=rasterized)
    finally:
        standalone.close()

    # Write companion parameters CSV
    file_name = str(Path(save_path).with_suffix("")) + "_parameters.csv"
    ifh.write_molecules_to_csv(
        islat_ref.molecules_dict,
        file_path=dirname(save_path),
        file_name=file_name,
    )

    notify_user(islat_ref, FullSpectrumView.SAVE_MESSAGE.format(path=save_path))

    return save_path