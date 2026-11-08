"""
SampleStackPlot -- every spectrum of a sample in one multi-panel full-spectrum figure.

Built on :class:`FullSpectrumPlot` (same wavelength panels, N Panels, plot
range, data-density scaling, overlays and theme) but each panel shows all the
spectra of a sample, each with the models of its own parameter set (see
:mod:`iSLAT.Modules.DataProcessing.SampleStack`).

Two layouts:

* ``"offset"`` -- every spectrum in the same panel, shifted up so the
  spectra do not overlap.  The first spectrum of the sample is at the top;
  each is labelled with its name at its zero-flux baseline.
* ``"rows"`` -- each panel is split into one row per spectrum, sharing the
  wavelength axis, each row with its own flux scale.

Three model modes:

* ``"data"`` -- observed spectra only.
* ``"total"`` -- plus the summed model of the visible molecules (filled).
* ``"molecules"`` -- plus every visible molecule's model (dashed).

Usage (notebook)::

    from iSLAT.Modules.DataProcessing.SampleStack import SampleStackBuilder
    spectra = SampleStackBuilder().build(islat)
    plot = SampleStackPlot(spectra, layout="rows", model_mode="molecules", n_panels=4)
    plot.generate_plot()
    plot.show()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from matplotlib import transforms
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.ticker import MaxNLocator

from .FullSpectrumPlot import FullSpectrumPlot
from .StackedSpectralPanel import StackedSpectralPanel
from .SpectralPanel import SpectralPanel, GapMode
from .SpectrumPanel import SpectrumPanel

if TYPE_CHECKING:
    from matplotlib.gridspec import SubplotSpec
    from iSLAT.Modules.DataProcessing.SampleStack import SampleSpectrum

#: Layouts and model modes, in the order the GUI lists them.
STACK_LAYOUTS: Tuple[str, ...] = ("offset", "rows")
STACK_MODEL_MODES: Tuple[str, ...] = ("data", "total", "molecules")

#: Tag of the spectrum-name labels.
STACK_LABEL_TAG = "_islat_stack_label"

#: Default space between offset spectra, as a fraction of the tallest one in the panel.
DEFAULT_STACK_GAP: float = 0.08
#: Smallest slot an offset spectrum gets, as a fraction of the tallest one in the
#: panel, so weak spectra (and their labels) do not crowd together.
MIN_SLOT_FRACTION: float = 0.25


def normalize_stack_layout(value: Any) -> str:
    """``"rows"`` or ``"offset"`` (the default for anything else)."""
    return "rows" if str(value).strip().lower() == "rows" else "offset"


def normalize_stack_model_mode(value: Any) -> str:
    """One of :data:`STACK_MODEL_MODES` (``"total"`` for anything unknown)."""
    mode = str(value).strip().lower()
    return mode if mode in STACK_MODEL_MODES else "total"


@dataclass
class StackModels:
    """The models drawn for one spectrum: per-molecule cache and the summed model."""
    mol_cache: List[tuple] = field(default_factory=list)
    summed_wave: Optional[np.ndarray] = None
    summed_flux: Optional[np.ndarray] = None


def compute_stack_models(spectrum: "SampleSpectrum", model_mode: str) -> StackModels:
    """Compute the models *model_mode* draws for *spectrum*, from its own parameter set."""
    model_mode = normalize_stack_model_mode(model_mode)
    molecules = getattr(spectrum, "molecules", None)
    if model_mode == "data" or molecules is None:
        return StackModels()
    models = StackModels()
    try:
        sw, sf = molecules.get_summed_flux(
            spectrum.wave_obs, visible_only=True, extend_to_global_range=True,
        )
        if sw is not None and sf is not None and len(sw) > 0:
            models.summed_wave = np.asarray(sw, dtype=float)
            models.summed_flux = np.asarray(sf, dtype=float)
    except Exception as exc:
        print(f"[SampleStackPlot] Could not compute the summed model for {spectrum.name}: {exc}")
    if model_mode == "molecules":
        models.mol_cache, _labels, _colors = FullSpectrumPlot.molecule_cache(
            molecules, spectrum.wave, spectrum.wave_obs,
        )
    return models


def _models_key(spectrum: "SampleSpectrum", model_mode: str) -> Tuple[Any, ...]:
    """Spectra sharing a parameter set share their models, unless the models follow each spectrum's sampling."""
    molecules = getattr(spectrum, "molecules", None)
    if molecules is not None and getattr(molecules, "match_spectral_sampling", False):
        return (model_mode, id(molecules), id(spectrum.wave_obs), id(spectrum.wave))
    return (model_mode, id(molecules))


def _primary(spectra: Sequence["SampleSpectrum"]) -> "SampleSpectrum":
    """The active spectrum, or the first one."""
    for spectrum in spectra:
        if getattr(spectrum, "is_active", False):
            return spectrum
    return spectra[0]


def _in_range(x: np.ndarray, xr: Tuple[float, float]) -> np.ndarray:
    return (x >= xr[0]) & (x <= xr[1])


#: Tag of the error-bar line drawn with each stacked spectrum (also tagged ``_islat_observed``).
ERRORBAR_TAG = "_islat_errorbar"


class StackSpectrumPanel(SpectrumPanel):
    """:class:`SpectrumPanel` that draws error bars as a single line.

    ``Axes.errorbar`` builds one path per error bar, which dominates the
    render time once several long spectra are stacked; one NaN-separated
    line looks the same and is many times faster.
    """

    def _plot_observed_spectrum(
        self,
        ax: Axes,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        error_data: Optional[np.ndarray] = None,
        color: Optional[str] = None,
        label: str = "Data",
        deduplicate: bool = False,
    ) -> None:
        from .BasePlot import BasePlot
        if flux_data is None or len(flux_data) == 0:
            return
        if color is None:
            color = self._get_theme_value("foreground", "black")
        if deduplicate:
            BasePlot._clear_tagged_artists(ax, "_islat_observed")
        zorder = self._get_theme_value("zorder_observed", 2)
        wave = np.asarray(wave_data, dtype=float)
        flux = np.asarray(flux_data, dtype=float)
        (line,) = ax.plot(wave, flux, color=color, linewidth=1, label=label, zorder=zorder)
        line._islat_observed = True
        if error_data is not None and len(error_data) == len(flux):
            err = np.asarray(error_data, dtype=float)
            xs = np.empty(3 * len(wave))
            ys = np.empty(3 * len(wave))
            xs[0::3] = wave
            xs[1::3] = wave
            xs[2::3] = np.nan
            ys[0::3] = flux - err
            ys[1::3] = flux + err
            ys[2::3] = np.nan
            (bars,) = ax.plot(xs, ys, color=color, linewidth=0.5, zorder=zorder)
            bars._islat_observed = True
            setattr(bars, ERRORBAR_TAG, True)


# ======================================================================
# Offset layout: one axes per panel, every spectrum on its own offset
# ======================================================================
class SampleOffsetPanel(StackSpectrumPanel):
    """A wavelength panel drawing every sample spectrum on its own vertical offset.

    The spectra are stacked from the last (bottom) to the first (top); each
    takes the height of its own data and models in the panel (at least
    :data:`MIN_SLOT_FRACTION` of the tallest), plus *stack_gap* times the
    tallest one as spacing.  :attr:`offsets`, :attr:`slots` and
    :attr:`stack_ylim` are set by :meth:`generate_plot`.
    """

    def __init__(
        self,
        spectra: Sequence["SampleSpectrum"],
        models: Sequence[StackModels],
        xmin: float,
        xmax: float,
        ax: Optional[Axes] = None,
        stack_gap: float = DEFAULT_STACK_GAP,
        **kwargs,
    ) -> None:
        primary = _primary(spectra)
        super().__init__(
            wave_data=primary.wave, flux_data=primary.flux, xmin=xmin, xmax=xmax,
            error_data=primary.err, molecules=primary.molecules, ax=ax, **kwargs,
        )
        self.spectra = list(spectra)
        self.models = list(models)
        self.stack_gap = float(stack_gap)
        self.offsets: List[float] = []
        #: ``(bottom, top)`` of each spectrum's slot in data coordinates.
        self.slots: List[Tuple[float, float]] = []
        self.stack_ylim: Optional[Tuple[float, float]] = None

    def spectrum_at(self, y: float) -> Optional["SampleSpectrum"]:
        """The spectrum whose slot is nearest to the height *y* (``None`` before rendering)."""
        if not self.slots:
            return None
        distances = [max(lo - y, y - hi, 0.0) for lo, hi in self.slots]
        return self.spectra[int(np.argmin(distances))]

    def _spectrum_data(self, spectrum: "SampleSpectrum"):
        """``(wave, flux, err)`` of *spectrum* in this panel (with gap breaks in ``GapMode.SKIP``)."""
        helper = SpectrumPanel(
            wave_data=spectrum.wave, flux_data=spectrum.flux,
            xmin=self.xmin, xmax=self.xmax, error_data=spectrum.err,
            gap_mode=self.gap_mode, gap_threshold=self.gap_threshold,
        )
        return helper.get_panel_data_with_gaps()

    @staticmethod
    def _extent(flux: np.ndarray, models: StackModels, xr: Tuple[float, float]) -> Tuple[float, float]:
        """``(low, high)`` of the data and drawn models in range; the low end is at most 0."""
        parts = [np.asarray(flux, dtype=float)]
        if models.summed_wave is not None and models.summed_flux is not None:
            parts.append(models.summed_flux[_in_range(models.summed_wave, xr)])
        for m_lam, m_flux, *_rest in models.mol_cache:
            parts.append(np.asarray(m_flux, dtype=float)[_in_range(np.asarray(m_lam), xr)])
        values = np.concatenate([p.ravel() for p in parts]) if parts else np.array([])
        values = values[np.isfinite(values)]
        if values.size == 0:
            return 0.0, 0.0
        return min(0.0, float(values.min())), max(0.0, float(values.max()))

    def compute_offsets(self, data: List[Tuple[np.ndarray, np.ndarray, Any]]) -> None:
        """Set :attr:`offsets` and :attr:`stack_ylim` for the panel data *data* (one per spectrum)."""
        xr = self.xlim
        extents = [self._extent(d[1], m, xr) for d, m in zip(data, self.models)]
        tallest = max([hi - lo for lo, hi in extents] + [0.0])
        gap = self.stack_gap * tallest if tallest > 0 else 0.1
        min_height = MIN_SLOT_FRACTION * tallest
        self.offsets = [0.0] * len(extents)
        self.slots = [(0.0, 0.0)] * len(extents)
        cursor = 0.0
        for i in reversed(range(len(extents))):
            lo, hi = extents[i]
            height = max(hi - lo, min_height)
            self.offsets[i] = cursor - lo
            self.slots[i] = (cursor, cursor + height)
            cursor += height + gap
        top = cursor - gap
        span = top if top > 0 else 1.0
        self.stack_ylim = (-0.02 * span, top + 0.02 * span)

    def generate_plot(self, **kwargs) -> None:
        ax = self._resolve_axes()
        xr = self.xlim
        data = [self._spectrum_data(s) for s in self.spectra]
        self.compute_offsets(data)

        base_lw = kwargs.get("model_linewidth", self._get_theme_value("model_linewidth", 1.2))
        alpha = kwargs.get("model_alpha", self._get_theme_value("model_alpha", 1.0))
        summed_color = self._get_theme_value("summed_spectra_color", "lightgray")
        for i, (spectrum, models, (wave, flux, err)) in enumerate(zip(self.spectra, self.models, data)):
            off = self.offsets[i]
            if len(flux) > 0:
                self._plot_observed_spectrum(ax, wave, np.asarray(flux, dtype=float) + off, err)
            for m_lam, m_flux, m_color, m_label, m_name in models.mol_cache:
                mask = _in_range(np.asarray(m_lam), xr)
                if np.any(mask):
                    (line,) = ax.plot(
                        np.asarray(m_lam)[mask], np.asarray(m_flux)[mask] + off,
                        linestyle="--", color=m_color, alpha=alpha,
                        linewidth=self.get_molecule_linewidth_by_name(m_name, base_lw),
                        label=m_label, zorder=self._get_theme_value("zorder_model", 3),
                    )
                    line._molecule_name = m_name
            if models.summed_wave is not None and models.summed_flux is not None:
                mask = _in_range(models.summed_wave, xr)
                if np.any(mask) and np.any(models.summed_flux[mask] > 0):
                    fill = ax.fill_between(
                        models.summed_wave[mask], off, models.summed_flux[mask] + off,
                        color=summed_color, alpha=1.0, label="Sum",
                        zorder=self._get_theme_value("zorder_summed", 1),
                    )
                    fill._islat_summed = True
            self._draw_label(ax, spectrum, off)
        ax.set_xlim(*xr)

    def _draw_label(self, ax: Axes, spectrum: "SampleSpectrum", baseline: float) -> None:
        fg = self._get_theme_value("foreground", "black")
        txt = ax.text(
            0.004, baseline, spectrum.name,
            transform=transforms.blended_transform_factory(ax.transAxes, ax.transData),
            fontsize=6, color=fg, ha="left", va="bottom",
            fontweight="bold" if getattr(spectrum, "is_active", False) else "normal",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=self._get_theme_value("background", "white"),
                edgecolor="none", alpha=0.6,
            ),
            zorder=100, clip_on=True,
        )
        setattr(txt, STACK_LABEL_TAG, True)


# ======================================================================
# The stacked figure
# ======================================================================
class SampleStackPlot(FullSpectrumPlot):
    """Multi-panel figure stacking every spectrum of a sample.

    Parameters
    ----------
    spectra : sequence of SampleSpectrum
        The spectra, in display order (top to bottom), e.g. from
        :meth:`SampleStackBuilder.build`.  The active one (or the first)
        defines the default wavelength range and the data-density panels.
    layout : {"offset", "rows"}
        Offset spectra in each panel, or one row per spectrum.
    model_mode : {"data", "total", "molecules"}
        Which models are drawn with each spectrum.
    stack_gap : float
        Spacing between offset spectra, as a fraction of the tallest.
    **kwargs
        Forwarded to :class:`FullSpectrumPlot` (``n_panels``, ``xlim_range``,
        ``theme``, ``line_list``, the display options, ...).  In the offset
        layout the Y-limit options (``y_limits``, ``y_data_sync``,
        ``y_autoscale_source``) do not apply: each panel fits its stack.
    """

    def __init__(
        self,
        spectra: Sequence["SampleSpectrum"],
        layout: str = "offset",
        model_mode: str = "total",
        stack_gap: float = DEFAULT_STACK_GAP,
        **kwargs,
    ) -> None:
        spectra = list(spectra)
        if not spectra:
            raise ValueError("SampleStackPlot needs at least one spectrum.")
        primary = _primary(spectra)
        super().__init__(
            wave_data=primary.wave,
            flux_data=primary.flux,
            molecules=primary.molecules,
            error_data=primary.err,
            wave_data_obs=primary.wave_obs,
            **kwargs,
        )
        self.spectra: List["SampleSpectrum"] = spectra
        self.layout: str = "offset"
        self.model_mode: str = "total"
        self.set_options(layout=layout, model_mode=model_mode)
        self.stack_gap: float = float(stack_gap)
        self._models: List[StackModels] = []
        self._axes_spectrum: Dict[Axes, "SampleSpectrum"] = {}

    # ------------------------------------------------------------------
    # Mutators (call generate_plot afterwards)
    # ------------------------------------------------------------------
    def set_spectra(self, spectra: Sequence["SampleSpectrum"]) -> bool:
        """Replace the spectra, keeping the display range.  Returns ``True`` if the panel edges changed."""
        spectra = list(spectra)
        if not spectra:
            raise ValueError("SampleStackPlot needs at least one spectrum.")
        self.spectra = spectra
        primary = self.primary
        return self.update_data(
            wave_data=primary.wave, flux_data=primary.flux, molecules=primary.molecules,
            error_data=primary.err, wave_data_obs=primary.wave_obs,
        )

    def set_options(self, layout: Optional[str] = None, model_mode: Optional[str] = None) -> None:
        """Change the layout and / or model mode."""
        if layout is not None:
            self.layout = normalize_stack_layout(layout)
            self.y_axis_label = (
                "Flux Density (Jy) + offset" if self.layout == "offset" else "Flux Density (Jy)"
            )
        if model_mode is not None:
            self.model_mode = normalize_stack_model_mode(model_mode)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def primary(self) -> "SampleSpectrum":
        """The active spectrum (or the first)."""
        return _primary(self.spectra)

    def spectrum_for_axes(self, ax: Axes) -> Optional["SampleSpectrum"]:
        """The spectrum drawn on row axes *ax* (``"rows"`` layout), else ``None``."""
        return self._axes_spectrum.get(ax)

    def spectrum_at(self, ax: Optional[Axes], y: Optional[float] = None) -> Optional["SampleSpectrum"]:
        """The spectrum shown at height *y* (data coordinates) on *ax*, in either layout."""
        if ax is None:
            return None
        spectrum = self._axes_spectrum.get(ax)
        if spectrum is not None or y is None:
            return spectrum
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if isinstance(panel, SampleOffsetPanel) and panel.ax is ax:
                    return panel.spectrum_at(float(y))
        return None

    def row_axes(self) -> List[Tuple[Axes, "SampleSpectrum"]]:
        """``(axes, spectrum)`` for every row of every panel (``"rows"`` layout)."""
        return list(self._axes_spectrum.items())

    def axes_by_spectrum(self) -> List[Tuple["SampleSpectrum", List[Axes]]]:
        """Every spectrum with its row axes (``"rows"`` layout), in display order."""
        groups: Dict[int, List[Axes]] = {}
        for ax, spectrum in self._axes_spectrum.items():
            groups.setdefault(id(spectrum), []).append(ax)
        return [(s, groups[id(s)]) for s in self.spectra if id(s) in groups]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        self.subplots.clear()
        self._axes_spectrum.clear()
        computed: Dict[Tuple[Any, ...], StackModels] = {}
        self._models = []
        for spectrum in self.spectra:
            key = _models_key(spectrum, self.model_mode)
            if key not in computed:
                computed[key] = compute_stack_models(spectrum, self.model_mode)
            self._models.append(computed[key])
        kwargs.setdefault(
            "model_linewidth", self._get_theme_value("full_spectrum_model_linewidth", 1.0),
        )
        kwargs.setdefault(
            "model_alpha", self._get_theme_value("full_spectrum_model_alpha", 0.8),
        )
        StackedSpectralPanel.generate_plot(self, **kwargs)

        legend_ax = self._legend_axes
        if legend_ax is not None:
            labels: List[str] = []
            colors: List[str] = []
            if self.model_mode == "molecules" and self.primary.molecules is not None:
                visible = self.primary.molecules.get_visible_molecules(return_objects=True)
                labels = [self.get_molecule_display_name(m) for m in visible]
                colors = [self.get_molecule_color(m) for m in visible]
            self.legend_strategy.build_legend(legend_ax, self.fig, labels, colors)

    def update_panels_inplace(self) -> None:
        """Re-render (the stack is always rebuilt: offsets depend on every model)."""
        self.generate_plot()

    def _create_cell(
        self,
        idx: int,
        xmin: float,
        xmax: float,
        gs_slot: "SubplotSpec",
        **kwargs,
    ) -> List[SpectralPanel]:
        common = dict(
            gap_mode=self.gap_mode, gap_threshold=self.gap_threshold,
            x_scaling=self.x_scaling, theme=self.theme,
        )
        if self.layout == "offset":
            ax = self.fig.add_subplot(gs_slot)
            panel = SampleOffsetPanel(
                self.spectra, self._models, xmin, xmax,
                ax=ax, stack_gap=self.stack_gap, **common,
            )
            self.subplots[idx] = ax
            return [panel]

        inner = GridSpecFromSubplotSpec(
            len(self.spectra), 1, subplot_spec=gs_slot, hspace=0.0,
        )
        axes: List[Axes] = []
        panels: List[SpectralPanel] = []
        for k, (spectrum, models) in enumerate(zip(self.spectra, self._models)):
            ax = self.fig.add_subplot(inner[k, 0], sharex=axes[0] if axes else None)
            panels.append(StackSpectrumPanel(
                wave_data=spectrum.wave, flux_data=spectrum.flux,
                xmin=xmin, xmax=xmax, error_data=spectrum.err,
                molecules=spectrum.molecules, mol_cache=models.mol_cache,
                summed_wave=models.summed_wave, summed_flux=models.summed_flux,
                wave_data_obs=spectrum.wave_obs, ax=ax, **common,
            ))
            axes.append(ax)
            self._axes_spectrum[ax] = spectrum
        self.subplots[idx] = tuple(axes)
        return panels

    def _post_render_cell(
        self,
        idx: int,
        cell_panels: List[SpectralPanel],
        is_last: bool,
    ) -> None:
        fg = self._get_theme_value("foreground", "black")
        rows = self.layout == "rows"
        for p_idx, panel in enumerate(cell_panels):
            ax = panel.ax
            if ax is None:
                continue
            self._apply_display_visibility(ax)
            lims = self._stack_panel_ylim(panel)
            if lims is not None:
                ax.set_ylim(*lims)
            ax.tick_params(axis="x", labelsize=7)
            ax.tick_params(axis="y", labelsize=6 if rows else 7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            if rows:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            is_bottom = p_idx == len(cell_panels) - 1
            if not is_bottom:
                ax.tick_params(axis="x", labelbottom=False)
            elif is_last:
                ax.set_xlabel("Wavelength (\u03bcm)", color=fg)
            if rows:
                self._draw_row_label(ax, self._axes_spectrum.get(ax))

        if self.gap_mode is GapMode.SKIP and rows:
            for panel in cell_panels:
                panel.draw_gap_indicators()

        if self.atomic_lines is not None and len(self.atomic_lines) > 0:
            self._plot_cell_atomic_lines(cell_panels, self.atomic_lines, "_islat_atomic_line")
        if self.line_list is not None and len(self.line_list) > 0:
            self._plot_cell_saved_lines(
                cell_panels, self.line_list, "_islat_saved_line", self._saved_line_labels_shown(),
            )

    def _draw_row_label(self, ax: Axes, spectrum: Optional["SampleSpectrum"]) -> None:
        if spectrum is None:
            return
        txt = ax.text(
            0.004, 0.94, spectrum.name,
            transform=ax.transAxes, fontsize=6, ha="left", va="top",
            color=self._get_theme_value("foreground", "black"),
            fontweight="bold" if getattr(spectrum, "is_active", False) else "normal",
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor=self._get_theme_value("background", "white"),
                edgecolor="none", alpha=0.6,
            ),
            zorder=100,
        )
        setattr(txt, STACK_LABEL_TAG, True)

    # ------------------------------------------------------------------
    # Y limits
    # ------------------------------------------------------------------
    def _stack_panel_ylim(self, panel: SpectralPanel) -> Optional[Tuple[float, float]]:
        """Y limits of *panel*: the stack's own range, or the row rule for its spectrum."""
        if isinstance(panel, SampleOffsetPanel):
            return panel.stack_ylim
        spectrum = self._axes_spectrum.get(panel.ax)
        if spectrum is None:
            return None
        return self.row_ylim(panel.ax, spectrum)

    def row_ylim(self, ax: Axes, spectrum: "SampleSpectrum") -> Optional[Tuple[float, float]]:
        """Y limits of the row *ax* showing *spectrum*, under the plot's Y options.

        Manual :attr:`y_limits` win; then :attr:`y_data_sync` uses this
        spectrum's data min / max (whole spectrum, all panels or this
        panel); then the ``"model"`` source fits the drawn model; otherwise
        the spectrum's peak in the panel plus :attr:`ymax_factor`.
        """
        from .PlotView import data_y_range, fit_y_limits_to_data, fit_y_limits_to_model
        if self.y_limits is not None:
            return self.y_limits
        if self.y_data_sync is not None:
            if self.y_data_sync == "global":
                lims = data_y_range(spectrum.wave, spectrum.flux)
            elif self.y_data_sync == "panel":
                lims = data_y_range(spectrum.wave, spectrum.flux, [ax.get_xlim()])
            else:
                ranges = list(zip(self._panel_edges, self._panel_ends))
                lims = data_y_range(spectrum.wave, spectrum.flux, ranges)
            if lims is not None:
                return lims
        if self.y_autoscale_source == "model":
            lims = fit_y_limits_to_model(ax) or fit_y_limits_to_data(ax, spectrum.wave, spectrum.flux)
            if lims is not None:
                return lims
        xr = ax.get_xlim()
        flux = np.asarray(spectrum.flux, dtype=float)[_in_range(np.asarray(spectrum.wave), xr)]
        flux = flux[np.isfinite(flux)]
        if flux.size == 0:
            return (-0.005, 0.1)
        peak = float(flux.max())
        return (-0.005, peak + peak * self.ymax_factor)

    def stack_ylims(self) -> List[Tuple[Axes, Optional[Tuple[float, float]]]]:
        """``(axes, limits)`` for every axes, recomputed with the current options."""
        out: List[Tuple[Axes, Optional[Tuple[float, float]]]] = []
        for cell_panels in self.panels.values():
            for panel in cell_panels:
                if panel.ax is not None:
                    out.append((panel.ax, self._stack_panel_ylim(panel)))
        return out

    # ------------------------------------------------------------------
    # Overlays: labels only on the top row of each panel
    # ------------------------------------------------------------------
    @staticmethod
    def _plot_cell_saved_lines(cell_panels, line_data, tag: str, show_labels: bool) -> None:
        for p_idx, panel in enumerate(cell_panels):
            if hasattr(panel, "plot_saved_lines"):
                panel.plot_saved_lines(line_data, tag=tag, show_labels=show_labels and p_idx == 0)

    @staticmethod
    def _plot_cell_atomic_lines(cell_panels, atomic_df, tag: str) -> None:
        for p_idx, panel in enumerate(cell_panels):
            if p_idx == 0:
                panel.plot_atomic_lines(atomic_df, tag=tag)
                continue
            ax = panel.ax
            if ax is None:
                continue
            xr = ax.get_xlim()
            waves = np.asarray(atomic_df["wave"], dtype=float)
            for wave in waves[_in_range(waves, xr)]:
                line = ax.axvline(wave, linestyle="--", color="tomato", alpha=0.7)
                setattr(line, tag, True)

    def plot_saved_lines(
        self,
        line_data: Any,
        tag: str = "_islat_saved_line",
        show_labels: Optional[bool] = None,
    ) -> None:
        if line_data is None or (hasattr(line_data, "__len__") and len(line_data) == 0):
            return
        if show_labels is None:
            show_labels = self._saved_line_labels_shown()
        for cell_panels in self.panels.values():
            self._plot_cell_saved_lines(cell_panels, line_data, tag, show_labels)

    def plot_atomic_lines(self, atomic_df: Any, tag: str = "_islat_atomic_line") -> None:
        if atomic_df is None or (hasattr(atomic_df, "__len__") and len(atomic_df) == 0):
            return
        for cell_panels in self.panels.values():
            self._plot_cell_atomic_lines(cell_panels, atomic_df, tag)
