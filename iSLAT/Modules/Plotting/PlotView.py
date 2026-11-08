"""
PlotView - abstract interface for switchable plot views in the iSLAT GUI.

The iSLATPlot controller owns one *active_view* at a time.

Every user-facing action (toggle molecule, toggle summed spectrum, toggle legend, …) is forwarded to the active view's implementation, eliminating scattered ``if is_full_spectrum`` checks throughout the codebase.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

from .ArtistTags import OBSERVED_TAG, SUMMED_TAG, iter_molecule_lines, iter_tagged, set_tagged_visible

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

# ----------------------------------------------------------------------
# User wavelength markers (Shift+M)
# ----------------------------------------------------------------------
WAVELENGTH_MARKER_TAG = "_islat_wavelength_marker"
DEFAULT_WAVELENGTH_MARKER_COLOR = "magenta"
#: Label placement relative to the marker: just left of it, centered on it, or just right of it.
WAVELENGTH_MARKER_LABEL_SIDES = ("left", "center", "right")
DEFAULT_WAVELENGTH_MARKER_LABEL_SIDE = "center"
#: Matplotlib line-style names accepted for marker lines (single marks and range edges).
WAVELENGTH_MARKER_LINESTYLES = ("solid", "dashed", "dotted", "dashdot")
DEFAULT_WAVELENGTH_MARKER_LINESTYLE = "dashed"
_LINESTYLE_ALIASES = {"-": "solid", "--": "dashed", ":": "dotted", "-.": "dashdot", "dash-dot": "dashdot"}
#: Gap (points) between a side-placed label and its marker.
_LABEL_OFFSET_PTS = 3.0


def make_wavelength_marker(
    start: float,
    end: Optional[float] = None,
    label: str = "",
    color: Optional[str] = None,
    label_side: Optional[str] = None,
    linestyle: Optional[str] = None,
    label_color: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a validated marker dict.

    Keys: ``start``, ``end``, ``label``, ``color`` (line / band),
    ``label_color`` (defaults to ``color``), ``label_side`` (one of
    :data:`WAVELENGTH_MARKER_LABEL_SIDES`) and ``linestyle`` (one of
    :data:`WAVELENGTH_MARKER_LINESTYLES`; ``"-"``, ``"--"``, ``":"`` and
    ``"-."`` are accepted too).  A single wavelength has ``start == end``;
    reversed ranges are swapped.

    Raises
    ------
    ValueError
        If a bound is not a finite number, or a color, the label side or the
        line style is not recognised.
    """
    import numpy as np
    from matplotlib.colors import is_color_like

    start = float(start)
    end = start if end is None else float(end)
    if not (np.isfinite(start) and np.isfinite(end)):
        raise ValueError("Wavelengths must be finite numbers.")
    if end < start:
        start, end = end, start
    color = (color or "").strip() or DEFAULT_WAVELENGTH_MARKER_COLOR
    if not is_color_like(color):
        raise ValueError(f"'{color}' is not a valid color.")
    label_color = (label_color or "").strip() or color
    if not is_color_like(label_color):
        raise ValueError(f"'{label_color}' is not a valid label color.")
    label_side = (label_side or "").strip().lower() or DEFAULT_WAVELENGTH_MARKER_LABEL_SIDE
    if label_side not in WAVELENGTH_MARKER_LABEL_SIDES:
        raise ValueError(f"Label position must be one of: {', '.join(WAVELENGTH_MARKER_LABEL_SIDES)}.")
    linestyle = (linestyle or "").strip().lower() or DEFAULT_WAVELENGTH_MARKER_LINESTYLE
    linestyle = _LINESTYLE_ALIASES.get(linestyle, linestyle)
    if linestyle not in WAVELENGTH_MARKER_LINESTYLES:
        raise ValueError(f"Line style must be one of: {', '.join(WAVELENGTH_MARKER_LINESTYLES)}.")
    return {
        "start": start, "end": end, "label": str(label or "").strip(), "color": color,
        "label_color": label_color, "label_side": label_side, "linestyle": linestyle,
    }


def clear_wavelength_markers(ax: Any) -> None:
    """Remove every wavelength-marker artist from *ax*."""
    for artist in list(ax.get_children()):
        if hasattr(artist, WAVELENGTH_MARKER_TAG):
            artist.remove()


def draw_wavelength_markers(ax: Any, markers: Any) -> None:
    """Draw *markers* on *ax* as vertical lines (single) or shaded bands (ranges).

    Ranges get a translucent fill plus edge lines in the marker's line
    style.  Artists are collections in x-data / y-axes coordinates added
    with ``autolim=False``, so they span the full height and never change
    the axes' limits (``relim`` ignores collections).
    """
    from matplotlib.collections import LineCollection, PolyCollection

    trans = ax.get_xaxis_transform()
    for marker in markers or ():
        try:
            start, end = float(marker["start"]), float(marker["end"])
        except (KeyError, TypeError, ValueError):
            continue
        color = marker.get("color") or DEFAULT_WAVELENGTH_MARKER_COLOR
        linestyle = marker.get("linestyle") or DEFAULT_WAVELENGTH_MARKER_LINESTYLE
        linestyle = _LINESTYLE_ALIASES.get(linestyle, linestyle)
        if linestyle not in WAVELENGTH_MARKER_LINESTYLES:
            linestyle = DEFAULT_WAVELENGTH_MARKER_LINESTYLE

        artists = []
        if end > start:
            artists.append(PolyCollection(
                [[(start, 0.0), (end, 0.0), (end, 1.0), (start, 1.0)]], transform=trans,
                facecolors=color, edgecolors="none", alpha=0.2, zorder=0.5,
            ))
            artists.append(LineCollection(
                [[(start, 0.0), (start, 1.0)], [(end, 0.0), (end, 1.0)]], transform=trans,
                colors=color, linestyles=linestyle, linewidths=1.0, alpha=0.8, zorder=2.5,
            ))
        else:
            artists.append(LineCollection(
                [[(start, 0.0), (start, 1.0)]], transform=trans,
                colors=color, linestyles=linestyle, linewidths=1.2, zorder=2.5,
            ))
        for artist in artists:
            setattr(artist, WAVELENGTH_MARKER_TAG, True)
            ax.add_collection(artist, autolim=False)

        label = marker.get("label")
        if label:
            side = marker.get("label_side") or DEFAULT_WAVELENGTH_MARKER_LABEL_SIDE
            if side == "left":
                x, ha, dx = start, "right", -_LABEL_OFFSET_PTS
            elif side == "right":
                x, ha, dx = end, "left", _LABEL_OFFSET_PTS
            else:
                x, ha, dx = (start + end) / 2.0, "center", 0.0
            text = ax.annotate(
                label, xy=(x, 0.98), xycoords=trans,
                xytext=(dx, 0.0), textcoords="offset points",
                ha=ha, va="top", fontsize=8, color=marker.get("label_color") or color,
                clip_on=True, zorder=3,
            )
            setattr(text, WAVELENGTH_MARKER_TAG, True)


class WavelengthMarkerStoreMixin:
    """Wavelength-marker API for an object holding them in ``self.toggle_state``.

    The host provides ``toggle_state`` (a dict) and
    ``refresh_wavelength_markers()`` (redraw wherever the markers are shown).
    Used by the main plot controller and the Display Full Spectrum window, and
    by the Mark Wavelengths dialog as its *marker store*.
    """

    @property
    def wavelength_markers(self) -> list:
        """The user's wavelength markers (``{"start", "end", "label", "color"}`` dicts)."""
        return self.toggle_state.setdefault("wavelength_markers", [])

    def add_wavelength_marker(
        self,
        start: float,
        end: Optional[float] = None,
        label: str = "",
        color: Optional[str] = None,
        label_side: Optional[str] = None,
        linestyle: Optional[str] = None,
        label_color: Optional[str] = None,
    ) -> dict:
        """Mark a wavelength (*end* omitted / equal) or a range on every wavelength plot.

        *label_side* is ``"left"``, ``"center"`` or ``"right"`` of the marker;
        *linestyle* is ``"solid"``, ``"dashed"``, ``"dotted"`` or ``"dashdot"``;
        *label_color* defaults to *color*.
        Raises ``ValueError`` for non-finite bounds or an invalid option.
        """
        marker = make_wavelength_marker(start, end, label, color, label_side, linestyle, label_color)
        self.wavelength_markers.append(marker)
        self.refresh_wavelength_markers()
        return marker

    def remove_wavelength_marker(self, index: int) -> None:
        """Remove the marker at *index* (ignored when out of range)."""
        markers = self.wavelength_markers
        if 0 <= index < len(markers):
            del markers[index]
            self.refresh_wavelength_markers()

    def clear_wavelength_markers(self) -> None:
        """Remove every wavelength marker."""
        self.toggle_state["wavelength_markers"] = []
        self.refresh_wavelength_markers()

    def refresh_wavelength_markers(self) -> None:  # pragma: no cover - hosts override
        """Redraw the markers wherever they are shown."""


def fit_y_limits_to_data(ax: Any, wave: Any, flux: Any) -> Optional[Tuple[float, float]]:
    """Return ``(ymin, ymax)`` fitting *flux* inside *ax*'s current x-range.

    This is the historical main-plot auto-scale rule: the minimum and
    maximum observed flux in view, with one-eighth of the peak as headroom.
    When nothing falls in range the whole array's peak is used with a zero
    floor.  Returns ``None`` when there is no usable flux.
    """
    import numpy as np
    if wave is None or flux is None or len(flux) == 0:
        return None
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    wmin, wmax = ax.get_xlim()
    in_range = flux[(wave >= wmin) & (wave <= wmax)]
    if in_range.size == 0:
        top = np.nanmax(flux)
        bottom = 0.0
    else:
        top = np.nanmax(in_range)
        bottom = np.nanmin(in_range)
    if not np.isfinite(top) or not np.isfinite(bottom):
        return None
    return (float(bottom), float(top + top / 8))


def fit_y_limits_to_model(ax: Any) -> Optional[Tuple[float, float]]:
    """Return ``(ymin, ymax)`` fitting the *drawn model* inside *ax*'s x-range.

    Looks at the visible molecule lines (tagged ``_molecule_name``) and the
    summed-model fill (tagged ``_islat_summed``) already on the axes, so it
    reflects exactly what is plotted.  Headroom is one-eighth of the peak,
    as for the data rule.  Returns ``None`` when no model is drawn in range.
    """
    import numpy as np
    wmin, wmax = ax.get_xlim()
    top = -np.inf
    bottom = np.inf
    for line in iter_molecule_lines(ax):
        if not line.get_visible():
            continue
        x, y = line.get_data()
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        sel = (x >= wmin) & (x <= wmax) & np.isfinite(y)
        if np.any(sel):
            top = max(top, float(y[sel].max()))
            bottom = min(bottom, float(y[sel].min()))
    for coll in iter_tagged(ax, SUMMED_TAG, lines=False):
        if not coll.get_visible():
            continue
        for path in coll.get_paths():
            v = np.asarray(path.vertices, dtype=float)
            if v.size == 0:
                continue
            sel = (v[:, 0] >= wmin) & (v[:, 0] <= wmax) & np.isfinite(v[:, 1])
            if np.any(sel):
                top = max(top, float(v[sel, 1].max()))
                bottom = min(bottom, float(v[sel, 1].min()))
    if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
        return None
    return (bottom, top + top / 8)


# ----------------------------------------------------------------------
# Y axis synced to the observed data's min / max
# ----------------------------------------------------------------------
#: Scopes for ``toggle_state["y_data_sync"]``: the whole spectrum, the
#: active view's visible wavelength range (one range shared by all of its
#: panels), or each panel's own wavelength range.
Y_DATA_SYNC_SCOPES: Tuple[str, ...] = ("global", "view", "panel")


def normalize_y_data_sync(value: Any) -> Optional[str]:
    """Return *value* as a Y data-sync scope, or ``None`` when syncing is off.

    Accepts the scopes in :data:`Y_DATA_SYNC_SCOPES` in any case; anything
    else (``None``, ``"off"``, ...) means off.
    """
    if value is None:
        return None
    scope = str(value).strip().lower()
    return scope if scope in Y_DATA_SYNC_SCOPES else None


def data_y_range(
    wave: Any,
    flux: Any,
    x_ranges: Optional[Any] = None,
) -> Optional[Tuple[float, float]]:
    """Return the exact ``(min, max)`` of the finite *flux* values.

    With *x_ranges* (a sequence of ``(start, end)`` wavelength pairs) only
    the points whose wavelength lies in at least one range count.  Unlike
    the auto-scale rules no headroom is added; a flat selection is widened
    slightly, since an axis cannot show a zero-height range.  Returns
    ``None`` when no finite flux is selected.
    """
    import numpy as np
    from matplotlib.transforms import nonsingular
    if flux is None:
        return None
    flux = np.asarray(flux, dtype=float).ravel()
    keep = np.isfinite(flux)
    if x_ranges is not None:
        if wave is None:
            return None
        wave = np.asarray(wave, dtype=float).ravel()
        if wave.shape != flux.shape:
            return None
        in_range = np.zeros(flux.shape, dtype=bool)
        for start, end in x_ranges:
            lo, hi = sorted((float(start), float(end)))
            in_range |= (wave >= lo) & (wave <= hi)
        keep &= in_range
    if not np.any(keep):
        return None
    lo, hi = nonsingular(float(flux[keep].min()), float(flux[keep].max()), expander=0.05)
    return (float(lo), float(hi))


def data_synced_y_limits(
    axes: Any,
    wave: Any,
    flux: Any,
    scope: Any,
) -> List[Optional[Tuple[float, float]]]:
    """Return Y limits that sync each of *axes* to the observed data's min / max.

    * ``"global"``: every axes gets the min / max of all the data.
    * ``"view"``: every axes gets the min / max of the data inside any of
      the axes' x-ranges (one shared range for the whole view).
    * ``"panel"``: each axes gets the min / max inside its own x-range.

    One entry per axes, ``None`` where there is no data (or the scope is off).
    """
    axes = list(axes)
    scope = normalize_y_data_sync(scope)
    if not axes or scope is None:
        return [None] * len(axes)
    if scope == "panel":
        return [data_y_range(wave, flux, [ax.get_xlim()]) for ax in axes]
    x_ranges = None if scope == "global" else [ax.get_xlim() for ax in axes]
    return [data_y_range(wave, flux, x_ranges)] * len(axes)


class PlotView(ABC):
    """
    Abstract base for a swappable plot view inside the main iSLAT window.

    Each view owns a matplotlib *Figure* and a Tk *FigureCanvasTkAgg*.
    The controller (:class:`iSLATPlot`) calls these methods without knowing which concrete view is active.

    Subclasses must implement the lifecycle (:meth:`activate`,
    :meth:`deactivate`), rendering (:meth:`update_model_plot`), molecule
    callbacks, theme and canvas methods.  Selection and overlay-toggle
    methods default to no-ops, so views without those features need not
    override them.
    """

    #: The Tk canvas showing this view (``None`` until built).  Views that
    #: own their canvas assign it; :class:`ThreePanelView` overrides it with a
    #: property returning the controller's canvas.
    _canvas: Optional["FigureCanvasTkAgg"] = None

    def __init__(self, plot_manager: Any) -> None:
        """
        Parameters
        ----------
        plot_manager : iSLATPlot
            The controller that owns this view (or a stand-in providing
            ``islat``, ``theme``, ``toggle_state`` and ``control_bus``).
        """
        self._pm = plot_manager
        self._parent_frame: Any = None  # Tk frame the canvas is packed into
        self._initialised: bool = False  # True once the content has been built
        self._needs_refresh: bool = True  # Data changed since the last render

    @property
    def _islat(self) -> Any:
        """The top-level iSLAT object, read from the controller."""
        return self._pm.islat

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def activate(self, parent_frame: Any) -> None:
        """
        Make this view the visible one.

        Pack / display the view's canvas inside *parent_frame* and ensure the rendered content is up-to-date.
        """
        ...

    @abstractmethod
    def deactivate(self) -> None:
        """
        Hide this view (pack_forget the canvas).

        The view should **not** destroy its resources - it may be reactivated later.
        """
        ...

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    @abstractmethod
    def update_model_plot(
        self,
        wave_data: Any,
        flux_data: Any,
        molecules_dict: "MoleculeDict",
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Full re-render of the model spectrum (observed + molecules + sum).

        Called when a new spectrum is loaded, molecule parameters change globally, or the user switches into this view.
        """
        ...

    @abstractmethod
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
        """
        Lightweight update after a single molecule's visibility is toggled.

        Implementations should **not** reload data from disk - only update the artists that changed.

        Parameters
        ----------
        force_rerender : bool
            When *True* the molecule's artists must be re-rendered from current parameters (e.g. because parameters changed while the molecule was hidden).
            Implementations should remove the stale artists and create fresh ones instead of just toggling visibility.
        """
        ...

    # ------------------------------------------------------------------
    # Selection & line-inspection (no-op defaults)
    # ------------------------------------------------------------------
    def on_selection(self, xmin: float, xmax: float) -> None:
        """Handle a wavelength-range selection (span selector drag).

        The view should render the line inspection panel, populate the population diagram scatter points, and highlight the strongest line.
        The default is a no-op for views that do not support interactive selection (e.g. :class:`FullSpectrumView`).
        """

    def clear_selection(self) -> None:
        """Clear the current line-inspection selection and reset panels.  Default: no-op."""

    def clear_active_lines(self) -> None:
        """Remove all active-line artists (vlines, text, scatter) from the view.  Default: no-op."""

    # ------------------------------------------------------------------
    # Molecule lifecycle callbacks
    # ------------------------------------------------------------------
    @abstractmethod
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """The user selected a different active molecule.

        If a selection is active the view should re-run the line inspection / population diagram for the new molecule.
        Otherwise only the population-diagram title and base diagram need updating.
        """
        ...

    @abstractmethod
    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """A single molecule's parameter changed (not visibility).

        The view should update the spectrum if the molecule is visible, and refresh the line-inspection / population diagram if the molecule is the active one.
        """
        ...

    @abstractmethod
    def on_molecule_deleted(self, molecule_name: str) -> None:
        """A molecule was removed from the dict - update all panels."""
        ...

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    @abstractmethod
    def apply_theme(self, theme: dict) -> None:
        """
        Apply a new theme dictionary to this view.

        Implementations must propagate the theme to every owned figure, axes, canvas widget, and sub-plot delegate so that switching between views always reflects the current theme.

        Called by the controller's :meth:`iSLATPlot.apply_theme` and automatically on :meth:`activate` when the theme has changed since the view was last visible.
        """
        ...

    # ------------------------------------------------------------------
    # Toggle helpers (no-op defaults)
    # ------------------------------------------------------------------
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """
        Reconcile the view's visual state with the canonical *toggle_state* dict from the controller.

        Called by the controller when this view is **activated** so that
        overlays (atomic lines, saved lines, summed spectrum, legend) match
        the state the user set while a different view was visible.
        The default is a no-op for views without such overlays.
        """

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill across all relevant axes.  Default: no-op."""

    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Toggle the legend visibility on all relevant axes.  Default: no-op."""

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """Add or remove saved line annotations.  Default: no-op."""

    def toggle_atomic_lines(self, show: bool) -> None:
        """Add or remove atomic line annotations.  Default: no-op."""

    # ------------------------------------------------------------------
    # Observed-data visibility and Y-axis policy (shared implementation)
    # ------------------------------------------------------------------
    # Views opt in by yielding axes from the two iterator hooks below.
    # :class:`ToggleMixin` wires both to ``_iter_toggle_axes`` so every
    # toggle-aware view gets the behaviour; views without a spectrum keep
    # the empty defaults and the public methods become no-ops.

    def _iter_observed_axes(self):
        """Yield axes that carry observed-spectrum artists (``_islat_observed``)."""
        return iter(())

    def _iter_y_axes(self):
        """Yield axes governed by the Y auto-scale / manual-limit policy."""
        return iter(())

    def _set_observed_visibility(self, visible: bool, axes=None) -> bool:
        """Show / hide every observed-spectrum artist on *axes*.

        *axes* defaults to :meth:`_iter_observed_axes`.  Returns ``True``
        when at least one artist was found.
        """
        found = False
        for ax in (axes if axes is not None else self._iter_observed_axes()):
            if set_tagged_visible(ax, OBSERVED_TAG, visible):
                found = True
        return found

    def toggle_observed_data(self, visible: bool) -> None:
        """Show or hide the observed spectrum on this view's observed axes."""
        if self._set_observed_visibility(visible):
            self.draw()

    def _y_autoscale_source(self) -> str:
        """``"data"`` (default) or ``"model"``: what the Y auto-scale fits to."""
        pm = getattr(self, "_pm", None)
        state = getattr(pm, "toggle_state", None)
        if isinstance(state, dict) and state.get("y_autoscale_source") == "model":
            return "model"
        return "data"

    def _autoscaled_y_limits_for(self, ax: Any) -> Optional[Tuple[float, float]]:
        """Default auto-scale rule for *ax*.

        Fits the observed flux inside the x-range, or the drawn model when
        the auto-scale source is ``"model"`` (falling back to the data when
        no model is drawn there).
        """
        if self._y_autoscale_source() == "model":
            lims = fit_y_limits_to_model(ax)
            if lims is not None:
                return lims
        islat = getattr(self, "_islat", None)
        return fit_y_limits_to_data(
            ax,
            getattr(islat, "wave_data", None),
            getattr(islat, "flux_data", None),
        )

    def _autoscale_y_axes(self) -> bool:
        """Re-fit every Y axis.  Views with their own layout rules override this."""
        changed = False
        for ax in self._iter_y_axes():
            lims = self._autoscaled_y_limits_for(ax)
            if lims is not None:
                ax.set_ylim(*lims)
                changed = True
        return changed

    # --- Y axis synced to the observed data's min / max -------------------
    def _y_data_sync_scope(self, toggle_state: Optional[dict] = None) -> Optional[str]:
        """The ``y_data_sync`` scope of *toggle_state* (default: the controller's), or ``None``."""
        if not isinstance(toggle_state, dict):
            toggle_state = getattr(getattr(self, "_pm", None), "toggle_state", None)
        if not isinstance(toggle_state, dict):
            return None
        return normalize_y_data_sync(toggle_state.get("y_data_sync"))

    def _y_data_arrays(self) -> Tuple[Any, Any]:
        """``(wave, flux)`` of the observed spectrum in the Y axes' wavelength coordinates."""
        islat = getattr(self, "_islat", None)
        return getattr(islat, "wave_data", None), getattr(islat, "flux_data", None)

    def _iter_y_sync_only_axes(self):
        """Yield extra axes that follow the data min / max sync only.

        Unlike :meth:`_iter_y_axes` they ignore manual limits and the normal
        auto-scale rule, and they do not widen the view's range: for
        ``"global"`` and ``"view"`` they take the same limits as the view's
        Y axes, for ``"panel"`` the data in their own x-range.  Default: none.
        """
        return iter(())

    def _data_synced_y_limits_by_axes(self, scope: str) -> List[Tuple[Any, Optional[Tuple[float, float]]]]:
        """``(axes, limits)`` for every Y axis and sync-only axis under *scope* (``None`` = no data)."""
        axes = list(self._iter_y_axes())
        followers = list(self._iter_y_sync_only_axes())
        wave, flux = self._y_data_arrays()
        limits = data_synced_y_limits(axes, wave, flux, scope)
        if followers:
            if scope == "panel" or not axes:
                limits += data_synced_y_limits(followers, wave, flux, scope)
            else:
                limits += [limits[0]] * len(followers)   # the view's shared range
        return list(zip(axes + followers, limits))

    def _apply_data_synced_y_limits(self, scope: str) -> bool:
        """Set every Y axis (and sync-only axis) to the observed data's min / max for *scope*.

        Axes without data in their range keep their limits.  Returns
        ``True`` when any limits were set.
        """
        changed = False
        for ax, lims in self._data_synced_y_limits_by_axes(scope):
            if lims is not None:
                ax.set_ylim(*lims)
                changed = True
        return changed

    def y_axis_count(self) -> int:
        """Number of Y axes (panels) the data min / max sync applies to."""
        return (sum(1 for _ax in self._iter_y_axes())
                + sum(1 for _ax in self._iter_y_sync_only_axes()))

    def get_data_y_range(self, scope: str = "global") -> Optional[Tuple[float, float]]:
        """The observed data's ``(min, max)`` for *scope*, as one pair of limits.

        ``"global"`` covers the whole spectrum and ``"view"`` the visible
        range of every Y axis.  ``"panel"`` has one range per panel, so it
        only gives a single pair when the view has exactly one panel
        (see :meth:`y_axis_count`).  Returns ``None`` for views without Y
        axes, without data, or for an unknown scope.
        """
        scope = normalize_y_data_sync(scope)
        axes = list(self._iter_y_axes())
        if scope is None or not axes or (scope == "panel" and self.y_axis_count() != 1):
            return None
        wave, flux = self._y_data_arrays()
        if scope == "global":
            return data_y_range(wave, flux)
        return data_y_range(wave, flux, [ax.get_xlim() for ax in axes])

    def apply_y_limits(self, toggle_state: dict, autoscale_now: bool = True) -> bool:
        """Apply the controller's Y-axis policy to this view's Y axes.

        * ``y_autoscale`` on and *autoscale_now*: re-fit the axes - to the
          observed data's min / max when ``y_data_sync`` is set, otherwise
          with the view's auto-scale rule.
        * ``y_autoscale`` on but *autoscale_now* False: leave them alone.
        * ``y_autoscale`` off: pin every Y axis to ``y_limits``.

        Returns ``True`` when any limits were set.
        """
        if not toggle_state.get("y_autoscale", True):
            manual = toggle_state.get("y_limits")
            if manual is None:
                return False
            changed = False
            for ax in self._iter_y_axes():
                ax.set_ylim(*manual)
                changed = True
            return changed
        if not autoscale_now:
            return False
        scope = self._y_data_sync_scope(toggle_state)
        if scope is not None:
            return self._apply_data_synced_y_limits(scope)
        return self._autoscale_y_axes()

    def get_y_limits(self) -> Optional[Tuple[float, float]]:
        """Return ``(ymin, ymax)`` of the first Y axis, or ``None`` if the view has none."""
        for ax in self._iter_y_axes():
            lo, hi = ax.get_ylim()
            return (float(lo), float(hi))
        return None

    def _reapply_display_state(self, toggle_state: dict) -> None:
        """Re-impose observed visibility and manual / data-synced Y limits after a re-render.

        Re-rendering recreates observed artists (visible) and lets the plot
        pick its own Y limits, so these settings must be put back.  Call it
        at the end of every re-render path and from ``sync_toggle_state``.
        """
        if not isinstance(toggle_state, dict):
            return
        self._set_observed_visibility(toggle_state.get("observed", True))
        self.apply_y_limits(toggle_state, autoscale_now=False)
        if toggle_state.get("y_autoscale", True):
            scope = self._y_data_sync_scope(toggle_state)
            if scope is not None:
                self._apply_data_synced_y_limits(scope)
        self._apply_wavelength_markers(toggle_state)

    # ------------------------------------------------------------------
    # User wavelength markers
    # ------------------------------------------------------------------
    def _iter_marker_axes(self):
        """Yield wavelength axes that show user markers (defaults to the observed axes)."""
        return self._iter_observed_axes()

    def _apply_wavelength_markers(self, toggle_state: Optional[dict]) -> None:
        """Redraw ``toggle_state['wavelength_markers']`` on every marker axis."""
        markers = []
        if isinstance(toggle_state, dict):
            markers = toggle_state.get("wavelength_markers") or []
        for ax in self._iter_marker_axes():
            clear_wavelength_markers(ax)
            draw_wavelength_markers(ax, markers)

    def refresh_wavelength_markers(self) -> None:
        """Redraw the controller's wavelength markers on this view and flush the canvas."""
        state = getattr(getattr(self, "_pm", None), "toggle_state", None)
        self._apply_wavelength_markers(state)
        self.draw()

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------
    def save_figure(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        dpi: int | None = None,
        rasterized: bool = False,
        **kwargs,
    ) -> str | None:
        """
        Save the current view's figure to a file.

        The default implementation saves the figure returned by
        :meth:`get_figure`.
        Subclasses may override this to produce a *different* figure for export (e.g. with toggle state baked in).

        Parameters
        ----------
        save_path : str or None
            Destination path.  If *None* a file dialog is opened.
        file_format : str
            File format extension (``"pdf"``, ``"png"``, …).
        dpi : int or None
            Resolution.  *None* uses matplotlib's default.
        rasterized : bool
            If *True* axes are rasterized before saving (useful for PDFs
            with very dense data).
        **kwargs
            Extra keyword arguments forwarded to ``fig.savefig()``.

        Returns
        -------
        str or None
            The path that was saved to, or *None* if the user cancelled.
        """
        from pathlib import Path
        from tkinter import filedialog

        fig = self.get_figure()
        if fig is None:
            return None

        if save_path is None:
            save_path = filedialog.asksaveasfilename(
                title="Save Figure",
                defaultextension=f".{file_format}",
                filetypes=[(f"{file_format.upper()} files", f"*.{file_format}")],
            )
        if not save_path:
            return None

        if rasterized:
            for ax in fig.axes:
                ax.set_rasterized(True)

        save_kw = {"bbox_inches": "tight", "format": file_format}
        if dpi is not None:
            save_kw["dpi"] = dpi
        elif rasterized:
            save_kw["dpi"] = 300
        else:
            save_kw["dpi"] = 300  # high-quality default for all exports
        save_kw.update(kwargs)

        fig.savefig(save_path, **save_kw)
        return save_path

    # ------------------------------------------------------------------
    # Line-inspection helpers
    # ------------------------------------------------------------------
    def get_selected_line(self) -> Optional[Dict[str, Any]]:
        """Return the currently selected line info dict, or *None*.

        Views that track a selected-line (e.g. :class:`ThreePanelView`)
        should override this.  The default returns *None*.
        """
        return None

    # ------------------------------------------------------------------
    # Interaction context
    # ------------------------------------------------------------------
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Optional[Any]:
        """Build and return a ``tk.Menu`` for a right-click at *event*.

        The default creates a minimal menu containing only "Save Figure".
        Views that provide additional items should create their own menu,
        append items, and call :meth:`_append_save_figure_item` before
        returning.

        Parameters
        ----------
        event :
            The matplotlib ``MouseEvent`` that triggered the right-click.
        canvas_widget :
            The Tk widget backing the active canvas (used to anchor the menu
            and compute screen coordinates).
        """
        try:
            import tkinter as tk
        except ImportError:
            return None
        menu = tk.Menu(canvas_widget, tearoff=0)
        self._append_save_figure_item(menu)
        return menu

    def _append_save_figure_item(self, menu: Any) -> None:
        """Append a ``Save Figure…`` command to *menu*.

        Calls :meth:`save_figure` with a file-dialog so the user can choose
        the destination and format.
        """
        menu.add_separator()
        menu.add_command(label="Save Figure…", command=self.save_figure)

    def _register_control_fields(self) -> None:
        """Register this view's :class:`ControlField` objects on the :class:`ControlBus`.

        Called from :meth:`activate`.  The default is a no-op.  Views that
        expose view-specific controls (e.g. :class:`FullSpectrumView`'s
        n-panels entry) should override this and call
        ``bus.unregister_owner(self)`` in :meth:`deactivate`.
        """

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    def _notify_canvas_changed(self) -> None:
        """Tell the plot manager this view replaced its canvas or redrew its content.

        Lets the main navigation toolbar follow the new canvas and drop
        view history that no longer applies.
        """
        notify = getattr(getattr(self, "_pm", None), "on_view_canvas_changed", None)
        if callable(notify):
            notify(self)

    @abstractmethod
    def draw(self) -> None:
        """
        Flush all pending changes to the screen (``canvas.draw_idle()``).
        """
        ...

    @abstractmethod
    def get_canvas(self) -> "FigureCanvasTkAgg":
        """Return the Tk canvas widget for this view."""
        ...

    @abstractmethod
    def get_figure(self) -> "Figure":
        """Return the matplotlib Figure for this view."""
        ...