"""
SpectralPanel -- Abstract base class for a single panel of spectral data.

Represents a rectangular axes view of observed spectral data within a
bounded wavelength range ``[xmin, xmax]``.  Concrete subclasses (e.g.
:class:`LineInspectionPlot`) implement the actual rendering logic, but
every spectral panel shares the same fundamental contract:

* Store observed *wave_data* / *flux_data* (and optional *error_data*).
* Define a wavelength window via *xmin* / *xmax*.
* Be embeddable in a larger figure via an external *ax* parameter.
* Provide a :meth:`generate_plot` that draws content onto its axes.

The :class:`StackedSpectralPanel` composer creates and manages many
``SpectralPanel`` instances to build multi-panel stacked figures.
"""
from abc import abstractmethod
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List, TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure

from .BasePlot import BasePlot

class GapMode(Enum):
    """Strategy for handling gaps in observed wavelength/flux data.

    Attributes
    ----------
    CONNECT : str
        Default matplotlib behaviour -- all data points are connected
        by a continuous line, even across large wavelength jumps or
        NaN regions.
    SKIP : str
        Gaps are detected automatically (via NaN flux values or large
        wavelength jumps) and the line is broken.  A visual indicator
        (hatched shading) is drawn over each gap region so the viewer
        can see that data is missing.
    """
    CONNECT = "connect"
    SKIP = "skip"

class XScaling(Enum):
    """Strategy for distributing wavelength ranges across stacked panels.

    Controls how :class:`StackedSpectralPanel` computes the panel edges
    that divide the full wavelength range into rows.

    Attributes
    ----------
    WAVELENGTH : str
        Each panel covers an equal wavelength width
        ``(wave_max - wave_min) / n_panels``.  This is the default
        (current) behaviour.  Panels in data-sparse regions will
        appear mostly empty, while dense regions may be crowded.
    DATA_DENSITY : str
        Each panel contains approximately the same number of observed
        data points.  Panels in densely-sampled spectral regions cover
        a narrower wavelength range, and sparse regions cover a wider
        range.  This keeps the horizontal point-to-point spacing
        roughly constant across all panels.
    """
    WAVELENGTH = "wavelength"
    DATA_DENSITY = "data_density"

if TYPE_CHECKING:
    import pandas as pd
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class SpectralPanel(BasePlot):
    """
    Abstract base for a single spectral-data panel.

    Subclasses must implement :meth:`generate_plot` (inherited from
    :class:`BasePlot`) and optionally override :meth:`compute_ylim` to
    customise how vertical limits are determined.

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array matching *wave_data*.
    xmin : float
        Left wavelength bound of the panel.
    xmax : float
        Right wavelength bound of the panel.
    error_data : np.ndarray, optional
        Flux uncertainties matching *wave_data*.
    molecules : MoleculeDict, optional
        Molecule collection -- visible molecules are plotted.
    figsize : tuple, optional
        Figure size.  Defaults to ``(10, 4)``.
    ax : Axes, optional
        Pre-existing axes for embedding in a larger figure.  When
        provided the panel does **not** create its own figure.
    **kwargs
        Forwarded to :class:`BasePlot`.
    """
    #: Attribute tag used for gap-indicator artists so they can be
    #: selectively removed with :meth:`BasePlot._clear_tagged_artists`.
    _GAP_TAG = "_islat_gap_indicator"

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        error_data: Optional[np.ndarray] = None,
        molecules: Optional["MoleculeDict"] = None,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        gap_mode: GapMode | str = GapMode.CONNECT,
        gap_threshold: Optional[float] = None,
        x_scaling: "XScaling | str" = "wavelength",
        **kwargs,
    ):
        super().__init__(figsize=figsize or (10, 4), **kwargs)
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.xmin = xmin
        self.xmax = xmax
        self.error_data = (
            np.asarray(error_data) if error_data is not None else None
        )
        self.molecules = molecules
        self._external_ax: Optional[Axes] = ax
        self._ax: Optional[Axes] = None

        # Gap handling
        if isinstance(gap_mode, str):
            gap_mode = GapMode(gap_mode)
        self.gap_mode: GapMode = gap_mode
        self.gap_threshold: Optional[float] = gap_threshold

        # Horizontal-axis scaling (stored for informational use; the
        # actual panel-edge logic lives in StackedSpectralPanel).
        if isinstance(x_scaling, str):
            x_scaling = XScaling(x_scaling)
        self.x_scaling: XScaling = x_scaling

    # ------------------------------------------------------------------
    # Panel range property
    # ------------------------------------------------------------------
    @property
    def xlim(self) -> Tuple[float, float]:
        """Wavelength limits of this panel."""
        return (self.xmin, self.xmax)

    # ------------------------------------------------------------------
    # Axes access
    # ------------------------------------------------------------------
    @property
    def ax(self) -> Optional[Axes]:
        """The matplotlib Axes used by this panel (may be *None* before
        :meth:`generate_plot` is called)."""
        return self._ax

    # ------------------------------------------------------------------
    # Masking helper
    # ------------------------------------------------------------------
    def get_panel_mask(self) -> np.ndarray:
        """Boolean mask selecting data points inside ``[xmin, xmax]``."""
        return (self.wave_data >= self.xmin) & (self.wave_data <= self.xmax)

    def get_panel_data(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Return ``(wave, flux, error)`` for points inside the panel range.

        *error* is ``None`` when :attr:`error_data` was not provided.
        """
        mask = self.get_panel_mask()
        err = self.error_data[mask] if self.error_data is not None else None
        return self.wave_data[mask], self.flux_data[mask], err

    # ------------------------------------------------------------------
    # Default y-limit computation (overridable)
    # ------------------------------------------------------------------
    def compute_ylim(
        self,
        ymax_factor: float = 0.2,
    ) -> Tuple[float, float]:
        """Compute ``(ymin, ymax)`` for this panel.

        The default implementation examines the observed flux within the
        panel range and adds *ymax_factor* headroom.  Subclasses may
        override this to incorporate model data, residuals, etc.

        Parameters
        ----------
        ymax_factor : float
            Fractional padding above the peak flux (0.2 = 20 %).

        Returns
        -------
        tuple[float, float]
        """
        mask = self.get_panel_mask()
        if np.any(mask):
            finite = np.isfinite(self.flux_data[mask])
            if np.any(finite):
                peak = float(np.nanmax(self.flux_data[mask][finite]))
                return (-0.005, peak + peak * ymax_factor)
        return (-0.005, 0.1)

    # ------------------------------------------------------------------
    # Gap detection and handling
    # ------------------------------------------------------------------
    def detect_gaps(
        self,
        wave: Optional[np.ndarray] = None,
        flux: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Detect gaps in the wavelength/flux data.

        A gap is defined as:
        * A run of NaN values in the flux array, or
        * A wavelength step larger than *threshold* times the median
          step size.

        Parameters
        ----------
        wave, flux : np.ndarray, optional
            Arrays to scan.  Defaults to the panel-masked data.
        threshold : float, optional
            Multiplier on the median step for detecting jumps.
            Defaults to :attr:`gap_threshold`, or ``3.0`` when that is
            *None*.

        Returns
        -------
        list[tuple[float, float]]
            ``(gap_start_wave, gap_end_wave)`` for every detected gap.
        """
        if wave is None or flux is None:
            mask = self.get_panel_mask()
            wave = self.wave_data[mask]
            flux = self.flux_data[mask]
        if len(wave) < 2:
            return []

        if threshold is None:
            threshold = self.gap_threshold if self.gap_threshold is not None else 3.0

        gaps: List[Tuple[float, float]] = []

        # 1. NaN runs
        is_nan = np.isnan(flux)
        if np.any(is_nan):
            # Find contiguous NaN blocks
            diff = np.diff(is_nan.astype(int))
            starts = np.where(diff == 1)[0] + 1   # index of first NaN
            ends = np.where(diff == -1)[0] + 1     # index after last NaN
            # Handle leading / trailing NaN
            if is_nan[0]:
                starts = np.concatenate(([0], starts))
            if is_nan[-1]:
                ends = np.concatenate((ends, [len(is_nan)]))
            for s, e in zip(starts, ends):
                w0 = float(wave[max(s - 1, 0)])
                w1 = float(wave[min(e, len(wave) - 1)])
                gaps.append((w0, w1))

        # 2. Large wavelength jumps
        dw = np.diff(wave)
        median_dw = float(np.nanmedian(dw)) if len(dw) > 0 else 0.0
        if median_dw > 0:
            # The step-based threshold must be at least 5 % of the
            # *panel* span so that small irregular spacing in an
            # otherwise continuous spectrum isn't misidentified as a
            # gap.  Using the panel span (xmax - xmin) rather than the
            # data span prevents false positives when data is clustered
            # at one end of a wide panel.
            panel_span = float(self.xmax - self.xmin)
            data_span = float(wave[-1] - wave[0])
            reference_span = max(panel_span, data_span)
            min_gap_width = 0.05 * reference_span
            step_gap_width = threshold * median_dw
            effective_gap_width = max(step_gap_width, min_gap_width)
            big = np.where(dw > effective_gap_width)[0]
            for idx in big:
                g_start = float(wave[idx])
                g_end = float(wave[idx + 1])
                # Avoid duplicating a gap already captured by NaN scan
                if not any(
                    (gs <= g_start <= ge) or (gs <= g_end <= ge)
                    for gs, ge in gaps
                ):
                    gaps.append((g_start, g_end))

        # Sort by start wavelength
        gaps.sort(key=lambda g: g[0])
        return gaps

    def get_panel_data_with_gaps(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Like :meth:`get_panel_data` but inserts NaN breaks at gaps.

        When :attr:`gap_mode` is :attr:`GapMode.SKIP`, each detected
        gap is replaced by a ``NaN`` sentinel so matplotlib draws a
        broken line.  Otherwise this is identical to
        :meth:`get_panel_data`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray | None]
        """
        wave, flux, err = self.get_panel_data()
        if self.gap_mode is not GapMode.SKIP or len(wave) < 2:
            return wave, flux, err

        gaps = self.detect_gaps(wave, flux)
        if not gaps:
            return wave, flux, err

        # Insert NaN break-points at each gap boundary
        new_wave = list(wave)
        new_flux = list(flux)
        new_err = list(err) if err is not None else None
        offset = 0
        for g_start, g_end in gaps:
            # Find insertion point: the first index where wave > g_start
            idx = np.searchsorted(wave, g_start, side="right")
            ins = idx + offset
            mid = (g_start + g_end) / 2.0
            new_wave.insert(ins, mid)
            new_flux.insert(ins, np.nan)
            if new_err is not None:
                new_err.insert(ins, np.nan)
            offset += 1

        result_wave = np.array(new_wave)
        result_flux = np.array(new_flux)
        result_err = np.array(new_err) if new_err is not None else None
        return result_wave, result_flux, result_err

    def panel_has_data(self) -> bool:
        """Return *True* if this panel has any finite observed flux.

        Useful for :class:`StackedSpectralPanel` to decide whether to
        skip an empty cell when ``gap_mode`` is ``SKIP``.
        """
        mask = self.get_panel_mask()
        if not np.any(mask):
            return False
        return bool(np.any(np.isfinite(self.flux_data[mask])))

    def draw_gap_indicators(
        self,
        gaps: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """Draw visual break indicators for each gap in the data.

        **Edge gaps** (gaps that touch the left or right panel boundary)
        are handled by *tightening* the axes x-limits so the gap is
        removed from the visible area entirely.  A small text label is
        placed at the affected edge to show what was skipped.

        **Internal gaps** (data on both sides) are blanked out with a
        background-colored rectangle, diagonal break lines are drawn
        at both edges, and a text annotation is placed in the center
        showing the skipped wavelength range.

        Artists are tagged with :attr:`_GAP_TAG` so they can be removed
        later with :meth:`remove_gap_indicators`.

        Parameters
        ----------
        gaps : list[tuple[float, float]], optional
            Explicit gap list.  Defaults to :meth:`detect_gaps`.
        """
        ax = self.ax
        if ax is None:
            return
        if gaps is None:
            gaps = self.detect_gaps()
        if not gaps:
            return

        xr_lo, xr_hi = self.xlim
        bg = self._get_theme_value("background", "white")
        fg = self._get_theme_value("foreground", "black")

        # ---- 1. Tighten xlim to collapse edge gaps --------------------
        # An "edge gap" is one whose start or end coincides (within a
        # small tolerance) with the panel boundary.  We shrink the
        # visible x-range so those gaps fall entirely outside the axes.
        panel_span = xr_hi - xr_lo
        tol = panel_span * 0.005  # 0.5 % of panel width

        new_lo, new_hi = xr_lo, xr_hi
        edge_annotations: List[Tuple[str, float, str]] = []  # side, wave, label

        for g_start, g_end in gaps:
            if g_end < xr_lo or g_start > xr_hi:
                continue
            touches_left = (g_start - tol) <= new_lo
            touches_right = (g_end + tol) >= new_hi

            if touches_left and not touches_right:
                # Gap at the left edge: move visible start past the gap
                new_lo = max(new_lo, g_end)
                lbl = f"\u2702 {g_start:.3f}\u2013{g_end:.3f} \u03bcm"
                edge_annotations.append(("left", new_lo, lbl))
            elif touches_right and not touches_left:
                # Gap at the right edge: pull visible end before the gap
                new_hi = min(new_hi, g_start)
                lbl = f"\u2702 {g_start:.3f}\u2013{g_end:.3f} \u03bcm"
                edge_annotations.append(("right", new_hi, lbl))

        # Add a small margin so data points don't sit on the axes edge.
        margin = (new_hi - new_lo) * 0.012
        new_lo = max(xr_lo, new_lo - margin)
        new_hi = min(xr_hi, new_hi + margin)

        if new_lo < new_hi:
            ax.set_xlim(new_lo, new_hi)

        # Place small annotations at the edges that were tightened.
        ylo, yhi = ax.get_ylim()
        for side, wave_pos, lbl in edge_annotations:
            ha = "left" if side == "left" else "right"
            x_pos = new_lo if side == "left" else new_hi
            txt = ax.text(
                x_pos, yhi, lbl,
                fontsize=4.5,
                color=fg,
                alpha=0.50,
                ha=ha,
                va="top",
                fontstyle="italic",
                zorder=92,
            )
            setattr(txt, self._GAP_TAG, True)

        # ---- 2. Internal gaps (data on both sides) --------------------
        # Re-read the (potentially updated) xlim.
        vis_lo, vis_hi = ax.get_xlim()

        for g_start, g_end in gaps:
            # Skip gaps that are now entirely outside the visible range
            # (including those collapsed by edge-tightening above).
            if g_end <= vis_lo or g_start >= vis_hi:
                continue

            lo = max(g_start, vis_lo)
            hi = min(g_end, vis_hi)

            # 2a. White-out the gap region
            span = ax.axvspan(
                lo, hi,
                facecolor=bg,
                alpha=1.0,
                edgecolor="none",
                linewidth=0.0,
                zorder=90,
            )
            setattr(span, self._GAP_TAG, True)

            # 2b. Diagonal break marks at each edge
            ylo, yhi = ax.get_ylim()
            y_range = yhi - ylo
            break_width = (hi - lo) * 0.08
            d = y_range * 0.04

            for edge in (lo, hi):
                bw = break_width if edge == lo else -break_width
                line, = ax.plot(
                    [edge - bw, edge + bw],
                    [ylo - d, yhi + d],
                    color="gray",
                    linewidth=1.2,
                    alpha=0.7,
                    zorder=91,
                    clip_on=False,
                )
                setattr(line, self._GAP_TAG, True)

            # 2c. Text annotation showing the skipped range
            mid_x = (lo + hi) / 2.0
            mid_y = (ylo + yhi) / 2.0
            lbl = f"\u2702 {g_start:.3f}\u2013{g_end:.3f} \u03bcm"
            txt = ax.text(
                mid_x, mid_y, lbl,
                fontsize=5.5,
                color=fg,
                alpha=0.55,
                ha="center",
                va="center",
                fontstyle="italic",
                rotation=90,
                zorder=92,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor=bg,
                    edgecolor="none",
                    alpha=0.8,
                ),
            )
            setattr(txt, self._GAP_TAG, True)

    def remove_gap_indicators(self) -> None:
        """Remove previously drawn gap-indicator artists."""
        ax = self.ax
        if ax is None:
            return
        # The gap indicators include patches (axvspan), lines (break
        # marks), and texts (wavelength annotations).
        self._clear_tagged_artists(
            ax, self._GAP_TAG, lines=True, collections=True, texts=True,
        )
        for artist in ax.patches[:]:
            if hasattr(artist, self._GAP_TAG):
                artist.remove()

    # ------------------------------------------------------------------
    # Post-render annotation helpers
    # ------------------------------------------------------------------
    def plot_atomic_lines(
        self,
        atomic_df: "pd.DataFrame",
        tag: str = "_islat_atomic_line",
    ) -> None:
        """Draw atomic-line markers on the panel's axes.

        Uses the current *visible* x-range (which may have been
        tightened by :meth:`draw_gap_indicators`) so that markers
        are not placed in cropped gap regions.
        """
        ax = self.ax
        if ax is None:
            return
        xr = tuple(ax.get_xlim())
        self._plot_atomic_lines(ax, atomic_df, xr=xr, tag=tag)

    def remove_atomic_lines(
        self,
        tag: str = "_islat_atomic_line",
    ) -> None:
        """Remove previously drawn atomic-line artists."""
        ax = self.ax
        if ax is None:
            return
        self._clear_tagged_artists(
            ax, tag, lines=True, collections=False, texts=True,
        )

    def plot_saved_lines(
        self,
        line_data: "pd.DataFrame",
        tag: str = "_islat_saved_line",
        show_labels: bool = True,
    ) -> None:
        """Draw saved-line annotations on the panel's axes.

        Uses the current *visible* x-range (which may have been
        tightened by :meth:`draw_gap_indicators`) so that markers
        are not placed in cropped gap regions.  *show_labels* ``False``
        draws the lines without their text labels.
        """
        ax = self.ax
        if ax is None:
            return
        xr = tuple(ax.get_xlim())
        ymin, ymax = ax.get_ylim()
        self._plot_line_annotations(
            ax, line_data, xr, ymin, ymax, tag=tag, show_labels=show_labels,
        )

    def remove_saved_lines(
        self,
        tag: str = "_islat_saved_line",
    ) -> None:
        """Remove previously drawn saved-line artists."""
        ax = self.ax
        if ax is None:
            return
        self._clear_tagged_artists(
            ax, tag, lines=True, collections=True, texts=True,
        )

    # ------------------------------------------------------------------
    # Convenience: change range and regenerate
    # ------------------------------------------------------------------
    def set_range(self, xmin: float, xmax: float) -> None:
        """Update the panel wavelength range and regenerate the plot."""
        self.xmin = xmin
        self.xmax = xmax
        self.generate_plot()

    # ------------------------------------------------------------------
    # Resolve / prepare axes
    # ------------------------------------------------------------------
    def _resolve_axes(self) -> Axes:
        """Return an Axes to draw on.

        If an external axes was supplied at construction time it is
        returned directly (and cleared).  Otherwise a new figure + axes
        pair is created.
        """
        if self._external_ax is not None:
            self._ax = self._external_ax
        else:
            self._ensure_figure()
            self.fig.clf()
            self._ax = self.fig.add_subplot(111)
        self._ax.clear()
        return self._ax