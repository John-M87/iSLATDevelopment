"""
ToggleMixin - shared toggle-state management for PlotView implementations.

Both :class:`ThreePanelView` and :class:`FullSpectrumView` need to
reconcile visual state (atomic lines, saved lines, summed spectrum,
legend, observed-data visibility, Y-axis limits) with the controller's
canonical ``toggle_state`` dict.

This mixin extracts the common *orchestration* logic so each view only
has to implement a few small hooks describing **where** its axes live
and **how** to add/remove overlay artists.

Usage::

    class MyView(ToggleMixin, PlotView):
        # implement the required hooks ...

Hook methods (must be implemented by the concrete view):

    ``_iter_toggle_axes()``
        Yield every axes object whose artists should be toggled.

    ``_add_atomic_line_artists()``
        Render atomic-line markers.

    ``_remove_atomic_line_artists()``
        Remove previously-rendered atomic-line markers.

    ``_add_saved_line_artists()``
        Render saved-line markers.

    ``_remove_saved_line_artists()``
        Remove previously-rendered saved-line markers.

    ``_load_saved_line_data()``
        Load / refresh saved-line data from disk.

    ``_toggle_ready() -> bool``
        Return ``False`` when the view is not initialised and toggles
        should be silently ignored.

    ``draw()``
        Flush pending changes to the canvas.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Optional

class ToggleMixin:
    """Mixin providing toggle-state reconciliation for :class:`PlotView` subclasses."""

    # ------------------------------------------------------------------
    # Hooks - concrete views must implement these
    # ------------------------------------------------------------------
    @abstractmethod
    def _iter_toggle_axes(self):
        """Yield the axes whose artists are affected by toggles."""
        ...

    @abstractmethod
    def _add_atomic_line_artists(self) -> None:
        ...

    @abstractmethod
    def _remove_atomic_line_artists(self) -> None:
        ...

    @abstractmethod
    def _add_saved_line_artists(self) -> None:
        ...

    @abstractmethod
    def _remove_saved_line_artists(self) -> None:
        ...

    @abstractmethod
    def _load_saved_line_data(self) -> Any:
        """Return the saved-line data (DataFrame or similar)."""
        ...

    @abstractmethod
    def _toggle_ready(self) -> bool:
        """Return True when the view is initialised and toggles can proceed."""
        ...

    @abstractmethod
    def draw(self) -> None:
        ...

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    def _set_summed_visibility(self, visible: bool) -> None:
        """Show / hide the summed-spectrum fill on all toggle axes."""
        for ax in self._iter_toggle_axes():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.set_visible(visible)

    # Observed-data / Y-policy hooks (see PlotView): every toggle axis takes part.
    def _iter_observed_axes(self):
        return self._iter_toggle_axes()

    def _iter_y_axes(self):
        return self._iter_toggle_axes()

    def _set_legend_visibility(self, visible: bool) -> None:
        """Show / hide the legend on all toggle axes.

        Subclasses may override this to delegate to a
        :class:`LegendStrategy` instead.
        """
        for ax in self._iter_toggle_axes():
            leg = ax.get_legend()
            if leg is not None:
                leg.set_visible(visible)

    # ------------------------------------------------------------------
    # Public toggle API (satisfies :class:`PlotView` abstract methods)
    # ------------------------------------------------------------------
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """Reconcile visual state with the controller's *toggle_state* dict."""
        if not self._toggle_ready():
            return

        # Atomic lines
        self._remove_atomic_line_artists()
        if toggle_state.get("atomic_lines", False):
            self._add_atomic_line_artists()

        # Saved lines
        self._remove_saved_line_artists()
        if toggle_state.get("saved_lines", False):
            self._add_saved_line_artists()

        # Summed spectrum
        self._set_summed_visibility(toggle_state.get("summed", True))

        # Legend
        self._set_legend_visibility(toggle_state.get("legend", True))

        # Observed-data visibility and manual Y limits
        self._reapply_display_state(toggle_state)

        self.draw()

    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill across all toggle axes."""
        if not self._toggle_ready():
            return
        self._set_summed_visibility(visible)
        self.draw()

    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Toggle the legend visibility on all toggle axes."""
        if not self._toggle_ready():
            return
        if visible is None:
            # Flip based on the first axes that has a legend
            for ax in self._iter_toggle_axes():
                leg = ax.get_legend()
                if leg is not None:
                    visible = not leg.get_visible()
                    break
            else:
                visible = True
        self._set_legend_visibility(visible)
        self.draw()

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """Add or remove saved-line annotations."""
        if not self._toggle_ready():
            return
        if show:
            if loaded_lines is not None:
                self._set_saved_line_data(loaded_lines)
            else:
                self._set_saved_line_data(self._load_saved_line_data())
            self._add_saved_line_artists()
        else:
            self._remove_saved_line_artists()
        self.draw()

    def toggle_atomic_lines(self, show: bool) -> None:
        """Add or remove atomic-line annotations."""
        if not self._toggle_ready():
            return
        if show:
            self._add_atomic_line_artists()
        else:
            self._remove_atomic_line_artists()
        self.draw()

    # ------------------------------------------------------------------
    # Optional hook - saved-line data storage
    # ------------------------------------------------------------------
    def _set_saved_line_data(self, data: Any) -> None:
        """Store saved-line data.  Default stores on ``self.line_data``."""
        self.line_data = data  # type: ignore[attr-defined]