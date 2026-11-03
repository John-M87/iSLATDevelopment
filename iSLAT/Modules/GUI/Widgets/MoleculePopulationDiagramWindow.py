"""
MoleculePopulationDiagramWindow - popout population diagram for one molecule.

Opened from the molecule control panel's right-click menu.  It is the
:class:`~iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow.FilteredPopulationDiagramWindow`
UI (same renderer, right-click Plot Settings, "Sync Settings" from the main
diagram, status bar, teardown) showing *all* of the molecule's lines, and it
redraws live whenever that molecule's parameters or the global parameters
change.
"""
from __future__ import annotations

import tkinter as tk
from typing import Any, Callable, Dict, Optional

from iSLAT.Modules.GUI.Widgets.FilteredPopulationDiagramWindow import (
    FilteredPopulationDiagramWindow,
    _population_data_length,
)

#: Parameter changes that do not affect the population diagram.
_IGNORED_PARAMS = frozenset({"is_visible", "is_bold"})


class MoleculePopulationDiagramWindow(FilteredPopulationDiagramWindow):
    """Live population diagram of every line of *mol_obj*.

    Use :meth:`open` to get one window per molecule.
    """

    #: Open windows keyed by ``id(molecule)`` so reopening raises the existing one.
    _open_windows: Dict[int, "MoleculePopulationDiagramWindow"] = {}

    def __init__(self, parent, mol_obj, islat=None, on_close: Callable = None):
        super().__init__(parent, mol_obj, islat=islat, on_close=on_close)
        self._mol_cb = self._on_molecule_parameter_changed
        self._global_cb = self._on_global_parameter_changed
        self._molecules_dict = getattr(islat, "molecules_dict", None)
        self._register_callbacks()
        self._render()

    @classmethod
    def open(cls, parent, mol_obj, islat=None) -> "MoleculePopulationDiagramWindow":
        """Raise the molecule's open window, or create one."""
        existing = cls._open_windows.get(id(mol_obj))
        if existing is not None and existing.mol_obj is mol_obj and existing._alive():
            existing.deiconify()
            existing.lift()
            existing.focus_set()
            return existing
        win = cls(parent, mol_obj, islat=islat)
        cls._open_windows[id(mol_obj)] = win
        return win

    # ------------------------------------------------------------------
    # Construction hooks
    # ------------------------------------------------------------------
    def _window_title(self, mol_name: str) -> str:
        return f"Population Diagram: {mol_name}"

    def _default_figure_name(self) -> str:
        return f"{getattr(self.mol_obj, 'name', 'molecule')}_population_diagram"

    def _create_population_plot(self, color: Optional[str]) -> Any:
        """Bind the plot to the molecule itself so every redraw reads its current state."""
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        return PopulationDiagramPlot(
            molecule=self.mol_obj,
            ax=self._ax,
            fig=self._fig,
            theme=self._theme() or None,
        )

    # ------------------------------------------------------------------
    # Live updates
    # ------------------------------------------------------------------
    def _register_callbacks(self) -> None:
        cls = type(self.mol_obj)
        if hasattr(cls, "add_molecule_parameter_change_callback"):
            cls.add_molecule_parameter_change_callback(self._mol_cb)
        if self._molecules_dict is not None and hasattr(
                self._molecules_dict, "add_global_parameter_change_callback"):
            self._molecules_dict.add_global_parameter_change_callback(self._global_cb)

    def _unregister_callbacks(self) -> None:
        cls = type(self.mol_obj)
        try:
            cls.remove_molecule_parameter_change_callback(self._mol_cb)
        except Exception:
            pass
        if self._molecules_dict is not None:
            try:
                self._molecules_dict.remove_global_parameter_change_callback(self._global_cb)
            except Exception:
                pass

    def _on_molecule_parameter_changed(self, molecule_name, parameter_name,
                                       old_value=None, new_value=None) -> None:
        if parameter_name in _IGNORED_PARAMS:
            return
        if parameter_name == "name" and old_value == getattr(self, "_shown_name", None):
            self.title(self._window_title(str(new_value)))
        elif molecule_name != getattr(self.mol_obj, "name", None):
            return
        self.refresh()

    def _on_global_parameter_changed(self, parameter_name=None, old_value=None,
                                     new_value=None) -> None:
        self.refresh()

    def refresh(self, delay_ms: int = 120) -> None:
        """Redraw from the molecule's current state, coalescing rapid calls."""
        if not self._alive():
            return
        if self._redraw_job is not None:
            try:
                self.after_cancel(self._redraw_job)
            except (tk.TclError, ValueError):
                pass
        self._redraw_job = self.after(max(int(delay_ms), 1), self._render)

    def _render(self) -> None:
        """Recompute (lazily) and redraw the diagram for every line of the molecule."""
        self._redraw_job = None
        if not self._alive() or self._pdp is None:
            return
        mol = self.mol_obj
        self._shown_name = getattr(mol, "name", None)
        try:
            if hasattr(mol, "calculate_intensity"):
                mol.calculate_intensity()
            self._pdp.generate_plot()
        except Exception as exc:
            self._status_var.set(f"Could not draw: {exc}")
            return
        self._draw_idle()
        components = getattr(self._pdp, "_component_data", None) or []
        shown = sum(_population_data_length(c) or 0 for c in components)
        self._status_var.set(
            f"Showing {shown} lines." if shown else "No population-diagram data available.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        """Drop the live-update callbacks, then tear down like the filtered window."""
        self._unregister_callbacks()
        if type(self)._open_windows.get(id(self.mol_obj)) is self:
            del type(self)._open_windows[id(self.mol_obj)]
        super().close()
