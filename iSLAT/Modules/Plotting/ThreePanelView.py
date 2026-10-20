"""
ThreePanelView - :class:`PlotView` implementation for the standard 3-panel layout.

Panels:
    1. Full spectrum overview  (ax1)
    2. Line inspection zoom    (ax2)
    3. Population diagram      (ax3)

This view **composes** a :class:`MainPlotGrid` in *borrowed-axes* mode for all spectrum-panel rendering, mirroring how :class:`FullSpectrumView` composes a :class:`FullSpectrumPlot`. 
The axes and canvas are still owned by the :class:`iSLATPlot` controller
- :class:`MainPlotGrid` renders onto them without calling ``fig.clf()`` so that cached references in ``InteractionHandler`` stay valid.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, List

import numpy as np

from .PlotView import PlotView
from .BasePlot import BasePlot
from .ToggleMixin import ToggleMixin
from .MainPlotGrid import MainPlotGrid
from .LineInspectionPlot import LineInspectionPlot
from .PopulationDiagramContextMixin import PopulationDiagramContextMixin
from .LineInspectionContextMixin import LineInspectionContextMixin

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

import iSLAT.Constants as c
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
#from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer
from iSLAT.Modules.DataProcessing.spectral_utils import flux_integral

# Import debug configuration
try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    class _Fallback:
        def verbose(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): print(f"WARNING: {a}")
        def error(self, *a, **k): print(f"ERROR: {a}")
        def trace(self, *a, **k): pass
    debug_config = _Fallback()

class ThreePanelView(ToggleMixin, PlotView, PopulationDiagramContextMixin, LineInspectionContextMixin):
    """
    Standard 3-panel GUI view backed by a :class:`MainPlotGrid`.

    The grid is created lazily in *borrowed-axes* mode on first render:
    it receives the controller's ``ax1``/``ax2``/``ax3`` and renders directly onto them.
    Toggle-state management (atomic lines, saved lines, summed spectrum, legend) is provided by :class:`ToggleMixin`.
    """
    def __init__(self, plot_manager: Any) -> None:
        """
        Parameters
        ----------
        plot_manager : iSLATPlot
            The main controller that owns fig, canvas, ax1-3.
        """
        self._pm = plot_manager  # short alias for the controller
        self._needs_refresh: bool = True  # Set True when data changes; cleared after re-render

        # MainPlotGrid in borrowed-axes mode - created lazily in
        # _ensure_grid() because wave_data/flux_data are not available
        # until the first render call.
        self._grid: MainPlotGrid | None = None

        # Active-line state (line inspection markers + pop-diagram scatter)
        self.active_lines: List[Any] = []
        self.selected_line: Optional[Dict[str, Any]] = None
        self._pick_event_connected: bool = False

        # Multi-molecule line inspection state
        # Stores the last (xmin, xmax) selection so comparison-molecule
        # changes can re-trigger on_selection without user interaction.
        self._current_selection: Optional[Tuple[float, float]] = None
        # Per-molecule scatter collections keyed by molecule name:
        #   {mol_name: (PathCollection, scatter_point_count)}
        self._active_scatter_collections: Dict[str, Any] = {}
        # Cache of {mol_name: line_data} from the last on_selection call, used when switching the pop-diagram on line-click.
        self._mol_line_data_cache: Dict[str, List[Any]] = {}
        self._comparison_callback_registered: bool = False

    # ------------------------------------------------------------------
    # Line-inspection helpers
    # ------------------------------------------------------------------
    def get_selected_line(self) -> Optional[Dict[str, Any]]:
        """Return the currently selected line info dict, or *None*."""
        return self.selected_line

    # ------------------------------------------------------------------
    # Helpers (private, short-hand access to controller state)
    # ------------------------------------------------------------------
    def _has_tagged_artists(self, tag: str) -> bool:
        """Return True if *ax1* contains any artist with the given tag."""
        for artist in list(self.ax1.lines) + list(self.ax1.texts):
            if hasattr(artist, tag):
                return True
        return False

    def _ensure_grid(self) -> MainPlotGrid:
        """Return the composed :class:`MainPlotGrid`, creating it lazily.

        The grid is constructed in *borrowed-axes* mode using the
        controller's ``ax1``/``ax2``/``ax3``. Because it never calls
        ``fig.clf()``, all external references to these axes stay valid.
        """
        if self._grid is not None:
            return self._grid

        islat = self._islat
        wave = getattr(islat, 'wave_data', np.array([]))
        flux = getattr(islat, 'flux_data', np.array([]))
        err = getattr(islat, 'err_data', None)
        mol_dict = getattr(islat, 'molecules_dict', None)
        active_mol = getattr(islat, 'active_molecule', None)

        self._grid = MainPlotGrid(
            wave_data=wave,
            flux_data=flux,
            error_data=err,
            molecules=mol_dict,
            active_molecule=active_mol,
            theme=self._pm.theme,
            ax_spectrum=self._pm.ax1,
            ax_inspection=self._pm.ax2,
            ax_popdiagram=self._pm.ax3,
        )
        return self._grid

    @property
    def _islat(self):
        return self._pm.islat

    @property
    def _canvas(self) -> "FigureCanvasTkAgg":
        return self._pm.canvas

    @property
    def ax1(self) -> "Axes":
        return self._pm.ax1

    @property
    def ax2(self) -> "Axes":
        return self._pm.ax2

    @property
    def ax3(self) -> "Axes":
        return self._pm.ax3

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, parent_frame: Any) -> None:
        """Show the original 3-panel canvas and refresh."""
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Register view-specific controls with the ControlBus
        self._register_control_fields()

        # Register comparison-molecule change callback once
        if not self._comparison_callback_registered:
            islat = self._islat
            mol_dict = getattr(islat, 'molecules_dict', None)
            target = mol_dict if mol_dict is not None else islat
            if hasattr(target, 'add_comparison_molecule_change_callback'):
                target.add_comparison_molecule_change_callback(
                    self._on_comparison_molecules_changed
                )
                self._comparison_callback_registered = True

        # Sync theme in case it changed while the other view was active.
        self.apply_theme(self._pm.theme)

        if self._needs_refresh:
            # Data changed while we were inactive - full re-render
            self._do_update_model_plot()
            self._needs_refresh = False
        else:
            # Simple view toggle - just sync overlay state
            self.sync_toggle_state(self._pm.toggle_state)

        # Restore the span selector and active line selection
        self._restore_line_selection()

    def deactivate(self) -> None:
        """Hide the original canvas."""
        self._canvas.get_tk_widget().pack_forget()

        # Unregister all fields this view registered across all surfaces
        bus = getattr(self._pm, 'control_bus', None)
        if bus is not None:
            bus.unregister_owner(self)

    # ------------------------------------------------------------------
    # Theme
    # ------------------------------------------------------------------
    def apply_theme(self, theme: dict) -> None:
        """Apply *theme* to the three-panel figure, axes, and canvas.

        Delegates to the controller's :meth:`_apply_plot_theming` which
        already handles figure/axes/spine/tick coloring.
        """
        # Keep the controller's theme reference in sync
        self._pm.theme = theme

        # Propagate theme to the grid so future renders pick up the colors.
        if self._grid is not None:
            self._grid.theme = theme
            if self._grid.pop_diagram_panel is not None:
                self._grid.pop_diagram_panel.theme = theme

        # Restyle figure / axes / data artists
        self._pm._apply_plot_theming()

        # Restyle canvas widget background
        try:
            self._canvas.get_tk_widget().configure(
                bg=theme.get("background", "#181A1B")
            )
        except Exception:
            pass

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        self._do_update_model_plot()
        self._needs_refresh = False

    def _update_spectrum_panel_only(self) -> None:
        """Re-render ax1 (spectrum overview) without touching ax2 or ax3.

        Used by :meth:`on_molecule_parameter_changed` so that an in-progress
        line-inspection or population-diagram selection is preserved.
        """
        islat = self._islat
        if not hasattr(islat, 'molecules_dict') or len(islat.molecules_dict) == 0:
            return

        mol_dict = islat.molecules_dict
        wave_data_obs = islat.wave_data_original
        wave_data = mol_dict.apply_stellar_rv(wave_data_obs)
        islat.wave_data = wave_data

        grid = self._ensure_grid()
        grid.wave_data = wave_data
        grid.flux_data = islat.flux_data
        grid.molecules = mol_dict
        grid.active_molecule = getattr(islat, 'active_molecule', None)
        grid.error_data = getattr(islat, 'err_data', None)
        grid.wave_data_obs = wave_data_obs

        # Re-render only the spectrum panel - leaves inspection + pop panels alone.
        grid._render_spectrum_panel()
        grid.apply_theme_to_figure()
        self._pm.make_span_selector()

        # Reapply the legend toggle - _render_spectrum_panel always rebuilds
        # the legend so we must re-hide it when the user has toggled it off.
        if not self._pm.legend_toggle:
            leg = self.ax1.get_legend()
            if leg is not None:
                leg.set_visible(False)

        self._reapply_display_state(self._pm.toggle_state)

    def _do_update_model_plot(self) -> None:
        """Internal full re-render via the composed :class:`MainPlotGrid`.

        Data is fetched from the ``iSLAT`` controller, pushed into the
        grid's ``update_data()`` method (which clears and re-renders the
        spectrum panel on the borrowed axes), and then toggle states are
        reconciled on top.
        """
        islat = self._islat

        if not hasattr(islat, 'molecules_dict') or len(islat.molecules_dict) == 0:
            # No molecules - clear all molecule artists from the spectrum axes.
            if self._grid is not None:
                self._grid.clear_all_molecule_lines()
            else:
                # Grid not yet created - clear directly from ax1
                for line in self._pm.ax1.lines[:]:
                    if getattr(line, "_molecule_name", None) is not None:
                        line.remove()
            self._canvas.draw_idle()
            return

        mol_dict = islat.molecules_dict

        # Always work from the original observer-frame wavelengths.
        wave_data_obs = islat.wave_data_original

        # RV-corrected (rest-frame) wavelengths for the display x-axis.
        wave_data = mol_dict.apply_stellar_rv(wave_data_obs)
        islat.wave_data = wave_data

        # Push data into the grid and re-render all three panels.
        grid = self._ensure_grid()
        grid.update_data(
            wave_data=wave_data,
            flux_data=islat.flux_data,
            molecules=mol_dict,
            active_molecule=getattr(islat, 'active_molecule', None),
            error_data=getattr(islat, 'err_data', None),
            wave_data_obs=wave_data_obs,
        )

        # Respect summed_toggle - hide the summed fill if the user toggled it off.
        if not self._pm.summed_toggle:
            BasePlot._clear_tagged_artists(
                grid.ax_spectrum, "_islat_summed", lines=False,
            )

        # Overlay saved / atomic lines if their toggles are on
        if self._pm.atomic_toggle:
            self.toggle_atomic_lines(True)

        if self._pm.line_toggle:
            self._add_saved_line_artists()

        # Respect the legend toggle state
        if not self._pm.legend_toggle:
            for ax in (self.ax1, self.ax2, self.ax3):
                leg = ax.get_legend()
                if leg is not None:
                    leg.set_visible(False)

        self._reapply_display_state(self._pm.toggle_state)

        # Recreate span selector and redraw
        self._pm.make_span_selector()
        self._canvas.draw_idle()

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
        """
        Fast incremental update - toggle one molecule's artists on ax1.

        Delegates to :meth:`MainPlotGrid.handle_molecule_visibility_change`
        which handles artist toggling, summed-spectrum update, and legend
        rebuild on the borrowed spectrum axes.
        """
        grid = self._ensure_grid()
        # Keep the grid's molecules reference in sync
        grid.molecules = molecules_dict
        grid.wave_data_obs = getattr(self._islat, 'wave_data_original', wave_data)

        grid.handle_molecule_visibility_change(
            molecule_name=molecule_name,
            is_visible=is_visible,
            force_rerender=force_rerender,
        )

        # Make sure that the toggle legend state is respected
        self._set_legend_visibility(self._pm.legend_toggle)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Selection & line-inspection
    # ------------------------------------------------------------------
    def on_selection(self, xmin: float, xmax: float) -> None:
        """Handle a span-selector drag - render line inspection + population diagram.

        All active molecules (primary + comparison) have their lines rendered
        in ax2 using each molecule's own color.  The population diagram (ax3)
        shows only the primary active molecule until the user clicks on a line.
        """
        debug_config.verbose("line_inspection", f"on_selection called", xmin=xmin, xmax=xmax)

        self._current_selection = (xmin, xmax)

        active_mol = getattr(self._islat, 'active_molecule', None)
        if active_mol is None:
            return

        all_mols = self._get_all_active_molecules()

        # Collect line data for every active molecule
        mol_line_data: List[Tuple["Molecule", List[Any]]] = []
        self._mol_line_data_cache = {}
        for mol in all_mols:
            try:
                ld = self._get_molecule_line_data(mol, xmin, xmax)
                if ld:
                    mol_line_data.append((mol, ld))
                    self._mol_line_data_cache[mol.name] = ld
            except Exception as e:
                debug_config.warning("three_panel_view", f"Could not get line data for {mol.name}: {e}")

        if not mol_line_data:
            if hasattr(self._islat, 'GUI') and hasattr(self._islat.GUI, 'data_field'):
                names = ", ".join(m.name for m in all_mols)
                self._islat.GUI.data_field.insert_text(
                    f"No transitions found for [{names}] in the selected range"
                )
            self.clear_active_lines()
            self._render_population_diagram_base()
            self._canvas.draw_idle()
            return

        # Clear previous active lines and render fresh ones
        self.clear_active_lines()
        self._render_line_inspection_multi(xmin, xmax, mol_line_data)

        # Population diagram: primary active molecule only
        primary_line_data = self._mol_line_data_cache.get(active_mol.name, [])
        if primary_line_data:
            self._render_population_diagram_with_lines(primary_line_data, active_mol)
        else:
            self._render_population_diagram_base()

        # Highlight strongest line AFTER both panels are populated
        self._highlight_strongest_line()

        # Connect pick event once
        if not self._pick_event_connected:
            self._canvas.mpl_connect('pick_event', self._on_pick_line)
            self._pick_event_connected = True

        self._canvas.draw_idle()
        debug_config.verbose("line_inspection", "on_selection completed")

    def clear_selection(self) -> None:
        """Clear the current line-inspection selection and reset panels 2-3."""
        self._current_selection = None
        self._mol_line_data_cache = {}
        self.clear_active_lines()
        self.ax2.clear()
        self._render_population_diagram_base()
        self._canvas.draw_idle()

    def clear_active_lines(self) -> None:
        """Remove all active-line artists (vlines, text, scatter)."""
        grid = self._ensure_grid()
        grid.clear_active_lines(self.active_lines)
        self._active_scatter_collections = {}

    # ------------------------------------------------------------------
    # Molecule lifecycle callbacks
    # ------------------------------------------------------------------
    def on_active_molecule_changed(
        self,
        new_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """The user selected a different active molecule."""
        debug_config.info("active_molecule", "ThreePanelView.on_active_molecule_changed()")

        if new_molecule is not None:
            self.ax3.set_title(f'{new_molecule.displaylabel} Population diagram')

        self.clear_active_lines()

        if current_selection is not None:
            xmin, xmax = current_selection
            self.on_selection(xmin, xmax)
        else:
            self._render_population_diagram_base()
            self._canvas.draw_idle()

    def on_molecule_parameter_changed(
        self,
        molecule_name: str,
        parameter_name: str,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """A molecule parameter changed - update spectrum + possibly line inspection."""
        # Visibility changes are handled separately
        if parameter_name == 'is_visible':
            return

        mol_dict = getattr(self._islat, 'molecules_dict', None)
        if mol_dict is None or molecule_name not in mol_dict:
            return

        molecule = mol_dict[molecule_name]

        if molecule.is_visible:
            # Re-render only the spectrum panel (ax1) to avoid wiping ax2/ax3.
            self._update_spectrum_panel_only()

        active_mol = getattr(self._islat, 'active_molecule', None)
        is_active = (active_mol is not None
                     and hasattr(active_mol, 'name')
                     and active_mol.name == molecule_name)
        is_comparison = molecule_name in [
            getattr(m, 'name', None)
            for m in getattr(self._islat, 'comparison_molecules', [])
        ]

        sel = current_selection or self._current_selection
        if sel is not None:
            # Always restore line inspection + pop diagram after spectrum update
            # so that ax2/ax3 are not left blank.
            self.on_selection(*sel)
        else:
            if is_active or is_comparison:
                self._render_population_diagram_base()
            self._canvas.draw_idle()

    def on_molecule_deleted(self, molecule_name: str) -> None:
        """A molecule was removed - clear and rebuild everything."""
        if self._grid is not None:
            self._grid.clear_all_molecule_lines()
        else:
            for line in self._pm.ax1.lines[:]:
                if getattr(line, "_molecule_name", None) is not None:
                    line.remove()

        active_mol = getattr(self._islat, 'active_molecule', None)
        if (active_mol is not None
                and hasattr(active_mol, 'name')
                and active_mol.name == molecule_name):
            self.clear_active_lines()

        self._do_update_model_plot()

    # ------------------------------------------------------------------
    # Private helpers - line inspection rendering
    # ------------------------------------------------------------------
    def _get_line_threshold(self) -> float:
        """Return the line-intensity threshold (0-1) from user settings.

        Falls back to 0.3 (30%) when ``user_settings`` is unavailable.
        """
        return getattr(self._islat, 'user_settings', {}).get(
            'line_threshold', 0.3,
        )

    def _get_all_active_molecules(self) -> List["Molecule"]:
        """Return the primary molecule followed by all comparison molecules.

        Delegates to MoleculeDict.get_active_set() when available, with a
        manual fallback for test scenarios that provide a plain islat stub.
        """
        mol_dict = getattr(self._islat, 'molecules_dict', None)
        if mol_dict is not None and hasattr(mol_dict, 'get_active_set'):
            return mol_dict.get_active_set()
        # Fallback: reconstruct manually from islat attributes
        result: List["Molecule"] = []
        primary = getattr(self._islat, 'active_molecule', None)
        if primary is not None:
            result.append(primary)
        for mol in getattr(self._islat, 'comparison_molecules', []):
            if mol is not None and mol not in result:
                result.append(mol)
        return result

    def _on_comparison_molecules_changed(self, _comparison_list: Any = None) -> None:
        """Re-run line inspection when the comparison-molecule set changes."""
        if self._current_selection is not None:
            xmin, xmax = self._current_selection
            self.on_selection(xmin, xmax)

    def _get_molecule_line_data(
        self, molecule: "Molecule", xmin: float, xmax: float,
    ) -> List[Tuple["MoleculeLine", float, Optional[float]]]:
        """Get molecule lines in a wavelength range (pure data-access)."""
        return self._pm.get_molecule_line_data(molecule, xmin, xmax)

    def _render_line_inspection(
        self,
        xmin: float,
        xmax: float,
        line_data: List[Tuple["MoleculeLine", float, Any]],
    ) -> None:
        """Render the line-inspection panel (ax2) with vertical markers.

        Kept for backward compatibility - delegates to
        :meth:`_render_line_inspection_multi` with a single-molecule list.
        """
        active_mol = self._islat.active_molecule
        if active_mol is None:
            return
        self._render_line_inspection_multi(xmin, xmax, [(active_mol, line_data)])

    def _render_line_inspection_multi(
        self,
        xmin: float,
        xmax: float,
        mol_line_data: List[Tuple["Molecule", List[Any]]],
    ) -> None:
        """Render line inspection for *all* active molecules.

        Draws the observed spectrum + the primary (first) molecule's model
        overlay once, then overlays comparison molecules' models, and
        finally adds per-molecule vertical line markers each in the
        molecule's own color.

        Parameters
        ----------
        xmin, xmax : float
            Wavelength bounds of the inspection window.
        mol_line_data : list[(Molecule, line_data)]
            Ordered list of ``(molecule, line_data_list)`` pairs.
            The first entry is the primary active molecule.
        """
        if not mol_line_data:
            return

        primary_mol = mol_line_data[0][0]
        comparison_mols = [mol for mol, _ in mol_line_data[1:]]

        fit_result = getattr(self._pm, 'fit_result', None)
        grid = self._ensure_grid()
        grid.render_line_inspection_plot(
            wave_data=self._islat.wave_data,
            flux_data=self._islat.flux_data,
            xmin=xmin, xmax=xmax,
            active_molecule=primary_mol,
            fit_result=fit_result,
            additional_molecules=comparison_mols if comparison_mols else None,
        )

        # Compute max_y for line-height scaling
        data_mask = (self._islat.wave_data >= xmin) & (self._islat.wave_data <= xmax)
        data_region_y = self._islat.flux_data[data_mask]
        max_y = (
            float(np.nanmax(data_region_y))
            if len(data_region_y) > 0
            else (self.ax2.get_ylim()[1] / 1.1)
        )

        # Add vertical line markers per molecule, each in its own color
        threshold = self._get_line_threshold()
        for mol, line_data in mol_line_data:
            mol_color = getattr(mol, 'color', None) or grid._get_theme_value(
                "active_scatter_line_color", "green"
            )
            grid.render_active_line_markers(
                line_data, self.active_lines, max_y,
                threshold=threshold,
                color=mol_color,
                molecule_name=mol.name,
                molecule_color=mol_color,
            )

    def _render_population_diagram_base(self) -> None:
        """Render the base population diagram for the active molecule."""
        grid = self._ensure_grid()
        grid.render_population_diagram_for_molecule(self._islat.active_molecule)

    def _render_population_diagram_with_lines(
        self,
        line_data: List[Tuple["MoleculeLine", float, Any]],
        molecule: Optional["Molecule"] = None,
    ) -> None:
        """Render pop-diagram base + scatter points for *line_data*.

        Parameters
        ----------
        line_data : list
            Line data triples for the molecule to display.
        molecule : Molecule, optional
            The molecule whose diagram is shown.  Defaults to
            ``islat.active_molecule`` when not provided.
        """
        if molecule is None:
            molecule = self._islat.active_molecule
        grid = self._ensure_grid()
        grid.render_population_diagram_for_molecule(molecule)
        if line_data and molecule is not None:
            mol_color = getattr(molecule, 'color', None) or grid._get_theme_value(
                "active_scatter_line_color", "green"
            )
            sc = grid.render_active_line_scatter(
                line_data, self.active_lines, molecule,
                threshold=self._get_line_threshold(),
                color=mol_color,
            )
            # Store scatter collection keyed by molecule name
            if sc is not None:
                # Count only the points belonging to *this* collection; entries
                # for other molecules may still hold stale scatter references.
                point_count = len([
                    e for e in self.active_lines
                    if len(e) > 2 and e[2] is sc
                ])
                self._active_scatter_collections[molecule.name] = (sc, point_count)

    # ------------------------------------------------------------------
    # Private helpers - pick / highlight interaction
    # ------------------------------------------------------------------
    def _on_pick_line(self, event: Any) -> None:
        """Handle line pick events - self-contained interaction logic."""
        picked_value = self._handle_line_pick_event(event)
        if picked_value:
            self.selected_line = picked_value
            self._display_line_info(picked_value)

            # Shift+click on a scatter point → trigger a line-inspection
            # selection centered on the picked wavelength, exactly as if the
            # user had drawn a span in the full-spectrum view.
            mouse_event = getattr(event, 'mouseevent', None)
            key = getattr(mouse_event, 'key', None) if mouse_event is not None else None
            is_shift = key is not None and 'shift' in str(key).lower()
            artist_axes = getattr(event.artist, 'axes', None)
            is_scatter_pick = (
                hasattr(event.artist, 'get_offsets')
                and artist_axes is self.ax3
            )
            if is_shift and is_scatter_pick:
                lam = picked_value.get('lam')
                if lam is not None:
                    self._trigger_inspection_from_wavelength(float(lam))
                    return  # draw_idle will be called inside the trigger

        # Shift+click on a base (non-active) scatter point on ax3 -
        # picked_value will be None here because the artist is not in
        # active_lines, but the scatter may still carry a wavelength tag.
        mouse_event = getattr(event, 'mouseevent', None)
        key = getattr(mouse_event, 'key', None) if mouse_event is not None else None
        is_shift = key is not None and 'shift' in str(key).lower()
        artist_axes = getattr(event.artist, 'axes', None)
        is_ax3_scatter = (
            hasattr(event.artist, 'get_offsets')
            and artist_axes is self.ax3
        )
        if is_shift and is_ax3_scatter:
            wav_arr = getattr(event.artist, '_islat_scatter_wavelengths', None)
            ind = getattr(event, 'ind', None)
            if wav_arr is not None and ind is not None and len(ind) > 0:
                lam = float(wav_arr[ind[0]])
                self._trigger_inspection_from_wavelength(lam)
                return

        self._canvas.draw_idle()

    def _trigger_inspection_from_wavelength(self, lam: float) -> None:
        """center the line-inspection panel on *lam* and trigger on_selection.

        The window width is derived (in priority order) from:
        1. The current inspection selection width (``_current_selection``).
        2. The current ax2 x-span (if ax2 has been rendered).
        3. A default of ±1 % of the wavelength (~3 resolution elements at R~100).

        Delegates to ``plot_manager.onselect`` so the full selection pipeline
        runs (current_selection stored, fit result cleared, active view
        notified) exactly as a span-drag from the full-spectrum panel does.
        """
        # Determine half-window
        if self._current_selection is not None:
            sel_xmin, sel_xmax = self._current_selection
            half_w = (sel_xmax - sel_xmin) / 2.0
        else:
            try:
                x0, x1 = self.ax2.get_xlim()
                half_w = (x1 - x0) / 2.0 if (x1 - x0) > 0 else lam * 0.01
            except Exception:
                half_w = lam * 0.01

        # Clamp half_w to a minimum of 0.001 µm to avoid degenerate selections
        half_w = max(half_w, 0.001)

        xmin = lam - half_w
        xmax = lam + half_w

        # Delegate through the plot_manager so the selection state is fully
        # updated (current_selection, toggle_state, fit_result cleared …)
        pm = self._pm
        if hasattr(pm, 'onselect'):
            pm.onselect(xmin, xmax)
        else:
            # Fallback: call on_selection directly on this view
            self.on_selection(xmin, xmax)

        self._canvas.draw_idle()

    def _handle_line_pick_event(self, event: Any) -> Any:
        """Handle line pick events and highlight the selected line.

        When multiple molecules are active, picking a line belonging to a
        comparison molecule will switch ``islat.active_molecule`` to that
        molecule and re-render the population diagram for it.

        Returns the value data dict of the picked line, or *None*.
        """
        import matplotlib.colors as mcolors

        picked_value = None
        picked_mol_name: Optional[str] = None
        picked_artist = event.artist
        grid = self._ensure_grid()

        # Rebuild _active_scatter_collections from active_lines before every
        # pick so that a set_axes / color_by regeneration (which creates a
        # new scatter artist via the cache replay) never leaves stale
        # references that cause missed highlights or data-field updates.
        self._active_scatter_collections = {}
        for _entry in self.active_lines:
            _sc = _entry[2] if len(_entry) > 2 else None
            _val = _entry[3] if len(_entry) > 3 else None
            if _sc is not None and _val is not None:
                _mname = _val.get('molecule_name', '')
                if _mname and _mname not in self._active_scatter_collections:
                    # count all entries that share this scatter object
                    _count = sum(
                        1 for e in self.active_lines
                        if len(e) > 2 and e[2] is _sc
                    )
                    self._active_scatter_collections[_mname] = (_sc, _count)

        # Determine which scatter collection (if any) was clicked
        scatter_point_clicked: Optional[int] = None
        for mol_name, (sc, _) in self._active_scatter_collections.items():
            if picked_artist is sc and hasattr(event, 'ind') and len(event.ind) > 0:
                scatter_point_clicked = event.ind[0]
                break

        # Reset all line/text artists to their per-molecule color and find the pick
        for line, text_obj, scatter, value in self.active_lines:
            mol_color = (value.get('molecule_color') or
                         grid._get_theme_value("active_scatter_line_color", 'green')) if value else 'green'
            if line is not None:
                line.set_color(mol_color)
            if text_obj is not None:
                text_obj.set_color(mol_color)

            is_line_picked = (picked_artist is line or picked_artist is text_obj)
            point_idx = value.get('_scatter_point_index', None) if value else None
            # The scatter index is only meaningful within the picked collection;
            # entries of other molecules may carry the same index.
            is_scatter_picked = (
                scatter_point_clicked is not None
                and scatter is picked_artist
                and point_idx == scatter_point_clicked
            )
            is_picked = is_line_picked or is_scatter_picked

            if is_picked:
                picked_value = value
                picked_mol_name = value.get('molecule_name') if value else None
                if line is not None:
                    line.set_color('orange')
                if text_obj is not None:
                    text_obj.set_color('orange')

        # Reset all scatter collections to their molecule color
        for mol_name, (sc, count) in self._active_scatter_collections.items():
            mol_dict = getattr(self._islat, 'molecules_dict', {})
            mol = mol_dict.get(mol_name) if mol_dict else None
            base_color = (getattr(mol, 'color', None) or
                          grid._get_theme_value("active_scatter_line_color", 'green'))
            if count > 0:
                sc.set_facecolors([mcolors.to_rgba(base_color)] * count)

        # Highlight picked scatter point
        if picked_value is not None:
            picked_scatter_idx = picked_value.get('_scatter_point_index')
            # Find the scatter collection for the picked molecule
            if picked_mol_name and picked_mol_name in self._active_scatter_collections:
                sc, count = self._active_scatter_collections[picked_mol_name]
                mol_dict = getattr(self._islat, 'molecules_dict', {})
                mol = mol_dict.get(picked_mol_name) if mol_dict else None
                base_color = (getattr(mol, 'color', None) or
                              grid._get_theme_value("active_scatter_line_color", 'green'))
                colors = [mcolors.to_rgba(base_color)] * count
                if picked_scatter_idx is not None and picked_scatter_idx < count:
                    colors[picked_scatter_idx] = mcolors.to_rgba('orange')
                sc.set_facecolors(colors)

        # Update the population diagram only when the pick belongs to a
        # *different* molecule than the one currently shown (comparison pick).
        # For same-molecule picks the scatter color was already updated above;
        # calling _render_population_diagram_with_lines here would call
        # generate_plot → clear axes → replay cache, causing scatter doubling
        # and compounding the title font size on every click.
        if picked_mol_name is not None:
            active_mol = getattr(self._islat, 'active_molecule', None)
            active_mol_name = getattr(active_mol, 'name', None)
            if picked_mol_name != active_mol_name:
                mol_dict = getattr(self._islat, 'molecules_dict', {})
                picked_mol = mol_dict.get(picked_mol_name) if mol_dict else None
                if picked_mol is not None:
                    line_data = self._mol_line_data_cache.get(picked_mol_name, [])
                    self._render_population_diagram_with_lines(line_data, picked_mol)

        return picked_value

    def _reapply_selected_line_highlight(self) -> None:
        """Re-apply the orange highlight for ``selected_line`` after a plot regeneration.

        Called after ``set_axes`` / ``color_by`` replays the cache so the
        previously-selected line stays highlighted rather than reverting to
        the base molecule color.
        """
        import matplotlib.colors as mcolors

        if not self.active_lines or self.selected_line is None:
            return

        grid = self._ensure_grid()

        # Rebuild _active_scatter_collections from the freshly-created artists.
        self._active_scatter_collections = {}
        for _entry in self.active_lines:
            _sc = _entry[2] if len(_entry) > 2 else None
            _val = _entry[3] if len(_entry) > 3 else None
            if _sc is not None and _val is not None:
                _mname = _val.get('molecule_name', '')
                if _mname and _mname not in self._active_scatter_collections:
                    _count = sum(
                        1 for e in self.active_lines
                        if len(e) > 2 and e[2] is _sc
                    )
                    self._active_scatter_collections[_mname] = (_sc, _count)

        # Reset all vlines to their base molecule color.
        for line, text_obj, scatter, value in self.active_lines:
            mol_color = (value.get('molecule_color') or
                         grid._get_theme_value("active_scatter_line_color", 'green')) if value else 'green'
            if line is not None:
                line.set_color(mol_color)
            if text_obj is not None:
                text_obj.set_color(mol_color)

        # Reset all scatter collections to their base color.
        for mol_name, (sc, count) in self._active_scatter_collections.items():
            mol_dict = getattr(self._islat, 'molecules_dict', {})
            mol = mol_dict.get(mol_name) if mol_dict else None
            base_color = (getattr(mol, 'color', None) or
                          grid._get_theme_value("active_scatter_line_color", 'green'))
            if count > 0:
                sc.set_facecolors([mcolors.to_rgba(base_color)] * count)

        # Identify the selected entry by matching the wavelength key.
        sel_lam = self.selected_line.get('lam')
        for line, text_obj, scatter, value in self.active_lines:
            if value is None or value.get('lam') != sel_lam:
                continue
            # Highlight the vline.
            if line is not None:
                line.set_color('orange')
            if text_obj is not None:
                text_obj.set_color('orange')
            # Highlight the scatter point.
            mol_name = value.get('molecule_name')
            scatter_idx = value.get('_scatter_point_index')
            if mol_name and mol_name in self._active_scatter_collections:
                sc, count = self._active_scatter_collections[mol_name]
                mol_dict = getattr(self._islat, 'molecules_dict', {})
                mol = mol_dict.get(mol_name) if mol_dict else None
                base_color = (getattr(mol, 'color', None) or
                              grid._get_theme_value("active_scatter_line_color", 'green'))
                colors = [mcolors.to_rgba(base_color)] * count
                if scatter_idx is not None and scatter_idx < count:
                    colors[scatter_idx] = mcolors.to_rgba('orange')
                sc.set_facecolors(colors)
            break

    def _highlight_strongest_line(self) -> None:
        """Find and highlight the strongest line in active_lines."""
        import matplotlib.colors as mcolors

        if not self.active_lines:
            return

        grid = self._ensure_grid()

        # Reset all to per-molecule color
        for line, text_obj, scatter, value in self.active_lines:
            mol_color = (value.get('molecule_color') or
                         grid._get_theme_value("active_scatter_line_color", 'green')) if value else 'green'
            if line is not None:
                line.set_color(mol_color)
            if text_obj is not None:
                text_obj.set_color(mol_color)

        # Find strongest overall line
        highest_intensity = -float('inf')
        strongest = None
        strongest_scatter_idx = None

        for line, text_obj, scatter, value in self.active_lines:
            intensity = value.get('intensity', 0) if value else 0
            if intensity > highest_intensity:
                highest_intensity = intensity
                strongest = (line, text_obj, scatter, value)
                strongest_scatter_idx = value.get('_scatter_point_index', None) if value else None

        # Reset scatter collections to their molecule color
        for mol_name, (sc, count) in self._active_scatter_collections.items():
            mol_dict = getattr(self._islat, 'molecules_dict', {})
            mol = mol_dict.get(mol_name) if mol_dict else None
            base_color = (getattr(mol, 'color', None) or
                          grid._get_theme_value("active_scatter_line_color", 'green'))
            if count > 0:
                colors = [mcolors.to_rgba(base_color)] * count
                sc.set_facecolors(colors)
                sc.set_zorder(1)

        # Highlight the strongest line orange
        if strongest is not None:
            line, text_obj, scatter, value = strongest
            if line is not None:
                line.set_color('orange')
            if text_obj is not None:
                text_obj.set_color('orange')
            self.selected_line = value

            # Highlight in the correct scatter collection
            strongest_mol_name = value.get('molecule_name') if value else None
            if strongest_mol_name and strongest_mol_name in self._active_scatter_collections:
                sc, count = self._active_scatter_collections[strongest_mol_name]
                mol_dict = getattr(self._islat, 'molecules_dict', {})
                mol = mol_dict.get(strongest_mol_name) if mol_dict else None
                base_color = (getattr(mol, 'color', None) or
                              grid._get_theme_value("active_scatter_line_color", 'green'))
                colors = [mcolors.to_rgba(base_color)] * count
                if strongest_scatter_idx is not None and strongest_scatter_idx < count:
                    colors[strongest_scatter_idx] = mcolors.to_rgba('orange')
                sc.set_facecolors(colors)

            if value:
                self._display_line_info(value)

    def _display_line_info(self, value: Dict[str, Any], clear_data_field: bool = True) -> None:
        """Display line information in the GUI data field.

        Formats line properties via :meth:`LineInspectionPlot.get_line_info`
        and enriches with observed / model flux integrals when a selection
        range is active.
        """
        islat = self._islat

        # --- flux integrals in the selected range ----------------------
        data_flux = None
        model_flux = None
        current_selection = self._pm.toggle_state.get("current_selection")
        if current_selection is not None:
            xmin, xmax = current_selection
            err_data = getattr(islat, 'err_data', None)
            line_flux, _ = flux_integral(
                lam=islat.wave_data,
                flux=islat.flux_data,
                lam_min=xmin, lam_max=xmax,
                err=err_data,
            )
            data_flux = line_flux[0] if isinstance(line_flux, (list, tuple)) else line_flux
            active_mol = getattr(islat, 'active_molecule', None)
            if active_mol is not None:
                molecule_wave, molecule_flux_arr = active_mol.get_flux(return_wavelengths=True)
                model_flux, _ = flux_integral(
                    lam=molecule_wave,
                    flux=molecule_flux_arr,
                    lam_min=xmin, lam_max=xmax,
                    err=None,
                )
                _, mol_opacity_arr = active_mol.get_tau(return_wavelengths=True)
                opacity_in_range, _ = flux_integral(
                    lam=molecule_wave, flux=mol_opacity_arr,
                    lam_min=xmin, lam_max=xmax, err=None,
                )

        # --- build line info dict + formatted string -------------------
        if 'formatted_text' in value:
            class _Line2:
                pass
            _l2 = _Line2()
            _l2.lam = value.get('lam')
            _l2.e_up = value.get('e_up')
            _l2.e_low = value.get('e_low')
            _l2.a_stein = value.get('a_stein')
            _l2.g_up = value.get('g_up')
            _l2.g_low = value.get('g_low')
            _l2.lev_up = value.get('up_lev')
            _l2.lev_low = value.get('low_lev')
            info = LineInspectionPlot.get_line_info(
                _l2,
                intensity=value.get('intensity', 0),
                tau=value.get('tau'),
                data_flux_in_range=data_flux,
                model_flux_in_range=model_flux,
                opacity_in_range=opacity_in_range,
                molecule=getattr(islat, 'active_molecule', None),
            )
        else:
            class _Line:
                pass
            _l = _Line()
            _l.lam = value.get('lam')
            _l.e_up = value.get('e_up', value.get('e'))
            _l.e_low = value.get('e_low')
            _l.a_stein = value.get('a_stein', value.get('a'))
            _l.g_up = value.get('g_up', value.get('g'))
            _l.g_low = value.get('g_low')
            _l.lev_up = value.get('up_lev')
            _l.lev_low = value.get('low_lev')
            info = LineInspectionPlot.get_line_info(
                _l,
                intensity=value.get('intensity', value.get('inten', 0)),
                tau=value.get('tau'),
                data_flux_in_range=data_flux,
                model_flux_in_range=model_flux,
                opacity_in_range=opacity_in_range,
                molecule=getattr(islat, 'active_molecule', None),
            )

        info_str = LineInspectionPlot.format_line_info(info)

        # --- push to GUI data-field ------------------------------------
        if (hasattr(islat, 'GUI') and hasattr(islat.GUI, 'data_field') and
                islat.GUI.data_field is not None):
            try:
                if hasattr(islat.GUI.data_field, 'text') and islat.GUI.data_field.text.winfo_exists():
                    islat.GUI.data_field.insert_text(info_str, clear_after=clear_data_field)
            except Exception as e:
                debug_config.warning("three_panel_view", f"Could not update data field: {e}")

    # ------------------------------------------------------------------
    # ToggleMixin hooks
    # ------------------------------------------------------------------
    def _toggle_ready(self) -> bool:
        """ThreePanelView is always ready (it delegates to pre-existing axes)."""
        return True

    def _iter_observed_axes(self):
        """Only the main spectrum panel carries the toggleable observed data."""
        yield self.ax1

    def _iter_y_axes(self):
        """The Y auto-scale / manual-limit policy governs the main spectrum panel."""
        yield self.ax1

    def _iter_toggle_axes(self):
        """Yield the three fixed axes."""
        yield self.ax1
        yield self.ax2
        yield self.ax3

    def _add_atomic_line_artists(self) -> None:
        """Render atomic lines on ax1 using BasePlot helpers."""
        atomic_data = load_atomic_lines()
        if atomic_data.empty:
            return
        BasePlot._plot_atomic_lines(self.ax1, atomic_data, tag="_islat_atomic_line")

    def _remove_atomic_line_artists(self) -> None:
        """Remove previously plotted atomic line artists from ax1."""
        BasePlot._clear_tagged_artists(
            self.ax1, "_islat_atomic_line", lines=True, collections=False, texts=True,
        )

    def _add_saved_line_artists(self) -> None:
        """Render saved lines on ax1 using BasePlot helpers."""
        loaded_lines = getattr(self, 'line_data', None)
        if loaded_lines is None:
            loaded_lines = self._load_saved_line_data()
        if loaded_lines is None or (hasattr(loaded_lines, 'empty') and loaded_lines.empty):
            return
        theme = self._pm.theme
        BasePlot._plot_saved_line_markers(
            self.ax1,
            loaded_lines,
            tag="_islat_saved_line",
            lam_color=theme.get("saved_line_color", theme.get("saved_line_color_one", "red")),
            range_color=theme.get("saved_line_color_two", "orange"),
        )

    def _remove_saved_line_artists(self) -> None:
        """Remove previously plotted saved line artists from ax1."""
        BasePlot._clear_tagged_artists(
            self.ax1, "_islat_saved_line", lines=True, collections=False, texts=False,
        )

    def _load_saved_line_data(self):
        """Load saved-line data from disk."""
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        return ifh.read_line_saves(file_name=self._islat.input_line_list)

    # ------------------------------------------------------------------
    # Override sync_toggle_state for idempotent behaviour
    # ------------------------------------------------------------------
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """Reconcile visual state, skipping redundant add/remove operations.

        The base :class:`ToggleMixin` always removes-then-adds; here we
        check ``_has_tagged_artists`` first for a lighter touch when the
        view is merely being re-activated.
        """
        # Atomic lines
        if toggle_state.get("atomic_lines", False):
            if not self._has_tagged_artists("_islat_atomic_line"):
                self._add_atomic_line_artists()
        else:
            if self._has_tagged_artists("_islat_atomic_line"):
                self._remove_atomic_line_artists()

        # Saved lines
        if toggle_state.get("saved_lines", False):
            if not self._has_tagged_artists("_islat_saved_line"):
                self._set_saved_line_data(self._load_saved_line_data())
                self._add_saved_line_artists()
        else:
            if self._has_tagged_artists("_islat_saved_line"):
                self._remove_saved_line_artists()

        # Summed spectrum
        self._set_summed_visibility(toggle_state.get("summed", True))

        # Legend
        self._set_legend_visibility(toggle_state.get("legend", True))

        # Observed-data visibility and manual Y limits (main panel)
        self._reapply_display_state(toggle_state)

        self.draw()

    # ------------------------------------------------------------------
    # Selection restoration
    # ------------------------------------------------------------------
    def _restore_line_selection(self) -> None:
        """Restore the span selector and line inspection from toggle_state."""
        sel = self._pm.toggle_state.get("current_selection")
        if sel is not None:
            xmin, xmax = sel
            # Rebuild span selector so it exists on the (possibly-cleared) axes
            self._pm.make_span_selector()
            # Restore the visual span extents
            if hasattr(self._pm, 'interaction_handler'):
                span = getattr(self._pm.interaction_handler, 'span_selector', None)
                if span is not None:
                    try:
                        span.set_visible(True)
                        span.extents = (xmin, xmax)
                        span.update()
                    except Exception:
                        pass
            # Re-run the line inspection / population diagram
            self.on_selection(xmin, xmax)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Interaction context
    # ------------------------------------------------------------------
    def build_context_menu(self, event: Any, canvas_widget: Any) -> Any:
        """Return a ``tk.Menu`` appropriate for the right-clicked axes, or ``None``."""
        try:
            import tkinter as tk
        except ImportError:
            return None

        if event.inaxes == self.ax2:
            menu = self._build_line_inspection_menu(canvas_widget)
            if menu is not None:
                self._append_save_figure_item(menu)
            return menu

        if event.inaxes == self.ax3:
            pdp = getattr(self._grid, 'pop_diagram_panel', None) if self._grid is not None else None
            _raw_draw_idle = self._pm.canvas.draw_idle

            def draw_idle():
                """Re-highlight the selected line then refresh the canvas."""
                self._reapply_selected_line_highlight()
                _raw_draw_idle()

            menu = self._build_population_diagram_menu(pdp, canvas_widget, draw_idle)
            if menu is not None:
                self._append_save_figure_item(menu)
            return menu

        if event.inaxes == self.ax1:
            menu = tk.Menu(canvas_widget, tearoff=0)
            pm = self._pm

            def _open_in_full_spectrum():
                try:
                    xmin, xmax = pm.ax1.get_xlim()
                except Exception:
                    xmin, xmax = None, None
                # Mirror the toggle_full_spectrum logic so the back-button works.
                pm._pre_fullspectrum_view_name = "Three Panel"
                pm.switch_view("Full Spectrum")
                if xmin is not None:
                    try:
                        pm._full_spectrum_view._set_display_range(xmin, xmax)
                    except Exception as exc:
                        debug_config.warning(
                            "three_panel_view",
                            f"build_context_menu: _set_display_range failed: {exc}",
                        )

            menu.add_command(
                label="Open in Full Spectrum view",
                command=_open_in_full_spectrum,
            )
            self._append_save_figure_item(menu)
            return menu

        # Unrecognised axes - minimal menu with just Save Figure
        menu = tk.Menu(canvas_widget, tearoff=0)
        self._append_save_figure_item(menu)
        return menu

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas

    def get_figure(self) -> "Figure":
        return self._pm.fig