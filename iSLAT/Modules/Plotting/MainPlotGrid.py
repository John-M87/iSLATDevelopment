"""
MainPlotGrid - Three-panel composite plot for iSLAT spectral analysis.

Layout (GridSpec 2x2):
    Row 0, spanning both columns : full spectrum with molecule overlays
    Row 1, left column           : line inspection (zoomed region)
    Row 1, right column          : population (rotation) diagram

The grid can be used entirely standalone in a notebook / script, or its axes can be shared with the GUI embedding layer.
"""
from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .BasePlot import BasePlot
from .CompositePlot import CompositePlot
from .LineInspectionPlot import LineInspectionPlot
from .PopulationDiagramPlot import PopulationDiagramPlot
from .SpectrumPanel import SpectrumPanel

try:
    import iSLAT.Constants as c
except ImportError:  # pragma: no cover
    c = None  # constants only needed for population diagram scatter

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

class MainPlotGrid(CompositePlot):
    """
    Three-panel composite plot mirroring the default iSLAT GUI layout.

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array.
    molecules : MoleculeDict, optional
        Molecules to overlay.
    active_molecule : Molecule, optional
        The *active* molecule used for the population diagram and highlighted
        in the inspection panel.
    error_data : np.ndarray, optional
        Flux uncertainties.
    spectrum_range : tuple[float, float], optional
        ``(xmin, xmax)`` wavelength limits for the top spectrum panel.
        When *None* the full data range is used.  Can be updated later
        via :meth:`set_spectrum_range`.
    inspection_range : tuple[float, float], optional
        ``(xmin, xmax)`` for the line inspection sub-plot. If *None* the
        inspection and population panels are left empty until
        :meth:`set_inspection_range` is called.
    inspection_molecules : list[str] or bool, optional
        Controls which molecules appear in the line-inspection panel:

        - *None* or ``False`` (default) -- only the **active molecule** is
          shown.
        - ``True`` -- all visible molecules from the *MoleculeDict* are shown
          (same behaviour as standalone ``LineInspectionPlot``).
        - A list of molecule name strings (e.g. ``["H2O", "CO"]``) -- only
          those molecules are shown.
    line_data : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples for line markers
        in the inspection panel.
    line_list : pd.DataFrame, optional
        Saved-line list for annotations on the main spectrum.
    atomic_lines : pd.DataFrame, optional
        Atomic-line annotations.
    figsize : tuple, optional
        Figure size.  Defaults to ``(15, 8.5)``.
    fig : Figure, optional
        Existing figure for GUI embedding.
    ax_spectrum : Axes, optional
        Pre-existing axes for the top spectrum panel.  When provided
        together with *ax_inspection* and *ax_popdiagram* the grid
        operates in **borrowed-axes mode**: :meth:`generate_plot` will
        *not* call ``fig.clf()`` or create new axes - it renders
        directly onto the supplied axes.  This is the mode used by
        ``ThreePanelView`` for GUI embedding.
    ax_inspection : Axes, optional
        Pre-existing axes for the line-inspection panel.
    ax_popdiagram : Axes, optional
        Pre-existing axes for the population-diagram panel.
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        active_molecule: Optional["Molecule"] = None,
        error_data: Optional[np.ndarray] = None,
        spectrum_range: Optional[Tuple[float, float]] = None,
        inspection_range: Optional[Tuple[float, float]] = None,
        inspection_molecules: Optional[Union[bool, List[str]]] = None,
        line_data: Optional[List[Tuple["MoleculeLine", float, Any]]] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        figsize: Optional[Tuple[float, float]] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        ax_spectrum: Optional[Axes] = None,
        ax_inspection: Optional[Axes] = None,
        ax_popdiagram: Optional[Axes] = None,
        **kwargs,
    ):
        # When all three axes are supplied we are in "borrowed-axes" mode:
        # the caller owns the figure and axes - we just render onto them.
        # Pass the figure extracted from the first supplied axes so
        # BasePlot.fig is set correctly even in borrowed mode.
        borrowed = (
            ax_spectrum is not None
            and ax_inspection is not None
            and ax_popdiagram is not None
        )
        if borrowed:
            kwargs.setdefault("fig", ax_spectrum.figure)

        super().__init__(figsize=figsize or (15, 8.5), **kwargs)
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.error_data = np.asarray(error_data) if error_data is not None else None
        self.molecules = molecules
        self.active_molecule = active_molecule
        self.spectrum_range = spectrum_range
        self.inspection_range = inspection_range
        self.inspection_molecules = inspection_molecules
        self.line_data = line_data
        self.line_list = line_list
        self.atomic_lines = atomic_lines
        # Observer-frame wavelengths for MoleculeDict calls that apply
        # stellar RV internally.  Falls back to wave_data when not given.
        self.wave_data_obs = (np.asarray(wave_data_obs)
                              if wave_data_obs is not None
                              else self.wave_data)

        # Borrowed-axes mode flag
        self._borrowed_axes: bool = borrowed

        # Panel axes - either injected (borrowed) or created in generate_plot
        self.ax_spectrum: Optional[Axes] = ax_spectrum
        self.ax_inspection: Optional[Axes] = ax_inspection
        self.ax_popdiagram: Optional[Axes] = ax_popdiagram

        # Persistent panel objects (created lazily in _render_* methods;
        # reused across calls so artists are mutated rather than rebuilt).
        self.spectrum_panel: Optional[SpectrumPanel] = None
        self.inspection_panel: Optional[LineInspectionPlot] = None
        self.pop_diagram_panel: Optional[PopulationDiagramPlot] = None
        # Tracks the current PopulationDiagramPlot mode so we know when
        # to recreate the panel rather than just mutating its attributes.
        self._pdp_mode: Optional[str] = None  # 'molecule' | 'molecules'

    # ------------------------------------------------------------------
    @staticmethod
    def create_three_panel_axes(fig):
        """Create the standard iSLAT 3-panel layout on *fig*.

        Layout (GridSpec 2 by 2):
            Row 0, spanning both columns : full spectrum overview
            Row 1, left column           : line inspection zoom
            Row 1, right column          : population / rotation diagram

        Returns
        -------
        tuple[Axes, Axes, Axes]
            ``(ax_spectrum, ax_inspection, ax_popdiagram)``
        """
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1.5], figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        return ax1, ax2, ax3

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the three-panel layout.

        In **standalone mode** (no pre-existing axes were injected) the
        figure is cleared, three new axes are created via
        :meth:`create_three_panel_axes`, and the full theme is applied.

        In **borrowed-axes mode** (axes were supplied to ``__init__``)
        the existing axes are simply cleared and re-rendered - the
        figure layout and axes identities are *not* touched, so cached
        references held by ``InteractionHandler`` remain valid.
        """
        if self._borrowed_axes:
            # Borrowed mode - render onto the pre-existing axes.
            # Do NOT call fig.clf() or create_three_panel_axes().
            self._render_spectrum_panel()
            self._render_inspection_panel()
            self._render_population_panel()
            # Apply theme only to the axes (figure chrome is owned by caller)
            self.apply_theme_to_figure()
            return

        # Standalone mode - full figure rebuild.
        self._ensure_figure()
        self.fig.clf()

        # Reset persistent panels and the child-panel registry: the axes
        # objects are about to be recreated, so stale references must go.
        self.child_panels.clear()
        self.spectrum_panel = None
        self.inspection_panel = None
        self._close_pop_diagram_panel()
        self._pdp_mode = None

        (self.ax_spectrum,
         self.ax_inspection,
         self.ax_popdiagram) = self.create_three_panel_axes(self.fig)

        self._render_spectrum_panel()
        self._render_inspection_panel()
        self._render_population_panel()
        self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # CompositePlot abstract-method implementations
    # ------------------------------------------------------------------
    def _build_layout(self) -> tuple:
        """Create the three-panel GridSpec layout on ``self.fig``.

        Returns
        -------
        tuple[Axes, Axes, Axes]
            ``(ax_spectrum, ax_inspection, ax_popdiagram)``
        """
        axes = self.create_three_panel_axes(self.fig)
        self.ax_spectrum, self.ax_inspection, self.ax_popdiagram = axes
        return axes

    def _create_panels(self, layout: tuple) -> None:
        """Create and register the three persistent panel objects.

        .. note::
           :meth:`MainPlotGrid.generate_plot` overrides the default
           :class:`CompositePlot` orchestration and handles rendering
           directly via :meth:`_render_spectrum_panel` etc., so this
           method is only called when the default base-class
           ``generate_plot`` is explicitly invoked.
        """
        self._render_spectrum_panel()
        self._render_inspection_panel()
        self._render_population_panel()

    def refresh(self) -> None:
        """Full refresh of all three panels.

        Each panel's data attributes are updated from ``self`` before
        re-rendering, so this correctly reflects the latest molecule
        parameters, spectrum data, etc.
        """
        if self.ax_spectrum is not None:
            self._render_spectrum_panel()
        if self.ax_inspection is not None:
            self._render_inspection_panel()
        if self.ax_popdiagram is not None:
            self._render_population_panel()
        self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------
    def _render_spectrum_panel(self) -> None:
        ax = self.ax_spectrum

        # In borrowed-axes mode, preserve user zoom / pan limits across
        # re-renders so the display doesn't jump.  The sentinel (0, 1)
        # means "axes have never been rendered into yet" - skip restore.
        _restore_lims = False
        if self._borrowed_axes:
            _prev_xlim = ax.get_xlim()
            _prev_ylim = ax.get_ylim()
            if _prev_xlim != (0.0, 1.0) and _prev_ylim != (0.0, 1.0):
                _restore_lims = True

        if self.wave_data is None or len(self.wave_data) == 0:
            ax.clear()
            ax.set_title("No spectrum data loaded")
            return

        # Resolve wavelength range for this render.
        # Priority: explicit spectrum_range > global_wavelength_range from molecules > data bounds.
        if self.spectrum_range is not None:
            xmin_sp, xmax_sp = self.spectrum_range
        else:
            xmin_sp = float(np.nanmin(self.wave_data))
            xmax_sp = float(np.nanmax(self.wave_data))
            # Expand to the molecules' global_wavelength_range when the user
            # has extended min/max wave beyond the observed data.
            if self.molecules is not None:
                mol_range = getattr(self.molecules, '_global_wavelength_range', None)
                if mol_range is not None:
                    xmin_sp = min(xmin_sp, float(mol_range[0]))
                    xmax_sp = max(xmax_sp, float(mol_range[1]))

        # Build the molecule cache (list of pre-computed spectra) so that
        # SpectrumPanel can slice into it without recomputing molecule data.
        mol_cache: List[tuple] = []
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            # Resolve the "match spectral sampling" (pixel-res) toggle the
            # same way FullSpectrumPlot._build_mol_cache() does, so that
            # individual molecule lines honor it just like the summed fill
            # (get_summed_flux already applies this internally).
            use_interp = False
            target_wave = None
            ref_wave = self.wave_data_obs if self.wave_data_obs is not None else self.wave_data
            if ref_wave is not None and hasattr(self.molecules, 'get_matched_sampling_wavelengths'):
                use_interp, target_wave = self.molecules.get_matched_sampling_wavelengths(ref_wave)
                if not use_interp:
                    target_wave = None

            for mol in self.molecules.get_visible_molecules(return_objects=True):
                try:
                    lam, flux = self.get_molecule_spectrum_data(
                        mol, self.wave_data,
                        interpolate_to_input=use_interp,
                        target_wavelengths=target_wave,
                        extend_to_range=True,
                    )
                    if lam is not None and flux is not None and len(flux) > 0:
                        mol_cache.append((
                            np.asarray(lam),
                            np.asarray(flux),
                            self.get_molecule_color(mol),
                            self.get_molecule_display_name(mol),
                            getattr(mol, "name", "unknown"),
                        ))
                except Exception:
                    pass
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data_obs, visible_only=True,
                    extend_to_global_range=True,
                )
            except Exception:
                pass

        # Create the SpectrumPanel on the first call; on subsequent calls
        # mutate its data attributes so no artists are rebuilt from scratch.
        if self.spectrum_panel is None:
            self.spectrum_panel = SpectrumPanel(
                wave_data=self.wave_data,
                flux_data=self.flux_data,
                xmin=xmin_sp,
                xmax=xmax_sp,
                error_data=self.error_data,
                molecules=self.molecules,
                mol_cache=mol_cache,
                summed_wave=summed_wave,
                summed_flux=summed_flux,
                line_list=self.line_list,
                atomic_lines=self.atomic_lines,
                wave_data_obs=self.wave_data_obs,
                ax=ax,
                theme=self.theme,
            )
            self._register_panel("spectrum", self.spectrum_panel)
        else:
            self.spectrum_panel.wave_data = self.wave_data
            self.spectrum_panel.flux_data = self.flux_data
            self.spectrum_panel.error_data = self.error_data
            self.spectrum_panel.molecules = self.molecules
            self.spectrum_panel.mol_cache = mol_cache
            self.spectrum_panel.summed_wave = summed_wave
            self.spectrum_panel.summed_flux = summed_flux
            self.spectrum_panel.line_list = self.line_list
            self.spectrum_panel.atomic_lines = self.atomic_lines
            self.spectrum_panel.wave_data_obs = self.wave_data_obs
            self.spectrum_panel.theme = self.theme
            self.spectrum_panel.xmin = xmin_sp
            self.spectrum_panel.xmax = xmax_sp
            # Ensure the panel still targets the correct axes object
            # (may change after a standalone fig.clf() + create_three_panel_axes).
            self.spectrum_panel._external_ax = ax

        # Delegate core rendering to the SpectrumPanel (clears ax internally).
        self.spectrum_panel.generate_plot()

        # Post-render: annotations, axis labels, title and legend are NOT
        # handled by SpectrumPanel (it defers them to _post_render_cell in
        # stacked-panel layouts), so we apply them here.
        xr = (xmin_sp, xmax_sp)
        ymin = float(ax.get_ylim()[0]) if ax.get_ylim()[0] != 0 else -0.005
        ymax = float(ax.get_ylim()[1])

        if self.line_list is not None:
            self._plot_line_annotations(ax, self.line_list, xr, ymin, ymax)
        if self.atomic_lines is not None:
            self._plot_atomic_lines(ax, self.atomic_lines, xr=xr)

        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Flux density (Jy)")
        ax.set_title("Full Spectrum with Line Inspection")
        self._update_legend(ax)

        # Restore user zoom / pan limits in borrowed-axes mode
        if _restore_lims:
            ax.set_xlim(_prev_xlim)
            ax.set_ylim(_prev_ylim)

    # ------------------------------------------------------------------
    def _render_inspection_panel(self) -> None:
        ax = self.ax_inspection

        if self.inspection_range is None:
            ax.clear()
            ax.set_title("Line Inspection -- select a range")
            return

        xmin, xmax = self.inspection_range

        # Resolve which molecule(s) to show in the inspection panel.
        # Default: only the active molecule.
        lip_molecule = None
        lip_molecules = None

        if self.inspection_molecules is True:
            # Show all visible molecules
            lip_molecules = self.molecules
        elif isinstance(self.inspection_molecules, (list, tuple)):
            # Show a specific subset by name - build a lightweight copy
            if self.molecules is not None:
                from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict as _MD
                lip_molecules = _MD.__new__(_MD)
                dict.__init__(lip_molecules)
                # Initialise the internal caches that MoleculeDict.__init__
                # normally creates - without these, get_visible_molecules()
                # and other helpers will raise AttributeError.
                lip_molecules._visible_molecules = set()
                lip_molecules._summed_flux_cache = {}
                lip_molecules._global_parms = getattr(
                    self.molecules, "_global_parms", {}
                )
                for name in self.inspection_molecules:
                    if name in self.molecules:
                        lip_molecules[name] = self.molecules[name]
            if not lip_molecules:
                lip_molecule = self.active_molecule
                lip_molecules = None
        else:
            # Default (None / False): only the active molecule
            lip_molecule = self.active_molecule

        # Create the LineInspectionPlot on the first call; on subsequent
        # calls mutate its attributes so the same panel object is reused.
        if self.inspection_panel is None:
            self.inspection_panel = LineInspectionPlot(
                wave_data=self.wave_data,
                flux_data=self.flux_data,
                xmin=xmin,
                xmax=xmax,
                error_data=self.error_data,
                molecule=lip_molecule,
                molecules=lip_molecules,
                line_data=self.line_data,
                wave_data_obs=self.wave_data_obs,
                ax=ax,
                theme=self.theme,
            )
            self._register_panel("inspection", self.inspection_panel)
        else:
            self.inspection_panel.wave_data = self.wave_data
            self.inspection_panel.flux_data = self.flux_data
            self.inspection_panel.xmin = xmin
            self.inspection_panel.xmax = xmax
            self.inspection_panel.error_data = self.error_data
            self.inspection_panel.molecule = lip_molecule
            self.inspection_panel.molecules = lip_molecules
            self.inspection_panel.line_data = self.line_data
            self.inspection_panel.wave_data_obs = (
                np.asarray(self.wave_data_obs)
                if self.wave_data_obs is not None
                else self.wave_data
            )
            self.inspection_panel.theme = self.theme
            # Ensure the panel targets the correct axes object.
            self.inspection_panel._external_ax = ax

        self.inspection_panel.generate_plot()

    # ------------------------------------------------------------------
    def _close_pop_diagram_panel(self) -> None:
        """Discard the current population panel, dropping its MoleculeDict callbacks."""
        panel = self.pop_diagram_panel
        self.pop_diagram_panel = None
        if panel is None:
            return
        try:
            panel.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _render_population_panel(self) -> None:
        ax = self.ax_popdiagram

        if self.active_molecule is None and self.molecules is None:
            ax.clear()
            ax.set_title("Population Diagram -- no molecule selected")
            return

        # Determine the rendering mode and the molecule(s) to display.
        # Default: always show only the active molecule.
        # Fall back to all visible molecules only when there is no active
        # molecule to display.
        if self.active_molecule is not None:
            new_mode = "molecule"
            pdp_molecule = self.active_molecule
            pdp_molecules = None
        elif self.molecules is not None and len(self.molecules) > 0:
            new_mode = "molecules"
            pdp_molecule = None
            pdp_molecules = self.molecules
        else:
            ax.clear()
            ax.set_title("Population Diagram -- no molecule selected")
            return

        # Recreate the panel only when the mode switches (molecule ↔ molecules),
        # since PopulationDiagramPlot enforces mutual exclusivity at construction.
        # Otherwise reuse the existing instance by mutating its attributes.
        if self.pop_diagram_panel is None or self._pdp_mode != new_mode:
            self._close_pop_diagram_panel()
            self.pop_diagram_panel = PopulationDiagramPlot(
                molecule=pdp_molecule,
                molecules=pdp_molecules,
                highlight_lines=self.line_data,
                ax=ax,
                theme=self.theme,
            )
            self._pdp_mode = new_mode
            self._register_panel("popdiagram", self.pop_diagram_panel)
        else:
            if new_mode == "molecule":
                self.pop_diagram_panel.molecule = pdp_molecule
            else:
                self.pop_diagram_panel.molecules = pdp_molecules
            self.pop_diagram_panel.highlight_lines = self.line_data
            self.pop_diagram_panel.theme = self.theme
            # Ensure the panel targets the correct axes object.
            self.pop_diagram_panel._external_ax = ax

        self.pop_diagram_panel.generate_plot()

    # ------------------------------------------------------------------
    # Public update helpers
    # ------------------------------------------------------------------
    def set_inspection_range(self, xmin: float, xmax: float) -> None:
        """Update the inspection range and refresh the bottom panels."""
        self.inspection_range = (xmin, xmax)
        if self.ax_inspection is not None:
            self._render_inspection_panel()
        if self.ax_popdiagram is not None:
            self._render_population_panel()

    def set_spectrum_range(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
    ) -> None:
        """Set the wavelength range for the top spectrum panel.

        Pass *None* for both to reset to the full data range.
        """
        if xmin is None and xmax is None:
            self.spectrum_range = None
        else:
            lo = xmin if xmin is not None else float(np.nanmin(self.wave_data))
            hi = xmax if xmax is not None else float(np.nanmax(self.wave_data))
            self.spectrum_range = (lo, hi)
        if self.ax_spectrum is not None:
            self._render_spectrum_panel()

    def set_inspection_molecules(
        self, molecules: Optional[Union[bool, List[str]]] = None,
    ) -> None:
        """Control which molecules appear in the inspection panel.

        Parameters
        ----------
        molecules : None | False | True | list[str]
            *None* / *False* -- only the active molecule (default).
            *True* -- all visible molecules.
            A list of name strings -- only those molecules.
        """
        self.inspection_molecules = molecules
        if self.ax_inspection is not None:
            self._render_inspection_panel()

    # ==================================================================
    # Data update helpers (for GUI embedding)
    # ==================================================================
    def update_data(
        self,
        wave_data: Optional[np.ndarray] = None,
        flux_data: Optional[np.ndarray] = None,
        molecules: Optional["MoleculeDict"] = None,
        active_molecule: Optional["Molecule"] = None,
        error_data: Optional[np.ndarray] = None,
        wave_data_obs: Optional[np.ndarray] = None,
    ) -> None:
        """Replace data arrays in-place and refresh all panels.

        Only non-*None* arguments are updated; the rest keep their
        current values.  After updating, all three panels are
        re-rendered via :meth:`refresh`.

        Parameters
        ----------
        wave_data, flux_data, molecules, active_molecule, error_data,
        wave_data_obs :
            Same meaning as the constructor arguments.
        """
        if wave_data is not None:
            self.wave_data = np.asarray(wave_data)
        if flux_data is not None:
            self.flux_data = np.asarray(flux_data)
        if molecules is not None:
            self.molecules = molecules
        if active_molecule is not None:
            self.active_molecule = active_molecule
        if error_data is not None:
            self.error_data = np.asarray(error_data)
        if wave_data_obs is not None:
            self.wave_data_obs = np.asarray(wave_data_obs)
        elif wave_data is not None:
            # Keep observer-frame wavelengths in sync when only
            # wave_data was updated (caller did not pass wave_data_obs).
            self.wave_data_obs = self.wave_data

        self.refresh()

    # ------------------------------------------------------------------
    # Fast molecule visibility helpers (no full rebuild)
    # ------------------------------------------------------------------
    def set_molecule_visibility(
        self,
        molecule_name: str,
        visible: bool,
        ax: Optional[Axes] = None,
    ) -> bool:
        """Toggle visibility of an already-plotted molecule on *ax*.

        Searches for ``Line2D`` artists tagged with ``_molecule_name``
        and sets their visibility.  Returns *True* if any artists were
        found (i.e. the fast-path succeeded).

        Parameters
        ----------
        molecule_name : str
            Internal molecule name (``molecule.name``).
        visible : bool
            Target visibility state.
        ax : Axes, optional
            Axes to search.  Defaults to ``ax_spectrum``.
        """
        target = ax if ax is not None else self.ax_spectrum
        if target is None:
            return False
        found = False
        for line in target.lines:
            if getattr(line, "_molecule_name", None) == molecule_name:
                line.set_visible(visible)
                found = True
        return found

    def remove_molecule_lines(
        self,
        molecule_name: str,
        ax: Optional[Axes] = None,
    ) -> None:
        """Remove all ``Line2D`` artists for *molecule_name* from *ax*.

        Parameters
        ----------
        molecule_name : str
            Internal molecule name.
        ax : Axes, optional
            Defaults to ``ax_spectrum``.
        """
        target = ax if ax is not None else self.ax_spectrum
        if target is None:
            return
        for line in target.lines[:]:
            if getattr(line, "_molecule_name", None) == molecule_name:
                line.remove()

    def clear_all_molecule_lines(self, ax: Optional[Axes] = None) -> None:
        """Remove every molecule-tagged ``Line2D`` from *ax*.

        Parameters
        ----------
        ax : Axes, optional
            Defaults to ``ax_spectrum``.
        """
        target = ax if ax is not None else self.ax_spectrum
        if target is None:
            return
        for line in target.lines[:]:
            if getattr(line, "_molecule_name", None) is not None:
                line.remove()

    @property
    def _pdp(self) -> Optional["PopulationDiagramPlot"]:
        """Alias for :attr:`pop_diagram_panel` (used by :class:`InteractionHandler`)."""
        return self.pop_diagram_panel

    def handle_molecule_visibility_change(
        self,
        molecule_name: str,
        is_visible: bool,
        *,
        force_rerender: bool = False,
    ) -> None:
        """Handle a single molecule's visibility toggle on the spectrum panel.

        1. Fast-toggle existing artists (or create from scratch).
        2. Recompute the summed spectrum fill.
        3. Rebuild the legend.

        Parameters
        ----------
        molecule_name : str
            The molecule whose visibility changed.
        is_visible : bool
            New visibility state.
        force_rerender : bool
            When *True*, stale artists are destroyed and recreated from
            current parameters (e.g. after a parameter change while the
            molecule was hidden).
        """
        if self.ax_spectrum is None or self.molecules is None:
            return
        if molecule_name not in self.molecules:
            return

        molecule = self.molecules[molecule_name]
        ax = self.ax_spectrum

        # Destroy stale artists when forcing a re-render
        if force_rerender and is_visible:
            self.remove_molecule_lines(molecule_name, ax)

        # Try fast visibility toggle
        toggled = self.set_molecule_visibility(molecule_name, is_visible, ax)

        # If turning ON but no artists exist, plot from scratch
        if is_visible and not toggled:
            self.spectrum_panel._plot_molecule_spectrum(ax, molecule, self.wave_data)

        # Recompute summed spectrum
        self._update_summed_spectrum()

        # Remove lines for molecules that are no longer visible
        if self.molecules is not None:
            visible_names = {
                getattr(m, "name", None)
                for m in self.molecules.get_visible_molecules(return_objects=True)
            }
            for line in ax.lines[:]:
                name = getattr(line, "_molecule_name", None)
                if name is not None and name not in visible_names:
                    line.remove()

        # Rebuild legend
        self._update_legend(ax)

    def _update_summed_spectrum(self) -> None:
        """Recompute and redraw the summed-spectrum fill on ``ax_spectrum``.

        Removes existing ``_islat_summed`` collections, recomputes the
        visible-molecule sum, and re-plots if any visible molecules exist.
        """
        ax = self.ax_spectrum
        if ax is None:
            return

        # Remove old summed fill via BasePlot static helper
        from .BasePlot import BasePlot
        BasePlot._clear_tagged_artists(ax, "_islat_summed", lines=False)

        if self.molecules is None:
            return
        try:
            s_wave, s_flux = self.molecules.get_summed_flux(
                self.wave_data_obs, visible_only=True,
                extend_to_global_range=True,
            )
            if s_wave is not None and len(s_flux) > 0 and np.any(s_flux > 0):
                # Delegate to SpectrumPanel which owns _plot_summed_spectrum
                if self.spectrum_panel is not None:
                    self.spectrum_panel._plot_summed_spectrum(
                        ax, s_wave, s_flux, deduplicate=True,
                    )
                else:
                    # Fallback: minimal inline fill (spectrum_panel not yet created)
                    fill_color = self._get_theme_value("summed_spectra_color", "lightgray")
                    fill = ax.fill_between(
                        s_wave, 0, s_flux,
                        color=fill_color, alpha=1.0, label="Sum",
                        zorder=self._get_theme_value("zorder_summed", 1),
                    )
                    fill._islat_summed = True
        except Exception:
            pass

    # ==================================================================
    # Active-line marker rendering (for GUI interaction layer)
    # ==================================================================
    def render_active_line_markers(
        self,
        line_data: List[Tuple["MoleculeLine", float, Any]],
        active_lines: List[Any],
        max_y: float,
        *,
        threshold: float = 0.0,
        color: str = "green",
        molecule_name: str = "",
        molecule_color: str = "",
    ) -> None:
        """Plot vertical line markers in the inspection panel.

        Delegates to :meth:`LineInspectionPlot.render_active_lines` via
        the persistent :attr:`inspection_panel` instance.
        """
        if self.inspection_panel is None or self.ax_inspection is None:
            return
        self.inspection_panel.render_active_lines(
            line_data, active_lines,
            max_y=max_y,
            threshold=threshold,
            color=color,
            molecule_name=molecule_name,
            molecule_color=molecule_color,
        )

    def render_active_line_scatter(
        self,
        line_data: List[Tuple["MoleculeLine", float, Any]],
        active_lines: List[Any],
        molecule: "Molecule",
        *,
        threshold: float = 0.0,
        color: str = "green",
    ) -> Any:
        """Plot scatter points for active lines on the population diagram.

        Delegates to :meth:`PopulationDiagramPlot.render_active_lines` via the persistent :attr:`pop_diagram_panel` instance.

        Returns the :class:`PathCollection` scatter artist, or *None*.
        """
        if self.pop_diagram_panel is None or self.ax_popdiagram is None:
            return None
        return self.pop_diagram_panel.render_active_lines(
            line_data, active_lines,
            molecule=molecule,
            threshold=threshold,
            color=color,
        )

    # ==================================================================
    # GUI line-inspection & population-diagram rendering
    # ==================================================================
    def render_line_inspection_plot(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        active_molecule: Optional["Molecule"] = None,
        fit_result: Optional[Any] = None,
        molecules: Optional["MoleculeDict"] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        legend_visible: bool = True,
        additional_molecules: Optional[List["Molecule"]] = None,
    ) -> None:
        """Render the line-inspection panel (ax_inspection) with observed
        data and the active molecule model.

        Reuses an internal :class:`LineInspectionPlot` instance and
        optionally overlays fit results.  When *additional_molecules* are
        provided their model spectra are overlaid after the primary
        molecule (each in its own color).

        Parameters
        ----------
        wave_data, flux_data : np.ndarray
            Full observed spectrum arrays.
        xmin, xmax : float
            Wavelength bounds for the zoomed inspection region.
        active_molecule : Molecule, optional
            Active molecule to show in the panel.
        fit_result : tuple, optional
            ``(gauss_fit, fitted_wave, fitted_flux)`` to overlay.
        molecules : MoleculeDict, optional
            For matched-spectral-sampling look-ups.
        wave_data_obs : np.ndarray, optional
            Observer-frame wavelengths.
        legend_visible : bool
            Whether the legend should be shown.
        additional_molecules : list[Molecule], optional
            Comparison molecules whose model spectra are also overlaid.
        """
        ax = self.ax_inspection
        if ax is None:
            return
        ax.clear()

        if xmin is None or xmax is None or (xmax - xmin) < 0.0001:
            return

        if molecules is None:
            molecules = self.molecules
        if wave_data_obs is None:
            wave_data_obs = self.wave_data_obs

        # Reuse / create a LineInspectionPlot delegate
        if self.inspection_panel is None:
            self.inspection_panel = LineInspectionPlot(
                wave_data=wave_data,
                flux_data=flux_data,
                xmin=xmin,
                xmax=xmax,
                molecule=active_molecule,
                molecules=molecules,
                wave_data_obs=wave_data_obs,
                ax=ax,
                fig=self.fig,
                theme=self.theme,
                render_all_visible=False,
            )
        else:
            self.inspection_panel.wave_data = wave_data
            self.inspection_panel.flux_data = flux_data
            self.inspection_panel.xmin = xmin
            self.inspection_panel.xmax = xmax
            self.inspection_panel.molecule = active_molecule
            self.inspection_panel.molecules = molecules
            self.inspection_panel.wave_data_obs = (
                np.asarray(wave_data_obs) if wave_data_obs is not None
                else wave_data
            )
            self.inspection_panel.render_all_visible = False
            self.inspection_panel.theme = self.theme
        self.inspection_panel.generate_plot()

        # Overlay additional (comparison) molecules in their own colors
        if additional_molecules:
            use_interp = False
            target_wave = None
            if molecules is not None and hasattr(molecules, 'get_matched_sampling_wavelengths'):
                use_interp, target_wave = molecules.get_matched_sampling_wavelengths(
                    wave_data_obs if wave_data_obs is not None else wave_data
                )
                if not use_interp:
                    target_wave = None
            for mol in additional_molecules:
                if mol is not None:
                    self.inspection_panel._overlay_molecule(ax, mol, use_interp, target_wave)
            # Rebuild legend now that all molecules have been plotted
            self.inspection_panel._update_legend(ax)

        # Overlay fit results if present
        if fit_result is not None:
            current_ylim = ax.get_ylim()
            max_y = current_ylim[1] / 1.1 if current_ylim[1] > 0 else 0.15
            self.inspection_panel.render_fit_results(
                ax, fit_result, xmin, xmax,
                legend_visible=legend_visible,
            )

    def render_population_diagram_for_molecule(
        self,
        molecule: "Molecule",
        force_redraw: bool = False,
    ) -> None:
        """Render the base population diagram on ``ax_popdiagram``.

        Includes a simple cache so repeated calls with the same molecule
        and unchanged parameters are skipped.

        Parameters
        ----------
        molecule : Molecule
            The molecule to visualise.
        force_redraw : bool
            When *True* the cache is bypassed.
        """
        ax = self.ax_popdiagram
        if ax is None:
            return

        # Cache check - skip re-render if molecule and parameters unchanged.
        current_hash = None
        if molecule is not None and hasattr(molecule, '_compute_intensity_hash'):
            current_hash = (molecule.name, molecule._compute_intensity_hash())

        if not hasattr(self, '_pdp_molecule'):
            self._pdp_molecule = None
            self._pdp_cache_key = None

        if (not force_redraw
                and self._pdp_molecule is molecule
                and current_hash is not None
                and self._pdp_cache_key == current_hash):
            return

        self._pdp_molecule = molecule
        self._pdp_cache_key = current_hash

        try:
            if self.pop_diagram_panel is None or self._pdp_mode != "molecule":
                self._close_pop_diagram_panel()
                self.pop_diagram_panel = PopulationDiagramPlot(
                    molecule=molecule,
                    ax=ax,
                    fig=self.fig,
                    theme=self.theme,
                )
                self._pdp_mode = "molecule"
            else:
                self.pop_diagram_panel.molecule = molecule
                self.pop_diagram_panel.theme = self.theme
            self.pop_diagram_panel.generate_plot()
        except Exception as e:
            ax.clear()
            mol_label = (
                BasePlot.get_molecule_display_name(molecule) if molecule
                else "Unknown"
            )
            fg = self._get_theme_value("foreground", "black")
            ax.set_title(f"{mol_label} - Error in calculation", color=fg)

    def render_fit_results(
        self,
        fit_result: Any,
        xmin: float,
        xmax: float,
        max_y: float,
        *,
        user_settings: Optional[Dict[str, Any]] = None,
        legend_visible: bool = True,
    ) -> None:
        """Overlay Gaussian-fit results on the line-inspection panel.

        Delegates to :meth:`SpectrumPanel.render_fit_results` via the
        persistent :attr:`inspection_panel` instance.
        """
        ax = self.ax_inspection
        if ax is None:
            return
        if self.inspection_panel is not None:
            self.inspection_panel.render_fit_results(
                ax, fit_result, xmin, xmax,
                user_settings=user_settings,
                legend_visible=legend_visible,
            )
        else:
            # Fallback: delegate to SpectrumPanel static helpers directly
            from .SpectrumPanel import SpectrumPanel as _SP
            _SP._clear_old_fit_results(ax, xmin, xmax)

    def clear_active_lines(self, active_lines: List[Any]) -> None:
        """Remove all active-line artists (vlines, text, scatter) and clear the list.

        Overrides :meth:`BasePlot.clear_active_lines` to handle all three
        artist types (vline, text, scatter) stored across both the
        inspection and population-diagram panels.

        Parameters
        ----------
        active_lines : list
            Mutable list of ``[line_artist, text_artist, scatter_artist, info_dict]``
            entries.  The list is cleared in-place.
        """
        seen_scatter = set()
        for entry in active_lines:
            vline, text = entry[0], entry[1]
            sc = entry[2] if len(entry) > 2 else None
            for artist in (vline, text):
                if artist is not None and getattr(artist, 'axes', None) is not None:
                    try:
                        artist.remove()
                    except (ValueError, AttributeError):
                        pass
            if sc is not None and id(sc) not in seen_scatter:
                seen_scatter.add(id(sc))
                if getattr(sc, 'axes', None) is not None:
                    try:
                        sc.remove()
                    except (ValueError, AttributeError):
                        pass
        active_lines.clear()

        # Clear the pop-diagram scatter cache so the next generate_plot() call
        # (e.g. from _render_population_diagram_base) does NOT re-render the
        # active-line scatter that was just removed.
        if self.pop_diagram_panel is not None:
            self.pop_diagram_panel._active_lines_cache = None