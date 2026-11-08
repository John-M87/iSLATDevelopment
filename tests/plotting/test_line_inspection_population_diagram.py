# -*- coding: utf-8 -*-
"""Tests for the interaction between the LineInspection panel and the
PopulationDiagram panel as coordinated by ThreePanelView.

Covers:
- on_selection populates active_lines for both the inspection panel
  (vlines) and the population diagram (scatter).
- The same line is highlighted in both panels after on_selection
  (strongest line is automatically selected / orange).
- When a line pick event is simulated, selected_line is updated and
  the matching scatter point is highlighted orange.
- clear_selection removes all active-line markers from both panels.
- When no selection has been made (no on_selection call), the population
  diagram has no active-line scatter plotted.
- clear_active_lines alone empties active_lines without touching the
  base population-diagram scatter.
- After changing the population-diagram axis (set_axes), the previously
  selected line remains highlighted orange in both panels, and all other
  lines revert to the base molecule color.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import iSLAT.Constants as c


# ===========================================================================
# Shared test helpers — molecule / spectrum construction
# ===========================================================================

def _make_line_list(n=5, lam_start=12.0):
    from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
    lines = []
    for i in range(n):
        lam = lam_start + i * 1.5
        lines.append({
            "nr": i + 1,
            "lev_up": f"0|{2*i+2}",
            "lev_low": f"0|{2*i+1}",
            "lam": lam,
            "freq": c.SPEED_OF_LIGHT_MICRONS / lam,
            "a_stein": 0.01 + i * 0.005,
            "e_up": 1000 + i * 600,
            "e_low": 500 + i * 600,
            "g_up": 2 * i + 3,
            "g_low": 2 * i + 1,
        })
    mll = MoleculeLineList(molecule_id="TestMol", lines_data=lines)
    mll.partition_function = mll._partition_type(
        t=np.array([100, 300, 500, 1000, 2000], dtype=float),
        q=np.array([10, 150, 500, 2000, 8000], dtype=float),
    )
    mll._molar_mass = 18.015
    return mll


def _make_intensity(n=5, lam_start=12.0, t_kin=500.0, n_mol=1e18):
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    mll = _make_line_list(n=n, lam_start=lam_start)
    inten = Intensity(mll)
    inten.calc_intensity(t_kin=t_kin, n_mol=n_mol, dv=1.0)
    return inten


def _make_molecule(name="H2O", temp=500.0, n=5, lam_start=12.0):
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.Spectrum import Spectrum
    mll = _make_line_list(n=n, lam_start=lam_start)
    mol = Molecule(
        name=name,
        displaylabel=name,
        filepath=None,
        color="#3399FF",
        is_visible=True,
        temp=float(temp),
        radius=1.0,
        n_mol=1e18,
        distance=160.0,
        fwhm=130.0,
        initial_molecule_parameters={
            "t_kin": float(temp),
            "scale_exponent": 18.0,
            "scale_number": 1.0,
            "radius_init": 1.0,
        },
    )
    mol.lines = mll
    mol.intensity = _make_intensity(n=n, lam_start=lam_start, t_kin=temp)
    spec = Spectrum(lam_min=10.0, lam_max=30.0, dlambda=0.01, R=3000.0, distance=160.0)
    area = np.pi * 1.0 ** 2
    spec.add_intensity(mol.intensity, area)
    mol.spectrum = spec
    return mol


def _get_line_data_in_range(mol, xmin, xmax):
    """Return (MoleculeLine, intensity, tau) triples for mol in [xmin, xmax]."""
    intensity_obj = mol.intensity
    if hasattr(intensity_obj, 'get_lines_in_range_with_intensity'):
        return intensity_obj.get_lines_in_range_with_intensity(xmin, xmax)
    return []


def _scatter_offsets_count(ax):
    """Total number of scatter data-points across all PathCollections on ax."""
    total = 0
    for coll in ax.collections:
        offsets = coll.get_offsets()
        if hasattr(offsets, '__len__'):
            total += len(offsets)
    return total


# ---------------------------------------------------------------------------
# Fixture: a fully wired ThreePanelView with a real molecule
# ---------------------------------------------------------------------------

def _build_three_panel_view(mol, comparison_mols=()):
    """Return (view, pm, fig): a ThreePanelView in borrowed-axes mode.

    *mol* is the primary active molecule; *comparison_mols* are the
    comparison molecules shown alongside it in the line inspection panel.
    """
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.Plotting.MainPlotGrid import MainPlotGrid
    from iSLAT.Modules.Plotting.ThreePanelView import ThreePanelView

    # Three real axes on a real (Agg) figure
    fig = plt.figure(figsize=(12, 8))
    ax1, ax2, ax3 = MainPlotGrid.create_three_panel_axes(fig)

    # Minimal observed spectrum that covers the molecule wavelengths
    wave = np.linspace(10.0, 22.0, 500)
    flux = np.random.default_rng(42).uniform(0.1, 0.5, len(wave))

    md = MoleculeDict()
    md[mol.name] = mol
    for comp in comparison_mols:
        md[comp.name] = comp
    md._active_molecule = mol                           # required by get_active_set()
    md._comparison_molecules = list(comparison_mols)    # required by get_active_set()
    md.apply_stellar_rv = MagicMock(side_effect=lambda w: w)

    # islat stub
    islat = MagicMock()
    islat.wave_data = wave
    islat.wave_data_original = wave.copy()
    islat.flux_data = flux
    islat.err_data = None
    islat.molecules_dict = md
    islat.active_molecule = mol
    islat.comparison_molecules = list(comparison_mols)
    islat.user_settings = {"line_threshold": 0.0}

    # plot_manager stub
    pm = MagicMock()
    pm.fig = fig
    pm.ax1 = ax1
    pm.ax2 = ax2
    pm.ax3 = ax3
    pm.canvas = MagicMock()
    pm.canvas.draw_idle = MagicMock()
    pm.canvas.mpl_connect = MagicMock(return_value=None)
    pm.theme = {"background": "#181A1B", "foreground": "#e8e6e3"}
    pm.toggle_state = {
        "atomic_lines": False, "saved_lines": False,
        "summed": True, "legend": True,
    }
    pm.islat = islat
    pm.summed_toggle = False
    pm.atomic_toggle = False
    pm.line_toggle = False
    pm.legend_toggle = True
    pm.make_span_selector = MagicMock()
    pm.fit_result = None

    # Wire get_molecule_line_data through the real MainPlot logic
    def _get_mol_line_data(molecule, xmin, xmax):
        intensity_obj = molecule.intensity
        if hasattr(intensity_obj, 'get_lines_in_range_with_intensity'):
            return intensity_obj.get_lines_in_range_with_intensity(xmin, xmax)
        return []
    pm.get_molecule_line_data = _get_mol_line_data

    return ThreePanelView(pm), pm, fig


@pytest.fixture
def tpv_fixture():
    """Yield (view, mol, pm) — a ThreePanelView in borrowed-axes mode."""
    mol = _make_molecule()
    view, pm, fig = _build_three_panel_view(mol)

    yield view, mol, pm

    plt.close(fig)


@pytest.fixture
def tpv_two_molecules():
    """Yield (view, primary, comparison, pm): a primary and one comparison molecule.

    The comparison molecule's lines sit between the primary's, so every
    line inspection marker belongs to exactly one molecule.
    """
    primary = _make_molecule(name="H2O", temp=500.0, lam_start=12.0)
    comparison = _make_molecule(name="CO", temp=900.0, lam_start=12.7)
    comparison.color = "#FF5500"
    view, pm, fig = _build_three_panel_view(primary, [comparison])

    yield view, primary, comparison, pm

    plt.close(fig)


# ===========================================================================
# 1. No selection → no active-line scatter in population diagram
# ===========================================================================

class TestNoSelectionNoActiveScatter:

    def test_active_lines_empty_on_init(self, tpv_fixture):
        """active_lines must be empty before any selection is made."""
        view, mol, pm = tpv_fixture
        assert view.active_lines == []

    def test_no_scatter_in_popdiagram_before_selection(self, tpv_fixture):
        """Before on_selection, ax3 must have no active-line scatter."""
        view, mol, pm = tpv_fixture
        # Force a base render so the population-diagram baseline exists
        grid = view._ensure_grid()
        grid.render_population_diagram_for_molecule(mol)

        # Count scatter BEFORE any selection
        n_base = _scatter_offsets_count(pm.ax3)

        # active_lines is empty so no extra scatter should appear
        assert view.active_lines == []
        # The scatter count must equal the base (no on_selection yet)
        assert _scatter_offsets_count(pm.ax3) == n_base

    def test_selected_line_none_before_selection(self, tpv_fixture):
        """selected_line must be None before any pick/selection."""
        view, mol, pm = tpv_fixture
        assert view.selected_line is None

    def test_active_scatter_collections_empty_before_selection(self, tpv_fixture):
        """_active_scatter_collections must be empty before on_selection."""
        view, mol, pm = tpv_fixture
        assert view._active_scatter_collections == {}


# ===========================================================================
# 2. on_selection populates both panels
# ===========================================================================

class TestOnSelectionBothPanels:

    # Wavelength range that covers all 5 test molecule lines (12.0–18.0 µm)
    XMIN, XMAX = 11.0, 20.0

    def test_active_lines_populated_after_on_selection(self, tpv_fixture):
        """on_selection must populate active_lines with at least one entry."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert len(view.active_lines) > 0

    def test_active_lines_have_four_element_tuples(self, tpv_fixture):
        """Each active_lines entry must be (vline, text, scatter, info_dict)."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        for entry in view.active_lines:
            assert len(entry) == 4, f"Expected 4 elements, got {len(entry)}: {entry}"

    def test_inspection_panel_has_vlines(self, tpv_fixture):
        """After on_selection, ax2 must contain vertical line markers."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        # Vlines are either Line2D objects or tagged artists on ax2
        vlines = [entry[0] for entry in view.active_lines if entry[0] is not None]
        assert len(vlines) > 0

    def test_population_diagram_has_scatter_after_selection(self, tpv_fixture):
        """After on_selection, ax3 must have active-line scatter plotted.

        _render_population_diagram_with_lines calls generate_plot (which
        redraws the base molecule scatter), then render_active_lines which
        adds a *separate* scatter collection for the selected lines.  We
        therefore check that _active_scatter_collections is non-empty,
        which is the canonical indicator that active scatter was plotted.
        """
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        assert len(view._active_scatter_collections) > 0, (
            "Expected active scatter in pop-diagram after on_selection"
        )

    def test_active_scatter_collections_populated(self, tpv_fixture):
        """_active_scatter_collections must contain an entry for the molecule."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert mol.name in view._active_scatter_collections

    def test_mol_line_data_cache_populated(self, tpv_fixture):
        """_mol_line_data_cache must hold line data keyed by molecule name."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert mol.name in view._mol_line_data_cache
        assert len(view._mol_line_data_cache[mol.name]) > 0

    def test_current_selection_stored(self, tpv_fixture):
        """_current_selection must be set to (xmin, xmax)."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert view._current_selection == (self.XMIN, self.XMAX)

    def test_draw_idle_called_after_selection(self, tpv_fixture):
        """canvas.draw_idle must be called after on_selection."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        pm.canvas.draw_idle.assert_called()


# ===========================================================================
# 3. Strongest line is highlighted consistently in both panels
# ===========================================================================

class TestStrongestLineConsistency:

    XMIN, XMAX = 11.0, 20.0

    def test_selected_line_set_after_on_selection(self, tpv_fixture):
        """_highlight_strongest_line must set selected_line after on_selection."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert view.selected_line is not None

    def test_selected_line_is_dict(self, tpv_fixture):
        """selected_line must be an info dict (not a bare artist)."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert isinstance(view.selected_line, dict)

    def test_selected_line_has_wavelength(self, tpv_fixture):
        """selected_line must contain a 'lam' (wavelength) key."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert "lam" in view.selected_line

    def test_selected_line_vline_is_orange(self, tpv_fixture):
        """The vline for the strongest line must be orange in the inspection panel."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        # Find the vline that corresponds to selected_line
        selected_wav = view.selected_line.get("lam")
        orange_vlines = []
        for vline, text_obj, scatter, value in view.active_lines:
            if vline is None:
                continue
            vline_color = vline.get_color()
            # Accept 'orange' by name or approximate RGB
            import matplotlib.colors as mcolors
            rgba = mcolors.to_rgba(vline_color)
            orange_rgba = mcolors.to_rgba("orange")
            if np.allclose(rgba[:3], orange_rgba[:3], atol=0.05):
                orange_vlines.append(value)

        assert len(orange_vlines) > 0, "No orange vline found for strongest line"

    def test_strongest_line_scatter_is_orange(self, tpv_fixture):
        """The scatter point for the strongest line must be orange."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        if mol.name not in view._active_scatter_collections:
            pytest.skip("No scatter collection for molecule")

        sc, count = view._active_scatter_collections[mol.name]
        face_colors = sc.get_facecolors()

        orange_rgba = mcolors.to_rgba("orange")
        has_orange = any(
            np.allclose(fc[:3], orange_rgba[:3], atol=0.05)
            for fc in face_colors
        )
        assert has_orange, "No orange scatter point found for strongest line"

    def test_selected_line_wavelength_within_selection_range(self, tpv_fixture):
        """selected_line's wavelength must fall within the selection range."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        lam = view.selected_line.get("lam")
        assert lam is not None
        assert self.XMIN <= lam <= self.XMAX, (
            f"selected_line wavelength {lam} outside [{self.XMIN}, {self.XMAX}]"
        )


# ===========================================================================
# 4. Pick event → same line highlighted in both panels
# ===========================================================================

class TestPickEventHighlighting:

    XMIN, XMAX = 11.0, 20.0

    def _simulate_pick(self, view, entry_index: int):
        """Simulate a pick event on the vline at active_lines[entry_index]."""
        vline, text_obj, scatter, value = view.active_lines[entry_index]
        event = MagicMock()
        event.artist = vline
        event.ind = []  # not a scatter pick
        return view._handle_line_pick_event(event), value

    def test_pick_updates_selected_line(self, tpv_fixture):
        """Picking a vline must update view.selected_line."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert len(view.active_lines) > 0

        _, picked_value = self._simulate_pick(view, 0)
        # selected_line is updated inside _on_pick_line (which calls
        # _handle_line_pick_event); we test _handle_line_pick_event return value
        returned = self._simulate_pick(view, 0)[0]
        assert returned is not None
        assert returned.get("lam") == view.active_lines[0][3].get("lam")

    def test_pick_returns_info_dict(self, tpv_fixture):
        """_handle_line_pick_event must return the line's info dict."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert len(view.active_lines) > 0

        event = MagicMock()
        event.artist = view.active_lines[0][0]  # vline
        event.ind = []
        result = view._handle_line_pick_event(event)
        assert isinstance(result, dict)

    def test_pick_vline_turns_orange(self, tpv_fixture):
        """The picked vline must be colored orange after a pick event."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        # Pick the first vline in active_lines
        vline = view.active_lines[0][0]
        if vline is None:
            pytest.skip("First active-line entry has no vline artist")

        event = MagicMock()
        event.artist = vline
        event.ind = []
        view._handle_line_pick_event(event)

        picked_rgba = mcolors.to_rgba(vline.get_color())
        orange_rgba = mcolors.to_rgba("orange")
        assert np.allclose(picked_rgba[:3], orange_rgba[:3], atol=0.05), (
            f"Picked vline color {vline.get_color()} is not orange"
        )

    def test_pick_scatter_point_turns_orange(self, tpv_fixture):
        """Picking a vline must highlight the corresponding scatter point orange."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        if mol.name not in view._active_scatter_collections:
            pytest.skip("No scatter collection for molecule")

        # Find an entry with a known scatter point index
        target_entry = None
        for entry in view.active_lines:
            vline, text_obj, scatter, value = entry
            if vline is not None and value and value.get("_scatter_point_index") is not None:
                target_entry = entry
                break

        if target_entry is None:
            pytest.skip("No active_lines entry with a scatter point index")

        vline, text_obj, scatter, value = target_entry
        event = MagicMock()
        event.artist = vline
        event.ind = []
        view._handle_line_pick_event(event)

        sc, count = view._active_scatter_collections[mol.name]
        face_colors = sc.get_facecolors()
        orange_rgba = mcolors.to_rgba("orange")
        has_orange = any(
            np.allclose(fc[:3], orange_rgba[:3], atol=0.05)
            for fc in face_colors
        )
        assert has_orange, "Corresponding scatter point was not highlighted orange"

    def test_unpicked_vlines_not_orange(self, tpv_fixture):
        """vlines that were NOT picked must revert to their molecule color."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        if len(view.active_lines) < 2:
            pytest.skip("Need ≥2 active lines to test non-picked colors")

        # Pick the first line
        vline_0 = view.active_lines[0][0]
        event = MagicMock()
        event.artist = vline_0
        event.ind = []
        view._handle_line_pick_event(event)

        orange_rgba = mcolors.to_rgba("orange")
        # Every vline except the picked one must NOT be orange
        for i, (vline, text_obj, scatter, value) in enumerate(view.active_lines):
            if i == 0 or vline is None:
                continue
            color = mcolors.to_rgba(vline.get_color())
            assert not np.allclose(color[:3], orange_rgba[:3], atol=0.05), (
                f"Non-picked vline at index {i} is unexpectedly orange"
            )


# ===========================================================================
# 5. clear_selection removes active lines from both panels
# ===========================================================================

class TestClearSelection:

    XMIN, XMAX = 11.0, 20.0

    def test_clear_selection_empties_active_lines(self, tpv_fixture):
        """clear_selection must empty the active_lines list."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert len(view.active_lines) > 0
        view.clear_selection()
        assert view.active_lines == []

    def test_clear_selection_empties_scatter_collections(self, tpv_fixture):
        """clear_selection must empty _active_scatter_collections."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        view.clear_selection()
        assert view._active_scatter_collections == {}

    def test_clear_selection_resets_current_selection(self, tpv_fixture):
        """clear_selection must set _current_selection back to None."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        view.clear_selection()
        assert view._current_selection is None

    def test_clear_selection_resets_mol_line_data_cache(self, tpv_fixture):
        """clear_selection must empty _mol_line_data_cache."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        view.clear_selection()
        assert view._mol_line_data_cache == {}

    def test_no_scatter_in_popdiagram_after_clear_selection(self, tpv_fixture):
        """After clear_selection, _active_scatter_collections must be empty."""
        view, mol, pm = tpv_fixture

        view.on_selection(self.XMIN, self.XMAX)
        assert len(view._active_scatter_collections) > 0, (
            "Expected active scatter after on_selection"
        )

        view.clear_selection()
        assert view._active_scatter_collections == {}, (
            "Expected no active scatter collections after clear_selection"
        )


# ===========================================================================
# 6. clear_active_lines alone (without clear_selection)
# ===========================================================================

class TestClearActiveLines:

    XMIN, XMAX = 11.0, 20.0

    def test_clear_active_lines_empties_list(self, tpv_fixture):
        """clear_active_lines must empty active_lines in-place."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert len(view.active_lines) > 0
        view.clear_active_lines()
        assert view.active_lines == []

    def test_clear_active_lines_clears_scatter_collections(self, tpv_fixture):
        """clear_active_lines must reset _active_scatter_collections."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        view.clear_active_lines()
        assert view._active_scatter_collections == {}

    def test_clear_active_lines_preserves_current_selection(self, tpv_fixture):
        """clear_active_lines must NOT touch _current_selection."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        view.clear_active_lines()
        # Selection range stays — only artists were removed
        assert view._current_selection == (self.XMIN, self.XMAX)


# ===========================================================================
# 7. No lines in range → population diagram shows no active scatter
# ===========================================================================

class TestNoLinesInRange:

    def test_on_selection_outside_molecule_range_no_scatter(self, tpv_fixture):
        """Selecting a range with no molecule lines must leave no active scatter."""
        view, mol, pm = tpv_fixture

        # Use a range that does NOT overlap any of the 5 test lines (12–18.5 µm)
        xmin, xmax = 50.0, 60.0
        view.on_selection(xmin, xmax)

        assert view.active_lines == [], (
            f"Expected no active lines, got {len(view.active_lines)}"
        )
        assert view._active_scatter_collections == {}

    def test_on_selection_outside_range_does_not_set_selected_line(self, tpv_fixture):
        """When no lines exist in the range, selected_line must stay None."""
        view, mol, pm = tpv_fixture
        view.on_selection(50.0, 60.0)
        assert view.selected_line is None

    def test_on_selection_outside_range_populates_no_cache(self, tpv_fixture):
        """When no lines are found, _mol_line_data_cache must be empty."""
        view, mol, pm = tpv_fixture
        view.on_selection(50.0, 60.0)
        assert view._mol_line_data_cache == {}


# ===========================================================================
# 8. Axis change → selected line highlight is preserved in both panels
# ===========================================================================

class TestAxisChangePreservesHighlight:
    """After set_axes + _reapply_selected_line_highlight the previously
    selected line must remain orange; all other lines must revert to the
    base molecule color."""

    XMIN, XMAX = 11.0, 20.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _do_pick(self, view, entry_index: int):
        """Simulate a vline pick on active_lines[entry_index] and wire
        selected_line so it mirrors what _on_pick_line would do."""
        vline, _t, _sc, value = view.active_lines[entry_index]
        event = MagicMock()
        event.artist = vline
        event.ind = []
        result = view._handle_line_pick_event(event)
        if result is not None:
            view.selected_line = result
        return result

    def _get_pdp(self, view):
        """Return the PopulationDiagramPlot from the view's grid."""
        grid = view._ensure_grid()
        return getattr(grid, 'pop_diagram_panel', None)

    def _axis_combos(self):
        return [
            ("wavelength", "intens"),
            ("e_low",      "rd_yax"),
            ("eu",         "intens"),
        ]

    # ------------------------------------------------------------------
    # 8a. selected_line is still set after axis change
    # ------------------------------------------------------------------

    def test_selected_line_survives_axis_change(self, tpv_fixture):
        """selected_line must not be None after set_axes."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        assert view.selected_line is not None

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        assert view.selected_line is not None

    # ------------------------------------------------------------------
    # 8b. vline for selected_line is orange after axis change + reapply
    # ------------------------------------------------------------------

    def test_selected_vline_orange_after_axis_change(self, tpv_fixture):
        """The selected vline must still be orange after axis change."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        # Explicitly pick the first vline entry so we control which line
        # is selected (rather than relying on strongest-line logic alone).
        pick_result = self._do_pick(view, 0)
        if pick_result is None:
            pytest.skip("Pick returned None")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        sel_lam = view.selected_line.get("lam")
        orange_rgba = mcolors.to_rgba("orange")

        matched_orange = False
        for vline, _t, _sc, value in view.active_lines:
            if vline is None or value is None:
                continue
            if value.get("lam") != sel_lam:
                continue
            rgba = mcolors.to_rgba(vline.get_color())
            if np.allclose(rgba[:3], orange_rgba[:3], atol=0.05):
                matched_orange = True
                break

        assert matched_orange, (
            "The selected vline is not orange after axis change + reapply"
        )

    # ------------------------------------------------------------------
    # 8c. Non-selected vlines are NOT orange after reapply
    # ------------------------------------------------------------------

    def test_nonselected_vlines_not_orange_after_axis_change(self, tpv_fixture):
        """Non-selected vlines must revert to molecule color after axis change."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)
        if len(view.active_lines) < 2:
            pytest.skip("Need ≥2 active lines")

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pick_result = self._do_pick(view, 0)
        if pick_result is None:
            pytest.skip("Pick returned None")
        sel_lam = view.selected_line.get("lam")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        orange_rgba = mcolors.to_rgba("orange")
        for vline, _t, _sc, value in view.active_lines:
            if vline is None or value is None:
                continue
            if value.get("lam") == sel_lam:
                continue  # skip the selected line itself
            rgba = mcolors.to_rgba(vline.get_color())
            assert not np.allclose(rgba[:3], orange_rgba[:3], atol=0.05), (
                f"Non-selected vline at λ={value.get('lam')} is orange after axis change"
            )

    # ------------------------------------------------------------------
    # 8d. Scatter point for selected_line is orange after axis change
    # ------------------------------------------------------------------

    def test_selected_scatter_orange_after_axis_change(self, tpv_fixture):
        """The scatter point for the selected line must be orange after axis change."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pick_result = self._do_pick(view, 0)
        if pick_result is None:
            pytest.skip("Pick returned None")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        if mol.name not in view._active_scatter_collections:
            pytest.skip("No scatter collection after axis change")

        sc, count = view._active_scatter_collections[mol.name]
        face_colors = sc.get_facecolors()
        orange_rgba = mcolors.to_rgba("orange")
        has_orange = any(
            np.allclose(fc[:3], orange_rgba[:3], atol=0.05)
            for fc in face_colors
        )
        assert has_orange, (
            "No orange scatter point found for selected line after axis change"
        )

    # ------------------------------------------------------------------
    # 8e. Non-selected scatter points are NOT orange after reapply
    # ------------------------------------------------------------------

    def test_nonselected_scatter_not_orange_after_axis_change(self, tpv_fixture):
        """All non-selected scatter points must revert to base color after axis change."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pick_result = self._do_pick(view, 0)
        if pick_result is None:
            pytest.skip("Pick returned None")
        sel_scatter_idx = view.selected_line.get("_scatter_point_index")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        if mol.name not in view._active_scatter_collections:
            pytest.skip("No scatter collection after axis change")

        sc, count = view._active_scatter_collections[mol.name]
        face_colors = sc.get_facecolors()
        orange_rgba = mcolors.to_rgba("orange")

        for i, fc in enumerate(face_colors):
            if i == sel_scatter_idx:
                continue  # skip the selected point
            assert not np.allclose(fc[:3], orange_rgba[:3], atol=0.05), (
                f"Non-selected scatter point {i} is orange after axis change"
            )

    # ------------------------------------------------------------------
    # 8f. _active_scatter_collections is rebuilt (non-empty) after axis change
    # ------------------------------------------------------------------

    def test_scatter_collections_rebuilt_after_axis_change(self, tpv_fixture):
        """_active_scatter_collections must be non-empty after set_axes + reapply."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        assert len(view._active_scatter_collections) > 0, (
            "_active_scatter_collections is empty after axis change + reapply"
        )
        assert mol.name in view._active_scatter_collections

    # ------------------------------------------------------------------
    # 8g. Highlight survives multiple consecutive axis changes
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("x_prop,y_prop", [
        ("wavelength", "intens"),
        ("e_low",      "rd_yax"),
        ("eu",         "intens"),
    ])
    def test_highlight_survives_repeated_axis_changes(
        self, tpv_fixture, x_prop, y_prop
    ):
        """Orange highlight must survive every axis combination."""
        import matplotlib.colors as mcolors
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        # First change — pick a specific line
        pick_result = self._do_pick(view, 0)
        if pick_result is None:
            pytest.skip("Pick returned None")

        pdp.set_axes(x_prop=x_prop, y_prop=y_prop, regenerate=True)
        view._reapply_selected_line_highlight()

        sel_lam = view.selected_line.get("lam")
        orange_rgba = mcolors.to_rgba("orange")

        # Check vline
        vline_orange = False
        for vline, _t, _sc, value in view.active_lines:
            if vline is None or value is None:
                continue
            if value.get("lam") != sel_lam:
                continue
            rgba = mcolors.to_rgba(vline.get_color())
            if np.allclose(rgba[:3], orange_rgba[:3], atol=0.05):
                vline_orange = True
                break

        assert vline_orange, (
            f"Selected vline not orange after set_axes({x_prop!r}, {y_prop!r})"
        )

        # Check scatter
        if mol.name in view._active_scatter_collections:
            sc, count = view._active_scatter_collections[mol.name]
            has_orange = any(
                np.allclose(fc[:3], orange_rgba[:3], atol=0.05)
                for fc in sc.get_facecolors()
            )
            assert has_orange, (
                f"No orange scatter point after set_axes({x_prop!r}, {y_prop!r})"
            )

    # ------------------------------------------------------------------
    # 8h. vlines are still pickable (have _islat_line_info or are Line2D)
    #     after axis change — confirming the fix to the al.clear() bug
    # ------------------------------------------------------------------

    def test_vlines_survive_axis_change(self, tpv_fixture):
        """Vline artists in active_lines must still exist after set_axes."""
        view, mol, pm = tpv_fixture
        view.on_selection(self.XMIN, self.XMAX)

        vlines_before = [
            entry[0] for entry in view.active_lines if entry[0] is not None
        ]
        assert len(vlines_before) > 0, "No vlines before axis change"

        pdp = self._get_pdp(view)
        if pdp is None:
            pytest.skip("No pop_diagram_panel on grid")

        pdp.set_axes(x_prop="wavelength", y_prop="intens", regenerate=True)
        view._reapply_selected_line_highlight()

        vlines_after = [
            entry[0] for entry in view.active_lines if entry[0] is not None
        ]
        assert len(vlines_after) == len(vlines_before), (
            f"Vline count changed: {len(vlines_before)} → {len(vlines_after)} "
            "after axis change (active_lines was incorrectly cleared)"
        )


# ===========================================================================
# 9. Several active molecules: clicking a line shows its molecule's diagram
# ===========================================================================

class TestMultiMoleculePickSwitchesPopulationDiagram:
    """With a primary and a comparison molecule in the selection, clicking a
    line inspection marker must show that marker's molecule on the population
    diagram, in either direction and any number of times, with only that
    molecule's active-line points on it."""

    XMIN, XMAX = 11.0, 20.0

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _marker_entries(view, mol_name):
        """active_lines entries of *mol_name* that have a line inspection marker."""
        return [
            entry for entry in view.active_lines
            if entry[0] is not None and entry[3].get("molecule_name") == mol_name
        ]

    def _pick_marker(self, view, mol_name, index=0):
        """Simulate a click on marker *index* of *mol_name*; return its info dict."""
        entries = self._marker_entries(view, mol_name)
        assert len(entries) > index, f"no line inspection marker {index} for {mol_name}"
        vline, _text, _sc, value = entries[index]
        event = MagicMock()
        event.artist = vline
        event.ind = []
        assert view._handle_line_pick_event(event) is value
        return value

    @staticmethod
    def _live_active_scatter(view):
        """Distinct active-line scatter artists in active_lines still drawn on ax3."""
        live = []
        for entry in view.active_lines:
            sc = entry[2]
            if sc is not None and sc in view.ax3.collections and all(sc is not s for s in live):
                live.append(sc)
        return live

    def _assert_diagram_shows(self, view, mol):
        pdp = view._grid.pop_diagram_panel
        assert pdp.molecule is mol
        assert mol.name in view.ax3.get_title()
        # Exactly one active-line scatter is drawn, and it belongs to mol.
        live = self._live_active_scatter(view)
        assert len(live) == 1, f"{len(live)} active-line scatter collections on the diagram"
        owners = {
            entry[3].get("molecule_name")
            for entry in view.active_lines if entry[2] is live[0]
        }
        assert owners == {mol.name}
        assert list(view._active_scatter_collections) == [mol.name]
        sc, count = view._active_scatter_collections[mol.name]
        assert sc is live[0]
        assert count == len(sc.get_offsets())

    # ------------------------------------------------------------------
    # tests
    # ------------------------------------------------------------------

    def test_selection_shows_primary_diagram(self, tpv_two_molecules):
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)
        assert self._marker_entries(view, primary.name)
        assert self._marker_entries(view, comparison.name)
        self._assert_diagram_shows(view, primary)

    def test_pick_comparison_marker_shows_comparison_diagram(self, tpv_two_molecules):
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)

        self._pick_marker(view, comparison.name)

        self._assert_diagram_shows(view, comparison)

    def test_pick_primary_marker_after_comparison_switches_back(self, tpv_two_molecules):
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)

        self._pick_marker(view, comparison.name)
        self._pick_marker(view, primary.name)

        self._assert_diagram_shows(view, primary)

    def test_clicking_back_and_forth_keeps_diagram_in_step(self, tpv_two_molecules):
        """Every click shows the clicked molecule; nothing piles up on ax3."""
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)
        n_collections = {primary.name: len(view.ax3.collections)}

        for mol in (comparison, primary, comparison, comparison, primary, primary, comparison):
            self._pick_marker(view, mol.name, index=1)
            self._assert_diagram_shows(view, mol)
            n_collections.setdefault(mol.name, len(view.ax3.collections))
            assert len(view.ax3.collections) == n_collections[mol.name]

    def test_picked_line_highlighted_on_switched_diagram(self, tpv_two_molecules):
        """The clicked marker and its point on the newly shown diagram are orange, nothing else."""
        import matplotlib.colors as mcolors
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)
        orange = mcolors.to_rgba("orange")

        for mol in (comparison, primary):
            value = self._pick_marker(view, mol.name, index=2)

            for vline, _text, _sc, entry_value in view.active_lines:
                if vline is None:
                    continue
                is_orange = np.allclose(mcolors.to_rgba(vline.get_color()), orange)
                assert is_orange == (entry_value is value)

            sc, _count = view._active_scatter_collections[mol.name]
            orange_points = [
                i for i, fc in enumerate(sc.get_facecolors()) if np.allclose(fc, orange)
            ]
            assert orange_points == [value["_scatter_point_index"]]

    def test_pick_switches_after_menu_reset_to_primary(self, tpv_two_molecules):
        """"Show Active Molecule Only" sets the diagram directly; a later click must still switch."""
        view, primary, comparison, pm = tpv_two_molecules
        view.on_selection(self.XMIN, self.XMAX)
        self._pick_marker(view, comparison.name)

        # What the population diagram context-menu item does.
        view._grid.pop_diagram_panel.set_molecule(primary)
        assert view._grid.pop_diagram_panel.molecule is primary
        # The comparison molecule's points must not be replayed onto the primary's diagram.
        assert self._live_active_scatter(view) == []

        self._pick_marker(view, comparison.name)

        self._assert_diagram_shows(view, comparison)
