# -*- coding: utf-8 -*-
"""Species-scoped quantum-field options shared by the line-list filter window
and the population-diagram axis / color-by dialogs."""

import matplotlib
matplotlib.use("Agg")

from types import SimpleNamespace

import pytest

import iSLAT.Modules.DataTypes.Intensity  # noqa: F401  (registers physics axis entries)
import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401  (registers quantum-field entries)
from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry
from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry


def _quantum_fields(entries):
    return {e.quantum_field for e in entries if e.quantum_field is not None}


class TestQuantumStateRegistryFields:
    def test_registered_species_declares_fields(self):
        names = {f.name for f in QuantumStateRegistry.get_fields("H2O")}
        assert "J" in names

    def test_unknown_species_declares_nothing(self):
        assert QuantumStateRegistry.get_fields("NotAMolecule") == ()
        assert QuantumStateRegistry.get_fields(None) == ()

    def test_field_names_is_union(self):
        h2o = {f.name for f in QuantumStateRegistry.get_fields("H2O")}
        co = {f.name for f in QuantumStateRegistry.get_fields("CO")}
        assert QuantumStateRegistry.get_field_names(["H2O", "CO"]) == h2o | co
        assert QuantumStateRegistry.get_field_names([]) == set()


class TestPlotAxisRegistrySpeciesScope:
    def test_quantum_entries_carry_field_name(self):
        entry = PlotAxisRegistry.get("qn_upper:J")
        assert entry is not None
        assert entry.quantum_field == "J"
        assert PlotAxisRegistry.get("eu").quantum_field is None

    def test_no_species_returns_everything(self):
        assert PlotAxisRegistry.get_axis_options() == PlotAxisRegistry.get_axis_options(species=None)
        assert _quantum_fields(PlotAxisRegistry.get_axis_options())

    def test_empty_species_drops_all_quantum_entries(self):
        for entries in (PlotAxisRegistry.get_axis_options(species=[]),
                        PlotAxisRegistry.get_color_options(species=[])):
            assert entries, "physics entries must survive"
            assert _quantum_fields(entries) == set()
            assert all(e.quantum_field is None for e in entries)

    def test_species_limits_quantum_entries_to_its_schema(self):
        h2o = {f.name for f in QuantumStateRegistry.get_fields("H2O")}
        co = {f.name for f in QuantumStateRegistry.get_fields("CO")}
        assert h2o - co, "test needs a field H2O declares and CO does not"

        co_axis = PlotAxisRegistry.get_axis_options(species=["CO"])
        assert _quantum_fields(co_axis) <= co
        assert not (_quantum_fields(co_axis) & (h2o - co))

        both = PlotAxisRegistry.get_color_options(species=["CO", "H2O"])
        assert _quantum_fields(both) <= (h2o | co)
        assert _quantum_fields(both) & (h2o - co)

    def test_scoped_lists_preserve_registry_order(self):
        full = PlotAxisRegistry.get_axis_options()
        scoped = PlotAxisRegistry.get_axis_options(species=["H2O"])
        scoped_keys = [e.key for e in scoped]
        assert scoped_keys == [e.key for e in full if e.key in scoped_keys]

    def test_filter_window_and_registry_agree(self):
        from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import _quantum_field_specs
        window_fields = {s.field for s in _quantum_field_specs("H2O")}
        axis_fields = _quantum_fields(PlotAxisRegistry.get_color_options(species=["H2O"]))
        assert window_fields == axis_fields


class TestPopulationDiagramPlotMoleculeIds:
    @staticmethod
    def _mol(name, mol_id):
        return SimpleNamespace(name=name, color="#123456",
                               line_list=SimpleNamespace(molecule_id=mol_id))

    def test_single_molecule(self):
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        pdp = PopulationDiagramPlot(self._mol("water", "H2O"))
        assert pdp.get_molecule_ids() == ["H2O"]
        pdp.close()

    def test_multiple_molecules_dedupes_and_skips_unknown(self):
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        no_id = SimpleNamespace(name="mystery", color="k", line_list=None)
        pdp = PopulationDiagramPlot(molecules=[
            self._mol("water", "H2O"), self._mol("water2", "H2O"),
            self._mol("co", "CO"), no_id,
        ])
        assert pdp.get_molecule_ids() == ["H2O", "CO"]
        pdp.close()

    def test_bare_intensity_has_no_species(self):
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        pdp = PopulationDiagramPlot(intensity=SimpleNamespace())
        assert pdp.get_molecule_ids() == []
        pdp.close()

    def test_context_mixin_scopes_to_plot(self):
        from iSLAT.Modules.Plotting.PopulationDiagramPlot import PopulationDiagramPlot
        from iSLAT.Modules.Plotting.PopulationDiagramContextMixin import PopulationDiagramContextMixin
        pdp = PopulationDiagramPlot(self._mol("co", "CO"))
        assert PopulationDiagramContextMixin._plot_species(pdp) == ["CO"]
        assert PopulationDiagramContextMixin._plot_species(None) is None
        pdp.close()
