"""Which species' quantum-number schema a line list uses: file header, overrides, name prefixes."""

from pathlib import Path

import numpy as np
import pytest

import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401  (registers schemas)
from iSLAT.Modules.DataTypes.MoleculeLineList import MoleculeLineList
from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry as Registry
from iSLAT.Modules.FileHandling import line_list_species
from iSLAT.Modules.FileHandling.molecular_data_reader import parse_par_header, read_par_header

ROOT = Path(__file__).resolve().parents[1]
CO_PAR = ROOT / "iSLAT" / "DATAFILES" / "HITRANdata" / "data_HITEMP_2019_CO.par"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Keep aliases, overrides and line-list caches out of the real session and data folder."""
    monkeypatch.setattr(Registry, "_aliases", {})
    monkeypatch.setattr(line_list_species, "species_file_path", tmp_path / "LineListSpecies.json")
    monkeypatch.setattr(line_list_species, "_overrides", None)
    monkeypatch.setattr("iSLAT.Modules.FileHandling.line_cache_folder_path", str(tmp_path / "cache"))


def _par(tmp_path, name, header, molar_mass=True, n_lines=3):
    """A small .par file: *header* line, the CO partition function start and CO lines."""
    if not CO_PAR.exists():
        pytest.skip("CO HITRAN file not available")
    text = CO_PAR.read_text().splitlines()
    first_record = text.index("Number of lines") + 4
    body = ["3"] + [l for l in text[5:20] if l.strip() and not l.startswith("#")][:3]
    body += ["Number of lines", str(n_lines)] + text[first_record - 2:first_record + n_lines]
    lines = [header] + (["# Molar Mass: 27.994915"] if molar_mass else []) + ["# test", "# 2026-11-24"] + body
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


class TestHeaderParsing:
    @pytest.mark.parametrize("line, hitran_id, iso, names", [
        ("# HITRAN 2024 H2O; id:1; iso:1;gid:1", 1, 1, ["H2O"]),
        ("# HITEMP 2019 CO; id:5; iso:1;gid:26", 5, 1, ["CO"]),
        ("# HITRAN 2024 H2O + HITEMP; id:1; iso:1;gid:1", 1, 1, ["H2O"]),
        ("# HITRAN 12C17O; id:5; iso:4;gid:29", 5, 4, ["12C17O"]),
        ("# HITRAN CO_v=1-0; id:; iso:;gid:", None, None, ["CO_v=1-0"]),
        ("# HITRAN ; id:; iso:;gid:", None, None, []),
        ("# HITRAN 2020 C3H4; id:X; iso:1;gid:400", None, 1, ["C3H4"]),
        ("# HITRAN H2O; iso:2;gid:26", None, 2, ["H2O"]),  # gid is not id
        ("# Written by hand", None, None, []),
    ])
    def test_first_line(self, line, hitran_id, iso, names):
        assert parse_par_header([line, "# Molar Mass: 18.0"]) == {
            "hitran_id": hitran_id, "hitran_iso": iso, "names": names}

    def test_reads_only_the_header(self, tmp_path):
        path = _par(tmp_path, "x.par", "# HITRAN CO; id:5; iso:2;gid:27")
        assert read_par_header(path) == {"hitran_id": 5, "hitran_iso": 2, "names": ["CO"]}


class TestRegistry:
    def test_name_prefix(self):
        assert Registry.species_from_name("H2O_hot_v0") == "H2O"
        assert Registry.species_from_name("CO_v=1-0_P") == "CO"
        assert Registry.species_from_name("CO2") == "CO2"
        assert Registry.species_from_name("H2Ohot") is None
        assert type(Registry.get_schema("H2O_WARM")) is type(Registry.get_schema("H2O"))

    def test_alias_wins_and_follows_a_rename(self):
        Registry.set_species("CO_like_name", "H2O")
        assert Registry.resolve_species("CO_like_name") == "H2O"
        Registry.rename_species("CO_like_name", "new_name")
        assert Registry.resolve_species("new_name") == "H2O"
        assert Registry.resolve_species("CO_like_name") == "CO"
        Registry.set_species("new_name", None)
        assert Registry.resolve_species("new_name") is None

    def test_unknown_stays_generic(self):
        assert Registry.resolve_species("NotAMolecule") is None
        assert Registry.get_fields("NotAMolecule") == ()
        Registry.set_species("thing", "C3H4")  # not a registered species
        assert Registry.resolve_species("thing") is None


class TestLineListSpecies:
    def test_header_id_beats_the_name(self, tmp_path):
        ll = MoleculeLineList(molecule_id="CO2_misnamed", filename=_par(tmp_path, "a.par", "# HITRAN H2O; id:1; iso:1;gid:1"))
        assert ll.species == "H2O" and (ll.hitran_id, ll.hitran_iso) == (1, 1)

    def test_header_name_when_id_blank(self, tmp_path):
        ll = MoleculeLineList(molecule_id="x", filename=_par(tmp_path, "b.par", "# HITRAN CO_v=1-0_P; id:; iso:;gid:"))
        assert ll.species == "CO"

    def test_molecule_name_then_file_name(self, tmp_path):
        blank = "# HITRAN ; id:; iso:;gid:"
        assert MoleculeLineList(molecule_id="H2O_warm", filename=_par(tmp_path, "c.par", blank)).species == "H2O"
        assert MoleculeLineList(molecule_id="warm", filename=_par(tmp_path, "OH_lines.par", blank)).species == "OH"
        assert MoleculeLineList(molecule_id="warm", filename=_par(tmp_path, "d.par", blank)).species is None

    def test_overrides(self, tmp_path):
        path = _par(tmp_path, "e.par", "# HITRAN ; id:; iso:;gid:")
        line_list_species.set_species_override(path, "HCN")
        line_list_species.reload_species_overrides()  # as on the next start
        ll = MoleculeLineList(molecule_id="warm", filename=path)
        assert ll.species == "HCN"
        ll.species = "C2H2"
        assert ll.species == "C2H2" and Registry.resolve_species("warm") == "C2H2"
        ll.species = None
        assert ll.species == "HCN"

    def test_load_registers_the_species_for_its_molecule_name(self, tmp_path):
        ll = MoleculeLineList(molecule_id="warm_component", filename=_par(tmp_path, "f.par", "# HITRAN H2O; id:1; iso:1;gid:1"))
        ll._ensure_data_loaded()
        assert Registry.resolve_species("warm_component") == "H2O"
        assert type(ll.quantum_schema).__name__ == "NonLinearTriatomicSchema"

    def test_file_without_molar_mass_loads(self, tmp_path):
        ll = MoleculeLineList(molecule_id="CO", filename=_par(tmp_path, "g.par", "# HITRAN CO; id:5; iso:1;gid:26", molar_mass=False))
        ll._ensure_data_loaded()
        assert len(ll._raw_lines_data) == 3 and ll._molar_mass is None

    def test_from_file(self, tmp_path):
        ll = MoleculeLineList.from_file(_par(tmp_path, "h.par", "# HITRAN H2O; id:1; iso:1;gid:1"), molecule_id="warm")
        assert ll.species == "H2O" and Registry.resolve_species("warm") == "H2O"
        assert ll.copy().species == "H2O"


class TestCache:
    def test_ids_are_cached(self, tmp_path, monkeypatch):
        path = _par(tmp_path, "i.par", "# HITRAN CO; id:5; iso:3;gid:28")
        first = MoleculeLineList(molecule_id="CO_iso", filename=path)
        first._ensure_data_loaded()
        cache = Path(first._get_cache_path(str(path)))
        assert list(np.load(cache / "metadata.npy")[3:]) == [5.0, 3.0]

        def no_header_read(*args):
            raise AssertionError("the header should come from the cache")
        monkeypatch.setattr("iSLAT.Modules.FileHandling.molecular_data_reader.read_par_header", no_header_read)
        second = MoleculeLineList(molecule_id="CO_iso", filename=path)
        second._ensure_data_loaded()
        assert (second.hitran_id, second.hitran_iso, second.species) == (5, 3, "CO")

    def test_older_cache_reads_the_header(self, tmp_path):
        path = _par(tmp_path, "j.par", "# HITRAN CO; id:5; iso:1;gid:26")
        first = MoleculeLineList(molecule_id="old", filename=path)
        first._ensure_data_loaded()
        cache = Path(first._get_cache_path(str(path)))
        np.save(cache / "metadata.npy", np.load(cache / "metadata.npy")[:3])  # as written before
        second = MoleculeLineList(molecule_id="old", filename=path)
        second._ensure_data_loaded()
        assert second.hitran_id == 5 and second.species == "CO"


def test_written_par_keeps_the_species(tmp_path):
    ll = MoleculeLineList(molecule_id="CO_v1", filename=_par(tmp_path, "k.par", "# HITRAN ; id:; iso:;gid:"))
    out = tmp_path / "filtered.par"
    ll.write_par_file(out)
    assert out.read_text().splitlines()[0] == "# HITRAN CO; id:5; iso:;gid:"
    assert MoleculeLineList(molecule_id="anything", filename=out).species == "CO"


def test_rename_keeps_the_species(tmp_path):
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    path = _par(tmp_path, "l.par", "# HITRAN H2O; id:1; iso:1;gid:1")
    md = MoleculeDict()
    md.load_molecules([{"name": "warm", "file": str(path)}], {}, strategy="sequential")
    try:
        md["warm"]._ensure_lines_loaded()
        md["warm"].lines._ensure_data_loaded()
        md.rename_molecule("warm", "hot")
        assert Registry.resolve_species("hot") == "H2O"
    finally:
        Molecule.remove_molecule_parameter_change_callback(md._on_molecule_parameter_changed)


def test_hitran_number_lookup_unchanged():
    from iSLAT.Modules.Hitran_data import get_molecule_identifier
    assert [get_molecule_identifier(m) for m in ("H2O", "CO", "OH", "ClNO2")] == [1, 5, 13, 61]
    with pytest.raises(KeyError):
        get_molecule_identifier("NotAMolecule")
