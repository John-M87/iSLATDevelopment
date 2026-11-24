"""Molecule parameter sets (molsave CSV rows) that are applied in place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Optional, Union

import iSLAT.Constants as c

if TYPE_CHECKING:
    from .Molecule import Molecule
    from .MoleculeDict import MoleculeDict

__all__ = ["APPLIED_PARAMETERS", "SAVE_COLUMNS", "ApplyResult", "ParameterSet", "ParameterSetStore"]

SAVE_COLUMNS = (
    "Molecule Name", "File Path", "Molecule Label", "Temp", "Rad", "N_Mol", "Color", "Vis",
    "Dist", "StellarRV", "FWHM", "KeplerianFWHM", "InstrumentalProfile", "Broad", "RV Shift",
)

APPLIED_PARAMETERS = (
    "temp", "radius", "n_mol", "distance", "fwhm", "keplerian_fwhm",
    "instrumental_profile_key", "broad", "rv_shift", "is_visible",
)
_DISPLAY_ATTRIBUTES = ("color", "displaylabel")

Rows = Union[Mapping[str, Mapping[str, Any]], Iterable[Mapping[str, Any]]]


@dataclass
class ApplyResult:
    """Molecule names by what :meth:`ParameterSet.apply_to` did to them."""

    updated: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    added: List[str] = field(default_factory=list)
    replaced: List[str] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.updated or self.added or self.replaced or self.removed)

    def summary(self) -> str:
        parts = [f"{len(self.unchanged)} unchanged", f"{len(self.updated)} updated"]
        for label, names in (("added", self.added), ("rebuilt", self.replaced),
                             ("removed", self.removed), ("failed", self.failed)):
            if names:
                parts.append(f"{len(names)} {label}")
        return ", ".join(parts)


class ParameterSet:
    """The parameters of a set of molecules, as molsave CSV rows keyed by molecule name."""

    def __init__(self, rows: Rows, name: str = "") -> None:
        values = rows.values() if isinstance(rows, Mapping) else rows
        self.name = name
        self._rows: Dict[str, Dict[str, Any]] = {
            str(row["Molecule Name"]): dict(row) for row in values if row.get("Molecule Name")
        }

    @classmethod
    def capture(cls, molecules_dict: "MoleculeDict", name: str = "") -> "ParameterSet":
        return cls((_row_for(n, m, molecules_dict) for n, m in molecules_dict.items()), name=name)

    @classmethod
    def from_csv(cls, file_path: str, update_save_file_names: bool = False,
                 name: Optional[str] = None) -> Optional["ParameterSet"]:
        """Read a molsave CSV; ``None`` if it is missing or empty."""
        import os
        from iSLAT.Modules.FileHandling.iSLATFileHandling import read_from_user_csv

        if not os.path.exists(file_path):
            return None
        rows = read_from_user_csv(os.path.dirname(file_path), os.path.basename(file_path),
                                  update_save_file_names=update_save_file_names)
        if not rows:
            return None
        return cls(rows, name=os.path.basename(file_path) if name is None else name)

    @property
    def names(self) -> List[str]:
        return list(self._rows)

    def rows(self) -> List[Dict[str, Any]]:
        return [dict(row) for row in self._rows.values()]

    def __len__(self) -> int:
        return len(self._rows)

    def __contains__(self, molecule_name: object) -> bool:
        return molecule_name in self._rows

    def __repr__(self) -> str:
        return f"ParameterSet(name={self.name!r}, molecules={self.names!r})"

    def apply_to(self, molecules_dict: "MoleculeDict",
                 initial_molecule_parameters: Optional[Mapping[str, Any]] = None) -> ApplyResult:
        """Make *molecules_dict* match this set, keeping molecules whose line data is unchanged.

        Kept molecules only get the parameters that differ, so their computed models
        survive.  Other molecules are built from their rows; molecules not in the set
        are removed.  No change notifications are sent: refresh the GUI afterwards.
        """
        initial = initial_molecule_parameters or {}
        result = ApplyResult()
        active_name = getattr(molecules_dict.active_molecule, "name", None)

        globals_changed = False
        if self._rows:
            first = next(iter(self._rows.values()))
            globals_changed = molecules_dict.restore_saved_globals(
                _to_float(first.get("Dist"), c.DEFAULT_DISTANCE),
                _to_float(first.get("StellarRV"), c.DEFAULT_STELLAR_RV),
            )

        to_insert: Dict[str, "Molecule"] = {}
        for mol_name, row in self._rows.items():
            try:
                # Parsing the row like a full load guarantees the same values and defaults.
                fresh = molecules_dict.build_molecule(row, initial)
            except Exception as exc:
                result.failed[mol_name] = str(exc)
                continue
            existing = molecules_dict.get(mol_name)
            if existing is None:
                to_insert[mol_name] = fresh
                result.added.append(mol_name)
            elif not _same_line_data(existing, fresh):
                to_insert[mol_name] = fresh
                result.replaced.append(mol_name)
            elif _copy_parameters(fresh, existing):
                result.updated.append(mol_name)
            else:
                result.unchanged.append(mol_name)

        keep = set(result.updated) | set(result.unchanged)
        for mol_name in list(molecules_dict.keys()):
            if mol_name not in keep:
                del molecules_dict[mol_name]
                if mol_name not in to_insert:
                    result.removed.append(mol_name)
        for mol_name, molecule in to_insert.items():
            molecules_dict[mol_name] = molecule
        molecules_dict.reorder([n for n in self._rows if n in molecules_dict])

        if result.changed or globals_changed:
            molecules_dict.invalidate_summed_flux_cache()
        if active_name in molecules_dict and molecules_dict.active_molecule is not molecules_dict[active_name]:
            molecules_dict.active_molecule = molecules_dict[active_name]
        return result


def _row_for(mol_name: str, molecule: "Molecule", molecules_dict: Any) -> Dict[str, Any]:
    return {
        "Molecule Name": mol_name,
        "File Path": getattr(molecule, "filepath", getattr(molecule, "hitran_data", "")),
        "Molecule Label": getattr(molecule, "displaylabel", mol_name),
        "Temp": getattr(molecule, "temp", 600),
        "Rad": getattr(molecule, "radius", 0.5),
        "N_Mol": getattr(molecule, "n_mol", 1e17),
        "Color": getattr(molecule, "color", "#FF0000"),
        "Vis": getattr(molecule, "is_visible", True),
        "Dist": getattr(molecules_dict, "_global_dist", 140),
        "StellarRV": getattr(molecules_dict, "_global_stellar_rv", 0),
        "FWHM": getattr(molecule, "fwhm", 200),
        "KeplerianFWHM": getattr(molecule, "keplerian_fwhm", 0.0),
        "InstrumentalProfile": getattr(molecule, "instrumental_profile_key", "constant"),
        "Broad": getattr(molecule, "broad", 2.5),
        "RV Shift": getattr(molecule, "rv_shift", 0),
    }


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _same_line_data(existing: "Molecule", fresh: "Molecule") -> bool:
    # Line data or a partition function replaced this session isn't saved, so rebuild those.
    return (getattr(existing, "filepath", None) == getattr(fresh, "filepath", None)
            and getattr(existing, "_line_list_version", 0) == 0)


def _copy_parameters(source: "Molecule", target: "Molecule") -> bool:
    # Pass only differing values: setting a parameter marks the models dirty even if unchanged.
    changes = {p: getattr(source, p) for p in APPLIED_PARAMETERS
               if getattr(target, p) != getattr(source, p)}
    if changes:
        target.bulk_update_parameters(changes, skip_notification=True)
    display_changed = False
    for attribute in _DISPLAY_ATTRIBUTES:
        value = getattr(source, attribute, None)
        if getattr(target, attribute, None) != value:
            setattr(target, attribute, value)
            display_changed = True
    return bool(changes) or display_changed


class ParameterSetStore:
    """The parameter sets of a session, each with its in-memory edits and last saved version.

    One set is active: the molecules show it.  Opening or switching to another
    set first keeps the active set's current parameters in memory, so no edits
    are lost; saving and reverting work on the active set.  A set is modified
    when its parameters differ from its last saved (or opened) version.
    """

    def __init__(self) -> None:
        self._current: Dict[str, ParameterSet] = {}
        self._saved: Dict[str, ParameterSet] = {}
        self._paths: Dict[str, Optional[str]] = {}
        self.active: Optional[str] = None

    @property
    def names(self) -> List[str]:
        return list(self._current)

    def __contains__(self, name: object) -> bool:
        return name in self._current

    def path(self, name: Optional[str] = None) -> Optional[str]:
        return self._paths.get(name or self.active)

    def find(self, path: Optional[str]) -> Optional[str]:
        """The set opened from or saved to *path*, if any."""
        if not path:
            return None
        key = _path_key(path)
        return next((n for n, p in self._paths.items() if p and _path_key(p) == key), None)

    def is_modified(self, molecules_dict: Optional["MoleculeDict"] = None, name: Optional[str] = None) -> bool:
        """Whether set *name* (default: the active one, read from *molecules_dict*) has unsaved changes."""
        name = name or self.active
        if name not in self._saved:
            return False
        if name == self.active and molecules_dict is not None:
            current = ParameterSet.capture(molecules_dict)
        else:
            current = self._current[name]
        return current.rows() != self._saved[name].rows()

    def open(self, parameter_set: ParameterSet, molecules_dict: "MoleculeDict", apply,
             path: Optional[str] = None) -> str:
        """Make *parameter_set* (read from *path*) the active set; returns its name.

        A set already opened from *path* is switched to instead, keeping its
        in-memory edits.  A set without a path (e.g. the defaults) is applied
        fresh, replacing a path-less set of the same name.  *apply* applies a
        set to the molecules (``iSLAT.apply_parameter_set``).
        """
        existing = self.find(path)
        if existing:
            self.switch(existing, molecules_dict, apply)
            return existing
        self._stash_active(molecules_dict)
        apply(parameter_set)
        name = parameter_set.name or (os.path.basename(path) if path else "Parameters")
        if path or self._paths.get(name, "") is not None:
            name = self._unique_name(name)
        self._set(name, ParameterSet.capture(molecules_dict, name), path)
        return name

    def switch(self, name: str, molecules_dict: "MoleculeDict", apply) -> None:
        """Show set *name*, keeping the active set's current parameters in memory."""
        if name not in self._current:
            raise KeyError(f"No parameter set named {name!r}")
        if name == self.active:
            return
        self._stash_active(molecules_dict)
        apply(self._current[name])
        self.active = name

    def mark_saved(self, molecules_dict: "MoleculeDict", path: str) -> str:
        """Record that the molecules' parameters were written to *path*; returns that set's name.

        The set for *path* (created if new) becomes active and unmodified.  When
        another set was active, its edits now live in the saved set, so it goes
        back to its own last saved version.
        """
        target = self.find(path)
        if self.active and self.active != target:
            self._current[self.active] = self._saved[self.active]
        name = target or self._unique_name(os.path.basename(path))
        self._set(name, ParameterSet.capture(molecules_dict, name), path)
        return name

    def revert(self, molecules_dict: "MoleculeDict", apply) -> bool:
        """Restore the active set's last saved version; returns whether there was one."""
        if self.active not in self._saved:
            return False
        saved = self._saved[self.active]
        apply(saved)
        self._current[self.active] = saved
        return True

    def _set(self, name: str, snapshot: ParameterSet, path: Optional[str]) -> None:
        self._current[name] = self._saved[name] = snapshot
        self._paths[name] = path
        self.active = name

    def _stash_active(self, molecules_dict: "MoleculeDict") -> None:
        if self.active in self._current:
            self._current[self.active] = ParameterSet.capture(molecules_dict, self.active)

    def _unique_name(self, name: str) -> str:
        candidate, n = name, 2
        while candidate in self._current:
            candidate, n = f"{name} ({n})", n + 1
        return candidate


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path)))
