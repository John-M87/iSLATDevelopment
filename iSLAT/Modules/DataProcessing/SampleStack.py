"""
SampleStack -- the spectra of a sample, each paired with its own parameter set.

Used by the Full Spectrum view's *Stack Sample Spectra* option to draw every
spectrum of the sample (see the Sample Manager) together.  Each sample entry
is shown with the molecule parameters it would get if it were made active:

* The **active** spectrum uses the live ``molecules_dict`` (its parameter
  file and overrides are already applied, plus any edits since).
* An entry with a **parameter file** gets its own :class:`MoleculeDict`
  loaded from that CSV, with the entry's global overrides (distance, stellar
  RV, model pixel resolution) applied on top.
* An entry **without** a parameter file keeps the current parameters, like
  switching to it does.  Without overrides it simply shares the live
  ``molecules_dict``; with overrides it gets a copy that follows the live
  molecules and has the overrides applied.

Separate parameter sets are cached between renders and updated in place, so
repeated redraws do not reload the line lists.  Every change is made without
molecule parameter-change notifications: those callbacks are class-wide and
would otherwise make the GUI react as if the live molecules had changed.
Call :meth:`SampleStackBuilder.release` when the sets are no longer needed.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

#: Sample override keys (``ControlPanelFields.json`` global fields) and the
#: :class:`MoleculeDict` property each one sets.  Used when no field
#: configuration is available.
DEFAULT_OVERRIDE_PROPERTIES: Dict[str, str] = {
    "distance": "global_distance",
    "stellar_rv": "global_stellar_rv",
    "model_pixel_res": "global_model_pixel_res",
}

#: Global settings copied from the live molecules into every separate set.
_SHARED_GLOBALS: Tuple[str, ...] = (
    "global_wavelength_range",
    "global_intensity_calculation_method",
    "match_spectral_sampling",
)

#: Parameters (Molecule attributes) copied from a live molecule into its copy.
_COPIED_PARAMETERS: Tuple[str, ...] = (
    "temp", "n_mol", "radius", "fwhm", "broad",
    "rv_shift", "keplerian_fwhm", "instrumental_profile_key",
)


@dataclass
class SampleSpectrum:
    """One spectrum of the sample together with the molecules that model it.

    Attributes
    ----------
    path : str or None
        Spectrum file (``None`` for an in-memory spectrum).
    name : str
        Display name (the file name without its extension).
    wave_obs : np.ndarray
        Observer-frame wavelengths.
    wave : np.ndarray
        Rest-frame wavelengths (``wave_obs`` corrected for this entry's stellar RV).
    flux : np.ndarray
        Observed flux.
    err : np.ndarray or None
        Flux uncertainties.
    molecules : MoleculeDict or None
        The parameter set used for this spectrum's models.
    is_active : bool
        ``True`` for the spectrum currently loaded in iSLAT.
    parameter_source : str
        ``"active"``, ``"file"`` (own parameter file), ``"current"`` (copy of
        the current parameters, with overrides) or ``"shared"`` (the current
        parameters themselves).
    param_file : str or None
        The entry's parameter file, if any.
    overrides : dict
        The entry's global overrides (override key -> value).
    """
    path: Optional[str]
    name: str
    wave_obs: np.ndarray
    wave: np.ndarray
    flux: np.ndarray
    err: Optional[np.ndarray]
    molecules: Any
    is_active: bool = False
    parameter_source: str = "shared"
    param_file: Optional[str] = None
    overrides: Dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------
def same_path(a: Optional[str], b: Optional[str]) -> bool:
    """``True`` when *a* and *b* name the same file (case / separator insensitive on Windows)."""
    if not a or not b:
        return False
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def spectrum_display_name(path: Optional[str]) -> str:
    """The file name of *path* without its extension (``"spectrum"`` when unknown)."""
    if not path:
        return "spectrum"
    return os.path.splitext(os.path.basename(path))[0]


def _close(a: Any, b: Any) -> bool:
    """Equality for parameter values: floats within a relative 1e-9, everything else ``==``."""
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=0.0)
    except (TypeError, ValueError):
        return a == b


def _file_signature(path: Optional[str]) -> Tuple[Any, ...]:
    """``(path, mtime_ns, size)`` so a changed file is detected; ``(path,)`` when it cannot be read."""
    try:
        st = os.stat(path)
        return (os.path.normcase(os.path.abspath(path)), st.st_mtime_ns, st.st_size)
    except (OSError, TypeError):
        return (path,)


def sample_signature(islat: Any) -> Tuple[Any, ...]:
    """A value that changes whenever the stacked sample would change.

    Covers the loaded spectrum and, for every other sample entry, its file,
    parameter file (and that file's modification time) and overrides.  The
    loaded spectrum's own parameter file and overrides are left out: they are
    already applied to the live molecules, whose changes reach the plot
    through the usual parameter callbacks.
    """
    active = getattr(islat, "loaded_spectrum_file", None)
    params_map = getattr(islat, "sample_spectra_params", None) or {}
    overrides_map = getattr(islat, "sample_spectra_overrides", None) or {}
    entries = []
    for path in getattr(islat, "sample_spectra", None) or []:
        if same_path(path, active):
            entries.append((path, "active"))
            continue
        param_file = params_map.get(path)
        entries.append((
            path,
            _file_signature(path),
            _file_signature(param_file) if param_file else None,
            tuple(sorted((overrides_map.get(path) or {}).items())),
        ))
    return (active, tuple(entries))


def load_sample_spectrum(
    path: str,
    wave_scale: float = 1.0,
    flux_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a spectrum file the way :meth:`iSLAT.load_spectrum` does.

    Returns observer-frame ``(wave, flux, err)``.  The wavelengths and flux
    are multiplied by *wave_scale* / *flux_scale* (the ``wave_data_scalar``
    and ``flux_data_scalar`` user settings); a missing ``err`` column becomes
    10 % of the flux, as for the loaded spectrum.

    Raises
    ------
    ValueError
        If the file is empty or has no ``wave`` / ``flux`` columns.
    """
    from iSLAT.Modules.FileHandling.iSLATFileHandling import read_spectral_data
    df = read_spectral_data(path)
    if df is None or df.empty:
        raise ValueError(f"Could not read a spectrum from {path}")
    missing = [col for col in ("wave", "flux") if col not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} has no {', '.join(missing)} column")
    wave = np.asarray(df["wave"].values, dtype=float) * float(wave_scale)
    flux = np.asarray(df["flux"].values, dtype=float) * float(flux_scale)
    if "err" in df.columns:
        err = np.asarray(df["err"].values, dtype=float)
    else:
        err = np.abs(flux) * 0.1
    return wave, flux, err


def molecule_save_row(molecule: Any, distance: float, stellar_rv: float) -> Dict[str, Any]:
    """A parameter-file row (molsave CSV format) describing *molecule*'s current parameters."""
    name = getattr(molecule, "name", "molecule")
    row = {
        "Molecule Name": name,
        "File Path": getattr(molecule, "filepath", None),
        "Molecule Label": getattr(molecule, "displaylabel", name),
        "Temp": getattr(molecule, "temp", None),
        "Rad": getattr(molecule, "radius", None),
        "N_Mol": getattr(molecule, "n_mol", None),
        "Color": getattr(molecule, "color", None),
        "Vis": getattr(molecule, "is_visible", True),
        "Dist": distance,
        "StellarRV": stellar_rv,
        "FWHM": getattr(molecule, "fwhm", None),
        "KeplerianFWHM": getattr(molecule, "keplerian_fwhm", 0.0),
        "InstrumentalProfile": getattr(molecule, "instrumental_profile_key", "constant"),
        "Broad": getattr(molecule, "broad", None),
        "RV Shift": getattr(molecule, "rv_shift", 0.0),
    }
    source = getattr(molecule, "_line_data_source", None)
    if source:
        row["line_data_source"] = source
        row["line_format"] = getattr(molecule, "_line_format", None)
    return row


def _release_molecule_dict(molecules: Any) -> None:
    """Unregister the class-wide parameter callback a :class:`MoleculeDict` adds on creation."""
    if molecules is None:
        return
    try:
        from iSLAT.Modules.DataTypes.Molecule import Molecule
        Molecule.remove_molecule_parameter_change_callback(molecules._on_molecule_parameter_changed)
    except Exception:
        pass


def _set_global(molecules: Any, prop: str, value: Any) -> bool:
    """Set the global *prop* on *molecules* when it differs; ``True`` when it changed.

    The global setters update the molecules without parameter notifications.
    """
    if value is None:
        return False
    try:
        current = getattr(molecules, prop)
    except AttributeError:
        return False
    if prop == "global_wavelength_range":
        if current is not None and len(current) == 2 and all(_close(c, v) for c, v in zip(current, value)):
            return False
        value = (float(value[0]), float(value[1]))
    elif _close(current, value):
        return False
    setattr(molecules, prop, value)
    return True


def _sync_display(molecules: Any, live: Any) -> bool:
    """Copy colors and visibility of same-named live molecules into *molecules*.

    The control panel's visibility check boxes and colors then apply to every
    spectrum of the stack.  Returns ``True`` when anything changed.
    """
    changed = False
    for name, mol in molecules.items():
        src = live.get(name) if live is not None else None
        if src is None:
            continue
        color = getattr(src, "color", None)
        if color and getattr(mol, "color", None) != color:
            mol.color = color
        visible = bool(getattr(src, "is_visible", True))
        if bool(getattr(mol, "is_visible", True)) != visible:
            mol.bulk_update_parameters({"is_visible": visible}, skip_notification=True)
            changed = True
    return changed


# ----------------------------------------------------------------------
# Parameter-set cache entry
# ----------------------------------------------------------------------
@dataclass
class _ParameterSet:
    kind: str                 # "file" or "current"
    signature: Tuple[Any, ...]
    molecules: Any


class SampleStackBuilder:
    """Builds :class:`SampleSpectrum` entries for a sample and caches their parameter sets.

    Parameters
    ----------
    molecule_dict_factory : callable, optional
        Creates an empty parameter set (default: :class:`MoleculeDict`).
        Tests can pass a lightweight stand-in.
    spectrum_loader : callable, optional
        ``loader(path, wave_scale, flux_scale) -> (wave, flux, err)``
        (default: :func:`load_sample_spectrum`).
    """

    def __init__(
        self,
        molecule_dict_factory: Optional[Callable[..., Any]] = None,
        spectrum_loader: Optional[Callable[..., Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self._factory = molecule_dict_factory
        self._loader = spectrum_loader or load_sample_spectrum
        self._spectra: Dict[str, Tuple[Tuple[Any, ...], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        self._sets: Dict[str, _ParameterSet] = {}
        #: Problems met by the last :meth:`build` (one message per skipped entry).
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, islat: Any) -> List[SampleSpectrum]:
        """Return one :class:`SampleSpectrum` per spectrum of *islat*'s sample, in sample order.

        Without a sample (or when the loaded spectrum is not part of it) the
        loaded spectrum is included as the active entry.  Entries whose file
        cannot be read are skipped and reported in :attr:`warnings`.
        """
        self.warnings = []
        live = getattr(islat, "molecules_dict", None)
        active_path = getattr(islat, "loaded_spectrum_file", None)
        sample = [p for p in (getattr(islat, "sample_spectra", None) or []) if p]
        if not any(same_path(p, active_path) for p in sample):
            sample = ([active_path] if active_path or not sample else []) + sample
        params_map = getattr(islat, "sample_spectra_params", None) or {}
        overrides_map = getattr(islat, "sample_spectra_overrides", None) or {}
        properties = self._override_properties(islat)
        settings = getattr(islat, "user_settings", None) or {}
        wave_scale = float(settings.get("wave_data_scalar", 1.0) or 1.0)
        flux_scale = float(settings.get("flux_data_scalar", 1.0) or 1.0)

        entries: List[SampleSpectrum] = []
        used: set = set()
        active_added = False
        for path in sample:
            if not active_added and (path == active_path or same_path(path, active_path)):
                entries.append(self._active_entry(islat, path))
                active_added = True
                if path:
                    used.add(os.path.normcase(os.path.abspath(path)))
                continue
            key = os.path.normcase(os.path.abspath(path))
            if key in used:
                continue
            used.add(key)
            param_file = params_map.get(path)
            overrides = {k: v for k, v in (overrides_map.get(path) or {}).items() if v is not None}
            try:
                wave_obs, flux, err = self._load(path, wave_scale, flux_scale)
                molecules, source = self._parameter_set(
                    key, param_file, overrides, live, properties, islat,
                )
            except Exception as exc:
                self.warnings.append(f"{os.path.basename(path)}: {exc}")
                continue
            wave = molecules.apply_stellar_rv(wave_obs) if molecules is not None else wave_obs
            entries.append(SampleSpectrum(
                path=path,
                name=spectrum_display_name(path),
                wave_obs=wave_obs,
                wave=np.asarray(wave, dtype=float),
                flux=flux,
                err=err,
                molecules=molecules,
                is_active=False,
                parameter_source=source,
                param_file=param_file if source == "file" else None,
                overrides=overrides,
            ))
        self._discard_except(used)
        return entries

    def release(self) -> None:
        """Drop every cached spectrum and parameter set (and their class-wide callbacks)."""
        for pset in self._sets.values():
            _release_molecule_dict(pset.molecules)
        self._sets.clear()
        self._spectra.clear()

    # ------------------------------------------------------------------
    # Entries
    # ------------------------------------------------------------------
    @staticmethod
    def _active_entry(islat: Any, path: Optional[str]) -> SampleSpectrum:
        live = getattr(islat, "molecules_dict", None)
        wave_obs = np.array(getattr(islat, "wave_data_original", getattr(islat, "wave_data", [])), dtype=float, copy=True)
        wave = live.apply_stellar_rv(wave_obs) if live is not None else wave_obs
        err = getattr(islat, "err_data", None)
        return SampleSpectrum(
            path=path,
            name=spectrum_display_name(path or getattr(islat, "loaded_spectrum_name", None)),
            wave_obs=wave_obs,
            wave=np.asarray(wave, dtype=float),
            flux=np.array(getattr(islat, "flux_data", []), dtype=float, copy=True),
            err=np.asarray(err, dtype=float) if err is not None else None,
            molecules=live,
            is_active=True,
            parameter_source="active",
        )

    def _load(self, path: str, wave_scale: float, flux_scale: float):
        signature = _file_signature(path) + (wave_scale, flux_scale)
        key = signature[0]
        cached = self._spectra.get(key)
        if cached is not None and cached[0] == signature:
            return cached[1]
        data = self._loader(path, wave_scale, flux_scale)
        self._spectra[key] = (signature, data)
        return data

    # ------------------------------------------------------------------
    # Parameter sets
    # ------------------------------------------------------------------
    @staticmethod
    def _override_properties(islat: Any) -> Dict[str, str]:
        """Override key -> MoleculeDict property, from the GUI's global field configuration."""
        getter = getattr(islat, "_get_global_fields_config", None)
        if callable(getter):
            try:
                fields = getter() or {}
                props = {
                    key: cfg["property"] for key, cfg in fields.items()
                    if not str(key).startswith("_") and isinstance(cfg, dict) and cfg.get("property")
                }
                if props:
                    return props
            except Exception:
                pass
        return dict(DEFAULT_OVERRIDE_PROPERTIES)

    def _parameter_set(
        self,
        key: str,
        param_file: Optional[str],
        overrides: Dict[str, Any],
        live: Any,
        properties: Dict[str, str],
        islat: Any,
    ) -> Tuple[Any, str]:
        """Return ``(molecules, source)`` for a non-active sample entry."""
        effective = {
            properties[k]: float(v) for k, v in overrides.items() if k in properties
        }
        if param_file and os.path.exists(param_file):
            signature = ("file",) + _file_signature(param_file) + (tuple(sorted(effective.items())),)
            pset = self._sets.get(key)
            if pset is None or pset.kind != "file" or pset.signature != signature:
                self._drop(key)
                pset = _ParameterSet("file", signature, self._load_file_set(param_file, live, islat))
                self._sets[key] = pset
            self._sync_file_set(pset.molecules, live, effective)
            return pset.molecules, "file"

        if live is None:
            self._drop(key)
            return None, "shared"
        differs = {
            prop: value for prop, value in effective.items()
            if not _close(getattr(live, prop, None), value)
        }
        if not differs:
            self._drop(key)
            return live, "shared"
        pset = self._sets.get(key)
        if pset is None or pset.kind != "current":
            self._drop(key)
            pset = _ParameterSet("current", ("current",), self._new_set())
            self._sets[key] = pset
        self._sync_current_set(pset.molecules, live, effective, islat)
        return pset.molecules, "current"

    def _drop(self, key: str) -> None:
        pset = self._sets.pop(key, None)
        if pset is not None:
            _release_molecule_dict(pset.molecules)

    def _discard_except(self, keep: Iterable[str]) -> None:
        keep = set(keep)
        for key in [k for k in self._sets if k not in keep]:
            self._drop(key)
        for key in [k for k in self._spectra if k not in keep]:
            self._spectra.pop(key, None)

    def _new_set(self) -> Any:
        if self._factory is not None:
            return self._factory()
        from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
        return MoleculeDict()

    @staticmethod
    def _initial_parameters(islat: Any) -> Dict[str, Any]:
        try:
            params = getattr(islat, "initial_molecule_parameters", None)
            return dict(params) if params else {}
        except Exception:
            return {}

    @staticmethod
    def _share_globals(molecules: Any, live: Any) -> None:
        """Copy the model range, intensity method and matched-sampling flag from the live set."""
        if live is None:
            return
        for prop in _SHARED_GLOBALS:
            value = getattr(live, prop, None)
            if prop == "match_spectral_sampling":
                if value is not None and bool(getattr(molecules, prop, False)) != bool(value):
                    setattr(molecules, prop, bool(value))
            else:
                _set_global(molecules, prop, value)

    def _load_file_set(self, param_file: str, live: Any, islat: Any) -> Any:
        """A new parameter set loaded from *param_file* (a molsave CSV)."""
        from iSLAT.Modules.FileHandling.iSLATFileHandling import read_from_user_csv
        rows = read_from_user_csv(os.path.dirname(param_file), os.path.basename(param_file))
        if not rows:
            raise ValueError(f"no molecules in {os.path.basename(param_file)}")
        molecules = self._new_set()
        # Range, method and pixel resolution must be set before the molecules
        # are created, so none of them needs a (notifying) per-molecule update.
        self._share_globals(molecules, live)
        if live is not None:
            _set_global(molecules, "global_model_pixel_res", getattr(live, "global_model_pixel_res", None))
        results = molecules.load_molecules(
            list(rows.values()),
            self._initial_parameters(islat),
            update_global_parameters=True,     # distance / stellar RV from the file
            strategy="sequential",
        )
        if not results.get("success"):
            _release_molecule_dict(molecules)
            raise ValueError(f"could not load the molecules of {os.path.basename(param_file)}")
        return molecules

    def _sync_file_set(self, molecules: Any, live: Any, effective: Dict[str, float]) -> None:
        self._share_globals(molecules, live)
        if live is not None and "global_model_pixel_res" not in effective:
            _set_global(molecules, "global_model_pixel_res", getattr(live, "global_model_pixel_res", None))
        for prop, value in effective.items():
            _set_global(molecules, prop, value)
        if _sync_display(molecules, live):
            molecules._clear_all_caches()

    def _sync_current_set(self, molecules: Any, live: Any, effective: Dict[str, float], islat: Any) -> None:
        """Make *molecules* a copy of the live molecules with the overrides in *effective* applied."""
        self._share_globals(molecules, live)
        for prop in ("global_model_pixel_res", "global_distance", "global_stellar_rv"):
            _set_global(molecules, prop, effective.get(prop, getattr(live, prop, None)))

        changed = False
        for name in list(molecules.keys()):
            src = live.get(name)
            if src is None or getattr(src, "filepath", None) != getattr(molecules[name], "filepath", None):
                del molecules[name]
                changed = True

        missing = [name for name in live.keys() if name not in molecules]
        if missing:
            rows = [
                molecule_save_row(live[name], molecules.global_distance, molecules.global_stellar_rv)
                for name in missing
            ]
            molecules.load_molecules(
                rows, self._initial_parameters(islat),
                update_global_parameters=False, strategy="sequential",
            )
            changed = True

        for name, src in live.items():
            dst = molecules.get(name)
            if dst is None:
                continue
            updates = {
                p: getattr(src, p) for p in _COPIED_PARAMETERS
                if hasattr(src, p) and not _close(getattr(dst, p, None), getattr(src, p))
            }
            if updates:
                dst.bulk_update_parameters(updates, skip_notification=True)
                changed = True
            if getattr(src, "displaylabel", None) and dst.displaylabel != src.displaylabel:
                dst.displaylabel = src.displaylabel

        if _sync_display(molecules, live):
            changed = True
        if list(molecules.keys()) != [n for n in live.keys() if n in molecules]:
            try:
                molecules.reorder([n for n in live.keys() if n in molecules])
            except ValueError:
                pass
        if changed:
            molecules._clear_all_caches()
