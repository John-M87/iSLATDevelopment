"""Species assigned by the user to line-list files, for files whose header names none.

Stored in ``DATAFILES/CONFIG/LineListSpecies.json`` as ``{file name: species}``.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional, Union

from . import config_file_path

species_file_path = config_file_path / "LineListSpecies.json"

_overrides: Optional[Dict[str, str]] = None


def species_overrides() -> Dict[str, str]:
    """All overrides, read from disk once."""
    global _overrides
    if _overrides is None:
        try:
            with open(species_file_path, "r") as fh:
                data = json.load(fh)
            _overrides = {str(k): str(v) for k, v in data.items() if v} if isinstance(data, dict) else {}
        except (OSError, ValueError):
            _overrides = {}
    return _overrides


def species_override_for_file(filename: Union[str, Path, None]) -> Optional[str]:
    if not filename:
        return None
    return species_overrides().get(os.path.basename(str(filename)))


def set_species_override(filename: Union[str, Path], species: Optional[str]) -> None:
    """Assign *species* to the line-list file *filename* (``None`` removes it) and save."""
    overrides = species_overrides()
    name = os.path.basename(str(filename))
    if species:
        overrides[name] = species
    else:
        overrides.pop(name, None)
    os.makedirs(os.path.dirname(species_file_path), exist_ok=True)
    with open(species_file_path, "w") as fh:
        json.dump(overrides, fh, indent=2, sort_keys=True)


def reload_species_overrides() -> None:
    """Forget the in-memory copy so the file is read again on next use."""
    global _overrides
    _overrides = None
