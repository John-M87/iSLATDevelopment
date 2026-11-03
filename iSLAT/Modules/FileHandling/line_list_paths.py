"""Locating line-list files referenced by molecule lists.

Molecule lists store each molecule's line-list path.  Files inside
``DATAFILES`` are stored relative to it (e.g. ``HITRANdata/data_Hitran_CO.par``
or ``HITRANdata/<sub-folder>/<file>.par``) so saves stay
portable; files elsewhere are stored as absolute paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

from . import absolute_data_files_path, hitran_data_folder_path

PathLike = Union[str, os.PathLike]

#: Sub-folders of HITRANdata that hold derived files rather than line lists.
_SKIP_DIRS = {"cache", "__pycache__"}


def to_portable_line_list_path(path: PathLike) -> str:
    """Return the path to store for the line-list file at *path*.

    Relative to ``DATAFILES`` (keeping any sub-folders, in ``/`` form) when
    the file is inside it, otherwise absolute.  A relative *path* is taken
    relative to the current directory.
    """
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(Path(absolute_data_files_path).resolve()).as_posix()
    except ValueError:
        return str(resolved)


def find_existing_line_list(path: PathLike) -> Optional[str]:
    """Return the existing file *path* refers to, or ``None``.

    Tried in the same order the line-list loader uses: as given when
    absolute, otherwise relative to ``DATAFILES`` and then to the current
    directory.
    """
    text = os.path.expanduser(str(path))
    candidates = [text] if os.path.isabs(text) else [
        os.path.join(absolute_data_files_path, text), text,
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _files_named(name: str, root: PathLike) -> List[Path]:
    matches: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        if name in filenames:
            matches.append(Path(dirpath) / name)
    return matches


def repair_line_list_path(path: Optional[PathLike]) -> Optional[PathLike]:
    """Return *path*, or a corrected path when the file it names is missing.

    Older saves could record a line list in a ``HITRANdata`` sub-folder as
    ``HITRANdata/<file>``.  When *path* does not exist and exactly one file
    with the same name exists anywhere under ``HITRANdata``, that file's
    portable path is returned; otherwise *path* is returned unchanged.
    """
    if not path or find_existing_line_list(path) is not None:
        return path
    name = os.path.basename(str(path))
    if not name or not os.path.isdir(hitran_data_folder_path):
        return path
    matches = _files_named(name, hitran_data_folder_path)
    if len(matches) != 1:
        return path
    repaired = to_portable_line_list_path(matches[0])
    print(f"Line list '{path}' not found; using '{repaired}'.")
    return repaired
