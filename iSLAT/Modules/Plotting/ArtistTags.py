"""
ArtistTags - find, show / hide and remove tagged matplotlib artists.

iSLAT marks the artists it draws with plain attributes (``line._islat_observed
= True``, ``line._molecule_name = "CO"``, ...) so they can be found again
later, e.g. to toggle the observed data or redraw the summed model.  The
helpers here replace the hand-written ``for art in ax.lines: if hasattr(art,
tag)`` loops that every plot and view used to carry.

An artist carries a tag when it has the attribute at all (``hasattr``).  The
molecule tag is the exception: its value is the molecule name, and
:func:`iter_molecule_lines` only counts lines whose name is not ``None``.
"""
from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence, Union

#: Observed spectrum: the data line, error bars and caps.
OBSERVED_TAG = "_islat_observed"
#: Summed model of the visible molecules (a filled area).
SUMMED_TAG = "_islat_summed"
#: Molecule model line; the attribute value is the molecule name.
MOLECULE_TAG = "_molecule_name"
#: Saved-line markers and labels.
SAVED_LINE_TAG = "_islat_saved_line"
#: Atomic-line markers and labels.
ATOMIC_LINE_TAG = "_islat_atomic_line"
#: Shading of a range dragged out while picking panel ranges on the plot.
PICKED_RANGE_TAG = "_islat_picked_range"

Tags = Union[str, Sequence[str]]


def tag_artist(artist: Any, tag: str, value: Any = True) -> Any:
    """Stamp *artist* with *tag* (set to *value*) and return it."""
    setattr(artist, tag, value)
    return artist


def _as_tags(tag: Tags) -> tuple:
    return (tag,) if isinstance(tag, str) else tuple(tag)


def iter_tagged(
    ax: Any,
    tag: Tags,
    *,
    lines: bool = True,
    collections: bool = True,
    texts: bool = False,
    patches: bool = False,
) -> Iterator[Any]:
    """Yield the artists on *ax* that carry *tag* (or any of several tags).

    The keyword flags choose which artist lists are searched.  The lists are
    copied first, so callers may remove the yielded artists while iterating.
    """
    tags = _as_tags(tag)
    pools: List[Any] = []
    if lines:
        pools.extend(ax.lines)
    if collections:
        pools.extend(ax.collections)
    if texts:
        pools.extend(ax.texts)
    if patches:
        pools.extend(ax.patches)
    for artist in pools:
        if any(hasattr(artist, t) for t in tags):
            yield artist


def has_tagged(ax: Any, tag: Tags, **where: bool) -> bool:
    """Return ``True`` when *ax* has at least one artist carrying *tag*."""
    return next(iter_tagged(ax, tag, **where), None) is not None


def set_tagged_visible(ax: Any, tag: Tags, visible: bool, **where: bool) -> bool:
    """Show or hide every artist on *ax* carrying *tag*; ``True`` when any was found."""
    found = False
    for artist in iter_tagged(ax, tag, **where):
        artist.set_visible(visible)
        found = True
    return found


def remove_tagged(ax: Any, tag: Tags, **where: bool) -> int:
    """Remove every artist on *ax* carrying *tag* and return how many were removed."""
    removed = 0
    for artist in iter_tagged(ax, tag, **where):
        artist.remove()
        removed += 1
    return removed


# ----------------------------------------------------------------------
# Molecule model lines
# ----------------------------------------------------------------------
def iter_molecule_lines(ax: Any, name: Optional[str] = None) -> Iterator[Any]:
    """Yield the molecule model lines on *ax*: all of them, or only *name*'s."""
    for line in list(ax.lines):
        mol_name = getattr(line, MOLECULE_TAG, None)
        if mol_name is None:
            continue
        if name is None or mol_name == name:
            yield line


def set_molecule_visible(ax: Any, name: str, visible: bool) -> bool:
    """Show or hide *name*'s model lines on *ax*; ``True`` when any was found."""
    found = False
    for line in iter_molecule_lines(ax, name):
        line.set_visible(visible)
        found = True
    return found


def remove_molecule_lines(ax: Any, name: Optional[str] = None) -> None:
    """Remove *name*'s model lines from *ax* (every molecule's when *name* is ``None``)."""
    for line in iter_molecule_lines(ax, name):
        line.remove()
