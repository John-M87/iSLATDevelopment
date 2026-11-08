"""
ModelOverlays - compute and draw the molecule models laid over a spectrum.

Shared by every plot that shows model spectra (the full spectrum and its
residual / optical-depth / sample-stack variants, the three-panel grid and
the fit-lines grid):

* :func:`resolve_matched_sampling` - the "Match Pix. Sampling" settings
* :func:`build_molecule_traces` - one :class:`MoleculeTrace` per visible molecule
* :func:`summed_model_flux` - the visible molecules' summed model
* :func:`plot_molecule_trace` / :func:`draw_molecule_traces` - dashed model lines
* :func:`draw_summed_fill` - the filled summed model
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING, Any, Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple,
)

import numpy as np

from .ArtistTags import MOLECULE_TAG, SUMMED_TAG, remove_tagged, tag_artist

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

#: ``getter(molecule, wave_data, interpolate_to_input=..., target_wavelengths=..., extend_to_range=...)``
#: returning ``(wavelengths, values)`` - e.g. :meth:`BasePlot.get_molecule_spectrum_data`.
TraceGetter = Callable[..., Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


class MoleculeTrace(NamedTuple):
    """One molecule's model, ready to draw.

    A tuple, so code that unpacks the older ``(lam, flux, color, label, name)``
    cache entries keeps working.
    """
    lam: np.ndarray
    flux: np.ndarray
    color: Any
    label: str
    name: str


def molecule_display_name(molecule: "Molecule") -> str:
    """The user-facing label of *molecule*."""
    return getattr(molecule, "displaylabel", getattr(molecule, "name", "unknown"))


def molecule_color(molecule: "Molecule") -> str:
    """The color of *molecule* (``"blue"`` when it has none)."""
    color = getattr(molecule, "color", None)
    return color if color else "blue"


def resolve_matched_sampling(
    molecules: Optional["MoleculeDict"],
    ref_wave: Optional[np.ndarray],
) -> Tuple[Any, Optional[np.ndarray]]:
    """Return ``(interpolate_to_input, target_wavelengths)`` for drawing models.

    *ref_wave* are the observer-frame data wavelengths.  When matched spectral
    sampling is off (or *molecules* cannot tell) the models are not resampled
    and the target is ``None``.
    """
    if (molecules is None or ref_wave is None
            or not hasattr(molecules, "get_matched_sampling_wavelengths")):
        return False, None
    use_interp, target_wave = molecules.get_matched_sampling_wavelengths(ref_wave)
    return use_interp, (target_wave if use_interp else None)


def build_molecule_traces(
    molecules: Optional["MoleculeDict"],
    wave_data: Optional[np.ndarray],
    getter: TraceGetter,
    wave_data_obs: Optional[np.ndarray] = None,
    *,
    skip_errors: bool = False,
) -> Tuple[List[MoleculeTrace], List[str], List[str]]:
    """Compute a :class:`MoleculeTrace` for every visible molecule.

    Parameters
    ----------
    molecules : MoleculeDict or None
        The molecules; only the visible ones are used.
    wave_data : np.ndarray
        Display (rest-frame) wavelengths passed to *getter*.
    getter : callable
        Returns ``(wavelengths, values)`` for one molecule, e.g.
        :meth:`BasePlot.get_molecule_spectrum_data` (flux) or
        :meth:`BasePlot.get_molecule_tau_data` (optical depth).
    wave_data_obs : np.ndarray, optional
        Observer-frame wavelengths for matched spectral sampling
        (default: *wave_data*).
    skip_errors : bool
        Leave out a molecule whose model raises instead of propagating the error.

    Returns
    -------
    (traces, labels, colors)
        *labels* and *colors* list every visible molecule (for the legend),
        including any whose model came back empty.
    """
    traces: List[MoleculeTrace] = []
    if molecules is None:
        return traces, [], []
    visible = molecules.get_visible_molecules(return_objects=True)
    labels = [molecule_display_name(m) for m in visible]
    colors = [molecule_color(m) for m in visible]

    ref_wave = wave_data_obs if wave_data_obs is not None else wave_data
    use_interp, target_wave = resolve_matched_sampling(molecules, ref_wave)

    for mol in visible:
        try:
            lam, values = getter(
                mol, wave_data,
                interpolate_to_input=use_interp,
                target_wavelengths=target_wave,
                extend_to_range=True,
            )
            if lam is None or values is None or len(values) == 0:
                continue
            traces.append(MoleculeTrace(
                np.asarray(lam), np.asarray(values),
                molecule_color(mol), molecule_display_name(mol),
                getattr(mol, "name", "unknown"),
            ))
        except Exception:
            if not skip_errors:
                raise
    return traces, labels, colors


def summed_model_flux(
    molecules: Optional["MoleculeDict"],
    wave_data_obs: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """The visible molecules' summed model ``(wave, flux)``, or ``(None, None)``.

    *wave_data_obs* must be the observer-frame wavelengths; the returned
    wavelengths are rest-frame.  Covers the molecules' whole wavelength range
    (``extend_to_global_range``).  Errors give ``(None, None)``.
    """
    if molecules is None:
        return None, None
    try:
        wave, flux = molecules.get_summed_flux(
            wave_data_obs, visible_only=True, extend_to_global_range=True,
        )
    except Exception:
        return None, None
    return wave, flux


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def plot_molecule_trace(
    ax: "Axes",
    trace: Sequence[Any],
    xr: Tuple[float, float],
    *,
    linewidth: float,
    alpha: float,
    zorder: float,
    linestyle: str = "--",
    offset: float = 0.0,
) -> Optional["Line2D"]:
    """Draw the part of *trace* inside ``xr`` as a model line, shifted up by *offset*.

    *trace* is a :class:`MoleculeTrace` or any ``(lam, flux, color, label,
    name)`` tuple.  The line is tagged with the molecule name.  Returns
    ``None`` when nothing of the trace falls in range.
    """
    lam, flux, color, label, name = trace[:5]
    lam = np.asarray(lam)
    flux = np.asarray(flux)
    mask = (lam >= xr[0]) & (lam <= xr[1])
    if not np.any(mask):
        return None
    y = flux[mask] + offset if offset else flux[mask]
    (line,) = ax.plot(
        lam[mask], y,
        linestyle=linestyle, color=color, alpha=alpha,
        linewidth=linewidth, label=label, zorder=zorder,
    )
    tag_artist(line, MOLECULE_TAG, name)
    return line


def draw_molecule_traces(
    ax: "Axes",
    traces: Iterable[Sequence[Any]],
    xr: Tuple[float, float],
    *,
    alpha: float,
    zorder: float,
    linewidth: float = 1.0,
    linewidth_for: Optional[Callable[[str], float]] = None,
    linestyle: str = "--",
    offset: float = 0.0,
) -> List["Line2D"]:
    """Draw every trace in *traces* with :func:`plot_molecule_trace`.

    *linewidth_for(name)* gives a per-molecule line width (e.g. bold
    molecules); otherwise every line uses *linewidth*.  Returns the lines drawn.
    """
    lines: List["Line2D"] = []
    for trace in traces:
        lw = linewidth_for(trace[4]) if linewidth_for is not None else linewidth
        line = plot_molecule_trace(
            ax, trace, xr,
            linewidth=lw, alpha=alpha, zorder=zorder,
            linestyle=linestyle, offset=offset,
        )
        if line is not None:
            lines.append(line)
    return lines


def draw_summed_fill(
    ax: "Axes",
    wave: Optional[np.ndarray],
    flux: Optional[np.ndarray],
    *,
    color: Any = "lightgray",
    zorder: float = 1,
    xr: Optional[Tuple[float, float]] = None,
    baseline: float = 0.0,
    label: str = "Sum",
    replace: bool = False,
) -> Optional["PolyCollection"]:
    """Fill the summed model between *baseline* and ``flux + baseline``.

    Only the part inside *xr* is drawn when it is given.  Nothing is drawn
    (and ``None`` returned) when no flux in range is positive.  With
    *replace* the axes' existing summed fills are removed first, but only
    when a new one is drawn.  The fill is tagged as the summed model.
    """
    if wave is None or flux is None:
        return None
    wave = np.asarray(wave)
    flux = np.asarray(flux)
    if xr is not None:
        mask = (wave >= xr[0]) & (wave <= xr[1])
        wave, flux = wave[mask], flux[mask]
    if len(flux) == 0 or not np.any(flux > 0):
        return None
    if replace:
        remove_tagged(ax, SUMMED_TAG, lines=False)
    fill = ax.fill_between(
        wave, baseline, flux + baseline if baseline else flux,
        color=color, alpha=1.0, label=label, zorder=zorder,
    )
    tag_artist(fill, SUMMED_TAG)
    return fill
