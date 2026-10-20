"""
PopulationDiagramPlot - Boltzmann / rotation diagram.

Plots ``ln(4πF / (hv A_u g_u))`` vs upper-state energy *E_u* using the computed intensity data from one or more :class:`Molecule` / :class:`Intensity` objects.

Supports **multi-molecule / multi-component** overlays with automatic color coding, as well as **property-based color mapping** 
(e.g. color by upper-level energy, transition label, Einstein A coefficient, etc.).

Can be used standalone (notebook / script) or embedded in a GUI layout.
"""
from __future__ import annotations

import weakref
from typing import (
    Optional, Tuple, List, Dict, Any, Sequence, Union, TYPE_CHECKING,
)
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Normalize, LogNorm
import matplotlib

from .BasePlot import BasePlot
import iSLAT.Constants as c
from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine as _MoleculeLine
from iSLAT.Modules.DataTypes.PlotAxisRegistry import PlotAxisRegistry as _AxisReg

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

# Default color cycle used when the caller does not supply colors.
_DEFAULT_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Marker cycle for shape-by encoding
_DEFAULT_MARKERS: List[str] = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

class PopulationDiagramPlot(BasePlot):
    """
    Boltzmann / rotation diagram for one or more molecules.

    The plot can be created from:

    * A single :class:`Molecule` (original API, fully backward-compatible).
    * A single bare :class:`Intensity` together with physical metadata.
    * A *list* of :class:`Molecule` objects, each rendered in its own
      color.
    * A :class:`MoleculeDict`, where every visible molecule is plotted.

    After the diagram has been generated, the points can optionally be
    re-colored based on a stored molecular property (e.g. upper-level
    energy, transition label, Einstein-A coefficient) via
    :meth:`color_by`.

    Parameters
    ----------
    molecule : Molecule, optional
        Single molecule whose intensity data is used.  Mutually
        exclusive with *intensity* and *molecules*.
    molecules : list[Molecule] | MoleculeDict, optional
        Multiple molecules to overlay.  Each is color-coded
        automatically (or by its ``.color`` attribute).  Mutually
        exclusive with *molecule* and *intensity*.
    intensity : Intensity, optional
        Pre-built :class:`Intensity` object.  When provided, *radius*
        and *distance* must also be supplied.  Mutually exclusive with
        *molecule* and *molecules*.
    name : str, optional
        Display name used in the title when *intensity* is given
        directly.  Ignored when *molecule* / *molecules* is provided.
    color : str, optional
        Marker color when *intensity* is given directly.
    radius : float, optional
        Emitting radius in AU - required when *intensity* is given
        directly.  Defaults to ``1.0``.
    distance : float, optional
        Distance to source in pc - required when *intensity* is given
        directly.  Defaults to ``160.0``.
    highlight_lines : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples rendered as
        larger scatter points.
    figsize : tuple, optional
        Defaults to ``(6, 5)``.
    ax : Axes, optional
        Pre-existing axes for embedding.
    """
    def __init__(
        self,
        molecule: Optional["Molecule"] = None,
        *,
        molecules: Optional[Union[Sequence["Molecule"], "MoleculeDict"]] = None,
        intensity: Optional["Intensity"] = None,
        name: Optional[str] = None,
        color: Optional[str] = None,
        radius: Optional[float] = None,
        distance: Optional[float] = None,
        highlight_lines: Optional[
            List[Tuple["MoleculeLine", float, Optional[float]]]
        ] = None,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        super().__init__(figsize=figsize or (6, 5), **kwargs)

        # ---- Validate mutual-exclusion --------------------------------
        n_sources = sum(
            x is not None for x in (molecule, molecules, intensity)
        )
        if n_sources > 1:
            raise ValueError(
                "Provide only one of 'molecule', 'molecules', or "
                "'intensity', not more than one."
            )

        self.molecule = molecule

        # Multi-molecule storage (normalised to a list later)
        self._molecules_input = molecules

        # Bare-intensity path
        self._intensity_obj: Optional["Intensity"] = intensity
        self._intensity_name: str = name or "Unknown"
        self._intensity_color: Optional[str] = color
        self._intensity_radius: float = radius if radius is not None else 1.0
        self._intensity_distance: float = (
            distance if distance is not None else 160.0
        )

        self.highlight_lines = highlight_lines
        self._external_ax = ax

        # Cached per-component data (populated by generate_plot)
        self._component_data: List[Dict[str, Any]] = []

        # Active color-map state (set by color_by / clear_color_mapping)
        self._color_mapping: Optional[Dict[str, Any]] = None
        # Tracked colorbar so it can be removed on the next regeneration
        self._colorbar = None
        # Axis configuration
        self._x_prop: str = "eu"
        self._y_prop: str = "rd_yax"
        self._x_log: bool = False
        self._y_log: bool = False
        # When True, the plot is locked to all active/visible molecules in
        # _molecules_dict_ref and regenerates automatically when the set changes.
        self._all_molecules_mode: bool = False
        self._molecules_dict_ref = None   # MoleculeDict driving all-molecules mode
        self._molecules_change_cb = None  # registered callback handle
        self._comparison_change_cb = None  # registered callback handle
        # Cache for the last render_active_lines call so active scatter can
        # be restored after generate_plot clears the axes.
        self._active_lines_cache: Optional[Dict[str, Any]] = None
        # Marker-shape mapping (set by shape_by / clear_shape_mapping)
        self._shape_mapping: Optional[Dict[str, Any]] = None
        # Fixed marker size override (None = use per-mode default)
        self._marker_size: Optional[float] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def ax(self) -> Axes:
        return self._ax

    @property
    def component_data(self) -> List[Dict[str, Any]]:
        """Per-component data computed during the last ``generate_plot`` call.

        Each element is a dict with keys: ``'name'``, ``'color'``,
        ``'eu'``, ``'rd_yax'``, ``'wavelength'``, ``'intens'``,
        ``'a_stein'``, ``'g_up'``, ``'lev_up'``, ``'lev_low'``,
        ``'e_low'``, ``'valid_mask'``.
        """
        return self._component_data

    # ------------------------------------------------------------------
    # Active-lines API implementation
    # ------------------------------------------------------------------
    def render_active_lines(
        self,
        line_data: List[Tuple["MoleculeLine", float, Any]],
        active_lines: List[Any],
        *,
        molecule: Optional["Molecule"] = None,
        threshold: float = 0.0,
        color: str = "green",
        **kwargs,
    ) -> Any:
        """Render scatter points for *line_data* on the population diagram.

        Appends / updates entries in *active_lines* and returns the
        :class:`~matplotlib.collections.PathCollection` scatter artist.

        Parameters
        ----------
        line_data : list[tuple]
            ``(MoleculeLine, intensity, tau)`` triples.
        active_lines : list
            Mutable list updated in-place with scatter artist references.
        molecule : Molecule
            Active molecule (required for ``radius`` and ``distance``).
        threshold : float
            Fraction (0-1) of max intensity below which lines are excluded.
        color : str
            Scatter point color.

        Returns
        -------
        PathCollection or None
            The scatter artist, or *None* if nothing was plotted.
        """
        if (
            not line_data
            or self._ax is None
            or molecule is None
        ):
            return None

        intensities = [i for _, i, _ in line_data]
        max_intensity = max(intensities) if intensities else 1.0
        if max_intensity <= 0:
            return None

        radius = getattr(molecule, "radius", 1.0)
        distance = getattr(molecule, "distance", 140.0)
        area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
        dist = distance * c.PARSEC_CM
        beam_s = area / dist ** 2

        x_vals: List[float] = []
        y_vals: List[float] = []
        # Each entry: (info_dict, al_idx) where al_idx is the index of the
        # matching vline entry in active_lines, or None when no vline exists.
        value_data_list: List[Tuple[Dict[str, Any], Optional[int]]] = []

        from iSLAT.Modules.DataTypes.Intensity import Intensity as _Intensity

        mol_name = getattr(molecule, "name", "") if molecule else ""

        # Resolve molecule_id once for quantum-field axis lookups.
        _mol_id: Optional[str] = self._molecule_id_of(molecule)

        # Index the existing active_lines entries that belong to *this*
        # molecule by transition identity.  Positional matching is not safe:
        # active_lines is shared across molecules (the primary molecule's
        # entries come first) and LineInspectionPlot skips lines whose
        # lineheight rounds to zero, so both offsets and gaps occur.
        pending_by_key: Dict[Tuple[Any, Any, Any], List[int]] = {}
        for entry_idx, entry in enumerate(active_lines):
            entry_info = entry[3] if len(entry) > 3 else None
            if not isinstance(entry_info, dict):
                continue
            if (entry_info.get("molecule_name") or "") != mol_name:
                continue
            entry_key = (
                entry_info.get("lam"),
                entry_info.get("e_up"),
                entry_info.get("a_stein"),
            )
            pending_by_key.setdefault(entry_key, []).append(entry_idx)

        for line, intensity, tau_val in line_data:
            frac = intensity / max_intensity
            if frac < threshold:
                continue  # skipped by both panels

            # Locate the vline entry for this exact transition, if one exists.
            _matches = pending_by_key.get((line.lam, line.e_up, line.a_stein))
            current_al_idx = _matches.pop(0) if _matches else None

            if any(
                x is None
                for x in [intensity, line.a_stein, line.g_up, line.lam]
            ):
                continue  # no scatter point for this line

            F = intensity * beam_s
            freq = c.SPEED_OF_LIGHT_MICRONS / line.lam
            rd_yax = (
                np.log(4 * np.pi * F / (line.a_stein * c.PLANCK_CONSTANT * freq * line.g_up))
                if (F > 0 and line.a_stein > 0) else np.nan
            )

            # Delegate per-line metadata (including FWHM breakdown) to Intensity
            # so all physics computation lives on the data object.
            info = _Intensity.get_line_info(line, intensity, tau_val, molecule=molecule)
            info["rd_yax"] = rd_yax  # computed here using beam_s
            info["intensity_percent"] = frac * 100
            info["molecule_name"] = mol_name  # needed for pick-highlight routing

            # Build the axis-property lookup map; FWHM values come from info.
            _LINE_MAP = {
                "eu":                    line.e_up,
                "e_low":                 line.e_low,
                "rd_yax":                rd_yax,
                "wavelength":            line.lam,
                "intens":                intensity,
                "a_stein":               line.a_stein,
                "g_up":                  line.g_up,
                "g_low":                 line.g_low,
                "tau":                   tau_val,
                "fwhm_instrumental_kms": info.get("fwhm_instrumental_kms"),
                "fwhm_convolved_kms":    info.get("fwhm_convolved_kms"),
            }
            xv = _LINE_MAP.get(self._x_prop)
            if xv is None:
                xv = _AxisReg.resolve_scalar(self._x_prop, line, _mol_id)
            yv = _LINE_MAP.get(self._y_prop)
            if yv is None:
                yv = _AxisReg.resolve_scalar(self._y_prop, line, _mol_id)
            try:
                xv = float(xv) if xv is not None else None
                yv = float(yv) if yv is not None else None
            except (TypeError, ValueError):
                xv = yv = None
            if xv is None or yv is None or np.isnan(xv) or np.isnan(yv):
                continue  # no scatter point for this line

            x_vals.append(xv)
            y_vals.append(yv)
            value_data_list.append((info, current_al_idx))

        if not x_vals:
            return None

        ax = self._ax
        sc = ax.scatter(
            x_vals, y_vals, s=30,
            color=color, edgecolors="black", picker=True,
        )

        for scatter_idx, (info, al_idx) in enumerate(value_data_list):
            info["_scatter_point_index"] = scatter_idx
            if al_idx is not None and al_idx < len(active_lines):
                # Update the existing vline entry with scatter artist + pop-diagram info
                active_lines[al_idx][2] = sc
                active_lines[al_idx][3].update(info)
            else:
                # No corresponding vline (pop-diagram rendered standalone)
                active_lines.append([None, None, sc, info])

        # Cache params so generate_plot can restore the scatter after a
        # full axes clear (e.g. when set_axes or color_by is called).
        self._active_lines_cache = {
            "line_data": line_data,
            "active_lines": active_lines,
            "molecule": molecule,
            "threshold": threshold,
            "color": color,
        }

        return sc

    def clear_active_lines(self, active_lines: List[Any]) -> None:
        """Remove scatter artists from *active_lines* and clear the list.

        Parameters
        ----------
        active_lines : list
            Mutable list of ``[vline, text, scatter, info]`` entries.
        """
        seen_scatter = set()
        for entry in active_lines:
            sc = entry[2] if len(entry) > 2 else None
            if sc is not None and id(sc) not in seen_scatter:
                seen_scatter.add(id(sc))
                if getattr(sc, 'axes', None) is not None:
                    try:
                        sc.remove()
                    except (ValueError, AttributeError):
                        pass
        active_lines.clear()
        # Clear the cache so generate_plot won't re-render stale scatter.
        self._active_lines_cache = None

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Generate the population diagram."""
        # Remove any existing colorbar before (re-)drawing so stale
        # colorbar axes don't accumulate on the figure.
        if self._colorbar is not None:
            try:
                self._colorbar.remove()
            except Exception:
                pass
            self._colorbar = None

        if self._external_ax is not None:
            self._ax = self._external_ax
        else:
            self._ensure_figure()
            self.fig.clf()
            self._ax = self.fig.add_subplot(111)

        ax = self._ax
        ax.clear()

        fg = self._get_theme_value("foreground", "black")

        # ---- Resolve data sources ------------------------------------
        components = self._resolve_components()

        if not components:
            ax.set_title("No molecule selected")
            return

        # ---- Compute population-diagram values for each component ----
        self._component_data = []
        all_valid_x: List[np.ndarray] = []
        all_valid_y: List[np.ndarray] = []

        for comp in components:
            cdata = self._compute_component(comp)
            if cdata is not None:
                self._component_data.append(cdata)
                mask = cdata["valid_mask"]
                x_arr = self._get_axis_array(cdata, self._x_prop)
                y_arr = self._get_axis_array(cdata, self._y_prop)
                if np.any(mask) and x_arr is not None and y_arr is not None:
                    all_valid_x.append(x_arr[mask])
                    all_valid_y.append(y_arr[mask])

        if not self._component_data:
            title = components[0]["name"] if components else "Unknown"
            ax.set_title(f"{title} - No intensity data", color=fg)
            return

        if not all_valid_x or not all_valid_y:
            names = ", ".join(cd["name"] for cd in self._component_data)
            ax.set_title(
                f"{names} - No valid data for population diagram",
                color=fg,
            )
            return

        # ---- Render scatter points -----------------------------------
        if self._color_mapping is not None:
            self._render_colormapped(ax)
        else:
            self._render_by_component(ax)

        # ---- Highlighted lines (e.g. from an inspection selection) ---
        if self.highlight_lines and self._component_data:
            first = self._component_data[0]
            self._render_highlights(ax, first["beam_s"])

        # ---- Axis limits & labels ------------------------------------
        cat_valid_x = np.concatenate(all_valid_x)
        cat_valid_y = np.concatenate(all_valid_y)
        _all_x_arrs = [self._get_axis_array(cd, self._x_prop) for cd in self._component_data]
        _all_y_arrs = [self._get_axis_array(cd, self._y_prop) for cd in self._component_data]
        all_x_data = np.concatenate([a for a in _all_x_arrs if a is not None]) if any(a is not None for a in _all_x_arrs) else cat_valid_x
        all_y_data = np.concatenate([a for a in _all_y_arrs if a is not None]) if any(a is not None for a in _all_y_arrs) else cat_valid_y

        # Set scale BEFORE computing limits so matplotlib never receives
        # non-positive limits on a log axis (which causes the squish bug).
        ax.set_xscale("log" if self._x_log else "linear")
        ax.set_yscale("log" if self._y_log else "linear")

        def _finite(arr: np.ndarray) -> np.ndarray:
            """Return only finite (non-NaN, non-Inf) values."""
            return arr[np.isfinite(arr)]

        # For log axes restrict the data used for limit computation to
        # strictly positive values; fall back to auto-limits if none exist.
        if self._x_log:
            pos_x = _finite(cat_valid_x[cat_valid_x > 0])
            pos_all_x = _finite(all_x_data[all_x_data > 0])
            if len(pos_x) == 0 or len(pos_all_x) == 0:
                ax.autoscale(axis="x")
            else:
                _xlo = np.nanmin(pos_all_x)
                _xhi = np.nanmax(pos_x)
                _factor = (_xhi / _xlo) ** 0.05 if _xlo > 0 and _xhi > _xlo else 1.05
                ax.set_xlim(_xlo / _factor, _xhi * _factor)
        else:
            # X limits (linear)
            _fx_all = _finite(all_x_data)
            _fx_valid = _finite(cat_valid_x)
            if len(_fx_all) == 0 or len(_fx_valid) == 0:
                ax.autoscale(axis="x")
            elif self._x_prop == "eu":
                ax.set_xlim(np.nanmin(_fx_all) - 50, np.nanmax(_fx_valid))
            else:
                _xr = np.nanmax(_fx_valid) - np.nanmin(_fx_all)
                _xpad = _xr * 0.05 if _xr > 0 else max(abs(np.nanmin(_fx_all)) * 0.05, 1)
                ax.set_xlim(np.nanmin(_fx_all) - _xpad, np.nanmax(_fx_valid) + _xpad)

        if self._y_log:
            pos_y = _finite(cat_valid_y[cat_valid_y > 0])
            pos_all_y = _finite(all_y_data[all_y_data > 0])
            if len(pos_y) == 0 or len(pos_all_y) == 0:
                ax.autoscale(axis="y")
            else:
                _ylo = np.nanmin(pos_y)
                _yhi = np.nanmax(pos_all_y)
                _factor = (_yhi / _ylo) ** 0.05 if _ylo > 0 and _yhi > _ylo else 1.05
                ax.set_ylim(_ylo / _factor, _yhi * _factor)
        else:
            # Y limits (linear)
            _fy_all = _finite(all_y_data)
            _fy_valid = _finite(cat_valid_y)
            if len(_fy_all) == 0 or len(_fy_valid) == 0:
                ax.autoscale(axis="y")
            elif self._y_prop == "rd_yax":
                ax.set_ylim(np.nanmin(_fy_valid), np.nanmax(_fy_all) + 0.5)
            else:
                _yr = np.nanmax(_fy_all) - np.nanmin(_fy_valid)
                _ypad = _yr * 0.05 if _yr > 0 else max(abs(np.nanmin(_fy_valid)) * 0.05, 1)
                ax.set_ylim(np.nanmin(_fy_valid) - _ypad, np.nanmax(_fy_all) + _ypad)

        ax.set_ylabel(self._get_axis_label(self._y_prop), color=fg, labelpad=-1)
        ax.set_xlabel(self._get_axis_label(self._x_prop), color=fg)

        # ---- User-specified axis limit overrides ---------------------
        def _apply_lim_override(ax_obj, axis: str, lim_spec, data_arr: np.ndarray) -> None:
            """Apply a (mode, lo, hi) limit spec onto *ax_obj* for *axis* ('x'|'y')."""
            if lim_spec is None:
                return
            mode, lo_raw, hi_raw = lim_spec[0], lim_spec[1], lim_spec[2]
            finite = data_arr[np.isfinite(data_arr)]
            set_lim = ax_obj.set_xlim if axis == 'x' else ax_obj.set_ylim
            cur_lo, cur_hi = (ax_obj.get_xlim() if axis == 'x' else ax_obj.get_ylim())
            if mode == 'exact':
                new_lo = float(lo_raw) if lo_raw is not None else cur_lo
                new_hi = float(hi_raw) if hi_raw is not None else cur_hi
                set_lim(new_lo, new_hi)
            elif mode == 'percentile':
                if len(finite) == 0:
                    return
                new_lo = float(np.nanpercentile(finite, float(lo_raw))) if lo_raw is not None else cur_lo
                new_hi = float(np.nanpercentile(finite, float(hi_raw))) if hi_raw is not None else cur_hi
                set_lim(new_lo, new_hi)

        _cat_x = np.concatenate(all_valid_x) if all_valid_x else np.array([])
        _cat_y = np.concatenate(all_valid_y) if all_valid_y else np.array([])
        _apply_lim_override(ax, 'x', getattr(self, '_x_lim', None), _cat_x)
        _apply_lim_override(ax, 'y', getattr(self, '_y_lim', None), _cat_y)

        # Title: single component → use its name; multi → generic
        if len(self._component_data) == 1:
            title = f"{self._component_data[0]['name']} Population diagram"
        else:
            title = "Population diagram"
        ax.set_title(title, fontsize="medium", color=fg)

        # Legend for multi-component mode
        if len(self._component_data) > 1 and self._color_mapping is None:
            self._render_component_legend(ax)

        # Re-render any previously plotted active-line scatter so it
        # persists across axes changes and color_by calls.
        if self._active_lines_cache is not None:
            cache = self._active_lines_cache
            al = cache["active_lines"]
            # Do NOT call al.clear() here - that would destroy the vline
            # entries added by LineInspectionPlot.render_active_lines,
            # making the vlines in the inspection panel un-pickable after
            # an axis change.  Instead, just null out the stale scatter
            # references so render_active_lines updates them in-place.
            for entry in al:
                if len(entry) > 2:
                    entry[2] = None
            self.render_active_lines(
                cache["line_data"],
                al,
                molecule=cache["molecule"],
                threshold=cache["threshold"],
                color=cache["color"],
            )

        if self.fig is not None:
            self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # Component resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _molecule_id_of(molecule: Optional["Molecule"]) -> Optional[str]:
        """Return the species id used for quantum-schema lookups.

        Prefers the line list's ``molecule_id`` (the key HITRAN schemas are
        registered under), falling back to a ``molecule_id`` attribute on the
        molecule itself.
        """
        if molecule is None:
            return None
        ll = getattr(molecule, "line_list", None)
        mol_id = getattr(ll, "molecule_id", None) if ll is not None else None
        if mol_id is None:
            mol_id = getattr(molecule, "molecule_id", None)
        return mol_id

    def _input_molecules(self) -> List["Molecule"]:
        """Return the molecules currently driving this diagram, in plot order.

        Resolves the single-molecule, list and :class:`MoleculeDict` input
        modes the same way :meth:`_resolve_components` does.  Returns an
        empty list in bare-intensity mode.
        """
        if self.molecule is not None:
            return [self.molecule]
        mol_seq = self._molecules_input
        if mol_seq is None:
            return []
        # MoleculeDict → use active set when in all-molecules mode,
        # otherwise fall back to visible molecules.
        if hasattr(mol_seq, "get_active_set") and self._all_molecules_mode:
            mol_seq = mol_seq.get_active_set()
        elif hasattr(mol_seq, "get_visible_molecules"):
            mol_seq = mol_seq.get_visible_molecules(return_objects=True)
        elif hasattr(mol_seq, "values") and not isinstance(mol_seq, list):
            mol_seq = mol_seq.values()
        return list(mol_seq)

    def get_molecule_ids(self) -> List[str]:
        """Return the distinct species ids of the molecules on this diagram.

        GUI dialogs pass this to
        :meth:`~iSLAT.Modules.DataTypes.PlotAxisRegistry.PlotAxisRegistry.get_axis_options`
        so only quantum numbers these species actually define are offered.
        Molecules with no resolvable id are skipped.
        """
        ids: List[str] = []
        for mol in self._input_molecules():
            mid = self._molecule_id_of(mol)
            if mid is not None and mid not in ids:
                ids.append(mid)
        return ids

    def _resolve_components(self) -> List[Dict[str, Any]]:
        """Normalise the various input modes into a list of component dicts.

        Each dict has keys: ``'molecule'``, ``'intensity'``, ``'name'``,
        ``'color'``, ``'radius'``, ``'distance'``.
        """
        components: List[Dict[str, Any]] = []

        # --- One or more Molecules (single, list or MoleculeDict) ---
        molecules = self._input_molecules()
        if molecules:
            for idx, mol in enumerate(molecules):
                components.append(self._mol_to_component(mol, idx=idx))
            # Ensure distinct colors when multiple components share a
            # color - assign from the default cycle in that case.
            if len(components) > 1:
                self._deduplicate_colors(components)
            return components

        # --- Bare Intensity ---
        if self._intensity_obj is not None:
            components.append(
                {
                    "molecule": None,
                    "intensity": self._intensity_obj,
                    "name": self._intensity_name,
                    "color": self._intensity_color
                    or _DEFAULT_COLORS[0],
                    "radius": self._intensity_radius,
                    "distance": self._intensity_distance,
                }
            )
            return components

        return components

    @staticmethod
    def _deduplicate_colors(components: List[Dict[str, Any]]) -> None:
        """Assign distinct colors from the default cycle when components
        would otherwise share the same color."""
        seen: Dict[str, int] = {}
        has_duplicates = False
        for comp in components:
            c = comp["color"]
            seen[c] = seen.get(c, 0) + 1
            if seen[c] > 1:
                has_duplicates = True
        if has_duplicates:
            for idx, comp in enumerate(components):
                comp["color"] = _DEFAULT_COLORS[
                    comp.get("color_idx", idx) % len(_DEFAULT_COLORS)
                ]

    @staticmethod
    def _mol_to_component(mol: "Molecule", idx: int = 0) -> Dict[str, Any]:
        """Convert a single Molecule into a component dict."""
        name = BasePlot.get_molecule_display_name(mol)
        color = BasePlot.get_molecule_color(mol)
        return {
            "molecule": mol,
            "intensity": None,
            "name": name,
            "color": color,
            "color_idx": idx,
            "radius": getattr(mol, "radius", 1.0),
            "distance": getattr(mol, "distance", 160.0),
        }

    # ------------------------------------------------------------------
    # Per-component computation
    # ------------------------------------------------------------------
    def _compute_component(
        self, comp: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Compute population-diagram values for a single component.

        Delegates all physics to
        :meth:`~iSLAT.Modules.DataTypes.Intensity.Intensity.get_population_diagram_data`
        and annotates the result with the rendering metadata (name, color)
        that only the plotting layer knows.

        Returns a data dict or ``None`` if no valid data is available.
        """
        from iSLAT.Modules.DataTypes.Intensity import Intensity as _Intensity

        mol          = comp.get("molecule")
        intensity_obj = comp.get("intensity")
        radius       = comp["radius"]
        distance     = comp["distance"]

        if mol is not None:
            # Ensure intensity has been calculated
            if not hasattr(mol, "intensity") or mol.intensity is None:
                if hasattr(mol, "calculate_intensity"):
                    mol.calculate_intensity()
            intensity_from_mol = getattr(mol, "intensity", None)
            if intensity_from_mol is None:
                return None

            data = intensity_from_mol.get_population_diagram_data(
                radius, distance, molecule=mol, full_range=True
            )

            # Fall back to active-range when full-range returns all-NaN intensities
            if data is not None:
                intens_arr = data.get("intens")
                if (
                    intens_arr is not None
                    and not (np.isfinite(intens_arr) & (intens_arr > 0)).any()
                ):
                    data_active = intensity_from_mol.get_population_diagram_data(
                        radius, distance, molecule=mol, full_range=False
                    )
                    if data_active is not None:
                        active_intens = data_active.get("intens")
                        if (
                            active_intens is not None
                            and (np.isfinite(active_intens) & (active_intens > 0)).any()
                        ):
                            data = data_active

        elif intensity_obj is not None:
            data = intensity_obj.get_population_diagram_data(radius, distance)
        else:
            return None

        if data is None:
            return None

        # Resolve molecule_id for quantum-field color mapping.
        mol_id = self._molecule_id_of(mol)

        return {
            "name":      comp["name"],
            "color":     comp["color"],
            "molecule_id": mol_id,
            **data,
        }

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _render_by_component(self, ax: Axes) -> None:
        """Render each component as a single-color scatter series.

        When only one component is present, the default scatter color
        comes from the theme (``scatter_main_color``, typically a
        neutral grey) to match the original single-molecule appearance.
        In multi-component mode each molecule uses its own color so
        that the user can visually distinguish them.
        """
        single = len(self._component_data) == 1
        all_shape_results: List[Tuple[Any, str, str]] = []

        for cdata in self._component_data:
            color = (
                self._get_theme_value("scatter_main_color", "#838B8B")
                if single
                else cdata["color"]
            )
            x_arr = self._get_axis_array(cdata, self._x_prop)
            y_arr = self._get_axis_array(cdata, self._y_prop)
            if x_arr is None or y_arr is None:
                continue

            s_val = self._get_marker_size(0.5 if single else 5)
            wav = cdata.get("wavelength")

            if self._shape_mapping is not None:
                _SA = {"lam": "wavelength", "e_up": "eu"}
                shape_key = _SA.get(self._shape_mapping["prop"], self._shape_mapping["prop"])
                shape_vals = _AxisReg.resolve_array(shape_key, cdata)
                if shape_vals is not None and len(shape_vals) == len(x_arr):
                    results = self._scatter_with_shape_groups(
                        ax, x_arr, y_arr, shape_vals,
                        s=s_val, color=color, alpha=0.8, picker=True,
                        wav_arr=wav,
                    )
                    all_shape_results.extend(results)
                    continue

            sc = ax.scatter(
                x_arr,
                y_arr,
                s=s_val,
                color=color,
                label=cdata["name"],
                alpha=0.8,
                picker=True,
            )
            # Tag with per-point wavelengths so shift+click can look up
            # the line wavelength regardless of the current x/y axis choice.
            if wav is not None:
                sc._islat_scatter_wavelengths = np.asarray(wav)

        if self._shape_mapping is not None and all_shape_results:
            self._add_shape_legend(ax, all_shape_results)

    def _render_component_legend(self, ax: Axes) -> None:
        """Add a legend showing component names and colors."""
        from matplotlib.lines import Line2D

        handles = []
        for cdata in self._component_data:
            handles.append(
                Line2D(
                    [0], [0],
                    marker="o",
                    color="w",
                    markerfacecolor=cdata["color"],
                    markersize=6,
                    label=cdata["name"],
                    linewidth=0,
                )
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize="small",
            framealpha=0.7,
        )

    # ------------------------------------------------------------------
    # color-mapping API
    # ------------------------------------------------------------------
    def color_by(
        self,
        prop: str,
        *,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        pmin: Optional[float] = None,
        pmax: Optional[float] = None,
        log_scale: bool = False,
        regenerate: bool = True,
    ) -> None:
        """Re-color the scatter points by a molecular property.

        Parameters
        ----------
        prop : str
            The property name to color by.  Supported values:

            * ``'e_up'``   - upper-level energy (K)
            * ``'e_low'``  - lower-level energy (K)
            * ``'a_stein'``- Einstein A coefficient
            * ``'g_up'``   - upper-state degeneracy
            * ``'g_low'``  - lower-state degeneracy
            * ``'wavelength'`` / ``'lam'``  - line wavelength
            * ``'intens'`` - model intensity
            * ``'lev_up'`` - upper quantum-state label (categorical)
            * ``'lev_low'``- lower quantum-state label (categorical)
            * ``'tau'``     - line-center opacity
            * ``'component'`` - which molecule / component the point
              belongs to (categorical)
            * ``'molecule'`` - alias for ``'component'``; colors each
              point by which molecule it belongs to (categorical)
            * Any quantum-field name defined in the molecule's schema,
              e.g. ``"J"``, ``"v"``, ``"Ka"``, ``"Kc"``, ``"v1"``,
              ``"v2"`` - colors by upper-state quantum number value.
              Use ``"qn_upper:FIELD"`` / ``"qn_lower:FIELD"`` prefixes
              to explicitly target the upper or lower state (bare names
              default to the upper state).

        cmap : str
            Matplotlib colormap name (default ``'viridis'``).
        vmin, vmax : float, optional
            Explicit color-scale limits for continuous properties.
            Ignored when *pmin* / *pmax* are set.  Ignored for
            categorical properties.
        pmin, pmax : float, optional
            Percentile cutoffs (0-100) for the color-scale limits.
            When set, the color scale minimum / maximum is computed
            as the *pmin*-th / *pmax*-th percentile of the plotted
            values, overriding *vmin* / *vmax*.  For example,
            ``pmax=75`` caps the top color at the 75th-percentile
            value so that the upper 25 % of the distribution all
            receive the maximum color.
        log_scale : bool
            Use a logarithmic color-norm (default ``False``).
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._color_mapping = {
            "prop": prop,
            "cmap": cmap,
            "vmin": vmin,
            "vmax": vmax,
            "pmin": pmin,
            "pmax": pmax,
            "log_scale": log_scale,
        }
        if regenerate:
            self.generate_plot()

    def clear_color_mapping(self, *, regenerate: bool = True) -> None:
        """Remove any property-based color mapping and revert to
        per-component coloring.

        Parameters
        ----------
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._color_mapping = None
        if regenerate:
            self.generate_plot()

    # ------------------------------------------------------------------
    # Marker-shape mapping API
    # ------------------------------------------------------------------
    def shape_by(
        self,
        prop: str,
        *,
        n_bins: int = 5,
        regenerate: bool = True,
    ) -> None:
        """Vary the scatter marker shape by a molecular property.

        Parameters
        ----------
        prop : str
            Property key - the same set accepted by :meth:`color_by`.
        n_bins : int
            Number of quantile bins used for *continuous* properties
            (ignored for categorical ones).  Clamped to 2–10.
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._shape_mapping = {
            "prop": prop,
            "n_bins": max(2, min(10, int(n_bins))),
        }
        if regenerate:
            self.generate_plot()

    def clear_shape_mapping(self, *, regenerate: bool = True) -> None:
        """Remove marker-shape mapping and revert to uniform circles.

        Parameters
        ----------
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._shape_mapping = None
        if regenerate:
            self.generate_plot()

    # ------------------------------------------------------------------
    # Marker-size override API
    # ------------------------------------------------------------------
    def set_marker_size(self, size: float, *, regenerate: bool = True) -> None:
        """Override the scatter marker size for all points.

        Parameters
        ----------
        size : float
            Marker area in points\u00b2 (same unit as matplotlib ``s=``).
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._marker_size = float(size)
        if regenerate:
            self.generate_plot()

    def clear_marker_size(self, *, regenerate: bool = True) -> None:
        """Remove the marker-size override and revert to per-mode defaults.

        Parameters
        ----------
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._marker_size = None
        if regenerate:
            self.generate_plot()

    def _get_marker_size(self, default: float) -> float:
        """Return the effective marker size: override if set, else *default*."""
        return self._marker_size if self._marker_size is not None else default

    # ------------------------------------------------------------------
    # Shape-mapping helpers
    # ------------------------------------------------------------------
    def _resolve_shape_indices(
        self, shape_vals: np.ndarray
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """Return *(idx_arr, group_labels, markers)* for *shape_vals*.

        *idx_arr* has the same length as *shape_vals*; each element is an
        integer index into *group_labels* / *markers*.
        """
        _ALIASES: Dict[str, str] = {"lam": "wavelength", "e_up": "eu"}
        prop: str = self._shape_mapping["prop"]  # type: ignore[index]
        n_bins: int = int(self._shape_mapping.get("n_bins", 5))  # type: ignore[union-attr]
        internal_key = _ALIASES.get(prop, prop)
        kind = _AxisReg.get_kind(internal_key)

        if kind == "categorical":
            str_vals = np.asarray(shape_vals, dtype=str)
            unique_labels: List[str] = list(np.unique(str_vals))
            label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
            idx_arr = np.array(
                [label_to_idx[str(v)] for v in str_vals], dtype=int
            )
            markers = [
                _DEFAULT_MARKERS[i % len(_DEFAULT_MARKERS)]
                for i in range(len(unique_labels))
            ]
            return idx_arr, unique_labels, markers

        # Continuous: quantile binning
        float_vals = np.asarray(shape_vals, dtype=float)
        finite_mask = np.isfinite(float_vals)
        if not np.any(finite_mask):
            return (
                np.zeros(len(float_vals), dtype=int),
                ["all"],
                [_DEFAULT_MARKERS[0]],
            )
        quantiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(
            np.nanpercentile(float_vals[finite_mask], quantiles)
        )
        n_actual = max(len(edges) - 1, 1)
        idx_arr = np.zeros(len(float_vals), dtype=int)
        for _i, _v in enumerate(float_vals):
            if np.isfinite(_v):
                _b = int(np.searchsorted(edges[1:], _v, side="left"))
                idx_arr[_i] = min(_b, n_actual - 1)
        bin_labels = [
            f"{edges[j]:.3g}\u2013{edges[j + 1]:.3g}" for j in range(n_actual)
        ]
        markers = [
            _DEFAULT_MARKERS[j % len(_DEFAULT_MARKERS)] for j in range(n_actual)
        ]
        return idx_arr, bin_labels, markers

    def _scatter_with_shape_groups(
        self,
        ax: Axes,
        x_arr: np.ndarray,
        y_arr: np.ndarray,
        shape_vals: np.ndarray,
        *,
        s: float = 5,
        alpha: float = 0.8,
        picker: bool = True,
        color=None,
        c: Optional[np.ndarray] = None,
        norm=None,
        cmap=None,
        wav_arr: Optional[np.ndarray] = None,
    ) -> List[Tuple[Any, str, str]]:
        """Scatter with marker shape varying per group.

        Returns a list of *(artist, group_label, marker)* tuples for
        building a shape legend.
        """
        idx_arr, group_labels, markers = self._resolve_shape_indices(shape_vals)
        results: List[Tuple[Any, str, str]] = []
        for _gi, (label, marker) in enumerate(zip(group_labels, markers)):
            gmask = idx_arr == _gi
            if not np.any(gmask):
                continue
            gx, gy = x_arr[gmask], y_arr[gmask]
            kw: Dict[str, Any] = dict(
                s=s, alpha=alpha, picker=picker, marker=marker,
            )
            if c is not None:
                kw["c"] = np.asarray(c)[gmask]
                if norm is not None:
                    kw["norm"] = norm
                if cmap is not None:
                    kw["cmap"] = cmap
            else:
                kw["color"] = color if color is not None else "#838B8B"
            sc = ax.scatter(gx, gy, **kw)
            if wav_arr is not None:
                sc._islat_scatter_wavelengths = np.asarray(wav_arr)[gmask]
            results.append((sc, label, marker))
        return results

    def _add_shape_legend(
        self, ax: Axes, results: List[Tuple[Any, str, str]]
    ) -> None:
        """Add a legend for the marker-shape groups at lower-left."""
        from matplotlib.lines import Line2D

        prop: str = self._shape_mapping["prop"]  # type: ignore[index]
        prop_label = self._property_label(prop)
        handles = [
            Line2D(
                [0], [0],
                marker=marker,
                color="w",
                markerfacecolor="#606060",
                markeredgecolor="#606060",
                markersize=6,
                label=lbl,
                linewidth=0,
            )
            for _, lbl, marker in results
        ]
        if handles:
            ax.legend(
                handles=handles,
                title=prop_label,
                loc="lower left",
                fontsize="x-small",
                title_fontsize="x-small",
                framealpha=0.7,
            )

    # ------------------------------------------------------------------
    def _render_colormapped(self, ax: Axes) -> None:
        """Render all components with a single property-based colormap."""
        mapping = self._color_mapping
        if mapping is None:
            return

        prop = mapping["prop"]
        cmap_name = mapping["cmap"]
        vmin = mapping.get("vmin")
        vmax = mapping.get("vmax")
        pmin = mapping.get("pmin")
        pmax = mapping.get("pmax")
        log_scale = mapping.get("log_scale", False)

        # Resolve legacy aliases so old callers (e.g. passing "lam" or "e_up")
        # still work; the registry uses the canonical forms "wavelength" / "eu".
        _ALIASES = {"lam": "wavelength", "e_up": "eu"}
        internal_key: str = _ALIASES.get(prop, prop)
        display_prop = prop

        # ---- Dispatch by kind (categorical vs continuous) -----------
        kind = _AxisReg.get_kind(internal_key)

        if kind == "categorical":
            # 'molecule' is a convenience alias for 'component'
            render_prop = "component" if internal_key in ("molecule", "component") else internal_key
            self._render_categorical_colormap(ax, render_prop, cmap_name)
            return

        # ---- Continuous (includes quantum fields via resolve_array) --
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_vals: List[np.ndarray] = []
        all_wav: List[np.ndarray] = []
        all_shape_vals: List[np.ndarray] = []
        _SHAPE_ALIASES: Dict[str, str] = {"lam": "wavelength", "e_up": "eu"}

        for cdata in self._component_data:
            vals_arr = _AxisReg.resolve_array(internal_key, cdata)
            x_arr = self._get_axis_array(cdata, self._x_prop)
            y_arr = self._get_axis_array(cdata, self._y_prop)
            if vals_arr is None or x_arr is None or y_arr is None:
                continue

            # Apply the valid_mask so that only lines above the 1%-of-max
            # flux threshold are plotted and used for scale computation.
            mask = cdata.get("valid_mask")
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                x_arr = x_arr[mask]
                y_arr = y_arr[mask]
                vals_arr = vals_arr[mask]
                wav_arr = cdata.get("wavelength")
                if wav_arr is not None:
                    all_wav.append(np.asarray(wav_arr)[mask])
            else:
                wav_arr = cdata.get("wavelength")
                if wav_arr is not None:
                    all_wav.append(np.asarray(wav_arr))

            if len(x_arr) == 0:
                continue
            all_x.append(x_arr)
            all_y.append(y_arr)
            all_vals.append(vals_arr)

            # Shape-by: collect shape property values with same masking
            if self._shape_mapping is not None:
                _sp_key = _SHAPE_ALIASES.get(
                    self._shape_mapping["prop"], self._shape_mapping["prop"]
                )
                sv = _AxisReg.resolve_array(_sp_key, cdata)
                if sv is not None:
                    sv = np.asarray(sv)
                    if mask is not None:
                        sv = sv[mask]
                    all_shape_vals.append(sv[:len(x_arr)])
                else:
                    all_shape_vals.append(np.zeros(len(x_arr)))

        if not all_vals:
            self._render_by_component(ax)
            return

        eu_cat = np.concatenate(all_x)
        rd_cat = np.concatenate(all_y)
        val_cat = np.concatenate(all_vals)

        # Percentile cutoffs override explicit vmin/vmax when set.
        finite_vals = val_cat[np.isfinite(val_cat)]
        if len(finite_vals) == 0:
            self._render_by_component(ax)
            return

        if pmin is not None:
            vmin = float(np.nanpercentile(finite_vals, float(pmin)))
        elif vmin is None:
            vmin = float(np.nanmin(finite_vals))
        if pmax is not None:
            vmax = float(np.nanpercentile(finite_vals, float(pmax)))
        elif vmax is None:
            vmax = float(np.nanmax(finite_vals))

        if log_scale:
            pos_vals = finite_vals[finite_vals > 0]
            if len(pos_vals) == 0:
                pos_vals = np.array([1e-30])
            safe_vmin = max(vmin, float(np.nanmin(pos_vals))) if vmin <= 0 else vmin
            safe_vmin = max(safe_vmin, 1e-300)
            safe_vmax = max(vmax, safe_vmin * 10)
            norm = LogNorm(vmin=safe_vmin, vmax=safe_vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = matplotlib.colormaps.get_cmap(cmap_name)

        wav_cat = np.concatenate(all_wav) if all_wav else None
        label = self._property_label(display_prop)
        fig = ax.get_figure()

        if self._shape_mapping is not None and all_shape_vals:
            shape_cat = np.concatenate(all_shape_vals)
            shape_results = self._scatter_with_shape_groups(
                ax, eu_cat, rd_cat, shape_cat,
                s=self._get_marker_size(5), c=val_cat, norm=norm, cmap=cmap_obj,
                alpha=0.8, picker=True,
                wav_arr=wav_cat,
            )
            if shape_results and fig is not None:
                self._colorbar = fig.colorbar(
                    shape_results[0][0], ax=ax, label=label, pad=0.02
                )
            self._add_shape_legend(ax, shape_results)
        else:
            sc = ax.scatter(
                eu_cat, rd_cat,
                c=val_cat, s=self._get_marker_size(5),
                cmap=cmap_obj, norm=norm,
                alpha=0.8, picker=True,
            )
            if wav_cat is not None:
                sc._islat_scatter_wavelengths = wav_cat
            if fig is not None:
                self._colorbar = fig.colorbar(sc, ax=ax, label=label, pad=0.02)

    # ------------------------------------------------------------------
    def _render_categorical_colormap(
        self, ax: Axes, prop: str, cmap_name: str
    ) -> None:
        """Render scatter with categorical color mapping."""
        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_shape_vals: List[np.ndarray] = []
        _SHAPE_ALIASES: Dict[str, str] = {"lam": "wavelength", "e_up": "eu"}

        for cdata in self._component_data:
            x_arr = self._get_axis_array(cdata, self._x_prop)
            y_arr = self._get_axis_array(cdata, self._y_prop)
            if x_arr is None or y_arr is None:
                continue
            mask = cdata.get("valid_mask")
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                x_arr = x_arr[mask]
                y_arr = y_arr[mask]
            n = len(x_arr)
            if n == 0:
                continue
            if prop == "component":
                labels = np.full(n, cdata["name"], dtype=object)
            else:
                vals = cdata.get(prop)
                if vals is None:
                    labels = np.full(n, "unknown", dtype=object)
                else:
                    vals_arr = np.asarray(vals, dtype=str)
                    if mask is not None:
                        vals_arr = vals_arr[mask]
                    labels = vals_arr
            all_x.append(x_arr)
            all_y.append(y_arr)
            all_labels.append(labels)

            # Shape-by: collect with same masking
            if self._shape_mapping is not None:
                _sp_key = _SHAPE_ALIASES.get(
                    self._shape_mapping["prop"], self._shape_mapping["prop"]
                )
                sv = _AxisReg.resolve_array(_sp_key, cdata)
                if sv is not None:
                    sv = np.asarray(sv)
                    if mask is not None:
                        sv = sv[mask]
                    all_shape_vals.append(sv[:n])
                else:
                    all_shape_vals.append(np.zeros(n))

        if not all_x:
            return
        eu_cat = np.concatenate(all_x)
        rd_cat = np.concatenate(all_y)
        label_cat = np.concatenate(all_labels)

        unique_labels = np.unique(label_cat)

        # In all-molecules mode, use each molecule's own .color attribute
        # so the population diagram matches the control-panel colors.
        if self._all_molecules_mode and prop == "component":
            label_to_color = {
                cdata["name"]: cdata["color"]
                for cdata in self._component_data
            }
            # Fall back to colormap for any label that somehow lacks a color
            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name).resampled(
                max(len(unique_labels), 1)
            )
            for i, lbl in enumerate(unique_labels):
                if lbl not in label_to_color:
                    label_to_color[lbl] = cmap_obj(i / max(len(unique_labels) - 1, 1))
        else:
            cmap_obj = matplotlib.colormaps.get_cmap(cmap_name).resampled(
                max(len(unique_labels), 1)
            )
            label_to_color = {
                lbl: cmap_obj(i / max(len(unique_labels) - 1, 1))
                for i, lbl in enumerate(unique_labels)
            }

        colors = np.array([label_to_color[lbl] for lbl in label_cat])

        if self._shape_mapping is not None and all_shape_vals:
            shape_cat = np.concatenate(all_shape_vals)
            shape_results = self._scatter_with_shape_groups(
                ax, eu_cat, rd_cat, shape_cat,
                s=self._get_marker_size(5), c=colors, alpha=0.8, picker=True,
            )
        else:
            ax.scatter(eu_cat, rd_cat, c=colors, s=self._get_marker_size(5), alpha=0.8)
            shape_results = []

        # Legend for categories (limit to 20 entries max for readability)
        from matplotlib.lines import Line2D

        entries = list(unique_labels)
        if len(entries) > 20:
            entries = entries[:20]

        handles = [
            Line2D(
                [0], [0],
                marker="o", color="w",
                markerfacecolor=label_to_color[lbl],
                markersize=6, label=str(lbl), linewidth=0,
            )
            for lbl in entries
        ]
        if len(unique_labels) > 20:
            handles.append(
                Line2D(
                    [0], [0], marker="", color="w",
                    label=f"… +{len(unique_labels) - 20} more",
                    linewidth=0,
                )
            )
        ax.legend(
            handles=handles,
            loc="upper right",
            fontsize="x-small",
            framealpha=0.7,
            ncol=max(1, len(entries) // 10),
        )

        if self._shape_mapping is not None and shape_results:
            self._add_shape_legend(ax, shape_results)

    # ------------------------------------------------------------------
    @staticmethod
    def _property_label(prop: str) -> str:
        """LaTeX label for a property name (used for colorbar labels).

        Delegates to :class:`~iSLAT.Modules.DataTypes.PlotAxisRegistry.PlotAxisRegistry`.
        """
        return _AxisReg.get_axis_label(prop)

    @classmethod
    def _get_all_axis_labels(cls) -> Dict[str, str]:
        """Return all axis-option labels sourced from the registry."""
        return {e.key: e.label for e in _AxisReg.get_axis_options()}

    @classmethod
    def _get_axis_label(cls, prop: str) -> str:
        """Human-readable axis label for a given property name.

        Delegates to :class:`~iSLAT.Modules.DataTypes.PlotAxisRegistry.PlotAxisRegistry`.
        """
        return _AxisReg.get_axis_label(prop)

    @staticmethod
    def _get_axis_array(
        cdata: Dict[str, Any], prop: str
    ) -> Optional[np.ndarray]:
        """Extract a property array from a component data dict via the registry."""
        return _AxisReg.resolve_array(prop, cdata)

    def set_axes(
        self,
        x_prop: str = "eu",
        y_prop: str = "rd_yax",
        x_log: bool = False,
        y_log: bool = False,
        x_lim: tuple = None,
        y_lim: tuple = None,
        *,
        regenerate: bool = True,
    ) -> None:
        """Configure the plot axes.

        Parameters
        ----------
        x_prop : str
            Property key to use for the X axis.
        y_prop : str
            Property key to use for the Y axis.
        x_log : bool
            Whether to use a logarithmic scale for the X axis.
        y_log : bool
            Whether to use a logarithmic scale for the Y axis.
        x_lim : tuple or None
            Axis limit override for the X axis.  Three forms are accepted:

            * ``None`` - use matplotlib auto-limits (default).
            * ``('exact', min_val, max_val)`` - pin the axis to the given
              numeric values (either may be ``None`` to leave that bound
              automatic).
            * ``('percentile', p_min, p_max)`` - set the limits from the
              *p_min*-th and *p_max*-th percentile of the plotted X data
              (0-100, either may be ``None`` to leave that bound automatic).

        y_lim : tuple or None
            Same as *x_lim* but for the Y axis.
        regenerate : bool
            If ``True`` (default) the plot is regenerated immediately.
        """
        self._x_prop = x_prop
        self._y_prop = y_prop
        self._x_log = x_log
        self._y_log = y_log
        self._x_lim = x_lim   # None | ('exact', lo, hi) | ('percentile', plo, phi)
        self._y_lim = y_lim
        if regenerate:
            self.generate_plot()

    # ------------------------------------------------------------------
    # Highlighted-lines overlay
    # ------------------------------------------------------------------
    def _render_highlights(self, ax: Axes, beam_s: float) -> None:
        """Overlay highlighted lines as larger scatter points."""
        if not self.highlight_lines:
            return
        color = self._get_theme_value("active_scatter_line_color", "green")

        from iSLAT.Modules.DataTypes.Intensity import Intensity as _Intensity

        x_vals: List[float] = []
        y_vals: List[float] = []

        for line_obj, intensity, tau in self.highlight_lines:
            if any(
                v is None
                for v in [intensity, line_obj.a_stein, line_obj.g_up, line_obj.lam]
            ):
                continue

            # Pre-compute derived values used by multiple axis options
            F = intensity * beam_s
            freq = c.SPEED_OF_LIGHT_MICRONS / line_obj.lam
            rd = np.log(
                4 * np.pi * F
                / (line_obj.a_stein * c.PLANCK_CONSTANT * freq * line_obj.g_up)
            ) if (F > 0 and line_obj.a_stein > 0) else np.nan

            # Resolve the active molecule for FWHM lookup.
            _hl_mol = self.molecule or (
                list(self._molecules_input.get_visible_molecules(return_objects=True))[0]
                if self._molecules_input is not None
                and hasattr(self._molecules_input, "get_visible_molecules")
                else None
            )

            # Delegate per-line metadata (including FWHM) to Intensity.get_line_info
            # so all physics computation lives on the data object.
            _hl_info = _Intensity.get_line_info(line_obj, intensity, tau, molecule=_hl_mol)

            # Resolve molecule_id for quantum-field axis lookups.
            _hl_mol_id: Optional[str] = self._molecule_id_of(_hl_mol)

            def _resolve(prop: str) -> Optional[float]:
                """Map a property key to a scalar value for this line."""
                _LINE_MAP = {
                    "eu":                    line_obj.e_up,
                    "e_low":                 line_obj.e_low,
                    "rd_yax":                rd,
                    "wavelength":            line_obj.lam,
                    "intens":                intensity,
                    "a_stein":               line_obj.a_stein,
                    "g_up":                  line_obj.g_up,
                    "g_low":                 line_obj.g_low,
                    "tau":                   tau,
                    "fwhm_instrumental_kms": _hl_info.get("fwhm_instrumental_kms"),
                    "fwhm_convolved_kms":    _hl_info.get("fwhm_convolved_kms"),
                }
                val = _LINE_MAP.get(prop)
                if val is not None:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return None
                # Fall back to registry scalar resolver for quantum/extended props
                return _AxisReg.resolve_scalar(prop, line_obj, _hl_mol_id)

            xv = _resolve(self._x_prop)
            yv = _resolve(self._y_prop)
            if xv is None or yv is None or np.isnan(xv) or np.isnan(yv):
                continue
            x_vals.append(xv)
            y_vals.append(yv)

        if x_vals:
            ax.scatter(
                x_vals, y_vals, s=30, color=color,
                edgecolors="black", zorder=5,
            )

    # ------------------------------------------------------------------
    # Bare-intensity table builder
    # ------------------------------------------------------------------
    def _build_table_from_intensity(self) -> Optional[pd.DataFrame]:
        """Build an intensity table from the stored bare Intensity object.

        Backward-compatible wrapper around
        :meth:`_build_table_from_intensity_obj`.
        """
        return self._build_table_from_intensity_obj(self._intensity_obj)

    @staticmethod
    def _build_table_from_intensity_obj(
        intensity_obj: Optional["Intensity"],
    ) -> Optional[pd.DataFrame]:
        """Build an intensity table from an arbitrary Intensity object."""
        if intensity_obj is None:
            return None
        if hasattr(intensity_obj, "build_table"):
            df = intensity_obj.build_table(full_range=True)
        else:
            return None
        if df is not None and hasattr(df, "index") and not df.empty:
            return df
        return None

    # ------------------------------------------------------------------
    # Public mutators (backward-compatible)
    # ------------------------------------------------------------------
    def set_molecule(self, molecule: "Molecule") -> None:
        """Switch to a single molecule and regenerate."""
        self._exit_all_molecules_mode()
        self.molecule = molecule
        self._molecules_input = None
        self._intensity_obj = None
        self._color_mapping = None
        self.generate_plot()

    def set_molecules(
        self, molecules: Union[Sequence["Molecule"], "MoleculeDict"],
    ) -> None:
        """Switch to multiple molecules and regenerate.

        When *molecules* is a :class:`MoleculeDict` the plot enters
        *all-molecules mode*: it automatically colors each component by
        its own molecule color and re-renders whenever the active/visible
        set changes.

        Parameters
        ----------
        molecules : list[Molecule] | MoleculeDict
            The molecules to overlay on the population diagram.
        """
        self.molecule = None
        self._molecules_input = molecules
        self._intensity_obj = None
        self.generate_plot()

        # If we were handed a MoleculeDict, enter persistent all-molecules mode
        if hasattr(molecules, 'add_active_molecule_change_callback'):
            self._enter_all_molecules_mode(molecules)
        else:
            # Plain list - not persistent, clear any prior mode
            self._exit_all_molecules_mode()

    # ------------------------------------------------------------------
    # All-molecules mode helpers
    # ------------------------------------------------------------------
    def _enter_all_molecules_mode(self, molecules_dict) -> None:
        """Lock the plot to a MoleculeDict and register a change callback."""
        # Unregister any previous callback first
        self._exit_all_molecules_mode()
        self._all_molecules_mode = True
        self._molecules_dict_ref = molecules_dict
        # Automatically color by molecule (uses each mol's own color)
        self._color_mapping = {"prop": "molecule", "cmap": "tab10"}
        # Register for active-molecule and comparison-molecule changes.
        # The callbacks only hold weak references to this plot so a discarded
        # diagram can be garbage collected even if nobody calls close().
        self._molecules_change_cb = self._make_weak_callback(
            molecules_dict,
            "_on_molecules_changed",
            "remove_active_molecule_change_callback",
        )
        self._comparison_change_cb = self._make_weak_callback(
            molecules_dict,
            "_on_comparison_changed",
            "remove_comparison_molecule_change_callback",
        )
        try:
            molecules_dict.add_active_molecule_change_callback(
                self._molecules_change_cb
            )
        except Exception:
            pass
        try:
            molecules_dict.add_comparison_molecule_change_callback(
                self._comparison_change_cb
            )
        except Exception:
            pass

    def _make_weak_callback(self, molecules_dict, method_name: str, remover_name: str):
        """Build a callback that keeps only a weak reference to this plot."""
        self_ref = weakref.ref(self)
        dict_ref = weakref.ref(molecules_dict)

        def _dispatch(*args, **kwargs):
            plot = self_ref()
            if plot is None:
                # Plot was garbage collected - self-unregister from the dict.
                owner = dict_ref()
                if owner is not None:
                    try:
                        getattr(owner, remover_name)(_dispatch)
                    except Exception:
                        pass
                return
            getattr(plot, method_name)(*args, **kwargs)

        return _dispatch

    def _exit_all_molecules_mode(self) -> None:
        """Unregister change callbacks and clear all-molecules mode."""
        if self._molecules_dict_ref is not None:
            if self._molecules_change_cb is not None:
                try:
                    self._molecules_dict_ref.remove_active_molecule_change_callback(
                        self._molecules_change_cb
                    )
                except Exception:
                    pass
            if self._comparison_change_cb is not None:
                try:
                    self._molecules_dict_ref.remove_comparison_molecule_change_callback(
                        self._comparison_change_cb
                    )
                except Exception:
                    pass
        self._all_molecules_mode = False
        self._molecules_dict_ref = None
        self._molecules_change_cb = None
        self._comparison_change_cb = None

    def close(self) -> None:
        """Leave all-molecules mode (dropping callbacks) and release the figure."""
        self._exit_all_molecules_mode()
        super().close()

    def _on_molecules_changed(self, old_molecule=None, new_molecule=None) -> None:
        """Callback fired when the active molecule changes in all-molecules mode."""
        if not self._all_molecules_mode or self._molecules_dict_ref is None:
            return
        self._molecules_input = self._molecules_dict_ref
        self.generate_plot()

    def _on_comparison_changed(self, comparison_molecules_list=None) -> None:
        """Callback fired when comparison molecules change in all-molecules mode."""
        if not self._all_molecules_mode or self._molecules_dict_ref is None:
            return
        self._molecules_input = self._molecules_dict_ref
        self.generate_plot()

    def set_intensity(
        self,
        intensity: "Intensity",
        *,
        name: Optional[str] = None,
        color: Optional[str] = None,
        radius: Optional[float] = None,
        distance: Optional[float] = None,
    ) -> None:
        """Switch to a bare :class:`Intensity` object and regenerate.

        Parameters
        ----------
        intensity : Intensity
            The Intensity object to use.
        name, color, radius, distance
            Optional metadata overrides.  Values that are ``None`` keep
            their previously stored value.
        """
        self.molecule = None
        self._molecules_input = None
        self._intensity_obj = intensity
        if name is not None:
            self._intensity_name = name
        if color is not None:
            self._intensity_color = color
        if radius is not None:
            self._intensity_radius = radius
        if distance is not None:
            self._intensity_distance = distance
        self._color_mapping = None
        self.generate_plot()

    def add_molecule(self, molecule: "Molecule") -> None:
        """Add a molecule to an existing multi-molecule diagram.

        If the diagram currently shows a single molecule, it is
        converted to multi-molecule mode automatically.
        """
        existing: List["Molecule"] = []
        if self.molecule is not None:
            existing.append(self.molecule)
            self.molecule = None
        if self._molecules_input is not None:
            seq = self._molecules_input
            if hasattr(seq, "values"):
                existing.extend(seq.values())
            else:
                existing.extend(seq)
        existing.append(molecule)
        self._molecules_input = existing
        self._intensity_obj = None
        self.generate_plot()

    def remove_molecule(self, name: str) -> None:
        """Remove a molecule by name from the multi-molecule diagram."""
        if self._molecules_input is None:
            if self.molecule is not None and getattr(self.molecule, "name", None) == name:
                self.molecule = None
                self.generate_plot()
            return
        seq = self._molecules_input
        if hasattr(seq, "values"):
            seq = list(seq.values())
        else:
            seq = list(seq)
        filtered = [m for m in seq if getattr(m, "name", None) != name]
        if len(filtered) == 0:
            self._molecules_input = None
        elif len(filtered) == 1:
            self.molecule = filtered[0]
            self._molecules_input = None
        else:
            self._molecules_input = filtered
        self.generate_plot()