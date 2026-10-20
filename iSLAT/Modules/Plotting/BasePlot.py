"""
BasePlot - Abstract base class for all iSLAT plot types.

Provides common infrastructure for figure/axes management, theming, molecule rendering helpers, and show/save functionality.
All plot classes inherit from this so they can work both inside the GUI and as standalone matplotlib figures in scripts or Jupyter notebooks.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, Union, TYPE_CHECKING
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure as MplFigure
from matplotlib.lines import Line2D

import iSLAT.Constants as c

from iSLAT.Modules.Plotting.LegendStrategy import (
    LegendStrategy,
    MoleculeColorLegend,
    StandardLegend,
)

# Import display config to ensure rcParams are set early and to access the savefig DPI default.
try:
    from iSLAT.Modules.GUI.DisplayConfig import display_config as _display_config
except ImportError:
    _display_config = None

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

def _detect_system_theme() -> str:
    """Return ``'DarkTheme'`` or ``'LightTheme'`` based on the OS appearance.

    Works on macOS (via ``defaults``), Windows (via registry), and falls back to ``'LightTheme'`` on other platforms or on error.
    """
    import platform
    import subprocess
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            # "Dark" is returned when dark mode is active; command
            # *fails* when light mode is active (key doesn't exist).
            if result.returncode == 0 and "dark" in result.stdout.strip().lower():
                return "DarkTheme"
            return "LightTheme"
        elif system == "Windows":
            import winreg  # available only on Windows
            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            )
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            winreg.CloseKey(key)
            return "LightTheme" if value == 1 else "DarkTheme"
        else:
            # Linux / other - no universal API; default to light.
            return "LightTheme"
    except Exception:
        return "LightTheme"

# ---------------------------------------------------------------------------
# Default theme (used when no GUI theme is supplied)
# ---------------------------------------------------------------------------
DEFAULT_THEME: Dict[str, Any] = {
    "foreground": "black",
    "background": "white",
    "graph_fill_color": "white",
    "summed_spectra_color": "lightgray",
    "scatter_main_color": "#838B8B",
    "active_scatter_line_color": "green",
    "highlighted_line_color": "yellow",
    "saved_line_color": "red",
    "saved_line_color_one": "red",
    "saved_line_color_two": "orange",
    "default_molecule_color": "blue",
    "model_linewidth": 2,
    "model_alpha": 1,
    "full_spectrum_model_linewidth": 0.8,
    "full_spectrum_model_alpha": 0.8,
    "zorder_observed": 2,
    "zorder_summed": 1,
    "zorder_model": 3,
}

class BasePlot(ABC):
    """
    Abstract base class for all iSLAT plot types.

    Subclasses must implement :meth:`generate_plot`.
    Everything else (figure lifecycle, molecule helpers, theming, show/save) is provided by the base class.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches ``(width, height)``.
    theme : dict, optional
        Theme dictionary.  Falls back to :data:`DEFAULT_THEME`.
    fig : Figure, optional
        Pre-existing matplotlib Figure to render into (e.g. from a GUI).
        When *None* a new figure will be created on demand.
    """
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        theme: Optional[Dict[str, Any]] = None,
        fig: Optional[MplFigure] = None,
        legend_strategy: Optional[LegendStrategy] = None,
        **kwargs,
    ):
        self._figsize = figsize
        self.theme: Dict[str, Any] = theme if theme is not None else DEFAULT_THEME.copy()
        self.fig: Optional[MplFigure] = fig
        self._owns_figure = fig is None  # True when we create the figure ourselves
        self.legend_strategy: LegendStrategy = (
            legend_strategy if legend_strategy is not None else StandardLegend()
        )
        # Mutable list of active line artist entries: [vline, text, scatter, info_dict]
        self._active_lines: List[Any] = []

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------
    def _get_theme_value(self, key: str, default: Any = None) -> Any:
        """Return a value from the theme dict, or *default*."""
        return self.theme.get(key, default)

    def apply_theme_to_figure(self, fig: Optional[MplFigure] = None) -> None:
        """Apply theme background / foreground colors to a figure and all its axes.

        This sets the figure face color, axes face color, tick colors, label colors, title colors, and spine colors from the theme dictionary.
        Call this after ``generate_plot()`` (or at the end of it) to get a fully themed figure without manual per-axes work.

        Parameters
        ----------
        fig : Figure, optional
            The figure to style.  Defaults to ``self.fig``.
        """
        target = fig if fig is not None else self.fig
        if target is None:
            return
        fg = self._get_theme_value("foreground", "black")
        bg = self._get_theme_value("background", "white")
        graph_bg = self._get_theme_value("graph_fill_color", bg)

        target.set_facecolor(bg)
        for ax in target.axes:
            ax.set_facecolor(graph_bg)
            ax.tick_params(colors=fg, which="both")
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            for spine in ax.spines.values():
                spine.set_color(fg)

            # Recolor tagged data artists so they match the new theme
            summed_color = self._get_theme_value("summed_spectra_color", "lightgray")
            for artist in ax.lines:
                if getattr(artist, '_islat_observed', False):
                    artist.set_color(fg)
            for artist in ax.collections:
                if getattr(artist, '_islat_summed', False):
                    artist.set_facecolor(summed_color)
                    artist.set_edgecolor(summed_color)

            # Theme the legend via the strategy if one exists
            self.legend_strategy.apply_theme(ax, self.theme)

    @staticmethod
    def load_theme(name: str = "auto") -> Dict[str, Any]:
        """Load a theme dictionary from the bundled JSON theme files.

        Parameters
        ----------
        name : str
            Theme name (e.g. ``"DarkTheme"``, ``"LightTheme"``,
            ``"PastelBlue"``).  The special value ``"auto"`` selects
            ``"DarkTheme"`` or ``"LightTheme"`` depending on the
            operating system's current appearance setting.

        Returns
        -------
        dict
            The parsed theme dictionary, or :data:`DEFAULT_THEME` if the file cannot be found.
        """
        if name == "auto":
            name = _detect_system_theme()

        import json
        try:
            from iSLAT.Modules.FileHandling import theme_file_path
            theme_json = Path(str(theme_file_path)) / f"{name}.json"
            if theme_json.exists():
                with open(theme_json, "r") as fh:
                    return json.load(fh)
        except Exception:
            pass
        return DEFAULT_THEME.copy()

    # ------------------------------------------------------------------
    # Molecule helpers (shared across all plot types)
    # ------------------------------------------------------------------
    @staticmethod
    def get_molecule_display_name(molecule: "Molecule") -> str:
        """Return the user-facing label for a molecule."""
        return getattr(molecule, "displaylabel", getattr(molecule, "name", "unknown"))

    @staticmethod
    def get_molecule_color(molecule: "Molecule") -> str:
        """Return the color associated with a molecule."""
        color = getattr(molecule, "color", None)
        return color if color else "blue"

    #: Multiplier applied to the base model line width when a molecule is bold.
    BOLD_LINEWIDTH_FACTOR: float = 2.5

    @classmethod
    def get_molecule_linewidth(cls, molecule: Optional["Molecule"], base_linewidth: float) -> float:
        """Return the line width to draw *molecule* with.

        Molecules flagged ``is_bold`` are drawn ``BOLD_LINEWIDTH_FACTOR`` times
        thicker than *base_linewidth*; everything else uses *base_linewidth*.
        """
        # Compare against True explicitly so non-bool stand-ins (e.g. mocks) are not treated as bold.
        if molecule is not None and getattr(molecule, "is_bold", False) is True:
            return float(base_linewidth) * cls.BOLD_LINEWIDTH_FACTOR
        return float(base_linewidth)

    def get_molecule_linewidth_by_name(self, molecule_name: str, base_linewidth: float) -> float:
        """Like :meth:`get_molecule_linewidth` but looks the molecule up in ``self.molecules``.

        Used by the ``mol_cache`` rendering loops, which only carry the molecule *name*.
        Falls back to *base_linewidth* when no molecule collection is attached.
        """
        molecule = None
        single = getattr(self, "molecule", None)
        if single is not None and getattr(single, "name", None) == molecule_name:
            molecule = single
        else:
            molecules = getattr(self, "molecules", None)
            if molecules is not None:
                try:
                    molecule = molecules.get(molecule_name)
                except Exception:
                    molecule = None
        return self.get_molecule_linewidth(molecule, base_linewidth)

    @staticmethod
    def build_molecule_legend(
        ax: Axes,
        mol_labels: List[str],
        mol_colors: List[str],
        *,
        ncols: Optional[int] = None,
        fontsize: int = 10,
        bbox_to_anchor: Tuple[float, float] = (0.5, 0.99),
        use_figure_transform: bool = True,
    ) -> None:
        """Create (or replace) a text-only, per-molecule-colored legend.

        Each entry is shown as bold colored text with no visible handle patch, giving a compact color key above the plot.
        The legend texts are tagged with ``_islat_mol_color = True`` so that
        :meth:`apply_theme_to_figure` will not overwrite them with the foreground color.

        When *ncols* is ``None`` (the default), the number of columns is auto-computed so that the legend does not exceed the width of the axes.
        If the labels are long, items wrap into multiple rows.

        When *mol_labels* is empty the existing legend (if any) is removed and no new one is created.

        Parameters
        ----------
        use_figure_transform : bool
            When True (default), *bbox_to_anchor* is interpreted in figure coordinates.
            When False, it uses axes coordinates.
        """
        old = ax.get_legend()
        if old is not None:
            old.remove()

        if not mol_labels:
            return

        from matplotlib.patches import Patch

        # --- Auto-compute ncols to fit within the axes width ----------
        if ncols is None:
            fig = ax.get_figure()
            fig_w_in = fig.get_figwidth()

            # Estimate average label width in inches (approx 0.085 in per character at the given fontsize, scaled from 10pt base)
            char_w_in = 0.085 * (fontsize / 10.0)
            avg_label_w = sum(len(lbl) for lbl in mol_labels) / len(mol_labels)
            col_w_in = avg_label_w * char_w_in + 0.3  # extra for padding

            # Maximum columns that fit within the figure width
            max_cols = max(int(fig_w_in / col_w_in), 1)
            ncols = min(max_cols, len(mol_labels))

        handles = [Patch(facecolor='none', edgecolor='none') for _ in mol_colors]

        # Place the legend centered on the full figure width, pinned near the top of the figure so it sits above all panels.
        fig = ax.get_figure()
        transform = fig.transFigure if use_figure_transform else ax.transAxes
        leg = ax.legend(
            handles,
            mol_labels,
            loc="upper center",
            ncols=min(ncols, len(mol_labels)),
            handlelength=0,
            handletextpad=0,
            bbox_to_anchor=bbox_to_anchor,
            bbox_transform=transform,
            fontsize=fontsize,
            prop={"weight": "bold"},
            frameon=False,
        )
        for txt, col in zip(leg.get_texts(), mol_colors):
            txt.set_color(col)
            txt._islat_mol_color = True

    @staticmethod
    def _extend_sampling_grids(
        molecule: "Molecule",
        interp_wave: Optional[np.ndarray],
        wave_data: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Pad the matched-sampling grids so they span the molecule's wavelength range.

        *interp_wave* is the (rest-frame) grid the model is resampled onto and
        *wave_data* the display-frame grid returned to the caller; both are
        padded at the data's median spacing so a model range wider than the
        observed data (e.g. a lower *Min. Wave*) is still drawn.  Returns
        the inputs unchanged when no padding is needed.
        """
        from iSLAT.Modules.DataProcessing.spectral_utils import extend_grid_to_range
        rng = getattr(molecule, "wavelength_range", None)
        if interp_wave is None or rng is None:
            return interp_wave, wave_data
        base = np.asarray(interp_wave, dtype=float)
        ext = extend_grid_to_range(base, float(rng[0]), float(rng[1]))
        if ext is base or len(ext) == len(base):
            return interp_wave, wave_data
        if wave_data is None or wave_data is interp_wave:
            return ext, None if wave_data is None else ext
        # wave_data and interp_wave differ by the (constant) stellar-RV factor.
        wd = np.asarray(wave_data, dtype=float)
        ratio = 1.0
        if len(wd) == len(base) and len(base) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.nanmedian(wd / base)
            if np.isfinite(r) and r > 0:
                ratio = float(r)
        return ext, ext * ratio

    @staticmethod
    def get_molecule_spectrum_data(
        molecule: "Molecule",
        wave_data: Optional[np.ndarray] = None,
        interpolate_to_input: bool = False,
        target_wavelengths: Optional[np.ndarray] = None,
        extend_to_range: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve ``(wavelength, flux)`` from a *Molecule* object.

        This delegates to ``molecule.get_flux()`` and works regardless of whether the molecule has already been computed or not.

        Parameters
        ----------
        molecule : Molecule
            The molecule to query.
        wave_data : np.ndarray, optional
            Wavelength array passed through to ``get_flux``.
        interpolate_to_input : bool, default False
            When True the model flux is resampled onto *target_wavelengths* (or *wave_data* when *target_wavelengths* is None).
            This is used by the matched-spectral-sampling feature.
        target_wavelengths : np.ndarray, optional
            Rest-frame wavelength grid to interpolate onto.
            When *interpolate_to_input* is True and this is provided, the model
            is resampled to these wavelengths (typically the data wavelengths
            corrected for the global stellar RV) while the returned wavelength array is *wave_data* (the observer-frame grid).
        extend_to_range : bool, default False
            When interpolating, pad the grids so the model is also returned
            where the molecule's wavelength range reaches beyond the data
            (plotting only; exports keep the exact observed grid).
        """
        if molecule is None:
            return None, None
        try:
            interp_wave = target_wavelengths if target_wavelengths is not None else wave_data
            if interpolate_to_input and extend_to_range:
                interp_wave, wave_data = BasePlot._extend_sampling_grids(molecule, interp_wave, wave_data)
            lam, flux = molecule.get_flux(
                wavelength_array=interp_wave,
                return_wavelengths=True,
                interpolate_to_input=interpolate_to_input,
            )
            # When we interpolated to rest-frame target wavelengths, return
            # the observer-frame (wave_data) wavelengths so the line is plotted at the correct observed positions.
            if interpolate_to_input and target_wavelengths is not None and wave_data is not None:
                lam = wave_data
            return lam, flux
        except Exception as exc:
            print(f"[BasePlot] Could not get flux for "
                  f"{BasePlot.get_molecule_display_name(molecule)}: {exc}")
            return None, None

    @staticmethod
    def get_molecule_tau_data(
        molecule: "Molecule",
        wave_data: Optional[np.ndarray] = None,
        interpolate_to_input: bool = False,
        target_wavelengths: Optional[np.ndarray] = None,
        extend_to_range: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Retrieve ``(wavelength, tau)`` from a *Molecule* object.

        Mirrors :meth:`get_molecule_spectrum_data` but returns the convolved optical-depth profile instead of flux.

        Parameters
        ----------
        molecule : Molecule
            The molecule to query.
        wave_data : np.ndarray, optional
            Wavelength array passed through to ``get_tau``.
        interpolate_to_input : bool, default False
            When True the model tau is resampled onto *target_wavelengths* (or *wave_data* when *target_wavelengths* is None).
        target_wavelengths : np.ndarray, optional
            Rest-frame wavelength grid to interpolate onto.
        extend_to_range : bool, default False
            See :meth:`get_molecule_spectrum_data`.
        """
        if molecule is None:
            return None, None
        try:
            interp_wave = target_wavelengths if target_wavelengths is not None else wave_data
            if interpolate_to_input and extend_to_range:
                interp_wave, wave_data = BasePlot._extend_sampling_grids(molecule, interp_wave, wave_data)
            lam, tau = molecule.get_tau(
                wavelength_array=interp_wave,
                return_wavelengths=True,
                interpolate_to_input=interpolate_to_input,
            )
            if interpolate_to_input and target_wavelengths is not None and wave_data is not None:
                lam = wave_data
            return lam, tau
        except Exception as exc:
            print(f"[BasePlot] Could not get tau for "
                  f"{BasePlot.get_molecule_display_name(molecule)}: {exc}")
            return None, None

    @staticmethod
    def get_intensity_data(
        molecule: "Molecule",
        *,
        full_range: bool = False,
    ) -> Optional[pd.DataFrame]:
        """Return the Intensity table (DataFrame) from *molecule*.

        Triggers calculation if needed.

        Parameters
        ----------
        molecule : Molecule
            Molecule object whose intensity data is requested.
        full_range : bool, optional
            If ``True``, return all lines in the underlying HITRAN file with intensity computed for each.
            Defaults to ``False`` (active wavelength range only).

        Returns
        -------
        Optional[pd.DataFrame]
            Intensity table, or ``None`` when no data is available.
        """
        try:
            if molecule is None:
                return None
            if not hasattr(molecule, "intensity") or molecule.intensity is None:
                if hasattr(molecule, "calculate_intensity"):
                    molecule.calculate_intensity()
            intensity_obj = getattr(molecule, "intensity", None)
            if intensity_obj is None:
                return None

            # Prefer the new build_table() API when available
            if hasattr(intensity_obj, "build_table"):
                df = intensity_obj.build_table(full_range=full_range)
            else:
                # Fallback for older Intensity implementations
                table = getattr(intensity_obj, "get_table", None)
                if table is None:
                    return None
                df = table if isinstance(table, pd.DataFrame) else table

            if df is not None and hasattr(df, "index"):
                df.index = range(len(df.index))
            return df
        except Exception as exc:
            print(f"[BasePlot] Error getting intensity data: {exc}")
            return None

    # ------------------------------------------------------------------
    # Figure lifecycle helpers
    # ------------------------------------------------------------------
    def _ensure_figure(self, silent: bool = False, **subplot_kw) -> MplFigure:
        """Create the figure if it doesn't already exist.

        The figure is **always** created via :class:`~matplotlib.figure.Figure` - it is *not* registered with the pyplot state machine at creation time.
        This prevents "leaked" figure widgets in Jupyter notebooks when plot objects
        are used as intermediate data sources (e.g. the two source plots inside a :class:`CompositeStackedPanel`).

        When the user calls :meth:`show`, the figure is registered
        with pyplot **on demand** so that the active interactive backend (ipympl ``%matplotlib widget``, inline, TkAgg, …) can display it properly.

        Parameters
        ----------
        silent : bool
            Kept for backward compatibility.
            Has no effect - all figures are now created without pyplot registration.
        """
        if self.fig is None:
            kw: Dict[str, Any] = {"layout": "constrained"}
            if self._figsize is not None:
                kw["figsize"] = self._figsize
            self.fig = MplFigure(**kw)
            self._owns_figure = True
        return self.fig

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------
    def _plot_visible_molecules(
        self,
        ax: Axes,
        molecules: "MoleculeDict",
        wave_data: Optional[np.ndarray] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        linewidth: float = 1,
        alpha: float = 0.8,
        update_legend: bool = True,
    ) -> None:
        """Plot all visible molecules from a *MoleculeDict* on *ax*.

        When *molecules.match_spectral_sampling* is enabled, each molecule's
        model flux is individually resampled to the data wavelength grid
        (corrected for the global stellar RV) before plotting.

        Parameters
        ----------
        wave_data : np.ndarray, optional
            Display-frame wavelength array (used for x-axis positions).
        wave_data_obs : np.ndarray, optional
            Observer-frame wavelength array passed to ``get_matched_sampling_wavelengths``.
            Falls back to *wave_data* when not provided.
        """
        # get_matched_sampling_wavelengths expects observer-frame input.
        obs = wave_data_obs if wave_data_obs is not None else wave_data
        use_interp = False
        target_wave = None
        if obs is not None and hasattr(molecules, 'get_matched_sampling_wavelengths'):
            use_interp, target_wave = molecules.get_matched_sampling_wavelengths(obs)
            if not use_interp:
                target_wave = None  # don't pass target wavelengths when not interpolating

        visible = molecules.get_visible_molecules(return_objects=True)
        for mol in visible:
            self._plot_molecule_spectrum(
                ax, mol, wave_data=wave_data, linewidth=linewidth, alpha=alpha,
                interpolate_to_input=use_interp, target_wavelengths=target_wave,
            )
        if update_legend:
            self._update_legend(ax)

    @staticmethod
    def _update_legend(ax: Axes) -> None:
        """Add or update the legend on *ax*, excluding invisible artists.

        This static method is kept for backward compatibility with callers that invoke it directly.
        It delegates to :class:`StandardLegend`.
        """
        _fallback = StandardLegend()
        fig = ax.get_figure()
        _fallback.build_legend(ax, fig, [], [])

    @staticmethod
    def _clear_tagged_artists(
        ax: Axes,
        tag: str,
        *,
        lines: bool = True,
        collections: bool = True,
        texts: bool = False,
    ) -> None:
        """Remove every artist on *ax* that carries the attribute *tag*.

        Parameters
        ----------
        ax : Axes
            Target axes.
        tag : str
            Attribute name to look for (e.g. ``'_islat_observed'``).
        lines : bool
            Search ``ax.lines`` (includes ``Line2D`` artists).
        collections : bool
            Search ``ax.collections`` (includes ``LineCollection``,
            ``PolyCollection``, etc.).
        texts : bool
            Search ``ax.texts``.
        """
        if lines:
            for artist in ax.lines[:]:
                if hasattr(artist, tag):
                    artist.remove()
        if collections:
            for artist in ax.collections[:]:
                if hasattr(artist, tag):
                    artist.remove()
        if texts:
            for artist in ax.texts[:]:
                if hasattr(artist, tag):
                    artist.remove()

    @staticmethod
    def _plot_line_annotations(
        ax: Axes,
        line_data: pd.DataFrame,
        xr: Tuple[float, float],
        ymin: float,
        ymax: float,
        offset_label: float = 0.003,
        tag: Optional[str] = None,
    ) -> None:
        """
        Draw vertical dotted lines + labels for a line list DataFrame.

        The DataFrame must contain at least ``wave`` (or ``lam``) and ``species`` columns.

        Parameters
        ----------
        tag : str, optional
            When provided every created artist is stamped with
            ``setattr(artist, tag, True)`` so callers can later remove them with :meth:`_clear_tagged_artists`.
        """
        if line_data is None or len(line_data) == 0:
            return
        col = "wave" if "wave" in line_data.columns else ("lam" if "lam" in line_data.columns else None)
        if col is None:
            return
        lam_arr = line_data[col].values
        species_arr = line_data["species"].values if "species" in line_data.columns else [""] * len(lam_arr)
        line_id_arr = line_data["line"].values if "line" in line_data.columns else [""] * len(lam_arr)

        for i, lam in enumerate(lam_arr):
            if xr[0] < lam < xr[1]:
                vl = ax.vlines(lam, ymin, ymax, linestyles="dotted", color="grey", linewidth=0.7)
                txt = ax.text(
                    lam + offset_label,
                    ymax,
                    f"{species_arr[i]} {line_id_arr[i]}",
                    fontsize=6,
                    rotation=90,
                    va="top",
                    ha="left",
                    color="grey",
                )
                if tag is not None:
                    setattr(vl, tag, True)
                    setattr(txt, tag, True)

    @staticmethod
    def _plot_atomic_lines(
        ax: Axes,
        atomic_df: pd.DataFrame,
        xr: Optional[Tuple[float, float]] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Draw atomic line markers on *ax*.

        Parameters
        ----------
        tag : str, optional
            When provided every created artist is stamped with ``setattr(artist, tag, True)`` so callers can later remove them with :meth:`_clear_tagged_artists`.
        """
        if atomic_df is None or len(atomic_df) == 0:
            return
        if xr is not None:
            atomic_df = atomic_df[
                (atomic_df["wave"] >= xr[0]) & (atomic_df["wave"] <= xr[1])
            ]
        for _, row in atomic_df.iterrows():
            line_artist = ax.axvline(row["wave"], linestyle="--", color="tomato", alpha=0.7)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            text_artist = ax.text(
                row["wave"] + 0.006 * (xlim[1] - xlim[0]),
                ylim[1],
                f"{row['species']} {row.get('line', '')}",
                fontsize=8,
                rotation=90,
                va="top",
                ha="left",
                color="tomato",
            )
            if tag is not None:
                setattr(line_artist, tag, True)
                setattr(text_artist, tag, True)

    @staticmethod
    def _plot_saved_line_markers(
        ax: Axes,
        loaded_lines: pd.DataFrame,
        tag: str = "_islat_saved_line",
        lam_color: str = "red",
        range_color: str = "orange",
        alpha: float = 0.7,
    ) -> None:
        """Plot saved-line markers on *ax* with artist tagging.

        Handles both point markers (``lam`` column) and range markers (``xmin`` / ``xmax`` columns).
        Every created artist is stamped with *tag* so it can be removed later via :meth:`_clear_tagged_artists`.

        Parameters
        ----------
        ax : Axes
            Target axes.
        loaded_lines : DataFrame
            Saved-line data.  Expected columns: ``lam`` and optionally
            ``xmin``, ``xmax``.
        tag : str
            Attribute name stamped onto every created artist.
        lam_color, range_color : str
            colors for the center-wavelength and range-boundary markers.
        alpha : float
            Transparency of the markers.
        """
        if loaded_lines is None or loaded_lines.empty:
            return
        for _, line in loaded_lines.iterrows():
            if "lam" in line:
                art = ax.axvline(
                    line["lam"], color=lam_color, alpha=alpha,
                    linestyle=":",
                )
                setattr(art, tag, True)
            if "xmin" in line and "xmax" in line:
                for val in (line["xmin"], line["xmax"]):
                    art = ax.axvline(val, color=range_color, alpha=alpha)
                    setattr(art, tag, True)

    # ------------------------------------------------------------------
    # Active-lines API
    # ------------------------------------------------------------------
    @property
    def active_lines(self) -> List[Any]:
        """List of active line artist entries managed by this panel.

        Each entry is ``[vline_artist, text_artist, scatter_artist, info_dict]``.
        Use :meth:`render_active_lines` to populate and :meth:`clear_active_lines` to remove all artists.
        """
        return self._active_lines

    def render_active_lines(
        self,
        line_data: List[Any],
        active_lines: List[Any],
        **kwargs,
    ) -> Any:
        """Render active line markers or scatter points onto this panel.

        The default implementation is a no-op.  Subclasses should override:

        * :class:`LineInspectionPlot` - renders vertical dashed line markers
          with energy/A-coefficient labels.
        * :class:`PopulationDiagramPlot` - renders scatter points on the
          Boltzmann diagram.

        Parameters
        ----------
        line_data : list
            ``(MoleculeLine, intensity, tau)`` triples.
        active_lines : list
            Mutable list that receives ``[line, text, scatter, info]`` entries.
        **kwargs
            Subclass-specific keyword arguments (e.g. *max_y*, *threshold*,
            *color*, *molecule*, etc.).

        Returns
        -------
        Any
            Subclass-dependent return value (e.g. the scatter artist from
            :class:`PopulationDiagramPlot`).
        """
        return None

    def clear_active_lines(self, active_lines: List[Any]) -> None:
        """Remove all active-line artists and clear *active_lines*.

        The default implementation simply clears the list without removing matplotlib artists.
        Subclasses override this to also call ``artist.remove()`` on each stored artist before clearing.

        Parameters
        ----------
        active_lines : list
            The mutable active-lines list to clear (same object that was passed to :meth:`render_active_lines`).
        """
        active_lines.clear()

    # ------------------------------------------------------------------
    # Abstract API - subclasses must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_plot(self, **kwargs) -> None:
        """Generate or refresh the plot. Subclasses must implement this."""
        ...

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    @staticmethod
    def _in_notebook() -> bool:
        """Return *True* when running inside a Jupyter / IPython kernel.

        Safe to call even when IPython is not installed.
        """
        try:
            from IPython import get_ipython  # noqa: F811
        except ImportError:
            return False
        try:
            shell = get_ipython()
            if shell is None:
                return False
            return type(shell).__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    def show(self, block: bool = False) -> None:
        """Display the plot interactively.

        Figures are created without pyplot registration (see :meth:`_ensure_figure`),
        so this method registers the figure with pyplot **on demand** before displaying.
        In a Jupyter notebook this lets the active interactive backend (ipympl, inline, …) create the proper widget or image.
        Outside of a notebook the figure is registered with the GUI backend so ``plt.show()`` opens a window.
        """
        if self.fig is None:
            self.generate_plot()

        # Register the figure with pyplot if it isn't already.
        self._register_with_pyplot()

        if self._in_notebook():
            plt.show(block=block)
            return

        # Non-notebook: plt.show() opens a GUI window.
        plt.show(block=block)

    def _register_with_pyplot(self) -> None:
        """Ensure *self.fig* is known to pyplot's figure manager.

        If the figure was created via :class:`MplFigure` (which is always the case now), it has no ``number`` attribute and is unknown to pyplot.
        This method registers it so the active backend (ipympl widget, inline, TkAgg, …) can display it.

        Calling this more than once is safe - it's a no-op when the figure is already registered.
        """
        if self.fig is None:
            return

        # Already registered?
        try:
            fig_num = self.fig.number
        except AttributeError:
            fig_num = None

        if fig_num is not None and fig_num in plt.get_fignums():
            return

        # Register with the backend's figure manager.
        try:
            num = id(self.fig)
            manager = plt._backend_mod.new_figure_manager_given_figure(
                num, self.fig,
            )
            # Use _set_new_active_manager (not set_active) so the
            # manager gets its _cidgcf callback registered.  Without
            # this, Gcf.destroy_all() crashes because it expects
            # manager._cidgcf to exist for mpl_disconnect().
            plt._pylab_helpers.Gcf._set_new_active_manager(manager)
        except Exception:
            # Fallback for backends that don't support this.
            try:
                from IPython.display import display as ipy_display
                ipy_display(self.fig)
            except ImportError:
                pass

    def save(
        self,
        path: Union[str, Path],
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> Path:
        """Save the figure to *path*."""
        if self.fig is None:
            self.generate_plot()
        path = Path(path)
        save_kw: Dict[str, Any] = {"bbox_inches": bbox_inches}
        if dpi is not None:
            save_kw["dpi"] = dpi
        else:
            # Use the centralized high-quality default
            _default_dpi = _display_config.savefig_dpi if _display_config else 300
            save_kw["dpi"] = _default_dpi
        save_kw.update(kwargs)
        self.fig.savefig(str(path), **save_kw)
        return path

    def close(self) -> None:
        """Close the figure and free memory."""
        if self.fig is not None and self._owns_figure:
            try:
                # Try pyplot close first (works for pyplot-managed figs)
                plt.close(self.fig)
            except Exception:
                # Figure was created with MplFigure() - not in pyplot,
                # just clear it directly.
                try:
                    self.fig.clear()
                except Exception:
                    pass
        self.fig = None

    def get_figure(self) -> Optional[MplFigure]:
        """Return the underlying matplotlib *Figure*."""
        return self.fig

    # ------------------------------------------------------------------
    # Notebook / IPython rich-display integration
    # ------------------------------------------------------------------
    def _repr_png_(self) -> Optional[bytes]:
        """Return a PNG rendering of the figure for IPython/Jupyter.

        When an iSLAT plot object is the last expression in a notebook
        cell, Jupyter calls this method automatically so the figure renders inline - just like a plain matplotlib ``Figure``.
        """
        if self.fig is None:
            self.generate_plot()
        if self.fig is None:
            return None
        try:
            import io
            buf = io.BytesIO()
            dpi = _display_config.savefig_dpi if _display_config else 150
            self.fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
            buf.seek(0)
            return buf.getvalue()
        except Exception:
            return None

    def _repr_html_(self) -> Optional[str]:
        """Return an HTML ``<img>`` tag embedding the figure as base-64.

        Provides a second rich-display path for IPython/Jupyter frontends that prefer HTML over raw PNG.
        """
        png = self._repr_png_()
        if png is None:
            return None
        try:
            import base64
            b64 = base64.b64encode(png).decode("ascii")
            return f'<img src="data:image/png;base64,{b64}" />'
        except Exception:
            return None

    def _repr_mimebundle_(self, **kwargs) -> Optional[dict]:
        """IPython rich-display mimebundle for notebook rendering."""
        png = self._repr_png_()
        if png is None:
            return None
        return {"image/png": png}