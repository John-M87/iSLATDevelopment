"""
LineInspectionPlot - Zoomed-in view of a narrow wavelength region.

Shows the observed spectrum, overlaid molecule model(s), and optionally individual line markers with energy/A-coefficient labels.

Can be used standalone (notebook / script) or embedded in a GUI layout.
"""
from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .SpectrumPanel import SpectrumPanel

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

class LineInspectionPlot(SpectrumPanel):
    """
    Plot a narrow wavelength region with observed data and molecule models.

    Implements the :class:`SpectralPanel` abstract interface for a single
    zoomed-in wavelength region with optional molecule model overlays and
    individual line markers with energy / A-coefficient labels.

    Parameters
    ----------
    wave_data : np.ndarray
        Full wavelength array (the region is selected with *xmin*/*xmax*).
    flux_data : np.ndarray
        Full flux array matching *wave_data*.
    xmin, xmax : float
        Wavelength bounds for the inspection window.
    error_data : np.ndarray, optional
        Error values matching *wave_data*.
    molecule : Molecule, optional
        Single active molecule whose model is overlaid.
    molecules : MoleculeDict, optional
        If provided *all visible* molecules are plotted instead of just one.
    line_data : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples for line markers.
    line_threshold : float, optional
        Fraction (0-1) of the strongest line below which lines are hidden.
        Defaults to ``0.0`` (show all).
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 4)``.
    ax : Axes, optional
        Pre-existing axes to draw into (for embedding in a larger figure).
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        error_data: Optional[np.ndarray] = None,
        molecule: Optional["Molecule"] = None,
        molecules: Optional["MoleculeDict"] = None,
        line_data: Optional[List[Tuple["MoleculeLine", float, Optional[float]]]] = None,
        line_threshold: float = 0.0,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        render_all_visible: bool = True,
        **kwargs,
    ):
        super().__init__(
            wave_data=wave_data,
            flux_data=flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=error_data,
            molecules=molecules,
            wave_data_obs=wave_data_obs,
            figsize=figsize or (10, 4),
            ax=ax,
            **kwargs,
        )
        self.molecule = molecule
        self.line_data = line_data
        self.line_threshold = line_threshold
        # When True (default), all visible molecules from *molecules* are rendered.
        # When False, only *molecule* (singular) is rendered; *molecules* is still used for matched-spectral-sampling queries.
        self.render_all_visible: bool = render_all_visible
        # Observer-frame wavelengths for matched spectral sampling.
        # SpectrumPanel stores None when not provided; fall back to wave_data.
        if self.wave_data_obs is None:
            self.wave_data_obs = self.wave_data

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Generate (or regenerate) the line inspection plot."""
        ax = self._resolve_axes()

        fg = self._get_theme_value("foreground", "black")

        # Mask to the inspection range
        mask = (self.wave_data >= self.xmin) & (self.wave_data <= self.xmax)
        obs_wave = self.wave_data[mask]
        obs_flux = self.flux_data[mask]
        obs_err = self.error_data[mask] if self.error_data is not None else None

        # -- observed spectrum -------------------------------------------
        self._plot_observed_spectrum(ax, obs_wave, obs_flux, obs_err)

        max_y = float(np.nanmax(obs_flux)) if len(obs_flux) > 0 else 0.15

        # -- molecule model(s) ------------------------------------------
        # Determine matched spectral sampling settings (same logic as FullSpectrumPlot)
        # so the line inspection plot honours the "Match Pix. Sampling" toggle.
        use_interp = False
        target_wave = None
        if self.molecules is not None and hasattr(self.molecules, 'get_matched_sampling_wavelengths'):
            use_interp, target_wave = self.molecules.get_matched_sampling_wavelengths(self.wave_data_obs)
            if not use_interp:
                target_wave = None

        if self.molecules is not None and self.render_all_visible:
            # Standalone / notebook mode: render every visible molecule.
            visible = self.molecules.get_visible_molecules(return_objects=True)
            for mol in visible:
                mol_max = self._overlay_molecule(ax, mol, use_interp, target_wave)
                if mol_max is not None and len(obs_flux) == 0:
                    max_y = max(max_y, mol_max)
        elif self.molecule is not None:
            # GUI mode (render_all_visible=False) or single-molecule mode:
            # only render the active molecule.
            mol_max = self._overlay_molecule(ax, self.molecule, use_interp, target_wave)
            if mol_max is not None and len(obs_flux) == 0:
                max_y = max(max_y, mol_max)

        # -- individual line markers ------------------------------------
        if self.line_data:
            max_y = self._plot_line_markers(ax, self.line_data, max_y)

        # -- appearance -------------------------------------------------
        ax.set_xlim(self.xmin, self.xmax)
        if max_y > 0:
            ax.set_ylim(0, max_y * 1.1)
        ax.set_xlabel("Wavelength (μm)", color=fg)
        ax.set_ylabel("Flux density (Jy)", color=fg)
        ax.set_title("Line inspection plot", color=fg)
        self._update_legend(ax)

        # Apply full theme (backgrounds, spines, etc.) to the figure
        if self.fig is not None:
            self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _overlay_molecule(
        self,
        ax: Axes,
        molecule: "Molecule",
        interpolate_to_input: bool = False,
        target_wavelengths: Optional[np.ndarray] = None,
    ) -> Optional[float]:
        """Plot one molecule model in the inspection range.

        Returns the max flux value in the range, or *None* if nothing was plotted.
        """
        plot_lam, model_flux = self.get_molecule_spectrum_data(
            molecule, self.wave_data,
            interpolate_to_input=interpolate_to_input,
            target_wavelengths=target_wavelengths,
        )
        if plot_lam is None or model_flux is None:
            return None
        m = (plot_lam >= self.xmin) & (plot_lam <= self.xmax)
        if not np.any(m):
            return None
        model_wave_range = plot_lam[m]
        model_flux_range = model_flux[m]
        if len(model_wave_range) == 0 or len(model_flux_range) == 0:
            return None

        # Skip molecules whose flux is effectively zero in this range so they don't clutter the legend with a flat-line entry.
        peak = float(np.nanmax(np.abs(model_flux_range)))
        if peak < 1e-30:
            return None

        ax.plot(
            model_wave_range,
            model_flux_range,
            color=self.get_molecule_color(molecule),
            linestyle="--",
            linewidth=self.get_molecule_linewidth(molecule, 2),
            label=self.get_molecule_display_name(molecule),
        )
        return float(np.nanmax(model_flux_range))

    def _plot_line_markers(
        self,
        ax: Axes,
        line_data: List[Tuple["MoleculeLine", float, Optional[float]]],
        max_y: float,
    ) -> float:
        """Draw vertical dashed lines for individual molecular transitions."""
        if not line_data:
            return max_y

        intensities = [i for _, i, _ in line_data]
        max_intensity = max(intensities) if intensities else 1.0
        threshold_val = max_intensity * self.line_threshold

        color = self._get_theme_value("active_scatter_line_color", "green")

        for line_obj, intensity, _ in line_data:
            if intensity < threshold_val:
                continue
            lineheight = (intensity / max_intensity) * max_y if max_intensity > 0 else 0
            if lineheight <= 0:
                continue
            ax.vlines(
                line_obj.lam,
                0,
                lineheight,
                color=color,
                linestyle="dashed",
                linewidth=1,
            )
            ax.text(
                line_obj.lam,
                lineheight,
                f"{line_obj.e_up:.0f},{line_obj.a_stein:.3f}",
                fontsize="x-small",
                color=color,
                rotation=45,
            )
        return max_y

    # ------------------------------------------------------------------
    # Active-lines API implementation
    # ------------------------------------------------------------------
    def render_active_lines(
        self,
        line_data: List[Tuple["MoleculeLine", float, Any]],
        active_lines: List[Any],
        *,
        max_y: float = 0.1,
        threshold: float = 0.0,
        color: str = "green",
        molecule_name: str = "",
        molecule_color: str = "",
        **kwargs,
    ) -> None:
        """Render vertical dashed line markers for *line_data* on this panel.

        Creates ``vlines`` + text labels for each line that exceeds the intensity *threshold* (as a fraction of the strongest line) and appends ``[vline, text, None, info_dict]`` entries to *active_lines*.

        Parameters
        ----------
        line_data : list[tuple]
            ``(MoleculeLine, intensity, tau)`` triples.
        active_lines : list
            Mutable list that receives ``[vline, text, None, info]`` entries.
        max_y : float
            Scaling height for the strongest line.
        threshold : float
            Fraction (0-1) of max intensity below which lines are hidden.
        color : str
            color for the markers and labels.
        molecule_name : str
            Molecule name stored in the info dict.
        molecule_color : str
            Molecule color stored in the info dict.
        """
        # Use the already-resolved axes directly - do NOT call _resolve_axes() here as that clears the axes and would wipe the already-rendered observed spectrum and molecule model.
        ax = self._ax if self._ax is not None else self._external_ax
        if not line_data or ax is None:
            return

        intensities = [i for _, i, _ in line_data]
        max_intensity = max(intensities) if intensities else 1.0
        if max_intensity <= 0:
            return

        for line, intensity, tau_val in line_data:
            frac = intensity / max_intensity
            if frac < threshold:
                continue
            lineheight = frac * max_y
            if lineheight <= 0:
                continue

            vline = ax.vlines(
                line.lam, 0, lineheight,
                color=color, linestyle="dashed", linewidth=1, picker=True,
            )
            text = ax.text(
                line.lam, lineheight,
                f"{line.e_up:.0f},{line.a_stein:.3f}",
                fontsize="x-small", color=color, rotation=45,
                picker=True,
            )
            info = LineInspectionPlot.get_line_info(line, intensity, tau_val)
            info["lineheight"] = lineheight
            info["intensity_percent"] = frac * 100
            info["molecule_name"] = molecule_name
            info["molecule_color"] = molecule_color or color

            active_lines.append([vline, text, None, info])

    def clear_active_lines(self, active_lines: List[Any]) -> None:
        """Remove vline and text artists from *active_lines* and clear the list.

        Parameters
        ----------
        active_lines : list
            Mutable list of ``[vline, text, scatter, info]`` entries.
        """
        for entry in active_lines:
            for artist in entry[:2]:  # vline and text only
                if artist is not None and getattr(artist, 'axes', None) is not None:
                    try:
                        artist.remove()
                    except (ValueError, AttributeError):
                        pass
        active_lines.clear()

    # ------------------------------------------------------------------
    # Line information helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_line_info(
        line: "MoleculeLine",
        intensity: float,
        tau: Optional[float] = None,
        data_flux_in_range: Optional[float] = None,
        model_flux_in_range: Optional[float] = None,
        opacity_in_range: Optional[float] = None,
        molecule: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build a structured information dict for a single molecular line.

        Delegates to :meth:`~iSLAT.Modules.DataTypes.Intensity.Intensity.get_line_info` so that the data object owns the canonical implementation.
        This method is kept here for backward compatibility; all callers that already use ``LineInspectionPlot.get_line_info`` continue to work unchanged.

        Parameters
        ----------
        line : MoleculeLine
            The molecular transition.
        intensity : float
            Computed model intensity for this line.
        tau : float, optional
            Line opacity.
        data_flux_in_range : float, optional
            Observed (data) flux integral in the selection range
            (erg s^-1 cm^-2).
        model_flux_in_range : float, optional
            Model flux integral in the selection range (erg s^-1 cm^-2).
        opacity_in_range: float, optional
            Model optical depth in the selection range.
        molecule : Molecule, optional
            Active molecule.  When provided the instrumental, Keplerian,
            and convolved FWHM at the line wavelength are included in the
            output and formatted text.

        Returns
        -------
        dict
            Keys: ``lam``, ``e_up``, ``e_low``, ``a_stein``, ``g_up``,
            ``g_low``, ``up_lev``, ``low_lev``, ``intensity``, ``tau``,
            ``data_flux_in_range``, ``model_flux_in_range``,
            ``fwhm_instrumental_kms``, ``fwhm_keplerian_kms``,
            ``fwhm_convolved_kms``, ``formatted_text``.
        """
        from iSLAT.Modules.DataTypes.Intensity import Intensity as _Intensity
        return _Intensity.get_line_info(
            line, intensity, tau,
            data_flux_in_range=data_flux_in_range,
            model_flux_in_range=model_flux_in_range,
            opacity_in_range=opacity_in_range,
            molecule=molecule,
        )

    @staticmethod
    def format_line_info(info: Dict[str, Any]) -> str:
        """Return the pre-built formatted text from a :meth:`get_line_info` dict."""
        return info.get("formatted_text", "")

    @staticmethod
    def get_line_info_dataframe(
        line_data: List[Tuple["MoleculeLine", float, Optional[float]]],
    ) -> "pd.DataFrame":
        """Build a :class:`~pandas.DataFrame` with one row per molecular line.

        Delegates to :meth:`~iSLAT.Modules.DataTypes.Intensity.Intensity.get_line_info_dataframe` so that the data object owns the canonical implementation.
        This method is kept here for backward compatibility.

        Parameters
        ----------
        line_data : list of (MoleculeLine, intensity, tau)
            Line tuples as returned by
            ``Molecule.intensity.get_lines_in_range_with_intensity()``.

        Returns
        -------
        pd.DataFrame
            Columns: ``wavelength_um``, ``e_up_K``, ``e_low_K``,
            ``a_stein``, ``g_up``, ``g_low``, ``upper_level``,
            ``lower_level``, ``intensity``, ``tau``.
            Rows are sorted by wavelength.
        """
        from iSLAT.Modules.DataTypes.Intensity import Intensity as _Intensity
        return _Intensity.get_line_info_dataframe(line_data)