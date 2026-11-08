"""
OpticalDepthPanel -- Concrete :class:`SpectralPanel` for optical depth display.

Renders the convolved optical-depth profile τ(λ) for one or more molecules on a single axes.  Supports three display modes:

* **profile** (default) - continuous Gaussian-convolved τ(λ) with
  stacked ``fill_between`` per molecule.
* **stem** - vertical lines at each line-center wavelength with
  height equal to the per-line τ value.
* **fraction** - each molecule's fractional share of the total convolved
  τ per pixel, τᵢ / Σⱼ τⱼ (stacked fills sum to 1); see
  :meth:`OpticalDepthPanel.compute_fractional_shares`.

Used both as a standalone single-panel plot and as the cell type created by :class:`OpticalDepthSpectrumPlot`.
"""
import warnings
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes

from .SpectralPanel import SpectralPanel, GapMode
from .BasePlot import BasePlot
from .ArtistTags import iter_tagged

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

#: Attribute tagging the combined "Total τ" line (toggled like the summed model).
TOTAL_TAU_TAG = "_islat_total_tau"
#: Attribute tagging the per-line τ stems drawn in ``"stem"`` mode.
TAU_STEM_TAG = "_islat_tau_stem"
#: Y limits of the ``"fraction"`` mode (shares lie in [0, 1]; a little headroom keeps the top edge visible).
FRACTION_YLIM: Tuple[float, float] = (0.0, 1.05)
#: Axis labels of the ``"fraction"`` mode (full and short, for stacked sub-panels).
FRACTION_YLABEL = "Fraction of total τ"
FRACTION_YLABEL_SHORT = r"$\tau_i/\tau$"


class OpticalDepthPanel(SpectralPanel):
    """
    Single-axes panel showing optical depth τ(λ).

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array (used only by the base class for
        masking / gap detection; not plotted).
    xmin, xmax : float
        Wavelength bounds for this panel.
    error_data : np.ndarray, optional
        Flux uncertainties (forwarded to base; not plotted).
    molecules : MoleculeDict, optional
        Molecule collection -- visible ones are plotted.
    molecule : Molecule, optional
        Single molecule to plot (used when *molecules* is ``None``).
    mol_tau_cache : list[tuple], optional
        Pre-computed ``(wavelengths, tau, color, label, name)`` tuples
        from :meth:`OpticalDepthSpectrumPlot._build_mol_tau_cache`.
    display_mode : str, optional
        ``"profile"`` for convolved Gaussian profiles (default),
        ``"stem"`` for per-line markers, or ``"fraction"`` for each
        molecule's share of the total τ per pixel (y axis 0 to 1).
    log_scale : bool, optional
        Use logarithmic y-axis.  Default ``False``.  Not applied in
        ``"fraction"`` mode, whose axis is always linear.
    show_total : bool, optional
        Show combined τ line when multiple molecules are rendered.
        Default ``True``.  Not drawn in ``"fraction"`` mode (it would be 1).
    stacked_fill : bool, optional
        Use cumulative ``fill_between`` per molecule.  Default ``True``.
        In ``"fraction"`` mode the stacked fills sum to 1; otherwise each
        molecule's share is drawn as a line.
    tau_floor : float, optional
        Minimum τ value when *log_scale* is ``True``.  Default ``1e-10``.
    fraction_min_tau : float, optional
        In ``"fraction"`` mode, pixels whose total τ is at or below this
        value are left blank (the share is undefined there).  Default ``1e-6``.
    ax : Axes, optional
        Pre-existing axes to draw into.
    **kwargs
        Forwarded to :class:`SpectralPanel` / :class:`BasePlot`.
    """
    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        error_data: Optional[np.ndarray] = None,
        molecules: Optional["MoleculeDict"] = None,
        molecule: Optional["Molecule"] = None,
        mol_tau_cache: Optional[List[tuple]] = None,
        display_mode: str = "profile",
        log_scale: bool = False,
        show_total: bool = True,
        stacked_fill: bool = True,
        tau_floor: float = 1e-10,
        fraction_min_tau: float = 1e-6,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        super().__init__(
            wave_data=wave_data,
            flux_data=flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=error_data,
            molecules=molecules,
            ax=ax,
            **kwargs,
        )
        self.molecule: Optional["Molecule"] = molecule
        self.mol_tau_cache: List[tuple] = mol_tau_cache or []
        self.display_mode = display_mode
        self.log_scale = log_scale
        self.show_total = show_total
        self.stacked_fill = stacked_fill
        self.tau_floor = tau_floor
        self.fraction_min_tau = fraction_min_tau

    # ------------------------------------------------------------------
    def _ensure_mol_tau_cache(self) -> None:
        """Populate *mol_tau_cache* from *molecule*/*molecules* if empty.

        When an ``OpticalDepthPanel`` is used standalone (not embedded by ``OpticalDepthSpectrumPlot``) the cache starts empty.
        This method builds it on-demand using :meth:`BasePlot.get_molecule_tau_data`.
        """
        if self.mol_tau_cache:
            return

        mols: list = []
        if self.molecule is not None:
            mols.append(self.molecule)
        elif self.molecules is not None:
            mols = list(
                self.molecules.get_visible_molecules(return_objects=True)
            )

        for mol in mols:
            lam, tau = self.get_molecule_tau_data(mol, self.wave_data)
            if lam is not None and tau is not None and len(tau) > 0:
                self.mol_tau_cache.append((
                    lam, tau,
                    self.get_molecule_color(mol),
                    self.get_molecule_display_name(mol),
                    getattr(mol, "name", "unknown"),
                ))

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Render optical-depth profiles, fractional shares or stem markers."""
        self._ensure_mol_tau_cache()
        ax = self._resolve_axes()
        xr = self.xlim
        fg = self._get_theme_value("foreground", "black")
        fraction = self.display_mode == "fraction"

        if fraction:
            self._render_fraction(ax, xr)
        elif self.display_mode == "profile":
            self._render_profile(ax, xr)
        else:
            self._render_stem(ax, xr)

        ax.set_xlim(*xr)
        ax.set_ylabel(FRACTION_YLABEL if fraction else "Optical Depth (\u03c4)", color=fg, fontsize=8)

        if fraction:
            ax.set_ylim(*FRACTION_YLIM)
        elif self.log_scale:
            # A panel without τ > 0 warns here; its limits are set afterwards.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Data has no positive values")
                ax.set_yscale("log")

    # ------------------------------------------------------------------
    # Profile mode
    # ------------------------------------------------------------------
    def _render_profile(self, ax: Axes, xr: Tuple[float, float]) -> None:
        """Render convolved τ(λ) with stacked fills."""
        entries = self.mol_tau_cache
        if not entries:
            return

        cumulative = None
        total_tau = None

        for m_lam, m_tau, m_color, m_label, m_name in entries:
            m_mask = (m_lam >= xr[0]) & (m_lam <= xr[1])
            if not np.any(m_mask):
                continue

            lam_seg = m_lam[m_mask]
            tau_seg = m_tau[m_mask]

            # Accumulate total
            if total_tau is None:
                total_tau = np.zeros_like(tau_seg)
            if len(tau_seg) == len(total_tau):
                total_tau = total_tau + tau_seg

            if self.stacked_fill:
                if cumulative is None:
                    cumulative = np.zeros_like(tau_seg)
                bottom = cumulative.copy()
                cumulative = cumulative + tau_seg
                ax.fill_between(
                    lam_seg, bottom, cumulative,
                    color=m_color, alpha=0.45, label=m_label,
                    zorder=self._get_theme_value("zorder_model", 3),
                )
            else:
                ax.plot(
                    lam_seg, tau_seg,
                    color=m_color, alpha=0.8,
                    linewidth=self.get_molecule_linewidth_by_name(
                        m_name,
                        self._get_theme_value("full_spectrum_model_linewidth", 0.8),
                    ),
                    label=m_label,
                    zorder=self._get_theme_value("zorder_model", 3),
                )

        # Total τ line
        if self.show_total and total_tau is not None and len(entries) > 1:
            # Find the common wavelength segment
            ref_lam = entries[0][0]
            ref_mask = (ref_lam >= xr[0]) & (ref_lam <= xr[1])
            if np.any(ref_mask):
                total_line, = ax.plot(
                    ref_lam[ref_mask], total_tau,
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=self._get_theme_value("model_linewidth", 2) * 0.6,
                    alpha=0.7, linestyle="-",
                    label="Total \u03c4",
                    zorder=self._get_theme_value("zorder_model", 3) + 1,
                )
                setattr(total_line, TOTAL_TAU_TAG, True)

    # ------------------------------------------------------------------
    # Fraction mode
    # ------------------------------------------------------------------
    def compute_fractional_shares(
        self,
        xr: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, List[tuple]]:
        """Return each molecule's share of the total convolved τ per pixel.

        The pixels are those of the first molecule's τ grid within *xr*
        (default: the panel range); molecules on a different grid are
        interpolated onto it (zero outside their own range).  Negative or
        non-finite τ counts as zero.

        Returns
        -------
        wavelengths : np.ndarray
            The pixel wavelengths.
        shares : list[tuple]
            ``(share, color, label, name)`` per molecule, in cache order,
            where ``share = τᵢ / Σⱼ τⱼ`` - in [0, 1] and summing to 1 over
            the molecules - and NaN where the total τ is at or below
            :attr:`fraction_min_tau`.
        """
        self._ensure_mol_tau_cache()
        entries = self.mol_tau_cache
        if not entries:
            return np.array([]), []
        if xr is None:
            xr = self.xlim

        ref_lam = np.asarray(entries[0][0], dtype=float)
        in_range = (ref_lam >= xr[0]) & (ref_lam <= xr[1])
        lam = ref_lam[in_range]

        taus = []
        for m_lam, m_tau, *_ in entries:
            m_lam = np.asarray(m_lam, dtype=float)
            m_tau = np.asarray(m_tau, dtype=float)
            if m_lam.shape == ref_lam.shape and np.array_equal(m_lam, ref_lam):
                tau = m_tau[in_range]
            else:
                tau = np.interp(lam, m_lam, m_tau, left=0.0, right=0.0)
            taus.append(np.where(np.isfinite(tau) & (tau > 0), tau, 0.0))

        total = np.sum(taus, axis=0)
        defined = total > self.fraction_min_tau
        safe_total = np.where(defined, total, 1.0)
        shares = [
            (np.where(defined, tau / safe_total, np.nan), color, label, name)
            for tau, (_lam, _tau, color, label, name) in zip(taus, entries)
        ]
        return lam, shares

    def _render_fraction(self, ax: Axes, xr: Tuple[float, float]) -> None:
        """Render each molecule's share of the total τ (stacked to 1, or one line each)."""
        lam, shares = self.compute_fractional_shares(xr)
        if lam.size == 0:
            return
        zorder = self._get_theme_value("zorder_model", 3)
        if self.stacked_fill:
            bottom = np.zeros_like(lam)
            for share, color, label, _name in shares:
                top = bottom + share  # NaN (undefined share) leaves a gap
                # No edge: a molecule with no share in range would otherwise
                # leave a zero-height outline along its boundary.
                ax.fill_between(
                    lam, bottom, top, color=color, alpha=0.45, linewidth=0,
                    label=label, zorder=zorder,
                )
                bottom = top
        else:
            base_lw = self._get_theme_value("full_spectrum_model_linewidth", 0.8)
            for share, color, label, name in shares:
                ax.plot(
                    lam, share, color=color, alpha=0.8, label=label, zorder=zorder,
                    linewidth=self.get_molecule_linewidth_by_name(name, base_lw),
                )

    # ------------------------------------------------------------------
    # Stem mode
    # ------------------------------------------------------------------
    def _render_stem(self, ax: Axes, xr: Tuple[float, float]) -> None:
        """Render per-line τ as vertical lines."""
        molecules_to_plot: list = []

        if self.molecule is not None:
            molecules_to_plot.append(self.molecule)
        elif self.molecules is not None:
            molecules_to_plot = list(
                self.molecules.get_visible_molecules(return_objects=True)
            )

        for mol in molecules_to_plot:
            color = self.get_molecule_color(mol)
            label = self.get_molecule_display_name(mol)
            intensity_obj = getattr(mol, "intensity", None)
            if intensity_obj is None:
                continue

            if hasattr(intensity_obj, "get_lines_in_range_with_intensity"):
                lines = intensity_obj.get_lines_in_range_with_intensity(
                    xr[0], xr[1],
                )
            else:
                continue

            lam_vals = []
            tau_vals = []
            for ln, _intens, tau in lines:
                if tau is not None and tau > 0:
                    lam_vals.append(ln.lam)
                    tau_vals.append(tau)

            if lam_vals:
                # A log axis cannot show a stem starting at zero.
                bottom = self.tau_floor if self.log_scale else 0
                stems = ax.vlines(
                    lam_vals, bottom, tau_vals,
                    colors=color, alpha=0.7,
                    linewidth=0.8, label=label,
                    zorder=self._get_theme_value("zorder_model", 3),
                )
                setattr(stems, TAU_STEM_TAG, True)

    def stem_tau_max(self) -> float:
        """Return the tallest stem drawn in ``"stem"`` mode (0.0 when none)."""
        ax = self.ax
        if ax is None:
            return 0.0
        peak = 0.0
        for coll in iter_tagged(ax, TAU_STEM_TAG, lines=False):
            for seg in coll.get_segments():
                if len(seg):
                    top = float(np.nanmax(seg[:, 1]))
                    if np.isfinite(top):
                        peak = max(peak, top)
        return peak

    # ------------------------------------------------------------------
    def compute_ylim(self, ymax_factor: float = 0.2) -> Tuple[float, float]:
        """Compute appropriate y-limits for the optical-depth panel."""
        if self.display_mode == "fraction":
            return FRACTION_YLIM
        max_tau = 0.0

        for entry in self.mol_tau_cache:
            m_lam, m_tau = entry[0], entry[1]
            m_mask = (m_lam >= self.xlim[0]) & (m_lam <= self.xlim[1])
            if np.any(m_mask):
                finite = np.isfinite(m_tau[m_mask])
                if np.any(finite):
                    max_tau = max(max_tau, float(np.nanmax(m_tau[m_mask][finite])))

        if max_tau <= 0:
            max_tau = 1.0

        if self.log_scale:
            return (self.tau_floor, max_tau * 2.0)
        return (0.0, max_tau * 1.1)