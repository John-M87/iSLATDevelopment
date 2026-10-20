"""
OpticalDepthPanel -- Concrete :class:`SpectralPanel` for optical depth display.

Renders the convolved optical-depth profile τ(λ) for one or more molecules on a single axes.  Supports two display modes:

* **profile** (default) - continuous Gaussian-convolved τ(λ) with
  stacked ``fill_between`` per molecule.
* **stem** - vertical lines at each line-center wavelength with
  height equal to the per-line τ value.

Used both as a standalone single-panel plot and as the cell type created by :class:`OpticalDepthSpectrumPlot`.
"""
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes

from .SpectralPanel import SpectralPanel, GapMode
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

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
        ``"profile"`` for convolved Gaussian profiles (default) or
        ``"stem"`` for per-line markers.
    log_scale : bool, optional
        Use logarithmic y-axis.  Default ``False``.
    show_total : bool, optional
        Show combined τ line when multiple molecules are rendered.
        Default ``True``.
    stacked_fill : bool, optional
        Use cumulative ``fill_between`` per molecule.  Default ``True``.
    tau_floor : float, optional
        Minimum τ value when *log_scale* is ``True``.  Default ``1e-10``.
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
        """Render optical-depth profiles or stem markers."""
        self._ensure_mol_tau_cache()
        ax = self._resolve_axes()
        xr = self.xlim
        fg = self._get_theme_value("foreground", "black")

        if self.display_mode == "profile":
            self._render_profile(ax, xr)
        else:
            self._render_stem(ax, xr)

        ax.set_xlim(*xr)
        ax.set_ylabel("Optical Depth (\u03c4)", color=fg, fontsize=8)

        if self.log_scale:
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
                ax.plot(
                    ref_lam[ref_mask], total_tau,
                    color=self._get_theme_value("foreground", "black"),
                    linewidth=self._get_theme_value("model_linewidth", 2) * 0.6,
                    alpha=0.7, linestyle="-",
                    label="Total \u03c4",
                    zorder=self._get_theme_value("zorder_model", 3) + 1,
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
                ax.vlines(
                    lam_vals, 0, tau_vals,
                    colors=color, alpha=0.7,
                    linewidth=0.8, label=label,
                    zorder=self._get_theme_value("zorder_model", 3),
                )

    # ------------------------------------------------------------------
    def compute_ylim(self, ymax_factor: float = 0.2) -> Tuple[float, float]:
        """Compute appropriate y-limits for the optical-depth panel."""
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