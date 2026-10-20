import os
import platform
import traceback
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
from typing import TYPE_CHECKING, Any, Dict, Optional

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c
from ..GUIFunctions import create_button, create_menu_btn
from ..Tooltips import CreateToolTip, MenuToolTip
from .ResizableFrame import ResizableFrame
from iSLAT.Modules.GUI.Widgets.ChartWindow import MoleculeSelector
from iSLAT.Modules.GUI.Widgets.SampleManagerWindow import SampleManagerWindow
from iSLAT.Modules.GUI.Widgets.SortMoleculesWindow import SortMoleculesWindow
from iSLAT.Modules.GUI.Widgets.BulkApplyWindow import BulkApplyPropertiesWindow
from iSLAT.Modules.GUI.PlotGridWindow import PlotGridWindow
from iSLAT.Modules.GUI.FullSpectrumWindow import FullSpectrumWindow
from iSLAT.Modules.FileHandling.iSLATFileHandling import (
    write_molecules_to_csv, generate_csv, line_saves_file_path,
    line_saves_file_name, example_data_folder_path, read_from_user_csv
)
from iSLAT.Modules.FileHandling import save_folder_path, models_folder_path
from iSLAT.Modules.Plotting.FullSpectrumView import output_full_spectrum
from iSLAT.Modules.DataProcessing.Slabfit import SlabModel
from iSLAT.Modules.DataProcessing.BatchFittingService import BatchFittingService
from iSLAT.Modules.DataProcessing.DeblendingService import DeblendingService
from iSLAT.Modules.DataProcessing.LineSaveService import LineSaveService
from iSLAT.Modules.GUI.ControlSurface import ControlSurface

if TYPE_CHECKING:
    from iSLAT.Modules.Plotting.MainPlot import iSLATPlot

# ---------------------------------------------------------------------------
# TopBarSurface - ControlSurface implementation for the TopBar dynamic area
# ---------------------------------------------------------------------------
class TopBarSurface(ControlSurface):
    """Concrete :class:`ControlSurface` that renders :class:`ControlField` objects
    inside the TopBar's dynamic button area.

    Fields are packed left-to-right with ``side="left"``.  Typically used for
    :class:`ToggleField` instances that should appear as push-buttons next to
    the static TopBar buttons (e.g. "Toggle Residuals" when the full-spectrum
    view is active).

    Parameters
    ----------
    container:
        The ``tk.Frame`` that acts as the widget parent.
    """
    def __init__(self, container) -> None:
        super().__init__()
        self._container = container

    # ------------------------------------------------------------------
    # Internal rebuild
    # ------------------------------------------------------------------
    def _rebuild(self) -> None:
        from iSLAT.Modules.GUI.ControlField import RenderContext

        try:
            for child in self._container.winfo_children():
                child.destroy()
        except Exception:
            pass
        self._widget_refs.clear()

        for field in self._fields.values():
            try:
                widgets = field.build_widget(self._container, RenderContext.TOP_BAR)
            except Exception:
                widgets = []

            self._widget_refs[field.key] = widgets
            for w in widgets:
                try:
                    w.pack(side="left", padx=2, pady=2)
                except Exception:
                    pass

class TopBar(ResizableFrame):
    def __init__(
        self, 
        master: tk.Widget, 
        islat: Any, 
        theme: Dict[str, Any], 
        main_plot: 'iSLATPlot', 
        data_field: Any, 
        control_panel: Any,
        config: Dict[str, Any]
    ) -> None:
        # Initialize the ResizableFrame with theme
        super().__init__(master, theme=theme, borderwidth=1, relief="groove")
        
        self.master = master
        self.islat = islat
        self.main_plot = main_plot
        self.data_field = data_field
        self.config = config
        self.control_panel = control_panel

        # Initialize services for non-GUI logic
        self.batch_fitting_service = BatchFittingService()
        self.deblending_service = DeblendingService()
        self.line_save_service = LineSaveService()

        self.button_frame = tk.Frame(self, bg=self.theme.get("background", "#181A1B"))
        self.button_frame.grid(row=0, column=1)

        # Create buttons for options
        self._create_buttons()
        
        # Create and add toolbar 
        toolbar_frame = tk.Frame(self)
        toolbar_frame.grid(row=0, column=2, sticky="nsew")
        self.toolbar = self.main_plot.create_toolbar(toolbar_frame)

        # Dynamic button area: views register ToggleFields here on activation.
        # Placed after the matplotlib toolbar (column=3).
        self._dynamic_toggle_frame = tk.Frame(self, bg=self.theme.get("background", "#181A1B"))
        self._dynamic_toggle_frame.grid(row=0, column=3)
        self._top_bar_surface = TopBarSurface(self._dynamic_toggle_frame)

        self.atomic_toggle: bool = False
        self.line_toggle: bool = False
        
        # Apply initial theme
        # self.apply_theme(theme)
    
    def _create_buttons(self):
        """Create all the button widgets."""
        molecule_drpdwn = create_menu_btn(self.button_frame, self.theme, "Manage Molecules", 0, 0)

        btn_theme = self.theme.get("buttons", {}).get("DefaultBotton", {})
        molecule_menu = tk.Menu(molecule_drpdwn, tearoff=0,
            bg=btn_theme.get("background", "lightgray"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            activebackground=btn_theme.get("active_background", "gray"),
            activeforeground=self.theme.get("foreground", "#F0F0F0"),
        )
        molecule_menu.add_command(label="HITRAN Query", command=self.hitran_query)
        molecule_menu.add_command(label="Default Molecules", command=self.default_molecules)
        molecule_menu.add_command(label="Add Molecules", command=self.add_molecule)
        molecule_menu.add_command(label="Sort Molecules", command=self.sort_molecules)
        molecule_menu.add_command(label="Bulk Apply Properties", command=self.bulk_apply_properties)
        molecule_menu.add_command(label="Export Models", command=self.export_models)
        molecule_menu.add_separator()
        molecule_menu.add_command(label="Save Parameters (Ctrl+S)", command=self.save_parameters)
        molecule_menu.add_command(label="Load Parameters (Ctrl+L)", command=self.load_parameters)
        molecule_menu.add_command(label="Load Parameters From File (Ctrl+Shift+L)", command=self.load_parameters_from_file)
        molecule_menu.add_command(label="Import Parameters From File", command=self.import_parameters_from_file)
        molecule_menu.add_command(label="Save Parameters To File (Ctrl+Shift+S)", command=self.save_parameters_to_file)
        molecule_drpdwn.config(menu=molecule_menu)
        MenuToolTip(molecule_menu, {
            "HITRAN Query":                        "Open the HITRAN molecule selector\nto query and add molecules by species.",
            "Default Molecules":                   "Load the default set of molecules\nfor the current spectrum.",
            "Add Molecules":                       "Add a molecule from the available\nHITRAN files.",
            "Sort Molecules":                      "Sort the molecules in the control panel\nby a chosen property (temperature, radius,\ncolumn density, …), ascending or descending.",
            "Bulk Apply Properties":               "Set one or more properties on many molecules\nat once, optionally only the visible ones.",
            "Export Models":                       "Export current model spectra\nto CSV files for external use.",
            "Save Parameters (Ctrl+S)":            "Save current molecule parameters\nto the default save file\nfor this spectrum.",
            "Load Parameters (Ctrl+L)":            "Load molecule parameters from\nthe default save file\nfor this spectrum.",
            "Load Parameters From File (Ctrl+Shift+L)": "Load molecule parameters from\na user-selected CSV save file,\nreplacing existing molecules.",
            "Import Parameters From File":              "Load molecules from a CSV save file\nand add them to the existing set\nwithout replacing current molecules.",
            "Save Parameters To File (Ctrl+Shift+S)":   "Save current molecule parameters\nto a user-selected CSV file.",
        })

        spec_functions_drpwn = create_menu_btn(self.button_frame, self.theme, "Spectral Functions", 0, 1)
        spec_functions_menu = tk.Menu(spec_functions_drpwn, tearoff=0,
            bg=btn_theme.get("background", "lightgray"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            activebackground=btn_theme.get("active_background", "gray"),
            activeforeground=self.theme.get("foreground", "#F0F0F0"),
        )
        spec_functions_menu.add_command(label="Save Line", command=self.save_line)
        spec_functions_menu.add_command(label="Fit Line", command=self.fit_selected_line)
        spec_functions_menu.add_command(label="Fit Saved Lines", command=self.fit_saved_lines)
        spec_functions_menu.add_command(label="Fit Saved Lines To Sample", command=self.fit_saved_lines_to_sample)
        #spec_functions_menu.add_command(label="Fit from Batch Config", command=self.fit_from_batch_config)
        #spec_functions_menu.add_command(label="Find Single Lines", command=self.find_single_lines)
        spec_functions_menu.add_command(label="Single Slab Fit", command=self.single_slab_fit)
        spec_functions_menu.add_command(label="Line de-Blender", command=lambda: self.fit_selected_line(deblend=True))
        spec_functions_menu.add_separator()
        spec_functions_menu.add_command(label="Subtract Models from Data", command=self.subtract_models_from_data)
        spec_functions_menu.add_separator()
        spec_functions_menu.add_command(label="Output Full Spectrum (Ctrl+Shift+F)", command=lambda: output_full_spectrum(self.islat))
        spec_functions_menu.add_command(label="Display Full Spectrum (Ctrl+F)", command=lambda: FullSpectrumWindow(self.master, self.islat))
        spec_functions_menu.add_separator()
        spec_functions_menu.add_command(label="Center View\u2026 (C)", command=self.open_center_view_dialog)
        spec_functions_menu.add_separator()
        spec_functions_menu.add_command(label="Manage Sample\u2026", command=self.manage_sample)
        
        spec_functions_drpwn.config(menu=spec_functions_menu)
        MenuToolTip(spec_functions_menu, {
            "Save Line":                        "Save the currently selected line\nto the output line measurements file.",
            "Fit Line":                         "Fit a Gaussian to the currently\nselected spectral line.",
            "Fit Saved Lines":                  "Fit Gaussians to all lines\nin the saved line list.",
            "Fit Saved Lines To Sample":        "Fit saved lines across a sample\nof spectrum files.",
            "Single Slab Fit":                  "Run a single-slab LTE model fit\nusing the saved line measurements.",
            "Line de-Blender":                  "Fit and de-blend overlapping\nspectral lines in the selection.",
            "Subtract Models from Data":        "Subtract all visible molecule models\nfrom the observed spectrum.",
            "Output Full Spectrum (Ctrl+Shift+F)": "Save the full combined model spectrum\nto a file.",
            "Display Full Spectrum (Ctrl+F)":   "Open the full combined model spectrum\nin a separate window.",
            "Manage Sample\u2026":              "Open the sample manager to add,\nremove, and switch between spectra.",
            "Center View\u2026 (C)":           "Reposition the plot window: center + range,\ncenter + number of data points,\nor an exact start / end.\nKeybind: C",
        })

        # ── Views dropdown ─────────────────────────────────────────────
        views_drpwn = create_menu_btn(self.button_frame, self.theme, "Views", 0, 2)
        views_menu = tk.Menu(views_drpwn, tearoff=0,
            bg=btn_theme.get("background", "lightgray"),
            fg=self.theme.get("foreground", "#F0F0F0"),
            activebackground=btn_theme.get("active_background", "gray"),
            activeforeground=self.theme.get("foreground", "#F0F0F0"),
        )
        for _view_name in self.main_plot.views:
            views_menu.add_command(
                label=_view_name,
                command=lambda n=_view_name: self.main_plot.switch_view(n),
            )

        # Main-plot options: Y auto-scale / manual limits and observed-data
        # visibility.  The check marks are re-read from the plot controller
        # each time the menu opens so they always reflect the real state.
        views_menu.add_separator()
        self._y_autoscale_var = tk.BooleanVar(value=True)
        self._observed_var = tk.BooleanVar(value=True)
        views_menu.add_checkbutton(
            label="Auto-scale Y Axis",
            variable=self._y_autoscale_var,
            command=lambda: self.main_plot.set_y_autoscale(self._y_autoscale_var.get()),
        )
        self._y_model_var = tk.BooleanVar(value=False)
        views_menu.add_checkbutton(
            label="Auto-scale Y to Model Flux",
            variable=self._y_model_var,
            command=lambda: self.main_plot.set_y_autoscale_source(
                "model" if self._y_model_var.get() else "data"),
        )
        views_menu.add_command(label="Set Y Limits\u2026", command=self.open_y_limits_dialog)
        views_menu.add_checkbutton(
            label="Show Observed Data",
            variable=self._observed_var,
            command=lambda: self.main_plot.toggle_observed_data(self._observed_var.get()),
        )
        views_menu.configure(postcommand=self._sync_plot_option_vars)

        views_drpwn.config(menu=views_menu)
        MenuToolTip(views_menu, {
            "Three Panel":        "Standard 3-panel layout:\nfull spectrum, line inspection,\nand population diagram.",
            "Full Spectrum":      "Multi-panel full-spectrum view\nshowing the entire wavelength range.",
            "Population Diagram": "Standalone Boltzmann / rotation\ndiagram for the active molecule.",
            "Line Grid":          "Grid of individual line fits\nfrom the most recent fit run.",
            "Auto-scale Y Axis":  "ON: the main plot's Y axis follows\nthe data in view.\nOFF: it stays at the current / set limits.",
            "Auto-scale Y to Model Flux": "When auto-scaling, fit the Y axis to the\nplotted model (molecule lines and sum)\ninstead of the observed data.",
            "Set Y Limits\u2026":  "Pin the main plot's Y axis to\nspecific values (turns auto-scale off).",
            "Show Observed Data": "Show / hide the observed spectrum\non the main plot.",
        })
        # ──────────────────────────────────────────────────────────────

        saved_lines_tip = "Show saved lines\nfrom the 'Input Line List'\nKeybind: S"
        atomic_lines_tip = "Show atomic lines\nusing separation threshold\nset in the 'Line Separ.\nKeybind: A"
        #export_model_tip = "Export current\nmodels into csv files"
        toggle_legend_tip = "Turn legend on/off\nKeybind: L"
        toggle_full_spectrum_tip = "Toggle full spectrum view on/off\nKeybind: F\n\nOpen in new window: Ctrl+F"
        toggle_summed_tip = "Toggle summed model flux on/off\n(gray fill in plot)\nKeybind: M"
        create_button(self.button_frame, self.theme, "Saved Lines", self.toggle_saved_lines, 0, 3, tip_text=saved_lines_tip)
        create_button(self.button_frame, self.theme, "Atomic Lines", self.toggle_atomic_lines, 0, 4, tip_text=atomic_lines_tip)
        create_button(self.button_frame, self.theme, "Full Spectrum", self.toggle_full_spectrum, 0, 5, tip_text=toggle_full_spectrum_tip)
        create_button(self.button_frame, self.theme, "Total Model", self.toggle_summed_spectrum, 0, 6, tip_text=toggle_summed_tip)
        create_button(self.button_frame, self.theme, "Legend", self.main_plot.toggle_legend, 0, 7, tip_text=toggle_legend_tip)
        
        # Navigate buttons - compact with minimal padding
        retreat_tip = "Retreat the plot start\nby the current range value\nShortcut: <"
        advance_tip = "Advance the plot start\nby the current range value\nShortcut: >"
        
        self._retreat_btn = ttk.Button(self.button_frame, text="<", command=self.retreat_plot_start, width=2)
        self._retreat_btn.grid(row=0, column=9, padx=(4, 0), pady=2, sticky="nsew")
        CreateToolTip(self._retreat_btn, retreat_tip)
        
        self._advance_btn = ttk.Button(self.button_frame, text=">", command=self.advance_plot_start, width=2)
        self._advance_btn.grid(row=0, column=10, padx=(0, 1), pady=2, sticky="nsew")
        CreateToolTip(self._advance_btn, advance_tip)

    def save_line(self, save_type="selected"):
        """Save the currently selected line to the line saves file."""
        # Use service to extract line info
        selected_line_info, error_msg = self.line_save_service.extract_line_info_from_selection(
            self.main_plot, save_type,
            wave_data=self.islat.wave_data,
            flux_data=self.islat.flux_data,
            err_data=self.islat.err_data
        )
        
        if error_msg:
            self.data_field.insert_text(f"{error_msg}\n")
            return
        
        # Get selection bounds
        selected_wave = getattr(self.main_plot, 'current_selection', None) # use current_selection instead of selected_wave. 
                                                                           # current_selection includes the actual selection range and selected_wave is the nearest pixels in the wave_data array.
        xmin, xmax = self.line_save_service.get_selection_bounds(
            selected_wave,
            self.main_plot.current_selection,
            selected_line_info['lam']
        )
        
        # Format line info for saving
        line_info = self.line_save_service.format_line_for_save(
            selected_line_info,
            self.islat.active_molecule.name,
            xmin,
            xmax
        )
        
        try:
            if not self.islat.output_line_measurements:
                self.data_field.insert_text("No output line measurements file specified.\n")
                return
            ifh.save_line(line_info, file_name=self.islat.output_line_measurements)
            self.data_field.insert_text(f"Saved line at {line_info['lam']:.4f} μm\n")
        except Exception as e:
            self.data_field.insert_text(f"Error saving line: {e}\n")

    def save_all_lines_in_range(self):
        """Save all lines of the active molecule within the current selected range."""
        if not hasattr(self.main_plot, 'current_selection') or self.main_plot.current_selection is None:
            self.data_field.insert_text("No region selected for saving.\n")
            return

        xmin, xmax = self.main_plot.current_selection

        if not self.islat.output_line_measurements:
            self.data_field.insert_text("No output line measurements file specified.\n")
            return

        # Only use the active molecule
        molecules_to_save = []
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            molecules_to_save = [self.islat.active_molecule]

        if not molecules_to_save:
            self.data_field.insert_text("No active molecule to save lines from.\n")
            return

        saved_count = 0
        for mol in molecules_to_save:
            try:
                line_data = mol.intensity.get_lines_in_range_with_intensity(xmin, xmax)
            except Exception:
                continue

            for line_obj, intensity, tau in line_data:
                try:
                    line_info = self.line_save_service.format_line_for_save(
                        {
                            'lam': line_obj.lam,
                            'up_lev': getattr(line_obj, 'lev_up', ''),
                            'low_lev': getattr(line_obj, 'lev_low', ''),
                            'tau': tau if tau is not None else 0.0,
                            'intensity': intensity,
                            'inten': intensity,
                            'e_up': getattr(line_obj, 'e_up', 0.0),
                            'e_low': getattr(line_obj, 'e_low', 0.0),
                            'a_stein': getattr(line_obj, 'a_stein', 0.0),
                            'g_up': getattr(line_obj, 'g_up', 1.0),
                            'g_low': getattr(line_obj, 'g_low', 1.0),
                        },
                        mol.name,
                        xmin,
                        xmax,
                    )
                    ifh.save_line(line_info, file_name=self.islat.output_line_measurements, silent=True)
                    saved_count += 1
                except Exception as e:
                    self.data_field.insert_text(f"Error saving line at {getattr(line_obj, 'lam', '?'):.4f} μm: {e}\n")

        if saved_count:
            self.data_field.insert_text(f"Saved {saved_count} line(s) in range [{xmin:.4f}, {xmax:.4f}] μm.\n")
        else:
            self.data_field.insert_text("No lines found in the selected range.\n")

    def toggle_saved_lines(self):
        """Show saved lines as vertical dashed lines on the plot."""
        loaded_lines = ifh.read_line_saves(file_name=self.islat.input_line_list)
        if loaded_lines.empty:   
            self.data_field.insert_text("No saved lines found.\n")
            return
        try:
            # Let MainPlot flip the toggle and forward to the active view
            self.main_plot.toggle_saved_lines(loaded_lines=loaded_lines)
            # Keep TopBar in sync for any legacy readers
            self.line_toggle = self.main_plot.line_toggle
            
        except Exception as e:
            self.data_field.insert_text(f"Error loading saved lines: {e}\n")

    def fit_selected_line(self, deblend=False):
        """Fit the currently selected line using LMFIT."""
        pm = self.main_plot
        if not hasattr(pm, 'current_selection') or pm.current_selection is None:
            self.data_field.insert_text("No region selected for fitting.\n")
            return

        xmin, xmax = pm.current_selection

        try:
            import numpy as np
            fit_mask = (pm.islat.wave_data >= xmin) & (pm.islat.wave_data <= xmax)
            x_fit = pm.islat.wave_data[fit_mask]
            y_fit = pm.islat.flux_data[fit_mask]

            if len(x_fit) < 5:
                self.data_field.insert_text("Insufficient data points in selection for fitting.\n")
                return

            fit_kwargs = dict(xmin=xmin, xmax=xmax, deblend=deblend)
            if deblend:
                active_mol = getattr(pm.islat, 'active_molecule', None)
                fit_kwargs.update(
                    wave_data_full=pm.islat.wave_data,
                    err_data_full=pm.islat.err_data,
                    user_settings=getattr(pm.islat, 'user_settings', {}),
                    active_molecule_fwhm=getattr(active_mol, 'fwhm', None) if active_mol else None,
                    lines_with_intensity=(
                        active_mol.intensity.get_lines_in_range_with_intensity(xmin, xmax)
                        if active_mol and hasattr(active_mol, 'intensity')
                        else None
                    ),
                    line_threshold=(
                        pm.islat.user_settings.get('line_threshold', 0.03)
                        if getattr(pm.islat, 'user_settings', None) else 0.03
                    ),
                )

            fit_result, fitted_wave, fitted_flux = pm.fitting_engine.fit_gaussian_line(
                x_fit, y_fit, **fit_kwargs
            )
            pm.fit_result = (fit_result, fitted_wave, fitted_flux)

            # Render the fit via the active view's line-inspection panel
            if fit_result is not None and fitted_wave is not None:
                user_settings = getattr(pm.islat, 'user_settings', {})
                legend_visible = pm.legend_toggle
                grid = getattr(getattr(pm, '_three_panel_view', None), '_grid', None)
                if grid is not None and grid.inspection_panel is not None:
                    grid.inspection_panel.render_fit_results(
                        grid.ax_inspection,
                        pm.fit_result,
                        xmin, xmax,
                        user_settings=user_settings,
                        legend_visible=legend_visible,
                    )
                pm.canvas.draw_idle()

            if fit_result is not None and hasattr(fit_result, 'params'):
                line_params = pm.fitting_engine.extract_line_parameters()
                if deblend:
                    self._display_deblend_results(line_params)
                else:
                    self._display_single_gaussian_results(line_params)
            else:
                self.data_field.insert_text("Fit completed but no valid result object returned.\n", clear_after=False)

        except Exception as e:
            self.data_field.insert_text(f"Error during fitting: {e}\n", clear_after=False)
            self.data_field.insert_text(f"Traceback: {traceback.format_exc()}\n", clear_after=False)

    def _display_deblend_results(self, line_params):
        """Display and save deblended line fit results."""
        self.data_field.insert_text("\nDe-blended line fit results:\n", clear_after=False)
        
        selection = self.main_plot.current_selection
        if selection and len(selection) >= 2:
            xmin, xmax = selection[0], selection[-1]
        else:
            self.data_field.insert_text("Invalid selection for deblending.\n", clear_after=False)
            return
            
        line_info = self.islat.active_molecule.intensity.get_lines_in_range_with_intensity(xmin, xmax)
        
        # Extract deblended components using service
        components = self.deblending_service.extract_deblended_components(
            line_params,
            line_info,
            self.islat.active_molecule.name
        )
        
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        spectrum_base_name = os.path.splitext(spectrum_name)[0] if spectrum_name != "unknown" else "default"
        save_file_name = f"{spectrum_base_name}-{line_saves_file_name}"
        
        # Display each component
        for component in components:
            self.data_field.insert_text(f"\nComponent {component['index']+1}:\n", clear_after=False)
            display_msgs = self.deblending_service.format_component_display(component)
            for msg in display_msgs:
                self.data_field.insert_text(msg, clear_after=False)
        
        # Save components using service
        saved_count = self.deblending_service.save_deblended_components(components, save_file_name)
        
        if not components:
            self.data_field.insert_text("No components found in fit result.\n", clear_after=False)
        else:
            self.data_field.insert_text(f"\nDe-blended line fit completed with {len(components)} components!", clear_after=False)
            
            # Save summary files
            fit_result_summary = self.main_plot.fitting_engine.get_fit_results_summary()
            fit_results_components = self.main_plot.fitting_engine.get_fit_results_components()
            self.deblending_service.save_deblend_summary(
                fit_result_summary,
                fit_results_components,
                spectrum_base_name,
                line_saves_file_path
            )
            
            # Save plot
            figpath = os.path.join(line_saves_file_path, f"{spectrum_base_name}-deblend_plot.pdf")
            self.main_plot.save_fig(figpath, dpi=300)
            
            if saved_count > 0:
                self.data_field.insert_text(f"\nDe-blended line saved in /LINESAVES!", clear_after=False)

    def _display_single_gaussian_results(self, line_params):
        """Display single Gaussian fit results."""
        self.data_field.insert_text("\nGaussian fit results:\n", clear_after=False)
        
        display_msgs = self.deblending_service.format_single_gaussian_display(line_params)
        for msg in display_msgs:
            self.data_field.insert_text(msg, clear_after=False)

    def fit_saved_lines(self, multiple_files=False):
        """
        Fit all saved lines using batch fitting service.
        
        Args:
            multiple_files (bool): If True, allows user to select multiple spectrum files.
                                 If False, fits saved lines to the currently loaded spectrum.
        """
        if not self.islat.input_line_list:
            # Prompt user to load a line list first
            self.data_field.insert_text("No line list loaded. Please select a line list file.\n")
            from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
            result = load_input_line_list()
            
            if result is None:
                self.data_field.insert_text("No line list selected. Operation cancelled.\n")
                return
            
            file_path, file_name = result
            self.islat.input_line_list = file_path
            self.data_field.insert_text(f"Loaded line list: {file_name}\n")
            
            # Update the FileInteractionPane label if available
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'file_interaction_pane'):
                self.islat.GUI.file_interaction_pane.refresh()
        
        if not self.islat.output_line_measurements:
            self.data_field.insert_text("No output line measurements file configured. Using default\n")
        
        # Progress callback for GUI updates
        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)
        
        if multiple_files:
            # Ask user to select multiple spectrum files
            spectrum_files = filedialog.askopenfilenames(
                title="Select Spectrum Files to Fit Saved Lines",
                filetypes=[("All files", "*.*")],
                initialdir=example_data_folder_path
            )
            
            if not spectrum_files:
                self.data_field.insert_text("No spectrum files selected.\n")
                return
            
            # Use batch fitting service for multiple files
            plot_grid_list, output_folder = self.batch_fitting_service.fit_lines_to_multiple_spectra(
                saved_lines_file=self.islat.input_line_list,
                spectrum_files=list(spectrum_files),
                config=self.config,
                progress_callback=progress_callback,
                base_output_path=line_saves_file_path,
                get_mole_save_data=self.islat.get_mole_save_data
            )
            
            if plot_grid_list:
                # Check user setting for direct PDF save
                save_directly_to_pdf = self.config.get('save_fit_plot_grid_directly_to_PDF', False)
                
                if save_directly_to_pdf:
                    # Use the output folder created during batch processing
                    save_path = output_folder if output_folder else line_saves_file_path
                    self.batch_fitting_service.save_plot_grids_to_pdf(
                        plot_grid_list,
                        save_path,
                        progress_callback=progress_callback
                    )
                else:
                    # Open a new window to display the plot grid
                    PlotGridWindow(self.master, plot_grid_list, theme=self.theme)
        else:
            # Fit saved lines to the currently loaded spectrum
            self._perform_saved_lines_fit()

    def _perform_saved_lines_fit(self, spectrum_name=None, wavedata=None, fluxdata=None, err_data=None, plot_results=True, plot_grid=False):
        """Internal method to perform saved lines fitting on a single spectrum."""
        saved_lines_file = self.islat.input_line_list
        
        if not saved_lines_file:
            self.data_field.insert_text("No input line list file configured.\n")
            return None
        
        # Default to islat data when not explicitly provided
        if spectrum_name is None:
            spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        if wavedata is None:
            wavedata = self.islat.wave_data
        if fluxdata is None:
            fluxdata = self.islat.flux_data
        if err_data is None:
            err_data = getattr(self.islat, 'err_data', None)
        
        # Progress callback for GUI updates
        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)
        
        # Determine the output file and path from the user's selection
        output_file = None
        output_path = None
        if self.islat.output_line_measurements:
            output_file = os.path.basename(self.islat.output_line_measurements)
            output_path = os.path.dirname(self.islat.output_line_measurements)
        
        # Set the output folder before calling fit so LineAnalyzer uses it
        if output_path:
            self.batch_fitting_service._current_output_folder = output_path
        
        # Use batch fitting service
        fit_data = self.batch_fitting_service.fit_lines_to_spectrum(
            saved_lines_file=saved_lines_file,
            spectrum_name=spectrum_name,
            wavedata=wavedata,
            fluxdata=fluxdata,
            err_data=err_data,
            output_file=output_file,
            progress_callback=progress_callback
        )
        
        if fit_data:
            fit_results_csv_data, fit_results_data = fit_data
            
            # Display summary
            summary = self.batch_fitting_service.get_fit_summary(fit_results_csv_data)
            self.data_field.insert_text(
                f"Completed fitting {summary['successful_fits']} out of {summary['total_lines']} lines.\n",
                clear_after=False
            )
            self.data_field.insert_text(
                f"Results saved to: {self.islat.output_line_measurements}\n",
                clear_after=False
            )
            
            # Display progress for each line
            progress_msgs = self.batch_fitting_service.format_fit_progress(fit_results_csv_data)
            for msg in progress_msgs:
                self.data_field.insert_text(msg, clear_after=False)
            
            # Always build a FitLinesPlotGrid so the Line Grid view
            # reflects the latest results whether or not it is currently active.
            from iSLAT.Modules.Plotting.FitLinesPlotGrid import FitLinesPlotGrid
            _sname = spectrum_name or getattr(self.islat, 'loaded_spectrum_name', 'unknown')
            try:
                _grid = FitLinesPlotGrid(
                    fit_data=fit_data,
                    wave_data=wavedata,
                    flux_data=fluxdata,
                    err_data=err_data,
                    fit_line_uncertainty=self.config.get('fit_line_uncertainty', 3.0),
                    spectrum_name=_sname,
                )
                _grid.generate_plot()
                # Push into the embedded view so it refreshes immediately
                _grid_view = getattr(self.main_plot, '_fit_lines_grid_view', None)
                if _grid_view is not None:
                    _grid_view.set_plot_grids([_grid])
            except Exception as _e:
                print(f"Warning: could not build FitLinesPlotGrid: {_e}")
                _grid = None

            if plot_grid:
                return _grid

            if plot_results:
                from iSLAT.Modules.Plotting.SpectrumPanel import SpectrumPanel
                uncertainty_sigma = self.config.get('fit_line_uncertainty', 3.0)
                SpectrumPanel.plot_gaussian_fits(
                    self.main_plot.ax1,
                    fit_results_data,
                    color='lime',
                    uncertainty_sigma=uncertainty_sigma,
                )
                self.main_plot.canvas.draw_idle()
        else:
            self.data_field.insert_text("No lines found or no fits completed successfully.\n", clear_after=False)
        
        return None

    def fit_saved_lines_to_sample(self):
        """Fit saved line list to a number of spectrum files at once.
        
        If sample spectra have been added via the + button, those are used
        automatically. Otherwise falls back to the file-selection dialog.
        """
        if hasattr(self.islat, 'sample_spectra') and len(self.islat.sample_spectra) > 1:
            # Use the sample spectra list (already includes the primary)
            self._fit_saved_lines_to_files(list(self.islat.sample_spectra))
        else:
            self.fit_saved_lines(multiple_files=True)

    def _fit_saved_lines_to_files(self, spectrum_files):
        """Fit saved lines to a given list of spectrum files."""
        if not self.islat.input_line_list:
            self.data_field.insert_text("No line list loaded. Please select a line list file.\n")
            from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
            result = load_input_line_list()
            if result is None:
                self.data_field.insert_text("No line list selected. Operation cancelled.\n")
                return
            file_path, file_name = result
            self.islat.input_line_list = file_path
            self.data_field.insert_text(f"Loaded line list: {file_name}\n")
            
            # Update the FileInteractionPane label if available
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'file_interaction_pane'):
                self.islat.GUI.file_interaction_pane.refresh()

        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)

        self.data_field.insert_text(f"Fitting saved lines to {len(spectrum_files)} spectra...\n")

        plot_grid_list, output_folder = self.batch_fitting_service.fit_lines_to_multiple_spectra(
            saved_lines_file=self.islat.input_line_list,
            spectrum_files=spectrum_files,
            config=self.config,
            progress_callback=progress_callback,
            base_output_path=line_saves_file_path,
            get_mole_save_data=self.islat.get_mole_save_data
        )

        if plot_grid_list:
            save_directly_to_pdf = self.config.get('save_fit_plot_grid_directly_to_PDF', False)
            if save_directly_to_pdf:
                save_path = output_folder if output_folder else line_saves_file_path
                self.batch_fitting_service.save_plot_grids_to_pdf(
                    plot_grid_list, save_path, progress_callback=progress_callback
                )
            else:
                PlotGridWindow(self.master, plot_grid_list, theme=self.theme)

    def fit_from_batch_config(self):
        """Load the BatchFittingConfig.json and run the batch fitting pipeline.

        Uses the same GUI-loaded state as the regular fit-to-sample flow:
        - The active sample spectra (or file-dialog selection if none).
        - The currently loaded line list (prompts via file dialog if none).
        - The same parallel / PDF-save settings as the regular batch fit.
        The batch config provides overrides and toggle settings only.
        """
        from iSLAT.Modules.FileHandling.iSLATFileHandling import (
            load_batch_fitting_config,
        )

        batch_config = load_batch_fitting_config()

        # -- spectra: same logic as fit_saved_lines_to_sample ---------------
        if hasattr(self.islat, 'sample_spectra') and len(self.islat.sample_spectra) > 1:
            spectrum_files = list(self.islat.sample_spectra)
        else:
            spectrum_files_tuple = filedialog.askopenfilenames(
                title="Select Spectrum Files for Batch Config Fit",
                filetypes=[("All files", "*.*")],
                initialdir=example_data_folder_path,
            )
            if not spectrum_files_tuple:
                self.data_field.insert_text("No spectrum files selected.\n")
                return
            spectrum_files = list(spectrum_files_tuple)

        # -- line list: prefer config, fall back to GUI-loaded, then dialog --
        saved_lines: str | None = batch_config.get("saved_lines_file") or None
        if not saved_lines:
            if self.islat.input_line_list:
                saved_lines = self.islat.input_line_list
            else:
                self.data_field.insert_text(
                    "No line list in batch config or loaded in GUI. "
                    "Please select a line list file.\n"
                )
                from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
                result = load_input_line_list()
                if result is None:
                    self.data_field.insert_text("No line list selected. Operation cancelled.\n")
                    return
                file_path, file_name = result
                self.islat.input_line_list = file_path
                saved_lines = file_path
                self.data_field.insert_text(f"Loaded line list: {file_name}\n")
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'file_interaction_pane'):
                    self.islat.GUI.file_interaction_pane.refresh()

        # Write the resolved line list back into the config dict so the
        # service method does not need its own fallback logic.
        batch_config["saved_lines_file"] = saved_lines

        def progress_callback(msg):
            self.data_field.insert_text(msg, clear_after=False)

        self.data_field.insert_text(
            f"Running batch config: {len(spectrum_files)} spectra, "
            f"lines file: {os.path.basename(saved_lines)}\n"
        )

        plot_grid_list, output_folder = self.batch_fitting_service.fit_lines_from_batch_config(
            batch_config=batch_config,
            user_settings=self.config,
            progress_callback=progress_callback,
            get_mole_save_data=self.islat.get_mole_save_data,
            base_output_path=line_saves_file_path,
            spectrum_files=spectrum_files,
            molecules_dict=self.islat.molecules_dict,
        )

        if plot_grid_list:
            save_directly_to_pdf = self.config.get("save_fit_plot_grid_directly_to_PDF", False)
            if save_directly_to_pdf:
                save_path = output_folder if output_folder else line_saves_file_path
                self.batch_fitting_service.save_plot_grids_to_pdf(
                    plot_grid_list, save_path, progress_callback=progress_callback
                )
            else:
                PlotGridWindow(self.master, plot_grid_list, theme=self.theme)

        if output_folder:
            self.data_field.insert_text(f"Output saved to: {output_folder}\n")

    def find_single_lines(self):
        """Find isolated molecular lines (similar to single_finder function in original iSLAT)."""
        lines_to_show = 10

        try:
            self.data_field.clear()
            single_lines = self.main_plot.find_single_lines()
            #self.main_plot.plot_single_lines() # method no longer exists
            for i, line in enumerate(single_lines[:lines_to_show]):  # Show first lines_to_show lines
                self.data_field.insert_text(f"  Line {i+1}:", clear_after=False)
                for key, value in line.items():
                    self.data_field.insert_text(f"    {key}: {value}", clear_after=False)
                self.data_field.insert_text("\n", clear_after=False)
            
            if len(single_lines) > lines_to_show:
                self.data_field.insert_text(f"  ... and {len(single_lines) - lines_to_show} more lines\n", clear_after=False)
            
        except Exception as e:
            self.data_field.insert_text(f"Error finding single lines: {e}\n")

    def subtract_models_from_data(self):
        """Subtract visible models from the data spectrum and save result."""
        self.data_field.insert_text("Subtracting models from spectrum...\n", clear_after=True)
        
        try:
            result = self.islat.subtract_models_from_data(visible_only=True)
            if result:
                self.data_field.insert_text(f"Model subtraction complete.\n", clear_after=False)
            else:
                self.data_field.insert_text("Model subtraction failed. Check console for details.\n", clear_after=False)
        except Exception as e:
            self.data_field.insert_text(f"Error during model subtraction: {e}\n", clear_after=False)
    
    def single_slab_fit(self):
        """Run single slab fit analysis.
        
        When no saved line list is loaded, prompts the user to select one,
        runs the fit-saved-lines workflow first, and then uses the resulting
        fitted-line measurements as input for the slab fit.
        """
        ran_saved_lines_fit = False

        if not self.islat.input_line_list:
            # Prompt user to load a line list
            self.data_field.insert_text("No line list loaded. Please select a line list file.\n")
            from iSLAT.Modules.FileHandling.iSLATFileHandling import load_input_line_list
            result = load_input_line_list()

            if result is None:
                self.data_field.insert_text("No line list selected. Slab fit cancelled.\n")
                return

            file_path, file_name = result
            self.islat.input_line_list = file_path
            self.data_field.insert_text(f"Loaded line list: {file_name}\n")

            # Update the FileInteractionPane label if available
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'file_interaction_pane'):
                self.islat.GUI.file_interaction_pane.refresh()

            # Run fit saved lines to produce measurements with flux/error columns
            self.data_field.insert_text("Fitting saved lines before slab fit...\n", clear_after=False)
            self._perform_saved_lines_fit(plot_results=True)
            ran_saved_lines_fit = True

            # Determine the actual output file that fit saved lines wrote so the
            # slab fit can read the fitted flux/error columns from it.
            if self.islat.output_line_measurements:
                self.islat.input_line_list = self.islat.output_line_measurements
            else:
                # No explicit output was configured - reconstruct the default
                # path that save_fit_results used.
                from iSLAT.Modules.FileHandling import fit_save_lines_file_name
                spectrum_name = getattr(self.islat, 'loaded_spectrum_name', None)
                if spectrum_name is not None:
                    spectrum_base = os.path.splitext(spectrum_name)[0]
                    out_name = f"{spectrum_base}-{os.path.basename(file_path)}"
                else:
                    out_name = fit_save_lines_file_name
                output_path = self.batch_fitting_service._current_output_folder if hasattr(self.batch_fitting_service, '_current_output_folder') and self.batch_fitting_service._current_output_folder else str(line_saves_file_path)
                default_output = os.path.join(output_path, out_name)
                if os.path.exists(default_output):
                    self.islat.input_line_list = default_output
                    self.data_field.insert_text(f"Using fit results for slab fit: {out_name}\n", clear_after=False)
                else:
                    self.data_field.insert_text("Fit saved lines did not produce an output file. Cannot proceed with slab fit.\n")
                    return

        if self.islat.input_line_list is None:
            self.data_field.insert_text("No input line list available. Cannot perform slab fit.\n", clear_after=True)
            return

        self.data_field.insert_text("Running single slab fit analysis...\n", clear_after=False)

        try:
            try:
                if not self.islat.output_line_measurements:
                    self.data_field.insert_text(f"No output line measurements file specified.\nUsing default folder: {line_saves_file_path}.", clear_after=False)
                    output_folder = line_saves_file_path
                else:
                    output_folder = os.path.dirname(self.islat.output_line_measurements)
            except Exception as e:
                self.data_field.insert_text(f"Error determining output folder: {e}", clear_after=False)
                self.data_field.insert_text(f"Using default folder: {line_saves_file_path}", clear_after=False)
                output_folder = line_saves_file_path
            # Use the SlabModel class to perform the fit
            slab_model = SlabModel(
                source=self.islat.active_molecule,
                output_folder=output_folder,
                data_field=self.data_field,
                input_file=self.islat.input_line_list,
                flux_col_name=self.islat.user_settings.get("flux_col_name", "Flux_islat"),
                error_col_name=self.islat.user_settings.get("error_col_name", "Err_data")
            )
                
        except Exception as e:
            self.data_field.insert_text(f"Error loading single slab fit: {e}\n")
            return
        
        try:
            fitted_result = slab_model.fit()
        except Exception as e:
            self.data_field.insert_text(f"Error fitting slab model: {e}\n")
            return
        
        try:
            slab_model.update_source_parameters(fitted_result)
        except Exception as e:
            self.data_field.insert_text(f"Error updating molecule parameters: {e}\n")
            return

        try:
            if hasattr(self.islat, 'GUI') and self.islat.GUI is not None:
                if hasattr(self.islat.GUI, 'plot'):
                    self.islat.GUI.plot.update_model_plot()
            if self.control_panel is not None:
                self.control_panel._update_molecule_parameter_fields()
        except Exception as e:
            print(f"single_slab_fit: GUI refresh warning - {e}")

        try:
            slab_model.save_results(fitted_result)
        except Exception as e:
            self.data_field.insert_text(f"Error saving slab model results: {e}\n")
            return

    def export_models(self):
        """Export current models and data."""
        self.data_field.insert_text("Exporting current models...\n")
        
        # Create a new window for exporting the spectrum
        export_window = tk.Toplevel(self.master)
        export_window.title("Export Models")
        # Always on top
        export_window.attributes("-topmost", True)

        # Create a label in the new window
        label = tk.Label(export_window, text="Select a molecule:")
        label.grid(row=0, column=0, sticky="w")

        # Create a dropdown menu in the new window
        options = list(self.islat.molecules_dict.keys()) + ["SUM", "ALL"]
        dropdown_var = tk.StringVar()
        dropdown = ttk.Combobox(export_window, textvariable=dropdown_var, values=options)
        dropdown.set(options[0])
        dropdown.grid(row=1, column=0, sticky="w")

        # Matched pixel sampling checkbox — defaults to the current GUI/MoleculeDict state
        match_sampling_var = tk.BooleanVar(
            value=bool(getattr(self.islat.molecules_dict, 'match_spectral_sampling', False))
        )
        match_check = ttk.Checkbutton(
            export_window,
            text="Match pixel sampling",
            variable=match_sampling_var,
        )
        match_check.grid(row=2, column=0, sticky="w")

        # --- Save location -------------------------------------------------
        dir_var = tk.StringVar(value=str(models_folder_path))
        name_var = tk.StringVar(value=f"{dropdown_var.get()}_spec_output.csv")

        tk.Label(export_window, text="Save folder:").grid(row=3, column=0, sticky="w")
        dir_entry = ttk.Entry(export_window, textvariable=dir_var, width=40)
        dir_entry.grid(row=4, column=0, sticky="we")
        ttk.Button(
            export_window,
            text="Browse\u2026",
            command=lambda: self._browse_export_dir(export_window, dir_var),
        ).grid(row=4, column=1, padx=4)

        name_label = tk.Label(export_window, text="File name:")
        name_label.grid(row=5, column=0, sticky="w")
        name_entry = ttk.Entry(export_window, textvariable=name_var, width=40)
        name_entry.grid(row=6, column=0, sticky="we")

        def _on_selection_changed(*_args):
            """Keep the default file name in sync; ALL writes one file per molecule."""
            selection = dropdown_var.get()
            if selection == "ALL":
                name_var.set("")
                name_entry.state(["disabled"])
                name_label.config(text="File name: (one file per molecule)")
            else:
                name_entry.state(["!disabled"])
                name_label.config(text="File name:")
                name_var.set(f"{selection}_spec_output.csv")

        dropdown_var.trace_add("write", _on_selection_changed)
        export_window.columnconfigure(0, weight=1)

        # Create a button in the new window
        button = ttk.Button(
            export_window,
            text="Generate CSV",
            command=lambda: generate_csv(
                molecules_data=self.islat.molecules_dict,
                mol_name=dropdown_var.get(),
                data_field=self.data_field,
                output_dir=dir_var.get().strip() or str(models_folder_path),
                wave_data=self.islat.wave_data_original,
                match_pixel_sampling=match_sampling_var.get(),
                file_name=name_var.get().strip() or None,
            ),
        )
        button.grid(row=1, column=1)

    def _browse_export_dir(self, parent, dir_var):
        """Prompt for the model-export destination folder."""
        # Drop topmost so the native dialog is not hidden behind the window
        parent.attributes("-topmost", False)
        try:
            chosen = filedialog.askdirectory(
                title="Select export folder",
                initialdir=dir_var.get() or str(models_folder_path),
                parent=parent,
            )
        finally:
            parent.attributes("-topmost", True)
        if chosen:
            dir_var.set(chosen)

    def toggle_atomic_lines(self):
        """
        Show atomic lines as vertical dashed lines on the plot.
        """
        try:
            # Let MainPlot flip the toggle and forward to the active view
            self.main_plot.toggle_atomic_lines()
            # Keep TopBar in sync for any legacy readers
            self.atomic_toggle = self.main_plot.atomic_toggle

        except Exception as e:
            self.data_field.insert_text(f"Error displaying atomic lines: {e}\n")
            traceback.print_exc()

    def toggle_summed_spectrum(self):
        """
        Toggle the display of the summed spectral flux on the plot.
        """
        try:
            self.main_plot.toggle_summed_spectrum()
        except Exception as e:
            self.data_field.insert_text(f"Error toggling summed spectrum: {e}\n")
            traceback.print_exc()

    def toggle_full_spectrum(self):
        """
        Toggle the display of the full spectrum on the plot.
        """
        try:
            self.main_plot.toggle_full_spectrum()
        except Exception as e:
            self.data_field.insert_text(f"Error toggling full spectrum: {e}\n")
            traceback.print_exc()

    def hitran_query(self):
        """
        Open the HITRAN molecule selector window.
        """
        try:
            # Use the root window from the islat class for the MoleculeSelector
            root_window = getattr(self.islat, 'root', self.master)
            MoleculeSelector(root_window, self.data_field, user_settings=self.islat.user_settings, islat=self.islat)
        except Exception as e:
            print(f"Error opening HITRAN query: {e}")
            if self.data_field:
                self.data_field.insert_text(f"Error opening HITRAN query: {e}", console_print=True)
    
    def spectra_browser(self):
        print("Open spectra browser")

    def manage_sample(self):
        """Open the sample manager window."""
        SampleManagerWindow.open(self.master, self.islat, self.theme)


    def default_molecules(self):
        self.islat.load_default_molecules()

    def add_molecule(self):
        self.islat.add_molecule_from_hitran()

    def sort_molecules(self):
        """Open the popout window for sorting molecules in the control panel."""
        SortMoleculesWindow.open(
            self.master, self.islat, self.control_panel, self.theme
        )

    def bulk_apply_properties(self):
        """Open the popout window for bulk-applying properties to molecules."""
        BulkApplyPropertiesWindow.open(
            self.master, self.islat, self.control_panel, self.theme
        )

    def save_parameters(self, file_path = None, auto = False):
        """
        Save current molecule parameters to CSV file.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Sure you want to save? This will overwrite any previous save for this spectrum file."
        )
        if confirmed == "no":
            return
        
        # Get the loaded spectrum name for filename
        spectrum_name = getattr(self.islat, 'loaded_spectrum_name', 'unknown')
        
        from iSLAT.Modules.FileHandling import molsave_file_name

        try:
            # Save the current molecule parameters
            saved_file = write_molecules_to_csv(
                self.islat.molecules_dict, 
                loaded_spectrum_name=spectrum_name,
                file_name=molsave_file_name
            )
                
            # Also save to the general molecules list for session persistence
            #write_molecules_list_csv(self.islat.molecules_dict, loaded_spectrum_name=spectrum_name)
            
            if saved_file:
                # Update the data field to show success message
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Molecule parameters saved to: {saved_file}',
                        clear_after=True
                    )
                print(f"Molecule parameters saved successfully to: {saved_file}")
            else:
                print("Failed to save molecule parameters")
                
        except Exception as e:
            print(f"Error saving parameters: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error saving parameters: {str(e)}',
                    clear_after=True
                )
    
    def load_parameters(self):
        """
        Load molecule parameters from CSV file for the current spectrum.
        Uses the iSLAT class's _load_spectrum_parameters method.
        """
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            "Are you sure you want to load parameters? Make sure to save any unsaved changes!"
        )
        if confirmed == "no":
            return
        
        # Show loading message
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
            self.islat.GUI.data_field.insert_text(
                'Loading saved parameters, this may take a moment...',
                clear_after=True
            )
        
        # Use the iSLAT class method to load parameters
        self.islat._load_spectrum_parameters()
        
        # Update GUI components
        if hasattr(self.islat, 'GUI'):
            if hasattr(self.islat.GUI, 'plot'):
                self.main_plot.update_all_plots()
            if hasattr(self.islat.GUI, 'control_panel'):
                self.islat.GUI.control_panel.refresh_from_molecules_dict()

    def load_parameters_from_file(self):
        """
        Load molecule parameters from a user-selected CSV file in the SAVES folder.
        Opens a file dialog to let the user pick any save file.
        """
        # Open file dialog starting in the SAVES folder
        selected_file = filedialog.askopenfilename(
            title="Select a save file to load",
            initialdir=str(save_folder_path),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not selected_file:
            return  # User cancelled
        
        # Display confirmation dialog
        confirmed = messagebox.askquestion(
            "Confirmation",
            f"Are you sure you want to load parameters from:\n{os.path.basename(selected_file)}?\n\nMake sure to save any unsaved changes!"
        )
        if confirmed == "no":
            return
        
        # Show loading message
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
            self.islat.GUI.data_field.insert_text(
                'Loading saved parameters, this may take a moment...',
                clear_after=True
            )
        
        try:
            # Split selected file into directory and filename
            file_dir = os.path.dirname(selected_file)
            file_name = os.path.basename(selected_file)
            
            # Read molecule data from the selected file
            mole_save_data = read_from_user_csv(
                file_path=file_dir,
                file_name=file_name,
                update_save_file_names=self.islat.user_settings.get(
                    "update_save_file_names_in_save_csv", False
                )
            )
            
            if mole_save_data is None:
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Failed to read save file: {file_name}',
                        clear_after=True
                    )
                return
            
            # Clear existing molecules and load from file
            self.islat.molecules_dict.clear()
            self.islat.init_molecules(mole_save_data)
            self.islat._molecules_loaded = True
            
            # Update GUI components
            if hasattr(self.islat, 'GUI'):
                if hasattr(self.islat.GUI, 'plot'):
                    self.main_plot.update_all_plots()
                if hasattr(self.islat.GUI, 'control_panel'):
                    self.islat.GUI.control_panel.refresh_from_molecules_dict()
                if hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Loaded parameters from: {file_name}',
                        clear_after=False
                    )
            
            print(f"Successfully loaded parameters from: {selected_file}")
            
        except Exception as e:
            print(f"Error loading parameters from file: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error loading parameters: {str(e)}',
                    clear_after=True
                )

    def import_parameters_from_file(self):
        """
        Import molecule parameters from a user-selected CSV file and add them
        to the existing molecules without clearing or replacing anything.
        Opens a file dialog to let the user pick any save file.
        """
        selected_file = filedialog.askopenfilename(
            title="Select a parameter file to import",
            initialdir=str(save_folder_path),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not selected_file:
            return  # User cancelled

        # Show loading message
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
            self.islat.GUI.data_field.insert_text(
                'Importing parameters, this may take a moment...',
                clear_after=True
            )

        try:
            file_dir = os.path.dirname(selected_file)
            file_name = os.path.basename(selected_file)

            # Read molecule data from the selected file
            mole_save_data = read_from_user_csv(
                file_path=file_dir,
                file_name=file_name,
                update_save_file_names=self.islat.user_settings.get(
                    "update_save_file_names_in_save_csv", False
                )
            )

            if mole_save_data is None:
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Failed to read file: {file_name}',
                        clear_after=True
                    )
                return

            # Filter out molecules already present so we never overwrite them
            existing_names = set(self.islat.molecules_dict.keys())
            new_mole_data = {
                name: data
                for name, data in mole_save_data.items()
                if name not in existing_names
            }

            if not new_mole_data:
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        'No new molecules to import (all already loaded).',
                        clear_after=False
                    )
                return

            # Add new molecules without clearing the existing ones
            self.islat.init_molecules(new_mole_data)

            # Update GUI components
            if hasattr(self.islat, 'GUI'):
                if hasattr(self.islat.GUI, 'plot'):
                    self.main_plot.update_all_plots()
                if hasattr(self.islat.GUI, 'control_panel'):
                    self.islat.GUI.control_panel.refresh_from_molecules_dict()
                if hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Imported {len(new_mole_data)} molecule(s) from: {file_name}',
                        clear_after=False
                    )

            print(f"Successfully imported {len(new_mole_data)} molecule(s) from: {selected_file}")

        except Exception as e:
            print(f"Error importing parameters from file: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error importing parameters: {str(e)}',
                    clear_after=True
                )

    def save_parameters_to_file(self):
        """
        Save molecule parameters to a user-selected CSV file in the SAVES folder.
        Opens a save dialog to let the user choose the destination filename.
        """
        # Open save dialog starting in the SAVES folder
        selected_file = filedialog.asksaveasfilename(
            title="Save parameters to file",
            initialdir=str(save_folder_path),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not selected_file:
            return  # User cancelled

        try:
            file_dir = os.path.dirname(selected_file)
            file_name = os.path.basename(selected_file)

            saved_file = write_molecules_to_csv(
                self.islat.molecules_dict,
                file_path=file_dir,
                file_name=file_name
            )

            if saved_file:
                if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                    self.islat.GUI.data_field.insert_text(
                        f'Molecule parameters saved to: {saved_file}',
                        clear_after=True
                    )
                print(f"Molecule parameters saved successfully to: {saved_file}")
            else:
                print("Failed to save molecule parameters")

        except Exception as e:
            print(f"Error saving parameters to file: {e}")
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'data_field'):
                self.islat.GUI.data_field.insert_text(
                    f'Error saving parameters: {str(e)}',
                    clear_after=True
                )

    def retreat_plot_start(self, extra_amount: Optional[float] = None):
        """Retreat the plot start by the current range value."""
        if self.control_panel:
            self.control_panel.retreat_plot_start(extra_amount=extra_amount)
    
    def advance_plot_start(self, extra_amount: Optional[float] = None):
        """Advance the plot start by the current range value."""
        if self.control_panel:
            self.control_panel.advance_plot_start(extra_amount=extra_amount)
    
    def toggle_legend(self):
        #print("Toggled legend on plot")
        self.main_plot.toggle_legend()

    def open_center_view_dialog(self) -> None:
        """Open the Center View dialog (same as the 'C' keybind)."""
        handler = getattr(self.main_plot, 'interaction_handler', None)
        if handler is not None and hasattr(handler, 'open_center_view_dialog'):
            handler.open_center_view_dialog()

    # ------------------------------------------------------------------
    # Main-plot options (Views dropdown)
    # ------------------------------------------------------------------
    def _sync_plot_option_vars(self) -> None:
        """Refresh the Views-menu check marks from the plot controller's state."""
        try:
            self._y_autoscale_var.set(bool(getattr(self.main_plot, 'y_autoscale_toggle', True)))
            self._y_model_var.set(getattr(self.main_plot, 'y_autoscale_source', 'data') == 'model')
            self._observed_var.set(bool(getattr(self.main_plot, 'observed_toggle', True)))
        except Exception:
            pass

    def open_y_limits_dialog(self) -> None:
        """Open a small dialog to pin the main plot's Y axis to specific values."""
        pm = self.main_plot
        try:
            cur_lo, cur_hi = pm.get_y_limits()
        except Exception:
            cur_lo, cur_hi = 0.0, 1.0

        win = tk.Toplevel(self)
        win.title("Main Plot \u2014 Y Limits")
        win.resizable(False, False)
        win.grab_set()
        pad = {"padx": 8, "pady": 4}

        tk.Label(win, text="Y min (Jy):").grid(row=0, column=0, sticky="w", **pad)
        lo_var = tk.StringVar(value=f"{cur_lo:.4g}")
        lo_entry = tk.Entry(win, textvariable=lo_var, width=12)
        lo_entry.grid(row=0, column=1, sticky="w", **pad)

        tk.Label(win, text="Y max (Jy):").grid(row=1, column=0, sticky="w", **pad)
        hi_var = tk.StringVar(value=f"{cur_hi:.4g}")
        tk.Entry(win, textvariable=hi_var, width=12).grid(row=1, column=1, sticky="w", **pad)

        status = tk.Label(win, text="", fg="red")
        status.grid(row=2, column=0, columnspan=2, sticky="w", **pad)

        def _apply(*_):
            try:
                lo = float(lo_var.get().strip())
                hi = float(hi_var.get().strip())
            except ValueError:
                status.config(text="Enter numeric values for both limits.")
                return
            if lo >= hi:
                status.config(text="Y min must be less than Y max.")
                return
            pm.set_y_limits(lo, hi)
            win.destroy()

        def _auto():
            pm.set_y_autoscale(True)
            win.destroy()

        btns = tk.Frame(win)
        btns.grid(row=3, column=0, columnspan=2, pady=8)
        ttk.Button(btns, text="Apply", command=_apply).pack(side="left", padx=4)
        ttk.Button(btns, text="Auto-scale", command=_auto).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=4)

        win.bind("<Return>", _apply)
        win.bind("<Escape>", lambda _e: win.destroy())
        lo_entry.focus_set()
