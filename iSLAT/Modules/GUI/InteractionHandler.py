import numpy as np
import time
from matplotlib.widgets import SpanSelector
from typing import Optional, Tuple, Callable, Dict, Any

class InteractionHandler:
    """Mouse/keyboard events - handles all user interactions with plots"""
    
    def __init__(self, plot_manager):
        self.plot_manager = plot_manager
        self.islat = plot_manager.islat
        
        # Plot references
        self.fig = plot_manager.fig
        self.ax1 = plot_manager.ax1  # Main spectrum plot
        self.ax2 = plot_manager.ax2  # Line inspection plot
        self.ax3 = plot_manager.ax3  # Population diagram
        self.canvas = plot_manager.canvas
        
        # Interaction state
        self.span_selector: Optional[SpanSelector] = None
        self.selected_range: Optional[Tuple[float, float]] = None
        self.mouse_pressed = False
        self.last_click_time = 0
        self.double_click_threshold = 0.5  # seconds
        
        # Callbacks
        self.selection_callbacks: Dict[str, Callable] = {}
        self.click_callbacks: Dict[str, Callable] = {}
        self.zoom_callbacks: Dict[str, Callable] = {}
        
        # Initialize interactions
        self._setup_interactions()
    
    def _setup_interactions(self):
        """Set up all mouse and keyboard interactions"""
        self._setup_span_selector()
        self._setup_mouse_events()
        self._setup_keyboard_events()
        self._setup_plot_navigation()
    
    def _setup_span_selector(self):
        """Set up the span selector for wavelength range selection"""
        self.span_selector = SpanSelector(
            self.ax1,
            self._on_span_select,
            direction='horizontal',
            useblit=True,
            props=dict(alpha=0.3, facecolor='lime'),
            interactive=True,
            drag_from_anywhere=True
        )
    
    def _setup_mouse_events(self):
        """Set up mouse event handlers"""
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.canvas.mpl_connect('pick_event', self._on_pick)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # Connect to draw events to catch navigation changes
        self.canvas.mpl_connect('draw_event', self._on_draw)
    
    def _setup_keyboard_events(self):
        """Set up keyboard event handlers"""
        # Matplotlib key events (work when canvas has focus)
        self.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.canvas.mpl_connect('key_release_event', self._on_key_release)
        
        # Also bind at tkinter level for global keybindings that work even when full spectrum canvas or other widgets have focus
        self._setup_tkinter_keybindings()
    
    def _setup_plot_navigation(self):
        """Set up plot navigation callbacks"""
        # Connect to axis limit changes
        self.ax1.callbacks.connect('xlim_changed', self._on_xlim_changed)
        self.ax1.callbacks.connect('ylim_changed', self._on_ylim_changed)
        
        # Store the last known xlim to detect changes during draw events
        self._last_xlim = self.ax1.get_xlim() if self.ax1 else None
    
    def _on_span_select(self, xmin: float, xmax: float):
        """Handle span selection on main plot"""
        if xmin == xmax:
            self.clear_current_selection()
            return
        
        # Ensure proper order
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        
        self.selected_range = (xmin, xmax)
        
        # Store selection in plot manager
        if hasattr(self.plot_manager, 'set_active_range'):
            self.plot_manager.set_active_range(xmin, xmax)
        
        # Trigger callbacks
        self._trigger_selection_callbacks('span_select', xmin, xmax)
        
        # Update displays
        self._update_population_diagram_highlights(xmin, xmax)
    
    def _on_mouse_press(self, event):
        """Handle mouse press events"""
        self.mouse_pressed = True
        
        if event.inaxes is None:
            return
        
        # Don't interfere with span selector on main plot
        #if event.inaxes == self.ax1 and hasattr(self, 'span_selector') and self.span_selector.active:
            # Let the span selector handle the event on main plot
        #    return
        
        # Check for double-click on other plots
        current_time = time.time()
        is_double_click = (current_time - self.last_click_time) < self.double_click_threshold
        self.last_click_time = current_time
        
        # Handle different types of clicks
        if event.button == 1:  # Left click
            if is_double_click and event.inaxes != self.ax1:
                self._on_double_click(event)
            elif not is_double_click:  # Single click
                self._on_single_click(event)
        elif event.button == 3:  # Right click
            self._on_right_click(event)
    
    def _on_mouse_release(self, event):
        """Handle mouse release events"""
        self.mouse_pressed = False
    
    def _on_mouse_move(self, event):
        """Handle mouse move events"""
        if event.inaxes is None:
            return
        
        # Update cursor information
        if hasattr(self.plot_manager, 'update_cursor_info'):
            self.plot_manager.update_cursor_info(event.xdata, event.ydata)
    
    def _on_single_click(self, event):
        """Handle single click events"""
        # Don't trigger click callbacks on main plot to avoid interfering with span selector
        if event.inaxes == self.ax2:
            # Click on line inspection plot
            self._trigger_click_callbacks('click', event)
        elif event.inaxes == self.ax3:
            # Click on population diagram
            self._trigger_click_callbacks('click', event)
        # Don't handle clicks on main plot (ax1) to let span selector work
    
    def _on_double_click(self, event):
        """Handle double click events"""
        pass
    
    def _on_right_click(self, event):
        """Delegate context-menu building to the active view."""
        try:
            canvas_widget = self.canvas.get_tk_widget()
        except Exception:
            return

        menu = self.plot_manager.active_view.build_context_menu(event, canvas_widget)
        if menu is None:
            return

        try:
            x_root = canvas_widget.winfo_rootx() + int(event.x)
            y_root = canvas_widget.winfo_rooty() + int(canvas_widget.winfo_height() - event.y)
            menu.tk_popup(x_root, y_root)
        except Exception:
            pass
        finally:
            menu.grab_release()

    def _on_pick(self, event):
        """Handle pick events (clicking on plot elements)"""
        artist = event.artist
        
        # Get the axes that contains this artist
        artist_axes = None
        if hasattr(artist, 'axes'):
            artist_axes = artist.axes
        
        # Handle scatter plot picking in population diagram
        if hasattr(artist, 'get_offsets') and artist_axes == self.ax3:
            # This is a scatter plot in the population diagram
            self._handle_scatter_pick(event)
        
        # Handle line picking
        elif hasattr(artist, '_islat_line_info'):
            line_info = artist._islat_line_info
            self._trigger_click_callbacks('line_picked', line_info)
    
    def _handle_scatter_pick(self, event):
        """Handle clicking on scatter points in population diagram"""
        # Get the indices of picked points
        indices = event.ind
        if indices is None or len(indices) == 0:
            return
        
        # Get the first picked point index
        idx = indices[0]
        
        # Get line information from the active molecule's intensity table
        if hasattr(self.plot_manager.islat, 'active_molecule') and self.plot_manager.islat.active_molecule:
            molecule = self.plot_manager.islat.active_molecule
            if hasattr(molecule, 'intensity') and hasattr(molecule.intensity, 'get_table'):
                line_table = molecule.intensity.get_table
                if idx < len(line_table):
                    # Get line data for the clicked point
                    line_data = line_table.iloc[idx]
                    
                    # Create line info dictionary
                    line_info = {
                        'wavelength': line_data['lam'],
                        'intensity': line_data['intens'],
                        'e_up': line_data['e_up'],
                        'e_low': line_data['e_low'],
                        'a_stein': line_data['a_stein'],
                        'g_up': line_data['g_up'],
                        'g_low': line_data['g_low'],
                        'up_lev': line_data.get('lev_up', 'N/A'),
                        'low_lev': line_data.get('lev_low', 'N/A'),
                        'tau': line_data.get('tau', 'N/A')
                    }
                    
                    # Display line information in data field
                    self._display_scatter_line_info(line_info)
    
    def _display_scatter_line_info(self, line_info):
        """Display information about clicked scatter point"""
        if hasattr(self.plot_manager.islat, 'GUI') and hasattr(self.plot_manager.islat.GUI, 'data_field'):
            data_field = self.plot_manager.islat.GUI.data_field
            
            # Format and display line information
            info_text = f"Selected Line Information:\n"
            info_text += f"Wavelength: {line_info['wavelength']:.6f} μm\n"
            info_text += f"Intensity: {line_info['intensity']:.3e} Jy\n"
            info_text += f"Upper Energy: {line_info['e_up']:.1f} K\n"
            info_text += f"Einstein A: {line_info['a_stein']:.3e} s⁻¹\n"
            info_text += f"Statistical Weight: {line_info['g_up']}\n"
            info_text += f"Upper Level: {line_info['up_lev']}\n"
            info_text += f"Lower Level: {line_info['low_lev']}\n"
            if line_info['tau'] != 'N/A':
                info_text += f"Optical Depth: {line_info['tau']:.3f}\n"
            
            data_field.insert_text(info_text, console_print=True, clear_after=False)
    
    def _on_scroll(self, event):
        """Handle scroll events for zooming"""        
        pass
    
    def _on_key_press(self, event):
        """Handle key press events"""
        if event.key == 'h':
            # Toggle grid
            self._toggle_grid()
        # Note: 'f', 'l', 's', etc. are handled by tkinter keybindings
        # (_on_tk_keypress) — don't duplicate here to avoid double-triggering.
    
    def _on_key_release(self, event):
        """Handle key release events"""
        pass
    
    def _on_xlim_changed(self, ax):
        """Handle x-axis limit changes"""
        new_xlim = ax.get_xlim()
        xone = new_xlim[0]
        xtwo = new_xlim[1]
        
        # Update display range in iSLAT (only if changed to prevent infinite loops)
        if hasattr(self.islat, 'display_range'):
            current_range = self.islat.display_range
            new_range = (xone, xtwo)
            # Only update if the values are actually different (with small tolerance for floating point)
            if (not current_range or 
                abs(current_range[0] - new_range[0]) > 1e-10 or 
                abs(current_range[1] - new_range[1]) > 1e-10):
                self.islat.display_range = new_range

        # Trigger zoom callbacks
        self._trigger_zoom_callbacks('xlim_changed', new_xlim)
    
    def _on_ylim_changed(self, ax):
        """Handle y-axis limit changes"""
        new_ylim = ax.get_ylim()
        self._trigger_zoom_callbacks('ylim_changed', new_ylim)
    
    def _on_draw(self, event):
        """Handle draw events to catch navigation changes that don't trigger axis callbacks"""
        # Check if xlim has changed since last draw
        current_xlim = self.ax1.get_xlim()
        
        if self._last_xlim is None:
            # First time, just store the current xlim
            self._last_xlim = current_xlim
            return
            
        # Check if xlim has actually changed (with small tolerance for floating point)
        if (abs(current_xlim[0] - self._last_xlim[0]) > 1e-10 or 
            abs(current_xlim[1] - self._last_xlim[1]) > 1e-10):
            
            # xlim has changed, update display range
            self._last_xlim = current_xlim
            self._on_xlim_changed(self.ax1)
    
    def _update_population_diagram_highlights(self, xmin: float, xmax: float):
        """Update population diagram to highlight lines in selected range"""
        if hasattr(self.plot_manager, 'active_line_range'):
            self.plot_manager.active_line_range = (xmin, xmax)
        
        # Re-render population diagram with highlights
        if (hasattr(self.plot_manager, 'renderer') and 
            hasattr(self.islat, 'active_molecule') and 
            self.islat.active_molecule):
            self.plot_manager.renderer.render_population_diagram(
                self.islat.active_molecule, wave_range=(xmin, xmax)
            )
    
    def _auto_zoom_to_point(self, x: float, y: float, ax=None):
        """Auto-zoom to a point on the specified plot"""
        if x is None or y is None:
            return
        
        if ax is None:
            ax = self.ax1
        
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate new zoom range (zoom in by factor of 4)
        x_range = (current_xlim[1] - current_xlim[0]) / 4
        y_range = (current_ylim[1] - current_ylim[0]) / 4
        
        new_xlim = (x - x_range/2, x + x_range/2)
        new_ylim = (y - y_range/2, y + y_range/2)
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def _zoom_around_point(self, x: float, y: float, zoom_factor: float, ax):
        """Zoom around a specific point"""
        if x is None or y is None:
            return
        
        current_xlim = ax.get_xlim()
        current_ylim = ax.get_ylim()
        
        # Calculate new limits
        x_range = (current_xlim[1] - current_xlim[0]) * zoom_factor
        y_range = (current_ylim[1] - current_ylim[0]) * zoom_factor
        
        new_xlim = (x - x_range/2, x + x_range/2)
        new_ylim = (y - y_range/2, y + y_range/2)
        
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        self.canvas.draw_idle()
    
    def _reset_zoom(self):
        """Reset zoom to show all data"""
        if hasattr(self.islat, 'wave_data') and hasattr(self.islat, 'flux_data'):
            if self.islat.wave_data is not None and len(self.islat.wave_data) > 0:
                self.ax1.set_xlim(self.islat.wave_data.min(), self.islat.wave_data.max())
                
                if self.islat.flux_data is not None and len(self.islat.flux_data) > 0:
                    flux_min = np.nanmin(self.islat.flux_data)
                    flux_max = np.nanmax(self.islat.flux_data)
                    margin = (flux_max - flux_min) * 0.1
                    self.ax1.set_ylim(flux_min - margin, flux_max + margin)
                
                self.canvas.draw_idle()
    
    def _toggle_grid(self):
        """Toggle grid on/off"""
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.grid(not ax.get_grid)
        self.canvas.draw_idle()
    
    def _setup_tkinter_keybindings(self):
        """Set up tkinter-level keybindings that work globally"""
        # Get the root window - try multiple sources
        root = None
        
        # First try: parent_frame's toplevel (most reliable during init)
        if hasattr(self.plot_manager, 'parent_frame') and self.plot_manager.parent_frame:
            try:
                root = self.plot_manager.parent_frame.winfo_toplevel()
            except Exception:
                pass
        
        # Second try: GUI.master
        if root is None and hasattr(self.islat, 'GUI') and self.islat.GUI:
            if hasattr(self.islat.GUI, 'master'):
                root = self.islat.GUI.master
        
        if root is None:
            print("Warning: Could not set up tkinter keybindings - no root window found")
            return
        
        # Store root reference for later
        self._tk_root = root
        
        # debug print
        #print(f"[DEBUG] Setting up tkinter keybindings on root: {root}")
        
        # Use bind_all for global keybindings that work regardless of focus
        # Use KeyPress event with keysym check for more reliable handling
        root.bind_all('<KeyPress>', self._on_tk_keypress)
    
    def _on_tk_keypress(self, event):
        """Handle all key press events from tkinter"""
        import platform
        
        keysym = event.keysym.lower()
        
        # Check if focus is on an entry widget - don't interfere with typing
        # event.widget may be a string path if the widget was destroyed or during dialogs
        try:
            widget_class = event.widget.winfo_class()
        except AttributeError:
            return
        if widget_class in ('Entry', 'Text', 'TEntry', 'TCombobox'):
            return  # Don't consume event, let typing work
        
        # Handle 'f' key for full spectrum
        if keysym == 'f':
            # Check if Ctrl or Command modifier is pressed
            # On Windows/Linux: Control is state bit 2 (0x4)
            # On Mac: Command is state bit 3 (0x8)
            # Shift is state bit 0 (0x1)
            ctrl_pressed = False
            shift_pressed = bool(event.state & 0x1)
            if platform.system() == "Darwin":
                ctrl_pressed = bool(event.state & 0x8) or bool(event.state & 0x4)
            else:
                ctrl_pressed = bool(event.state & 0x4)
            
            if ctrl_pressed and shift_pressed:
                # Ctrl+Shift+F / Cmd+Shift+F - output full spectrum (save to file)
                #print("[DEBUG] Ctrl+Shift+F pressed, outputting full spectrum")
                self._output_full_spectrum()
            elif ctrl_pressed:
                # Ctrl+F / Cmd+F - open full spectrum window
                #print("[DEBUG] Ctrl+F pressed, opening full spectrum window")
                self._open_full_spectrum_window()
            else:
                # Just 'f' - toggle full spectrum mode
                #print(f"[DEBUG] 'f' key pressed alone, toggling full spectrum.")
                self._toggle_full_spectrum()
            
            return 'break'  # Prevent event from propagating
        
        # Handle arrow keys for molecule selection
        elif keysym == 'up':
            self._select_previous_molecule()
            return 'break'
        elif keysym == 'down':
            self._select_next_molecule()
            return 'break'
        
        # Handle Ctrl+S for save parameters, plain 's' for toggle saved lines
        elif keysym == 's':
            ctrl_pressed = False
            shift_pressed = bool(event.state & 0x1)
            if platform.system() == "Darwin":
                ctrl_pressed = bool(event.state & 0x8) or bool(event.state & 0x4)
            else:
                ctrl_pressed = bool(event.state & 0x4)
            
            if ctrl_pressed and shift_pressed:
                self._save_parameters_to_file()
            elif ctrl_pressed:
                self._save_parameters()
            else:
                self._toggle_saved_lines()
            return 'break'
        
        # Handle Ctrl+Shift+L for load parameters from file, Ctrl+L for load parameters, plain 'l' for toggle legend
        elif keysym == 'l':
            ctrl_pressed = False
            shift_pressed = bool(event.state & 0x1)
            if platform.system() == "Darwin":
                ctrl_pressed = bool(event.state & 0x8) or bool(event.state & 0x4)
            else:
                ctrl_pressed = bool(event.state & 0x4)
            
            if ctrl_pressed and shift_pressed:
                self._load_parameters_from_file()
            elif ctrl_pressed:
                self._load_parameters()
            else:
                self._toggle_legend()
            return 'break'
        
        # Handle left/right arrow keys for cycling through sample spectra
        elif keysym == 'left':
            self._cycle_spectrum_previous()
            return 'break'
        elif keysym == 'right':
            self._cycle_spectrum_next()
            return 'break'
        
        # Handle 'a' key for toggling atomic lines
        elif keysym == 'a':
            self._toggle_atomic_lines()
            return 'break'
        
        # Handle 'm' key for toggling summed spectrum
        elif keysym == 'm':
            self._toggle_summed_spectrum()
            return 'break'

        # Handle 'v' key for toggling molecule visibility
        # Shift+V = toggle ALL molecules, plain v = toggle active molecule
        elif keysym == 'v':
            shift_pressed = bool(event.state & 0x1)
            if shift_pressed:
                self._toggle_all_molecule_visibility()
            else:
                self._toggle_active_molecule_visibility()
            return 'break'

        # Handle 'r' key for toggling residual sub-panels
        elif keysym == 'r':
            self._toggle_residuals()
            return 'break'

        # Handle 't' key for toggling three panel view
        elif keysym == 't':
            self._toggle_three_panel()
            return 'break'

        # Handle 'c' key for opening the center-view dialog
        elif keysym == 'c':
            self._open_center_view_dialog()
            return 'break'

        # Handle 'n' key for advancing plot start by the current range
        # Shift+N = retreat (subtract range), plain N = advance (add range)
        #elif keysym == 'n':
        #    shift_pressed = bool(event.state & 0x1)
        #    if shift_pressed:
        #        self._retreat_plot_start()
        #    else:
        #        self._advance_plot_start()
        #    return 'break'
        
        #elif keysym == ',' or keysym == '<':
        elif keysym in [',', '<', 'comma', 'less']: # needed for mac for some reason
            shift_pressed = bool(event.state & 0x1)
            if shift_pressed:
                self._retreat_plot_start(extra_amount=float(self.islat.GUI.control_panel.plot_range_var.get())/2)
            else:
                self._retreat_plot_start()
            return 'break'
            #self._retreat_plot_start()
            #return 'break'

        elif keysym in ['.', '>', 'period', 'greater']:
            shift_pressed = bool(event.state & 0x1)
            if shift_pressed:
                self._advance_plot_start(extra_amount=float(self.islat.GUI.control_panel.plot_range_var.get())/2)
            else:
                self._advance_plot_start()
            return 'break'
            #self._advance_plot_start()
            #return 'break'

    def _cycle_spectrum_previous(self):
        """Switch to the previous spectrum in the sample list."""
        if hasattr(self.islat, 'sample_spectra') and self.islat.sample_spectra:
            self.islat.cycle_spectrum(-1)
            if hasattr(self.islat, 'GUI') and self.islat.GUI and hasattr(self.islat.GUI, 'file_interaction_pane'):
                self.islat.GUI.file_interaction_pane.update_sample_spectra_label()

    def _cycle_spectrum_next(self):
        """Switch to the next spectrum in the sample list."""
        if hasattr(self.islat, 'sample_spectra') and self.islat.sample_spectra:
            self.islat.cycle_spectrum(1)
            if hasattr(self.islat, 'GUI') and self.islat.GUI and hasattr(self.islat.GUI, 'file_interaction_pane'):
                self.islat.GUI.file_interaction_pane.update_sample_spectra_label()

    def _select_next_molecule(self):
        """Select the next molecule in the list"""
        if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
            return
        
        molecules = list(self.islat.molecules_dict.keys())
        if not molecules:
            return
        
        # Get current active molecule
        current = None
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            if hasattr(self.islat.active_molecule, 'name'):
                current = self.islat.active_molecule.name
            else:
                current = str(self.islat.active_molecule)
        
        # Find current index and get next
        try:
            current_idx = molecules.index(current)
            next_idx = (current_idx + 1) % len(molecules)
        except (ValueError, TypeError):
            next_idx = 0
        
        next_mol = molecules[next_idx]
        self._set_molecule_via_control_panel(next_mol)
    
    def _select_previous_molecule(self):
        """Select the previous molecule in the list"""
        if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
            return
        
        molecules = list(self.islat.molecules_dict.keys())
        if not molecules:
            return
        
        # Get current active molecule
        current = None
        if hasattr(self.islat, 'active_molecule') and self.islat.active_molecule:
            if hasattr(self.islat.active_molecule, 'name'):
                current = self.islat.active_molecule.name
            else:
                current = str(self.islat.active_molecule)
        
        # Find current index and get previous
        try:
            current_idx = molecules.index(current)
            prev_idx = (current_idx - 1) % len(molecules)
        except (ValueError, TypeError):
            prev_idx = len(molecules) - 1
        
        prev_mol = molecules[prev_idx]
        self._set_molecule_via_control_panel(prev_mol)
    
    def _set_molecule_via_control_panel(self, mol_name):
        """Set the active molecule through the control panel to update UI properly"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            control_panel = self.islat.GUI.control_panel
            if hasattr(control_panel, '_on_molecule_selected'):
                control_panel._on_molecule_selected(mol_name)
        else:
            # Fallback: set directly on islat
            self.islat.active_molecule = mol_name

    def _toggle_full_spectrum(self):
        """Toggle full spectrum mode on the main plot"""
        if hasattr(self.plot_manager, 'toggle_full_spectrum'):
            self.plot_manager.toggle_full_spectrum()

    def _toggle_three_panel(self):
        """Toggle three panel view on the main plot"""
        if hasattr(self.plot_manager, 'toggle_three_panel'):
            self.plot_manager.toggle_three_panel()
    
    def _toggle_summed_spectrum(self):
        """Toggle summed spectrum visibility on the main plot"""
        if hasattr(self.plot_manager, 'toggle_summed_spectrum'):
            self.plot_manager.toggle_summed_spectrum()

    def _toggle_residuals(self):
        """Toggle residual sub-panels in the full spectrum view."""
        if hasattr(self.plot_manager, 'toggle_residuals'):
            self.plot_manager.toggle_residuals()

    def _get_active_plot_range(self) -> Tuple[float, float]:
        """Return the current active (xmin, xmax) of the active plot view.

        Views that bind the Plot Start / Plot Range controls (full spectrum,
        dedicated line inspection) report their own window through
        ``_get_display_range``; otherwise the main-plot axes are used.
        """
        xmin, xmax = None, None
        view = getattr(self.plot_manager, 'active_view', None)
        getter = getattr(view, '_get_display_range', None)
        if callable(getter):
            try:
                lo, hi = getter()
                if lo is not None and hi is not None and float(hi) != float(lo):
                    xmin, xmax = float(lo), float(hi)
            except Exception:
                xmin, xmax = None, None
        if xmin is None or xmax is None:
            try:
                xmin, xmax = self.ax1.get_xlim()
            except Exception:
                xmin, xmax = None, None
        if xmin is None or xmax is None:
            if hasattr(self.islat, 'display_range') and self.islat.display_range:
                xmin, xmax = self.islat.display_range
            else:
                xmin, xmax = 0.0, 0.0
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        return float(xmin), float(xmax)

    def _apply_plot_range(self, start: float, end: float) -> None:
        """Show exactly [*start*, *end*] on the active plot view.

        Routed through the control panel's Plot Start / Plot Range inputs so
        every view (including the full-spectrum and line-inspection range
        bindings) stays in sync.
        """
        start, end = float(start), float(end)
        if start > end:
            start, end = end, start
        plot_range = end - start

        control_panel = None
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            control_panel = self.islat.GUI.control_panel

        if control_panel is not None:
            control_panel._set_var(
                control_panel.plot_start_var,
                control_panel._format_value(start, "display_range_start"),
            )
            control_panel._set_var(
                control_panel.plot_range_var,
                control_panel._format_value(plot_range, "display_range_range"),
            )
            control_panel._update_display_range()
        elif hasattr(self.islat, 'display_range'):
            self.islat.display_range = (start, end)
            if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
                self.islat.GUI.plot.match_display_range(match_y=True)

    def _apply_center_view(self, center: float, plot_range: float) -> None:
        """Center the plot view on *center* with the given *plot_range* width."""
        plot_range = abs(plot_range)
        self._apply_plot_range(center - plot_range / 2.0, center + plot_range / 2.0)

    # ------------------------------------------------------------------
    # Center-view helpers (data-point based windows)
    # ------------------------------------------------------------------
    def _sorted_wave_data(self) -> Optional[np.ndarray]:
        """Return the displayed wavelength grid sorted ascending, or ``None``."""
        wave = getattr(self.islat, 'wave_data', None)
        if wave is None or len(wave) == 0:
            return None
        wave = np.asarray(wave, dtype=float)
        wave = wave[np.isfinite(wave)]
        if wave.size == 0:
            return None
        return np.sort(wave)

    def _count_points_in_range(self, xmin: float, xmax: float) -> int:
        """Number of data points whose wavelength lies inside [*xmin*, *xmax*]."""
        wave = self._sorted_wave_data()
        if wave is None:
            return 0
        return int(np.count_nonzero((wave >= xmin) & (wave <= xmax)))

    def _range_from_center_points(
        self, center: float, n_points: int,
    ) -> Optional[Tuple[float, float]]:
        """Return the wavelength window spanning *n_points* data points around *center*.

        The point nearest *center* sits in the middle of the window; the
        window is clamped at the ends of the spectrum (so fewer points are
        covered there).  Returns ``None`` when there is no data or the
        window would be degenerate.
        """
        wave = self._sorted_wave_data()
        if wave is None:
            return None
        n_points = int(n_points)
        if n_points < 2:
            return None
        n = wave.size
        idx = int(np.argmin(np.abs(wave - float(center))))
        lo = idx - n_points // 2
        hi = lo + n_points - 1
        if lo < 0:
            hi = min(n - 1, hi - lo)
            lo = 0
        if hi > n - 1:
            lo = max(0, lo - (hi - (n - 1)))
            hi = n - 1
        start, end = float(wave[lo]), float(wave[hi])
        if end <= start:
            return None
        return start, end

    def open_center_view_dialog(self) -> None:
        """Open a popup to reposition the plot window.

        Three ways to describe the window are offered:

        * **Center + range** - a center wavelength and a width in µm.
        * **Center + data points** - a center wavelength and how many data
          points the window should span.
        * **Exact range** - explicit start and end wavelengths.

        Defaults come from the active view's current window.
        """
        import tkinter as tk
        from tkinter import ttk

        # Resolve the parent window
        root = getattr(self, '_tk_root', None)
        if root is None:
            if hasattr(self.plot_manager, 'parent_frame') and self.plot_manager.parent_frame:
                try:
                    root = self.plot_manager.parent_frame.winfo_toplevel()
                except Exception:
                    root = None
        if root is None and hasattr(self.islat, 'GUI') and getattr(self.islat.GUI, 'master', None):
            root = self.islat.GUI.master
        if root is None:
            return

        xmin, xmax = self._get_active_plot_range()
        current_center = (xmin + xmax) / 2.0
        current_range = xmax - xmin
        current_points = self._count_points_in_range(xmin, xmax)

        win = tk.Toplevel(root)
        win.title("Center View")
        win.resizable(False, False)
        win.transient(root)

        content = ttk.Frame(win)
        content.pack(fill="both", expand=True, padx=12, pady=12)

        mode_var = tk.StringVar(value="range")
        center_var = tk.StringVar(value=f"{current_center:.4f}")
        range_var = tk.StringVar(value=f"{current_range:.4f}")
        points_var = tk.StringVar(value=str(max(current_points, 2)))
        start_var = tk.StringVar(value=f"{xmin:.4f}")
        end_var = tk.StringVar(value=f"{xmax:.4f}")

        row = 0
        ttk.Label(content, text="Center wavelength (\u03bcm):").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6)
        )
        center_entry = ttk.Entry(content, textvariable=center_var, width=14)
        center_entry.grid(row=row, column=1, sticky="ew", pady=(0, 6))

        row += 1
        ttk.Radiobutton(content, text="Plot range (\u03bcm):", variable=mode_var,
                        value="range").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        range_entry = ttk.Entry(content, textvariable=range_var, width=14)
        range_entry.grid(row=row, column=1, sticky="ew", pady=(0, 6))

        row += 1
        ttk.Radiobutton(content, text="Data points:", variable=mode_var,
                        value="points").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        points_entry = ttk.Entry(content, textvariable=points_var, width=14)
        points_entry.grid(row=row, column=1, sticky="ew", pady=(0, 6))

        row += 1
        ttk.Separator(content, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(2, 8)
        )

        row += 1
        ttk.Radiobutton(content, text="Exact range (\u03bcm):", variable=mode_var,
                        value="exact").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=(0, 6))
        exact_frame = ttk.Frame(content)
        exact_frame.grid(row=row, column=1, sticky="ew", pady=(0, 6))
        start_entry = ttk.Entry(exact_frame, textvariable=start_var, width=8)
        start_entry.pack(side="left")
        ttk.Label(exact_frame, text=" to ").pack(side="left")
        end_entry = ttk.Entry(exact_frame, textvariable=end_var, width=8)
        end_entry.pack(side="left")

        row += 1
        error_var = tk.StringVar(value="")
        ttk.Label(content, textvariable=error_var, foreground="red").grid(
            row=row, column=0, columnspan=2, sticky="w", pady=(0, 6)
        )

        entries_by_mode = {
            "range": (center_entry, range_entry),
            "points": (center_entry, points_entry),
            "exact": (start_entry, end_entry),
        }
        all_entries = (center_entry, range_entry, points_entry, start_entry, end_entry)

        def _sync_entry_states(*_):
            active = entries_by_mode.get(mode_var.get(), ())
            for entry in all_entries:
                entry.configure(state="normal" if entry in active else "disabled")

        mode_var.trace_add("write", _sync_entry_states)
        _sync_entry_states()

        # Clicking into a field selects the matching mode.
        for mode, entries in entries_by_mode.items():
            for entry in entries:
                if entry is center_entry:
                    continue
                entry.bind("<FocusIn>", lambda _e, m=mode: mode_var.set(m), add="+")

        def _resolve_window() -> Optional[Tuple[float, float]]:
            """Translate the dialog state into (start, end), or set an error."""
            mode = mode_var.get()
            if mode == "exact":
                try:
                    start = float(start_var.get())
                    end = float(end_var.get())
                except ValueError:
                    error_var.set("Invalid start or end wavelength.")
                    return None
                if start == end:
                    error_var.set("Start and end must differ.")
                    return None
                return (start, end) if start < end else (end, start)

            try:
                center = float(center_var.get())
            except ValueError:
                error_var.set("Invalid center wavelength.")
                return None
            if mode == "points":
                try:
                    n_points = int(float(points_var.get()))
                except ValueError:
                    error_var.set("Invalid number of data points.")
                    return None
                if n_points < 2:
                    error_var.set("Use at least 2 data points.")
                    return None
                window = self._range_from_center_points(center, n_points)
                if window is None:
                    error_var.set("No spectrum data available for that window.")
                return window
            try:
                plot_range = float(range_var.get())
            except ValueError:
                error_var.set("Invalid plot range.")
                return None
            if plot_range <= 0:
                error_var.set("Plot range must be greater than 0.")
                return None
            return (center - plot_range / 2.0, center + plot_range / 2.0)

        def _apply(event=None):
            window = _resolve_window()
            if window is None:
                return
            self._apply_plot_range(*window)
            win.destroy()

        def _cancel(event=None):
            win.destroy()

        row += 1
        btn_frame = ttk.Frame(content)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="e")
        ttk.Button(btn_frame, text="Apply", command=_apply).pack(
            side="right", padx=(4, 0)
        )
        ttk.Button(btn_frame, text="Cancel", command=_cancel).pack(
            side="right", padx=4
        )

        win.bind("<Return>", _apply)
        win.bind("<Escape>", _cancel)

        # Center the dialog on screen and give the center entry focus.
        win.update_idletasks()
        x = win.winfo_screenwidth() // 2 - win.winfo_width() // 2
        y = win.winfo_screenheight() // 2 - win.winfo_height() // 2
        win.geometry(f"+{x}+{y}")
        center_entry.focus_set()
        center_entry.select_range(0, "end")
        try:
            win.grab_set()
        except Exception:
            pass

    # Backward-compatible private name (used by the 'c' key handler).
    _open_center_view_dialog = open_center_view_dialog

    def _advance_plot_start(self, extra_amount: Optional[float] = None):
        """Advance the plot start by the current plot range value."""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            self.islat.GUI.control_panel.advance_plot_start(extra_amount=extra_amount)

    def _retreat_plot_start(self, extra_amount: Optional[float] = None):
        """Retreat the plot start by the current plot range value."""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            self.islat.GUI.control_panel.retreat_plot_start(extra_amount=extra_amount)

    def _toggle_atomic_lines(self):
        """Toggle atomic lines visibility"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'toggle_atomic_lines'):
                self.islat.GUI.top_bar.toggle_atomic_lines()
    
    def _toggle_saved_lines(self):
        """Toggle saved lines visibility"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'toggle_saved_lines'):
                self.islat.GUI.top_bar.toggle_saved_lines()
    
    def _toggle_legend(self):
        """Toggle legend visibility on the main plot"""
        if hasattr(self.plot_manager, 'toggle_legend'):
            self.plot_manager.toggle_legend()

    def _toggle_active_molecule_visibility(self):
        """Toggle visibility of the currently active molecule."""
        if not hasattr(self.islat, 'active_molecule') or self.islat.active_molecule is None:
            return
        mol = self.islat.active_molecule
        mol_name = getattr(mol, 'name', None)
        if mol_name is None or mol_name not in self.islat.molecules_dict:
            return

        new_vis = not mol.is_visible

        # Update the model
        self.islat.molecules_dict.bulk_set_visibility(new_vis, [mol_name])

        # Keep the ControlPanel checkbox in sync
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            cp = self.islat.GUI.control_panel
            if hasattr(cp, 'mol_visibility') and mol_name in cp.mol_visibility:
                cp.mol_visibility[mol_name].set(new_vis)

        # Trigger the plot update
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
            self.islat.GUI.plot.on_molecule_visibility_changed(mol_name, new_vis)
    
    def _toggle_all_molecule_visibility(self):
        """Toggle visibility of ALL molecules (Shift+V).
        
        If any molecule is visible, hides all. Otherwise shows all.
        """
        if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
            return

        any_visible = any(mol.is_visible for mol in self.islat.molecules_dict.values())
        new_visibility = not any_visible
        all_names = list(self.islat.molecules_dict.keys())

        # Update the model
        self.islat.molecules_dict.bulk_set_visibility(new_visibility, all_names)

        # Keep all ControlPanel checkboxes in sync
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'control_panel'):
            cp = self.islat.GUI.control_panel
            if hasattr(cp, 'mol_visibility'):
                for mol_name in all_names:
                    if mol_name in cp.mol_visibility:
                        cp.mol_visibility[mol_name].set(new_visibility)

        # Trigger full plot update
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
            self.islat.GUI.plot.update_model_plot()

    def _open_full_spectrum_window(self):
        """Open a separate full spectrum window"""
        from iSLAT.Modules.GUI.FullSpectrumWindow import FullSpectrumWindow
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'master'):
            FullSpectrumWindow(self.islat.GUI.master, self.islat)
        else:
            print("[DEBUG] Could not open FullSpectrumWindow - GUI.master not found")
    
    def _output_full_spectrum(self):
        """Output full spectrum to file (same as menu command)"""
        from iSLAT.Modules.Plotting.FullSpectrumView import output_full_spectrum
        output_full_spectrum(self.islat)
    
    def _save_parameters(self):
        """Save parameters (same as menu command)"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'save_parameters'):
                self.islat.GUI.top_bar.save_parameters()
    
    def _load_parameters(self):
        """Load parameters (same as menu command)"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'load_parameters'):
                self.islat.GUI.top_bar.load_parameters()
    
    def _load_parameters_from_file(self):
        """Load parameters from a user-selected file in the SAVES folder"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'load_parameters_from_file'):
                self.islat.GUI.top_bar.load_parameters_from_file()

    def _save_parameters_to_file(self):
        """Save parameters to a user-selected file in the SAVES folder"""
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'top_bar'):
            if hasattr(self.islat.GUI.top_bar, 'save_parameters_to_file'):
                self.islat.GUI.top_bar.save_parameters_to_file()
    
    # Callback management
    def add_selection_callback(self, name: str, callback: Callable):
        """Add a callback for selection events"""
        self.selection_callbacks[name] = callback
    
    def remove_selection_callback(self, name: str):
        """Remove a selection callback"""
        self.selection_callbacks.pop(name, None)
    
    def add_click_callback(self, name: str, callback: Callable):
        """Add a callback for click events"""
        self.click_callbacks[name] = callback
    
    def remove_click_callback(self, name: str):
        """Remove a click callback"""
        self.click_callbacks.pop(name, None)
    
    def add_zoom_callback(self, name: str, callback: Callable):
        """Add a callback for zoom events"""
        self.zoom_callbacks[name] = callback
    
    def remove_zoom_callback(self, name: str):
        """Remove a zoom callback"""
        self.zoom_callbacks.pop(name, None)
    
    def _trigger_selection_callbacks(self, event_type: str, *args):
        """Trigger all selection callbacks"""
        for callback in self.selection_callbacks.values():
            try:
                # For span_select, only pass the coordinates, not the event_type
                if event_type == 'span_select' and len(args) >= 2:
                    callback(args[0], args[1])  # xmin, xmax
                else:
                    callback(*args)
            except Exception as e:
                print(f"Error in selection callback: {e}")
    
    def _trigger_click_callbacks(self, event_type: str, *args):
        """Trigger all click callbacks"""
        for callback in self.click_callbacks.values():
            try:
                # For click events, typically pass the event object or coordinates
                if event_type == 'click' and len(args) == 1:
                    callback(args[0])  # event object
                else:
                    callback(*args)
            except Exception as e:
                print(f"Error in click callback: {e}")
    
    def _trigger_zoom_callbacks(self, event_type: str, *args):
        """Trigger all zoom callbacks"""
        for callback in self.zoom_callbacks.values():
            try:
                callback(event_type, *args)
            except Exception as e:
                print(f"Error in zoom callback: {e}")
    
    # Public interface
    def enable_interactions(self):
        """Enable all interactions"""
        if self.span_selector:
            self.span_selector.set_active(True)
    
    def disable_interactions(self):
        """Disable all interactions"""
        if self.span_selector:
            self.span_selector.set_active(False)
    
    def get_current_selection(self) -> Optional[Tuple[float, float]]:
        """Get the current wavelength selection"""
        return self.selected_range
    
    def set_selection(self, xmin: float, xmax: float):
        """Programmatically set a selection"""
        self.selected_range = (xmin, xmax)
        self._on_span_select(xmin, xmax)
    
    def clear_current_selection(self):
        """Clear the current selection"""
        self.selected_range = None
        if hasattr(self.plot_manager, 'clear_selection'):
            self.plot_manager.clear_selection()
    
    def get_interaction_info(self) -> Dict[str, Any]:
        """Get information about current interaction state"""
        return {
            'selected_range': self.selected_range,
            'mouse_pressed': self.mouse_pressed,
            'span_selector_active': self.span_selector.active if self.span_selector else False,
            'num_selection_callbacks': len(self.selection_callbacks),
            'num_click_callbacks': len(self.click_callbacks),
            'num_zoom_callbacks': len(self.zoom_callbacks)
        }
    
    # Additional callback methods expected by MainPlot
    def set_span_select_callback(self, callback: Callable):
        """Set callback for span selection events"""
        self.add_selection_callback('span_select', callback)
    
    def set_click_callback(self, callback: Callable):
        """Set callback for click events"""
        self.add_click_callback('click', callback)
    
    def create_span_selector(self, ax, color):
        """Create a span selector - compatibility method"""
        return self.span_selector
    
    def handle_click_event(self, event):
        """Handle click events - compatibility method"""
        self._on_mouse_press(event)