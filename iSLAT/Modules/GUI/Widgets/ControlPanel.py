#import traceback
import platform
import tkinter as tk
from tkinter import ttk, colorchooser, simpledialog, messagebox
from typing import Optional
import numpy as np
from iSLAT.Modules.DataTypes.Molecule import Molecule
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_control_panel_fields_config
from iSLAT.Modules.Debug import debug_config
from iSLAT.Modules.GUI.ControlSurface import ControlSurface
from ..GUIFunctions import create_wrapper_frame, create_scrollable_frame, ColorButton
#from .RegularFrame import RegularFrame
from ..Tooltips import CreateToolTip

# Platform-specific layout constants — Windows uses larger font metrics than macOS
_IS_WINDOWS = platform.system() == "Windows"

_ENTRY_LABEL_PADX = 3 if _IS_WINDOWS else 1
_ENTRY_FIELD_PADX = 3 if _IS_WINDOWS else 1
_MOL_BTN_WIDTH = 6 if _IS_WINDOWS else 4
_COLOR_VIS_SCROLL_WIDTH = 190 if _IS_WINDOWS else 160
_MOL_PARAM_SCROLL_WIDTH = 200 if _IS_WINDOWS else 170
_MATCH_BTN_PADX = 3 if _IS_WINDOWS else 1
# Column minimum pixel sizes for molecule visibility/color grid alignment
# Col 0 = drag handle, Col 1 = checkbox, Col 2 = name, Col 3 = delete, Col 4 = color
_VIS_COL_MINSIZES = (14, 30, 62, 30, 32) if _IS_WINDOWS else (12, 20, 42, 20, 22)

# ---------------------------------------------------------------------------
# Drag-and-drop reorder helpers (pure functions, unit-tested without Tk)
# ---------------------------------------------------------------------------
def drag_delta(src_idx: int, insert_after: int) -> int:
    """Row displacement implied by dropping row *src_idx* below row *insert_after*.

    *insert_after* is the index of the row the drop indicator sits beneath (``-1`` means above the first row), measured in the order *before* the move.
    Dropping a row onto its own slot, or onto the slot immediately above it, leaves it where it is.
    """
    new_idx = insert_after + 1 if insert_after < src_idx else insert_after
    return new_idx - src_idx


def shift_group_in_order(order, group, delta: int) -> list:
    """Move every name in *group* by *delta* rows within *order*.

    All members move by the same amount, so the group's shape (relative order and any gaps between members) is preserved and the remaining names fill the vacated slots in their existing order.
    *delta* is clamped so that no member leaves the list; a group already touching the edge it is pushed towards therefore does not move.
    A single-name group reduces to the usual remove-and-insert reorder.
    Returns a new list; *order* is not modified.
    """
    order = list(order)
    members = set(group)
    positions = [i for i, name in enumerate(order) if name in members]
    if not positions or delta == 0:
        return order
    n = len(order)
    delta = max(delta, -positions[0])
    delta = min(delta, n - 1 - positions[-1])
    if delta == 0:
        return order
    result = [None] * n
    for i in positions:
        result[i + delta] = order[i]
    others = iter(name for name in order if name not in members)
    for i in range(n):
        if result[i] is None:
            result[i] = next(others)
    return result


# ---------------------------------------------------------------------------
# ControlPanelSurface — ControlSurface implementation for the view-fields area
# ---------------------------------------------------------------------------
class ControlPanelSurface(ControlSurface):
    """Concrete :class:`ControlSurface` that renders :class:`ControlField` objects
    inside the ControlPanel's dynamic view-fields frame.

    Fields are laid out in a two-column grid:

    * Single-widget fields (e.g. :class:`ToggleField` checkbutton) span both
      columns.
    * Two-widget fields (label + control) place the label at column 0 and the
      control at column 1.
    * :class:`DisplayRangeField` produces no widgets but triggers
      ``on_display_range_changed`` so the ControlPanel can re-route its
      Plot Start / Plot Range inputs.

    Parameters
    ----------
    container:
        The ``ttk.Frame`` that acts as the widget parent (the existing
        ``_view_fields_frame``).
    on_display_range_changed:
        Optional callback invoked whenever the set of registered fields
        changes.  Receives the first :class:`DisplayRangeField` found (or
        ``None`` when none is registered).
    """
    def __init__(
        self,
        container,
        on_display_range_changed=None,
    ) -> None:
        super().__init__()
        self._container = container
        self._on_display_range_changed = on_display_range_changed

    # ------------------------------------------------------------------
    # Internal rebuild
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        from iSLAT.Modules.GUI.ControlField import DisplayRangeField, RenderContext

        # Tear down current widgets
        try:
            for child in self._container.winfo_children():
                child.destroy()
        except Exception:
            pass
        self._widget_refs.clear()

        display_range_field = None
        row = 0

        for field in self._fields.values():
            if isinstance(field, DisplayRangeField):
                display_range_field = field
                continue  # no visual widget

            try:
                widgets = field.build_widget(self._container, RenderContext.CONTROL_PANEL)
            except Exception:
                widgets = []

            self._widget_refs[field.key] = widgets

            if len(widgets) == 1:
                widgets[0].grid(
                    row=row, column=0, columnspan=2,
                    padx=_ENTRY_LABEL_PADX, pady=5, sticky="w",
                )
            elif len(widgets) >= 2:
                widgets[0].grid(row=row, column=0, padx=_ENTRY_LABEL_PADX, pady=5)
                widgets[1].grid(row=row, column=1, padx=_ENTRY_FIELD_PADX, sticky="w")

            if widgets:
                row += 1

        if self._on_display_range_changed is not None:
            try:
                self._on_display_range_changed(display_range_field)
            except Exception:
                pass

class ControlPanel(ttk.Frame):
    def __init__(self, master, islat, plot, data_field, font):
        super().__init__(master)
        
        self.master = master
        self.islat = islat
        self.plot = plot
        self.data_field = data_field
        self.font = font

        self.mol_dict = islat.molecules_dict
        self.updating = False

        self.mol_visibility = {}
        self.column_labels = {
            "": None,               # drag handle — no header text
            "On": "Turn on/off this\nmodel in the plot ",
            "Molecule": "Select active molecule",
            "Del.": "Remove this model\nfrom the GUI",
            "Color": "Change color\nfor this model"
            }
        
        self.label_frame = tk.LabelFrame(self, text="Spectrum and Models Control Panel", relief="solid", borderwidth=1)
        self.label_frame.grid(row=0, column=0, sticky="nsew", pady=0)
        self.label_frame.grid_rowconfigure(0,weight=1)

        temp_label = tk.Label(self)
        self.bg_color = temp_label.cget('bg')
        self.fg_color = temp_label.cget('fg')
        temp_label.destroy()
        fg_text = tk.Text()
        self.selected_color = "#007BFF"
        self.comparison_color = "#66B2FF"  # Lighter blue for comparison molecules
        self._shift_click_consumed = False  # Flag to prevent command firing on shift-click
        self._parameter_clipboard = None  # Stores {param: value} for Copy/Paste Parameters
        
        self.max_name_len = 4
        # Load field configurations from JSON file using iSLAT file handling
        self._load_field_configurations()

        # Initialize all UI components
        self._create_all_components()
        self._register_callbacks()

    def _create_all_components(self):
        """Create all control panel components in order"""
        gen_config_frame = self._create_general_config_frame()
        molecule_param_frame = self._create_molecule_param_frame()
        self.color_vis_frame = self._create_color_and_vis_frame()

        self._create_display_controls(gen_config_frame, 0, 0)
        self._create_wavelength_controls(gen_config_frame, 1, 0)  
        self._create_global_parameter_controls(gen_config_frame, 2, 0)  

        # Dynamic view-specific fields section (populated on view switch)
        # Row 5 — global params (start_row=2, row_offset=1) occupy rows 3-4
        self._view_fields_frame = ttk.Frame(gen_config_frame)
        self._view_fields_frame.grid(row=5, column=0, columnspan=4, sticky="nsew")
        self._view_field_entries = {}  # Kept for compatibility; superseded by _control_panel_surface
        self._current_display_range_binding = None

        # Surface that owns the view-fields area; registered on the ControlBus in Phase 6.
        self._control_panel_surface = ControlPanelSurface(
            self._view_fields_frame,
            on_display_range_changed=self._on_display_range_binding_changed,
        )

        self._create_molecule_specific_controls(molecule_param_frame, 0, 0)  # All other params here

        self._build_color_and_vis_controls(self.color_vis_frame) # my implementation

        self.grid_rowconfigure(1, weight=1)  # Because you placed the wrapper at row 1
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)

    def _create_general_config_frame(self):
        wrapper = create_wrapper_frame(self.label_frame, 1, 0, columnspan = 2)

        general_param_frame = ttk.Frame(wrapper)
        general_param_frame.grid(row=0, column=0, sticky="nsew")
        
        return general_param_frame
    
    def _create_molecule_param_frame(self):
       wrapper = create_wrapper_frame(self.label_frame, 0, 1)
       self._create_selected_frame(wrapper, 0, 0)
       molecule_param_frame = create_scrollable_frame(wrapper, height=250, width=_MOL_PARAM_SCROLL_WIDTH, horizontal=True, row=1, col=0)

       molecule_param_frame.rowconfigure(0, weight=1)
       molecule_param_frame.columnconfigure(0, weight=1)

       return molecule_param_frame

    def _create_color_and_vis_frame(self):
        wrapper = create_wrapper_frame(self.label_frame, 0, 0, sticky="nsew")

        color_vis_frame = create_scrollable_frame(wrapper, height=250, width=_COLOR_VIS_SCROLL_WIDTH, vertical=True)

        return color_vis_frame

    def _create_selected_frame(self, parent, row, col):
        selected_frame = tk.Frame(parent, bg = "darkgrey")
        selected_frame.grid(row=row, column = col, sticky="nsew")

        self.selected_label = tk.Label(selected_frame, background="darkgrey", anchor="center", justify="center")
        self.selected_label.grid(row=0, column=0, sticky="nsew")

        selected_frame.grid_rowconfigure(0, weight=1)
        selected_frame.grid_columnconfigure(0, weight=1)

    def _create_simple_entry(self, parent, label_text, initial_value, row, col, on_change_callback, width=7, param_name = None, tip_text = None):
        """Create a simple entry field with label and change callback"""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=col, padx=_ENTRY_LABEL_PADX, pady=5)

        if tip_text: 
            CreateToolTip(label, tip_text)
        
        var = tk.StringVar()
        value_str = initial_value
        if not isinstance(value_str, str):
            value_str = self._format_value(initial_value, param_name=param_name)
        
        var.set(value_str)
        
        entry = tk.Entry(
            parent, 
            textvariable=var, 
            width=width, # Turn back on to use the width from the json config
            justify="left", 
        )
        
        entry.grid(row=row, column=col + 1, padx=_ENTRY_FIELD_PADX, sticky="w")
        
        def on_change(*args):
            self.updating = True
            try:
                value = float(var.get())
                
                match param_name:
                    case "temp":
                        if value < 1:
                            self.data_field.insert_text("Cannot set temperature to less than 1")
                            return
                    case "n_mol":
                        if value == 0:
                            self.data_field.insert_text("Cannot set Column Density to 0")
                            return
                    case "radius":
                        if value <= 0: 
                            self.data_field.insert_text("Cannot set radius to 0 or negative")
                            return
                    
                on_change_callback(value)
                value_str = self._format_value(value, param_name)
                var.set(value_str)
                entry.configure(fg=self.fg_color, font=(self.font.cget("family"), self.font.cget("size"), "roman"))
            except ValueError as e:
                print(f"Error with new value: {e}")
                self.data_field.insert_text(f"Error with new value: {e}")
            finally:
                self.updating = False
        
        def on_write(*args):
            if self.updating: # Updating means that the entry variable is being updated from iSLAT and should not turn grey
                entry.configure(fg=self.fg_color, font=(self.font.cget("family"), self.font.cget("size"), "roman"))
                return
            try:
                new_entry = float(entry.get())
                old_entry = float(self._get_active_molecule_parameter_value(param_name))
            except (ValueError, TypeError): # If value is global parameter, turn grey
                entry.configure(fg="grey", font=(self.font.cget("family"), self.font.cget("size"), "italic"))
                return
            
            if new_entry == old_entry:
                entry.configure(fg=self.fg_color, font=(self.font.cget("family"), self.font.cget("size"), "roman"))
            else:
                entry.configure(fg="grey", font=(self.font.cget("family"), self.font.cget("size"), "italic"))

        
        entry.bind("<Return>", on_change)
        var.trace_add("write", on_write)
        
        return entry, var

    def _create_bound_parameter_entry(self, label_text, param_name, row, col, width=8):
        """Create an entry bound to a global parameter in molecules_dict"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return None, None
            
        molecules_dict = self.islat.molecules_dict
        current_value = getattr(molecules_dict, param_name, 0)
        
        def update_parameter(value_str):
            try:
                value = float(value_str)
                old_value = getattr(molecules_dict, param_name)
                if abs(old_value - value) > 1e-10:
                    setattr(molecules_dict, param_name, value)
            except (ValueError, AttributeError):
                pass
        
        return self._create_simple_entry(self, label_text, current_value, row, col, update_parameter, width, param_name=param_name)

    def _create_display_controls(self, parent, start_row, start_col):
        """Create plot start and range controls for display view"""
        plot_start_tip = "Start wavelength\nfor the upper plot\nunits: μm"
        plot_range_tip = "Wavelength range\nfor the upper plot\nunits: μm"

        # Plot start
        initial_start = getattr(self.islat, 'display_range', [4.5, 5.5])[0]
        self.plot_start_entry, self.plot_start_var = self._create_simple_entry(parent,
            "Plot start:", initial_start, start_row, start_col, lambda _: self._update_display_range(), param_name="display_range_start", tip_text=plot_start_tip)
        
        # Plot range
        display_range = getattr(self.islat, 'display_range', [4.5, 5.5])
        initial_range = round(display_range[1] - display_range[0], 2) # round to 2 decimal places
        self.plot_range_entry, self.plot_range_var = self._create_simple_entry(parent,
            "Plot range:", initial_range, start_row, start_col + 2, lambda _: self._update_display_range(), param_name="display_range_range", tip_text=plot_range_tip)

    def _create_wavelength_controls(self, parent, start_row, start_col):
        """Create wavelength range controls for model calculation range"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        min_wave_tip = "Minimum wavelength\nto calculate the models\nunits: μm"
        max_wave_tip = "Maximum wavelength\nto calculate the models\nunits: μm"
        molecules_dict = self.islat.molecules_dict
        min_wave, max_wave = molecules_dict.global_wavelength_range
        
        self.min_wavelength_entry, self.min_wavelength_var = self._create_simple_entry( parent,
            "Min. Wave:", min_wave, start_row, start_col, self._update_wavelength_range, tip_text=min_wave_tip)
        self.max_wavelength_entry, self.max_wavelength_var = self._create_simple_entry( parent,
            "Max. Wave:", max_wave, start_row, start_col + 2, self._update_wavelength_range, tip_text=max_wave_tip)

    def _create_global_parameter_controls(self, parent, start_row, start_col):
        """Create global parameter entry fields using MoleculeDict properties"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            label = tk.Label(parent, text="Global parameters not available")
            label.grid(row=start_row, column=start_col, columnspan=4, padx=5, pady=5)
            return

        # Store references for later updates
        self._global_parameter_entries = {}
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        
        for field_key, field_config in self.GLOBAL_FIELDS.items():
            # Calculate grid position (2 fields per row)
            row = start_row + row_offset + (col_offset // 2)
            col = start_col + (col_offset % 2) * 2
            
            entry, var = self._create_global_parameter_entry(
                parent,
                field_config['label'], 
                field_config['property'], 
                row, 
                col, 
                field_config['width'],
                tip_text=field_config['tip']
            )
            
            if entry and var:
                self._global_parameter_entries[field_config['property']] = (entry, var)
            
            # Add "Match Spectral Sampling" button next to model_pixel_res field
            if field_key == 'model_pixel_res':
                self._match_sampling_btn = ttk.Button(
                    parent, 
                    text="  Match Pix. Sampling", 
                    width=22,
                    command=self._toggle_match_spectral_sampling
                )
                self._match_sampling_btn.grid(row=row, column=start_col + 2, columnspan=2, padx=_MATCH_BTN_PADX, sticky="w")
                match_tip = ("Toggle matched spectral sampling.\n"
                             "When enabled, model flux is interpolated\n"
                             "pixel-by-pixel to match the spectrum's\n"
                             "wavelength grid") #(for data-model subtraction).\n"
                             #"MIRI and other spectra have uneven\n"
                             #"pixel sampling that varies with wavelength.")
                CreateToolTip(self._match_sampling_btn, match_tip)
            
            col_offset += 1

    def _build_color_and_vis_controls(self, parent):
        self.mol_frames = {}
        _row_pady = 3 if _IS_WINDOWS else 2

        # Header row — packed with fill="x" so it matches molecule row widths
        header_frame = tk.Frame(parent)
        header_frame.pack(fill="x")
        self._apply_vis_col_config(header_frame)

        for col, (label, tip_text) in enumerate(self.column_labels.items()):
            label_widget = tk.Label(header_frame, text=label)
            if tip_text:
                CreateToolTip(label_widget, tip_text)
            # Make the "On" label clickable to toggle all molecule visibility
            if label == "On":
                label_widget.config(cursor="hand2", fg="blue")
                label_widget.bind("<Button-1>", lambda e: self._toggle_all_molecule_visibility())
            label_widget.grid(row=0, column=col, sticky="ew", padx=3 if _IS_WINDOWS else 2)

        # Drag-and-drop state — reset on every rebuild
        self._drag_state = {
            'source': None,
            'target_idx': None,
            'indicator': None,
            'original_bgs': {},
        }

        # Molecule rows — each packed with fill="x" for consistent width,
        # using identical column config so columns align across rows.
        for row, (mol_name, mol_obj) in enumerate(self.mol_dict.items()):
            mol_name = str(mol_name)
            current_mol = mol_obj

            mol_frame = tk.Frame(parent)
            self.mol_frames[mol_name] = mol_frame
            mol_frame.pack(fill="x", pady=_row_pady)
            self._apply_vis_col_config(mol_frame)

            # --- Col 0: drag handle ---
            drag_handle = tk.Label(
                mol_frame,
                text="⠿",
                cursor="fleur",
                fg="#888888",
            )
            drag_handle.grid(row=0, column=0, sticky="nsew", padx=1)
            CreateToolTip(drag_handle, "Drag to reorder\nSelected molecules move together")
            drag_handle.bind("<ButtonPress-1>",   lambda e, n=mol_name: self._on_drag_start(n, e))
            drag_handle.bind("<B1-Motion>",        lambda e: self._on_drag_motion(e))
            drag_handle.bind("<ButtonRelease-1>",  lambda e: self._on_drag_release(e))

            # --- Col 1: visibility checkbox ---
            visibility_var = tk.BooleanVar()
            visibility_checkbox = ttk.Checkbutton(
                mol_frame,
                variable=visibility_var,
                command=lambda name=mol_name: self._on_visibility_changed(name)
            )
            visibility_checkbox.grid(row=0, column=1, sticky="nsew", pady=_row_pady)
            if mol_name not in self.mol_visibility:
                self.mol_visibility[mol_name] = visibility_var

            from ..GUIFunctions import _MOLNAME_STYLE, _DELETE_STYLE, configure_all_button_styles
            if hasattr(self, 'theme') and self.theme:
                configure_all_button_styles(self.theme)

            # --- Col 2: molecule name button ---
            mol_btn = ttk.Button(
                mol_frame,
                text=mol_name,
                width=_MOL_BTN_WIDTH,
                style=_MOLNAME_STYLE,
                command=lambda name=mol_name: self._on_molecule_selected(mol_name=name),
            )
            mol_btn.grid(row=0, column=2, sticky="ew", pady=_row_pady)
            # Shift-click toggles the molecule as a comparison molecule
            mol_btn.bind(
                "<Shift-Button-1>",
                lambda e, name=mol_name: self._on_molecule_shift_clicked(mol_name=name),
            )
            # Right-click opens the context menu
            # Button-3 = right-click on Windows/Linux; Button-2 and Control-Button-1 for macOS
            for _seq in ("<Button-3>", "<Button-2>", "<Control-Button-1>"):
                mol_btn.bind(
                    _seq,
                    lambda e, name=mol_name: self._show_molecule_context_menu(name, e),
                )
                mol_frame.bind(
                    _seq,
                    lambda e, name=mol_name: self._show_molecule_context_menu(name, e),
                )
            if len(mol_name) > self.max_name_len:
                CreateToolTip(mol_btn, mol_name, bg=self.bg_color)

            # --- Col 3: delete button ---
            delete_btn = ttk.Button(
                mol_frame,
                text="X",
                width=1,
                style=_DELETE_STYLE,
                command=lambda name=mol_name, frame=mol_frame: self._delete_molecule(mol_name=name, frame=frame),
            )
            delete_btn.grid(row=0, column=3, pady=_row_pady, sticky="ns")

            # --- Col 4: color button ---
            color_button = ColorButton(
                mol_frame,
                color=getattr(current_mol, 'color', None) or "blue",
            )
            color_button.add_command(command=lambda btn=color_button, name=mol_name: self._on_color_button_clicked(name, btn))
            color_button.grid(row=0, column=4, sticky="nsew")

            is_visible = getattr(self.islat.molecules_dict[mol_name], 'is_visible', False)
            visibility_var.set(is_visible)

        self._update_active_molecule_changes()

    # ------------------------------------------------------------------
    # Drag-and-drop reordering
    # ------------------------------------------------------------------

    def _on_drag_start(self, mol_name: str, event) -> None:
        """Record which molecule row the user started dragging.

        When the dragged row is part of the current selection (active molecule plus shift-clicked comparison molecules) the whole selection is dragged as a group, so every member row is highlighted.
        """
        self._drag_state['source'] = mol_name
        self._drag_state['target_idx'] = None
        self._drag_state['original_bgs'] = {}
        for name in self._drag_group(mol_name):
            frame = self.mol_frames.get(name)
            if frame is None:
                continue
            self._drag_state['original_bgs'][name] = frame.cget('bg')
            self._set_row_bg(frame, '#c8e6ff')
        # Create the drop-indicator line (created fresh each drag)
        self._drag_state['indicator'] = tk.Frame(
            self.color_vis_frame, height=2, bg='#0078d4'
        )

    def _drag_group(self, source: str) -> list:
        """Names that move together when *source* is dragged, in panel order.

        The selection (see :meth:`_get_selected_molecule_names`) moves as a group when the dragged row belongs to it; otherwise only the dragged row moves.
        """
        selected = self._get_selected_molecule_names()
        if source in selected and len(selected) > 1:
            return [name for name in self.mol_dict.keys() if name in selected]
        return [source]

    @staticmethod
    def _set_row_bg(frame, bg) -> None:
        """Recolour a molecule row frame and its child widgets."""
        try:
            frame.configure(bg=bg)
            for child in frame.winfo_children():
                try:
                    child.configure(bg=bg)
                except Exception:
                    pass
        except Exception:
            pass

    def _on_drag_motion(self, event) -> None:
        """Update the drop-indicator line as the cursor moves."""
        if not self._drag_state['source']:
            return
        indicator = self._drag_state['indicator']
        parent = self.color_vis_frame
        mol_names = list(self.mol_dict.keys())

        # Convert cursor y to parent-relative coordinates
        y = event.y_root - parent.winfo_rooty()

        # Determine insertion index: after which molecule index to insert
        insert_after = -1  # -1 => before the first row
        for i, name in enumerate(mol_names):
            frame = self.mol_frames.get(name)
            if frame is None:
                continue
            mid = frame.winfo_y() + frame.winfo_height() // 2
            if y > mid:
                insert_after = i

        self._drag_state['target_idx'] = insert_after

        # Re-pack the indicator at the correct position
        if indicator:
            indicator.pack_forget()
            if insert_after < 0:
                # Before the very first molecule row
                first = mol_names[0] if mol_names else None
                if first and first in self.mol_frames:
                    indicator.pack(fill='x', before=self.mol_frames[first])
                else:
                    indicator.pack(fill='x')
            elif insert_after < len(mol_names) - 1:
                next_name = mol_names[insert_after + 1]
                if next_name in self.mol_frames:
                    indicator.pack(fill='x', before=self.mol_frames[next_name])
                else:
                    indicator.pack(fill='x')
            else:
                # After the last molecule row
                indicator.pack(fill='x')

    def _on_drag_release(self, event) -> None:
        """Finalise the drop: reorder mol_dict and rebuild the panel.

        The dragged row's displacement in rows (:func:`drag_delta`) is applied to every row in its drag group (:meth:`_drag_group`, :func:`shift_group_in_order`), so a multi-molecule selection moves up or down as one block.
        """
        source = self._drag_state['source']
        if not source:
            return

        # Remove the indicator widget
        indicator = self._drag_state['indicator']
        if indicator:
            try:
                indicator.pack_forget()
                indicator.destroy()
            except Exception:
                pass

        # Restore the highlighted rows' backgrounds
        for name, original_bg in self._drag_state['original_bgs'].items():
            frame = self.mol_frames.get(name)
            if frame:
                self._set_row_bg(frame, original_bg or self.bg_color)

        target_idx = self._drag_state['target_idx']
        # Reset state before any early return
        self._drag_state = {'source': None, 'target_idx': None, 'indicator': None, 'original_bgs': {}}

        if target_idx is None:
            return

        mol_names = list(self.mol_dict.keys())
        if source not in mol_names:
            return

        delta = drag_delta(mol_names.index(source), target_idx)
        new_order = shift_group_in_order(mol_names, self._drag_group(source), delta)
        if new_order == mol_names:
            return  # dropped in place, or the group is already against the edge

        self.mol_dict.reorder(new_order)
        self._rebuild_color_and_vis_controls()

    def _apply_vis_col_config(self, frame):
        """Apply consistent column sizing for aligned visibility/color control columns."""
        for col, minsize in enumerate(_VIS_COL_MINSIZES):
            # col 2 is the molecule-name column — let it stretch
            frame.grid_columnconfigure(col, weight=(1 if col == 2 else 0), minsize=minsize)

    def _create_global_parameter_entry(self, parent, label_text, property_name, row, col, width=12, tip_text = None):
        """Create an entry bound to a global parameter in molecules_dict"""
        def update_global_parameter(value_str):
            if value_str in ["N/A", ""]:
                return
                
            if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
                return
                
            molecules_dict = self.islat.molecules_dict
            
            try:
                field_config = None
                for field_key, config in self.GLOBAL_FIELDS.items():
                    if config['property'] == property_name:
                        field_config = config
                        break
                
                if field_config:
                    value = field_config['datatype'](value_str)
                else:
                    value = float(value_str)
                    
                old_value = getattr(molecules_dict, property_name, None)
                
                if field_config and field_config['datatype'] == float:
                    try:
                        old_float_value = float(old_value) if old_value is not None else None
                    except (ValueError, TypeError):
                        old_float_value = None
                    
                    if old_float_value is None or abs(old_float_value - value) > 1e-10:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(molecules_dict, property_name, value)
                else:
                    if old_value != value:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(molecules_dict, property_name, value)
                            
            except (ValueError, AttributeError) as e:
                print(f"Error updating global {property_name}: {e}")
                self.data_field.insert_text(f"Error updating global {property_name}: {e}")
        
        # Get current value from molecules_dict
        current_value = 0.0
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            current_value = getattr(self.islat.molecules_dict, property_name, 0.0)
        
        return self._create_simple_entry(parent, label_text, current_value, row, col, update_global_parameter, width, param_name=property_name, tip_text=tip_text)

    def _toggle_match_spectral_sampling(self):
        """Toggle matched spectral sampling mode.
        
        When enabled, each molecule's rest-frame model flux is individually
        interpolated onto the data wavelength grid (accounting for the global
        stellar RV) and then the resampled models are summed.  This is
        essential for accurate data-model subtraction, especially for MIRI
        and other spectra with uneven pixel sampling that varies with
        wavelength.
        """
        if not hasattr(self.islat, 'wave_data') or self.islat.wave_data is None:
            self.data_field.insert_text("No spectrum loaded.")
            return
        
        if not hasattr(self.islat, 'molecules_dict') or not self.islat.molecules_dict:
            self.data_field.insert_text("No molecules loaded.")
            return
        
        # Toggle the flag - this will trigger the callback that updates plots
        current_state = self.islat.molecules_dict.match_spectral_sampling
        new_state = not current_state
        self.islat.molecules_dict.match_spectral_sampling = new_state
        
        # Update button appearance to indicate state
        if hasattr(self, '_match_sampling_btn'):
            if new_state:
                self._match_sampling_btn.configure(text="✓ Match Pix. Sampling")
            else:
                self._match_sampling_btn.configure(text="  Match Pix. Sampling")
        
    def _create_molecule_specific_controls(self, parent, start_row, start_col):
        """Create controls for molecule-specific parameters that update with active molecule"""
        # Store references for later updates
        self._molecule_parameter_entries = {}

        parameters_frame = tk.Frame(parent)
        parameters_frame.grid(row=0, column=0, sticky="nsew")
        
        # Create fields based on the class-level dictionary
        row_offset = 1
        col_offset = 0
        col = 0
        row = start_row 
        for field_key, field_config in self.MOLECULE_FIELDS.items():
            widget_type = field_config.get('widget_type', 'entry')

            if widget_type == 'optionmenu':
                widget, var = self._create_molecule_optionmenu(
                    parameters_frame,
                    field_config['label'],
                    field_config['attribute'],
                    row,
                    col,
                    options=field_config.get('options', []),
                    option_labels=field_config.get('option_labels', {}),
                    tip_text=field_config.get('tip'),
                )
            else:
                widget, var = self._create_molecule_parameter_entry(
                    parameters_frame,
                    field_config['label'], 
                    field_config['attribute'], 
                    row, 
                    col, 
                    tip_text=field_config['tip']
                )
            
            if widget and var:
                self._molecule_parameter_entries[field_config['attribute']] = (widget, var)
            row += 1
            col_offset += 1

        # default_btn = ttk.Button(parameters_frame, text="default parameters", command= lambda: self._update_molecule_parameter_fields(default=True))
        # default_btn.grid(row=row, column=col, columnspan=2, sticky="s")
        # CreateToolTip(default_btn, "reset current molecule's\nparameters to default values")
        parameters_frame.rowconfigure(row, weight=1)
    
    def reset_parameters_to_default(self):
        pass

    def _delete_molecule(self, mol_name = None, frame = None):
        mol_name = mol_name
        active_mol = self._get_active_molecule_object().name

        # print(f"destroying {mol_name}")
        self.data_field.insert_text(f"Deleting {mol_name}", clear_after = True)

        if mol_name == active_mol:
            live_dict = getattr(self.islat, 'molecules_dict', {})
            new_active = self.islat.user_settings.get("default_active_molecule", "H2O")
            # If the preferred default isn't present, pick any other molecule
            if new_active not in live_dict or new_active == mol_name:
                new_active = next(
                    (k for k in live_dict if k != mol_name),
                    None,
                )
            if new_active:
                print(f"setting {new_active} as active molecule")
                self.data_field.insert_text(f"setting {new_active} as active molecule", clear_after=False)
                self._set_active_molecule(mol_name=new_active)

        if frame is None:
            frame = self.mol_frames.get(mol_name)
        if frame is not None:
            frame.destroy()

        self.mol_frames.pop(mol_name, None)
        self.mol_visibility.pop(mol_name, None)
        del self.islat.molecules_dict[mol_name]

        # Delegate visual cleanup and rebuild to the plot controller
        self.plot.on_molecule_deleted(mol_name)

    def _create_molecule_parameter_entry(self, parent, label_text, param_name, row, col, width=7, tip_text = None):
        """Create an entry bound to the active molecule's parameter"""
        
        def update_active_molecule_parameter(value_str):
            # Skip updates for special cases
            if value_str in ["N/A", ""]:
                return
                
            if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
                return
                
            # Get the active molecule object
            active_mol = None
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                    active_mol = self.islat.molecules_dict[self.islat.active_molecule]
                elif hasattr(self.islat.active_molecule, 'name'):
                    active_mol = self.islat.active_molecule
            
            if not active_mol:
                return
                
            try:
                # Get field configuration for proper data type handling
                field_config = None
                for field_key, config in self.MOLECULE_FIELDS.items():
                    if config['attribute'] == param_name:
                        field_config = config
                        break
                
                # Convert value based on field configuration
                if field_config:
                    value = field_config['datatype'](value_str)
                else:
                    # Fallback for unknown parameters
                    value = float(value_str) if param_name in ["distance", "stellar_rv", "fwhm", "broad"] else str(value_str)
                    
                # Get old value for comparison
                old_value = getattr(active_mol, param_name, None)
                
                # Only update if the value actually changed
                if field_config and field_config['datatype'] == float:
                    # Convert old_value to float for proper comparison
                    try:
                        old_float_value = float(old_value) if old_value is not None else None
                    except (ValueError, TypeError):
                        old_float_value = None
                    
                    if old_float_value is None or abs(old_float_value - value) > 1e-10:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(active_mol, param_name, value)
                else:
                    if old_value != value:
                        # Use property setters to ensure proper cache invalidation and notifications
                        setattr(active_mol, param_name, value)
                            
            except (ValueError, AttributeError) as e:
                print(f"Error updating {param_name}: {e}")
                # self.data_field.insert_text(f"Error updating {param_name}: {e}")
        
        # Get initial value from active molecule
        initial_value = self._get_active_molecule_parameter_value(param_name)
        
        return self._create_simple_entry(parent, label_text, initial_value, row, col, update_active_molecule_parameter, width, param_name=param_name, tip_text=tip_text)

    def _create_molecule_optionmenu(self, parent, label_text, param_name, row, col,
                                     options=None, option_labels=None, tip_text=None):
        """Create an OptionMenu bound to a string attribute of the active molecule.

        Parameters
        ----------
        parent : tk widget
            Parent widget.
        label_text : str
            Label shown to the left of the menu.
        param_name : str
            Name of the molecule attribute to read/write.
        row, col : int
            Grid position.
        options : list[str]
            Sequence of valid option keys.
        option_labels : dict[str, str]
            Optional mapping from option key to human-readable label.  When
            provided the menu shows the label but stores/reads the key.
        tip_text : str or None
            Tooltip text for the label.
        """
        options = options or []
        option_labels = option_labels or {}

        # Build display labels (fall back to the key itself)
        display_labels = [option_labels.get(k, k) for k in options]
        # Reverse map: display label → key
        label_to_key = {option_labels.get(k, k): k for k in options}

        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=col, padx=_ENTRY_LABEL_PADX, pady=5)
        if tip_text:
            CreateToolTip(label, tip_text)

        # Get current value
        initial_key = self._get_active_molecule_parameter_value(param_name)
        initial_display = option_labels.get(str(initial_key), str(initial_key))

        var = tk.StringVar(value=initial_display)

        def on_option_changed(*_args):
            if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
                return
            active_mol = None
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                    active_mol = self.islat.molecules_dict[self.islat.active_molecule]
                elif hasattr(self.islat.active_molecule, 'name'):
                    active_mol = self.islat.active_molecule
            if not active_mol:
                return
            selected_display = var.get()
            selected_key = label_to_key.get(selected_display, selected_display)
            try:
                old_value = getattr(active_mol, param_name, None)
                if old_value != selected_key:
                    setattr(active_mol, param_name, selected_key)
            except Exception as e:
                print(f"Error updating {param_name}: {e}")

        menu = tk.OptionMenu(parent, var, *display_labels, command=lambda _: on_option_changed())
        menu.config(width=max(len(d) for d in display_labels) if display_labels else 10)
        menu.grid(row=row, column=col + 1, padx=_ENTRY_FIELD_PADX, sticky="w")

        # Store reverse map on the var so _update_molecule_parameter_fields can sync it
        var._label_to_key = label_to_key
        var._key_to_label = option_labels

        return menu, var

    def _load_field_configurations(self):
        """Load field configurations from JSON file using iSLAT file handling"""
        try:
            config = load_control_panel_fields_config()
            self.GLOBAL_FIELDS = config.get('global_fields', {})
            self.MOLECULE_FIELDS = config.get('molecule_fields', {})
        except Exception as e:
            print(f"Error loading control panel field configurations: {e}")
            # self.data_field.insert_text(f"Error loading control panel field configurations: {e}")
            # Use fallback default configurations
            self.GLOBAL_FIELDS = {}
            self.MOLECULE_FIELDS = {}

    def _register_callbacks(self):
        """Register callbacks for UI synchronization only"""
        try:
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                self.islat.molecules_dict.add_active_molecule_change_callback(self._on_active_molecule_change)
            
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                self.islat.molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
            
            Molecule.add_molecule_parameter_change_callback(self._on_molecule_parameter_change)

            # Views register their own fields via ControlBus in activate()/deactivate()

        except Exception as e:
            print(f"ControlPanel: Error registering callbacks: {e}")

    def _on_active_molecule_change(self, old_molecule, new_molecule):
        """Handle active molecule changes from the iSLAT callback system"""
        self._update_molecule_parameter_fields()
        self._update_active_molecule_changes()

    def _on_molecule_parameter_change(self, molecule_name, parameter_name, old_value, new_value):
        """Handle molecule parameter changes to update UI fields"""
        # Only update UI if this is the active molecule
        if (hasattr(self.islat, 'active_molecule') and 
            isinstance(self.islat.active_molecule, str) and 
            self.islat.active_molecule == molecule_name):
            
            # Update the specific parameter field if it exists
            if hasattr(self, '_molecule_parameter_entries') and parameter_name in self._molecule_parameter_entries:
                entry, var = self._molecule_parameter_entries[parameter_name]
                new_value_str = self._get_active_molecule_parameter_value(parameter_name)
                if var.get() != new_value_str:
                    self._set_var(var, new_value)
            
            # Update color and visibility controls if needed
            if parameter_name in ['color', 'is_visible']:
                self._update_active_molecule_changes()

    def _on_global_parameter_change(self, parameter_name, old_value, new_value):
        """Handle global parameter changes to update UI fields"""
        # Update the specific global parameter field if it exists
        if hasattr(self, '_global_parameter_entries') and parameter_name in self._global_parameter_entries:
            entry, var = self._global_parameter_entries[parameter_name]
            if var.get() != str(new_value):
                self._set_var(var, str(new_value))

    def _get_active_molecule_parameter_value(self, param_name) -> str:
        """Get the current value of a parameter from the active molecule"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return ""
        
        # Get the active molecule object
        active_mol = None
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                active_mol = self.islat.molecules_dict[self.islat.active_molecule]
            elif hasattr(self.islat.active_molecule, 'name'):
                active_mol = self.islat.active_molecule
        
        if not active_mol:
            return ""
            
        value = getattr(active_mol, param_name, "")
        return self._format_value(value, param_name)
        
    def _format_value(self, value, param_name) -> str:
        if not param_name:
            return f"{value:.2f}"
        
        try:
            # Get field configuration for proper formatting
            field_config = None
            for field_key, config in self.GLOBAL_FIELDS.items():
                if config['property'] == param_name:
                    field_config = config
                    break

            if field_config is None:
                for field_key, config in self.MOLECULE_FIELDS.items():
                    if config['attribute'] == param_name:
                        field_config = config
                        break
            
            # Format value based on field configuration
            if field_config and isinstance(value, (int, float)):
                return field_config['format'].format(value)
            elif isinstance(value, str):
                return value
            return f"{value:.6g}"
        except Exception as e:
            print(f"Error with formatting: {e}")
            return ""

    def _ensure_molecule_color_initialized(self, mol_obj):
        """Ensure molecule has a color assigned, using MoleculeWindow logic"""
        if not hasattr(mol_obj, 'color') or mol_obj.color is None:
            default_colors = self.theme.get("default_molecule_colors", ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"])
            
            # Use molecule index for consistent coloring, similar to MoleculeWindow
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                molecules_list = list(self.islat.molecules_dict.keys())
                try:
                    mol_index = molecules_list.index(mol_obj.name)
                except (ValueError, AttributeError):
                    mol_index = 0
            else:
                mol_index = 0
                
            mol_obj.color = default_colors[mol_index % len(default_colors)]

    def _on_visibility_changed(self, mol_name):
        """Handle visibility checkbox changes for individual molecule plotting"""
        
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        
        # Get the toggled molecule object
        selected_mol = self.islat.molecules_dict[mol_name]
        if not selected_mol:
            return
            
        new_visibility = self.mol_visibility[mol_name].get()
        molecule_name = selected_mol.name

        # if molecule was toggled on, set as new active molecule (maybe make this a setting)
        if new_visibility:
            self._on_molecule_selected(molecule_name)
        
        # Simply toggle this molecule's visibility - don't affect other molecules
        self.islat.molecules_dict.bulk_set_visibility(new_visibility, [molecule_name])
        
        debug_config.trace("control_panel", f"Set {molecule_name} visibility to {new_visibility}, actual value: {getattr(selected_mol, 'is_visible', 'UNDEFINED')}")
        
        # Trigger selective plot refresh to show/hide the molecule
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot') and hasattr(self.islat.GUI.plot, 'on_molecule_visibility_changed'):
            self.islat.GUI.plot.on_molecule_visibility_changed(molecule_name, new_visibility)
            debug_config.trace("control_panel", f"Triggered selective plot refresh for visibility change")

    def _toggle_all_molecule_visibility(self):
        """Toggle the visibility of all molecules at once"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
        
        # Determine new visibility state: if any molecule is visible, turn all off; otherwise turn all on
        any_visible = any(mol.is_visible for mol in self.islat.molecules_dict.values())
        new_visibility = not any_visible
        
        # Get all molecule names
        all_molecule_names = list(self.islat.molecules_dict.keys())
        
        # Update visibility for all molecules
        self.islat.molecules_dict.bulk_set_visibility(new_visibility, all_molecule_names)
        
        # Update all visibility checkboxes in the UI
        for mol_name in all_molecule_names:
            if mol_name in self.mol_visibility:
                self.mol_visibility[mol_name].set(new_visibility)
        
        # Trigger plot refresh — the active view handles both modes
        if hasattr(self.islat, 'GUI') and hasattr(self.islat.GUI, 'plot'):
            plot = self.islat.GUI.plot
            plot.update_model_plot()
        
        #print(f"ControlPanel: Toggled all molecules to visibility={new_visibility}")

    def _on_color_button_clicked(self, mol_name, btn):
        """Handle color button clicks to open color chooser"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        # Get the active molecule object
        selected_mol = self.islat.molecules_dict[mol_name]
        if not selected_mol:
            return
            
        # Get molecule name for the color chooser title
        mol_name = getattr(selected_mol, 'displaylabel', getattr(selected_mol, 'name', 'Molecule'))
        old_color = getattr(selected_mol,'color', "Blue")
        
        # Open color chooser
        color_code = colorchooser.askcolor(title=f"Pick color for {mol_name}", color=old_color)[1]
        if color_code:
            btn.change_color(color_code)
            
            # Update molecule color
            selected_mol.color = color_code
            
            # Manually trigger the molecule parameter change notification
            # since color is not a property with a setter
            if hasattr(selected_mol, '_notify_my_parameter_change'):
                selected_mol._notify_my_parameter_change('color', old_color, color_code)

    def _get_active_molecule_object(self) -> Molecule:
        """Get the active molecule object, similar to MoleculeWindow logic"""
        if not hasattr(self.islat, 'active_molecule') or not self.islat.active_molecule:
            return None
            
        if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
            if isinstance(self.islat.active_molecule, str) and self.islat.active_molecule in self.islat.molecules_dict:
                return self.islat.molecules_dict[self.islat.active_molecule]
            elif hasattr(self.islat.active_molecule, 'name'):
                return self.islat.active_molecule
        
        return None

    def _update_active_molecule_changes(self):
        """Update color button and visibility checkbox based on active molecule"""
        active_mol = self._get_active_molecule_object()
        self.selected_name = active_mol.name

        # Reset all frames to default, then highlight active + comparison
        for name, frame in self.mol_frames.items():
            frame.config(bg=self.bg_color)
        self.mol_frames[active_mol.name].config(bg=self.selected_color)
        # Re-apply comparison highlights
        for comp_mol in self.islat.comparison_molecules:
            comp_name = getattr(comp_mol, 'name', str(comp_mol))
            if comp_name in self.mol_frames and comp_name != active_mol.name:
                self.mol_frames[comp_name].config(bg=self.comparison_color)

        if len(active_mol.name) > self.max_name_len + 4:
            self.selected_name = active_mol.name[:self.max_name_len] + "..."
            CreateToolTip(self.selected_label, active_mol.name, bg = self.bg_color)

        self.selected_label.config(text=f"Selected Molecule: {self.selected_name}\nThermal Broadening: {active_mol.thermal_broad:.2g} km/s")

    # ------------------------------------------------------------------
    def _on_display_range_binding_changed(self, field):
        """Called by ``ControlPanelSurface`` when the :class:`DisplayRangeField` changes.

        Keeps ``_current_display_range_binding`` in sync so that
        ``_update_display_range`` continues to route reads/writes correctly.
        """
        if field is not None:
            self._current_display_range_binding = {
                "getter": field.getter,
                "setter": field.setter,
            }
            self._sync_display_range_from_binding()
        else:
            self._current_display_range_binding = None
            self._restore_display_range_from_islat()

    def _sync_display_range_from_binding(self):
        """Read the current view's display-range binding and update fields."""
        binding = self._current_display_range_binding
        if binding is None:
            return
        try:
            start, end = binding["getter"]()
            range_val = end - start
            self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
            self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))
        except Exception:
            pass

    def _restore_display_range_from_islat(self):
        """Restore Plot Start / Plot Range fields from islat.display_range."""
        try:
            start, end = self.islat.display_range
            range_val = end - start
            self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
            self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))
        except Exception:
            pass

    def _update_display_range(self, value_str=None):
        """Update display range bidirectionally between GUI and iSLAT class.

        When a view-specific display-range binding is active, reads/writes
        are routed through the binding's getter/setter instead of
        ``islat.display_range``.
        """
        # If value_str is a tuple, it's being called from iSLAT to update GUI
        if isinstance(value_str, tuple) and len(value_str) == 2:
            try:
                # Update GUI fields from iSLAT display_range (iSLAT -> GUI)
                start, end = value_str
                range_val = end - start #round(end - start, 2)
                
                self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
                self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))

            except Exception as e:
                print(f"Error updating display range GUI from iSLAT: {e}")
        else:
            try:
                start = float(self.plot_start_var.get())
                range_val = float(self.plot_range_var.get())

                binding = getattr(self, '_current_display_range_binding', None)
                if binding is not None:
                    # Route through the active view's binding
                    binding["setter"](start, start + range_val)
                elif hasattr(self.islat, 'display_range'):
                    new_display_range = (start, start + range_val)
                    if not hasattr(self.islat, '_display_range') or self.islat._display_range != new_display_range:
                        self.islat.display_range = new_display_range
                        self.islat.GUI.plot.match_display_range(match_y = True)
            except (ValueError, AttributeError):
                pass

    def advance_plot_start(self, extra_amount: Optional[float] = None):
        """Add the current plot range to the plot start value."""
        try:
            start = float(self.plot_start_var.get())
            #range_val = float(self.plot_range_var.get())
            if extra_amount is None:
                extra_amount = float(self.plot_range_var.get())
            new_start = start + extra_amount
            self._set_var(self.plot_start_var, self._format_value(new_start, "display_range_start"))
            self._update_display_range()
        except (ValueError, AttributeError):
            pass

    def retreat_plot_start(self, extra_amount: Optional[float] = None):
        """Subtract the current plot range from the plot start value."""
        try:
            start = float(self.plot_start_var.get())
            #range_val = float(self.plot_range_var.get())
            if extra_amount is None:
                extra_amount = float(self.plot_range_var.get())
            new_start = start - extra_amount
            self._set_var(self.plot_start_var, self._format_value(new_start, "display_range_start"))
            self._update_display_range()
        except (ValueError, AttributeError):
            pass

    def _update_wavelength_range(self, value_str=None):
        """Update wavelength range for model calculations (not display)"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            min_val = float(self.min_wavelength_var.get())
            max_val = float(self.max_wavelength_var.get())
            
            if min_val < max_val:
                molecules_dict = self.islat.molecules_dict
                molecules_dict.global_wavelength_range = (min_val, max_val)
        except (ValueError, AttributeError):
            pass

    def _update_molecule_parameter(self, value_str=None):
        """Update molecule-specific parameter UI fields for the active molecule"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        try:
            # Get the active molecule name
            active_molecule_name = getattr(self.islat, 'active_molecule', None)
            if not active_molecule_name:
                return
            
            molecules_dict = self.islat.molecules_dict
            molecule_obj = molecules_dict.get(active_molecule_name, None)
            if not molecule_obj:
                return
            
            # Update each molecule-specific parameter UI field
            for param_name, (entry, var) in self._molecule_parameter_entries.items():
                if hasattr(molecule_obj, param_name):
                    value = getattr(molecule_obj, param_name)
                    self._set_var(var, str(value))
        
        except Exception as e:
            print(f"Error updating molecule parameter UI fields: {e}")

    def _on_molecule_selected(self, mol_name, event=None):
        """Handle molecule selection - uses iSLAT's active_molecule property"""
        # If shift-click was consumed, skip normal selection
        if self._shift_click_consumed:
            self._shift_click_consumed = False
            return

        old_active_mol = self._get_active_molecule_object().name
        
        try:
            self.mol_frames[old_active_mol].config(bg = self.bg_color)
        except KeyError:
            old_active_mol = self.islat.user_settings.get("default_active_molecule", "H2O")
            self.mol_frames[old_active_mol].config(bg = self.bg_color)

        # Normal click: clear comparison molecules and reset their frame colors
        for comp_mol in self.islat.comparison_molecules:
            comp_name = getattr(comp_mol, 'name', str(comp_mol))
            if comp_name in self.mol_frames:
                self.mol_frames[comp_name].config(bg=self.bg_color)
        self.islat.molecules_dict.clear_comparison_molecules()

        self._set_active_molecule(mol_name= mol_name)

        # Explicitly refresh fields and highlighting in case the callback
        # chain was not yet set up or fires asynchronously.
        self._update_molecule_parameter_fields()
        self._update_active_molecule_changes()

    def _on_molecule_shift_clicked(self, mol_name):
        """Handle shift-click on a molecule button.

        - Shift-click on a molecule **not** in the active list: adds it to the
          list and makes it the new primary active molecule (the old primary is
          preserved as a comparison molecule).
        - Shift-click on an existing **comparison** molecule: removes it from
          the list (toggle-off).
        - Shift-click on the current **primary** active molecule: no-op.
        """
        self._shift_click_consumed = True

        active_mol = self._get_active_molecule_object()
        if active_mol and active_mol.name == mol_name:
            return  # Already the primary — nothing to do

        # Check whether this molecule is already a comparison
        comp_names = {
            getattr(m, 'name', None)
            for m in self.islat.comparison_molecules
        }

        if mol_name in comp_names:
            # Toggle-off: remove from the comparison list
            self.islat.molecules_dict.toggle_comparison_molecule(mol_name)
            if mol_name in self.mol_frames:
                self.mol_frames[mol_name].config(bg=self.bg_color)
        else:
            # New molecule: promote to primary (old primary → comparison)
            # _update_active_molecule_changes fires via the active_molecule
            # callback and repaints all frame colors correctly.
            self.islat.molecules_dict.promote_to_active_molecule(mol_name)

    # ------------------------------------------------------------------
    # Right-click context menu — molecule rows
    # ------------------------------------------------------------------
    def _show_molecule_context_menu(self, mol_name: str, event) -> None:
        """Post a right-click context menu for a molecule row."""
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="Duplicate",
            command=lambda: self._duplicate_molecule_action(mol_name),
        )
        menu.add_command(
            label="Delete",
            command=lambda: self._delete_molecule_action(mol_name),
        )

        # "Merge Selected" — visible only when 2+ molecules are selected
        # (active molecule + at least one comparison molecule).
        selected_names = self._get_selected_molecule_names()
        if len(selected_names) >= 2:
            menu.add_separator()
            menu.add_command(
                label=f"Merge Selected ({', '.join(selected_names)})",
                command=lambda: self._merge_molecules_action(selected_names),
            )

        menu.add_separator()
        menu.add_command(
            label="Edit Name",
            command=lambda: self._edit_molecule_name(mol_name),
        )
        menu.add_command(
            label="Edit Label",
            command=lambda: self._edit_molecule_label(mol_name),
        )
        menu.add_command(
            label="Unbold on Plot" if self._is_molecule_bold(mol_name) else "Bold on Plot",
            command=lambda: self._toggle_molecule_bold_action(mol_name),
        )
        menu.add_command(
            label="Filter Line List",
            command=lambda: self._filter_line_list_action(mol_name),
        )
        menu.add_command(
            label="View Line List",
            command=lambda: self._view_line_list_action(mol_name),
        )
        menu.add_command(
            label="Change Line List…",
            command=lambda: self._change_line_list_action(mol_name),
        )
        menu.add_separator()
        menu.add_command(
            label="Copy Parameters",
            command=lambda: self._copy_molecule_parameters(mol_name),
        )
        paste_state = tk.NORMAL if self._parameter_clipboard is not None else tk.DISABLED
        menu.add_command(
            label="Paste Parameters",
            state=paste_state,
            command=lambda: self._paste_molecule_parameters(mol_name),
        )
        menu.add_separator()
        menu.add_command(
            label="Export Model",
            command=lambda: self._export_molecule_action(mol_name),
        )
        try:
            menu.tk_popup(event.x_root, event.y_root, 0)
        finally:
            menu.grab_release()

    def _is_molecule_bold(self, mol_name: str) -> bool:
        """Return True if *mol_name* is currently drawn bold on the plot."""
        mol_dict = getattr(self.islat, 'molecules_dict', None)
        if mol_dict is None or mol_name not in mol_dict:
            return False
        return bool(getattr(mol_dict[mol_name], 'is_bold', False))

    def _toggle_molecule_bold_action(self, mol_name: str) -> None:
        """Toggle the molecule's bold flag.

        Setting ``is_bold`` fires the molecule parameter-change callback, which
        makes the active plot view redraw the molecule with a heavier line.
        """
        mol_dict = getattr(self.islat, 'molecules_dict', None)
        if mol_dict is None or mol_name not in mol_dict:
            print(f"ControlPanel: cannot bold unknown molecule '{mol_name}'.")
            return
        mol_obj = mol_dict[mol_name]
        mol_obj.is_bold = not getattr(mol_obj, 'is_bold', False)

    def _delete_molecule_action(self, mol_name: str) -> None:
        """Wrapper: call self._delete_molecule and report any failure."""
        result = self._delete_molecule(mol_name)
        if result is None:
            print(f"ControlPanel: failed to delete '{mol_name}'.")

    def _duplicate_molecule_action(self, mol_name: str) -> None:
        """Wrapper: call iSLAT.duplicate_molecule and report any failure."""
        result = self.islat.duplicate_molecule(mol_name)
        if result is None:
            print(f"ControlPanel: failed to duplicate '{mol_name}'.")

    def _get_selected_molecule_names(self) -> list:
        """Return the names of all currently \'selected\' molecules.

        Selection = the active molecule plus any comparison molecules that
        have been shift-clicked in the control panel.
        """
        names: list = []
        active = self._get_active_molecule_object()
        if active is not None:
            names.append(active.name)
        for comp in self.islat.comparison_molecules:
            comp_name = getattr(comp, 'name', str(comp))
            if comp_name not in names:
                names.append(comp_name)
        return names

    def _merge_molecules_action(self, mol_names: list) -> None:
        """Merge the line lists of *mol_names* into a new molecule."""
        if len(mol_names) < 2:
            self.data_field.insert_text("Select at least 2 molecules to merge (active + shift-click).")
            return

        self.data_field.insert_text(f"Merging {mol_names}\u2026", clear_after=False)
        new_name = self.islat.molecules_dict.merge_molecules(mol_names)
        if new_name is None:
            self.data_field.insert_text(f"Merge failed for {mol_names}.")
            return

        self.data_field.insert_text(f"Created merged molecule '{new_name}'.", clear_after=True)
        # Rebuild the UI to show the new molecule row
        self._rebuild_color_and_vis_controls()
        #self.plot.on_molecule_added(new_name, self.islat.molecules_dict)
        self.plot.update_model_plot()

    def _export_molecule_action(self, mol_name: str) -> None:
        """Export the named molecule's model spectrum to a CSV file."""
        try:
            from iSLAT.Modules.FileHandling.iSLATFileHandling import generate_csv
            generate_csv(
                molecules_data=self.islat.molecules_dict,
                mol_name=mol_name,
                data_field=self.data_field,
                wave_data=getattr(self.islat, 'wave_data_original', None),
            )
        except Exception as e:
            print(f"ControlPanel: export failed for '{mol_name}' — {e}")

    def _view_line_list_action(self, mol_name: str) -> None:
        """Open a read-only popout table of the molecule's line list."""
        mol_obj = self.islat.molecules_dict.get(mol_name)
        if mol_obj is None:
            return
        try:
            ll = mol_obj.line_list
        except Exception:
            ll = None
        if ll is None:
            msg = f"No line list loaded for '{mol_name}'."
            if self.data_field is not None:
                self.data_field.insert_text(msg)
            else:
                print(f"ControlPanel: {msg}")
            return
        from iSLAT.Modules.GUI.Widgets.LineListViewWindow import LineListViewWindow
        LineListViewWindow(self, mol_obj, self.data_field)

    def _filter_line_list_action(self, mol_name: str) -> None:
        """Open the LineListFilterWindow for the named molecule."""
        mol_obj = self.islat.molecules_dict.get(mol_name)
        if mol_obj is None:
            return
        try:
            ll = mol_obj.line_list
        except Exception:
            ll = None
        if ll is None:
            msg = f"No line list loaded for '{mol_name}'."
            if self.data_field is not None:
                self.data_field.insert_text(msg)
            else:
                print(f"ControlPanel: {msg}")
            return
        from iSLAT.Modules.GUI.Widgets.LineListFilterWindow import LineListFilterWindow
        LineListFilterWindow.open(
            self, mol_obj, self.data_field, islat=self.islat,
            theme=getattr(getattr(self.islat, "GUI", None), "theme", None),
        )

    def _change_line_list_action(self, mol_name: str) -> None:
        """Open a file dialog to pick a new .par line-list file for *mol_name*."""
        from tkinter import filedialog
        import os
        mol_obj = self.islat.molecules_dict.get(mol_name)
        if mol_obj is None:
            return
        initial_dir = None
        if getattr(mol_obj, 'filepath', None):
            initial_dir = os.path.dirname(mol_obj.filepath)
        filepath = filedialog.askopenfilename(
            title=f"Select line list file for '{mol_name}'",
            parent=self,
            initialdir=initial_dir,
            filetypes=[("HITRAN par files", "*.par"), ("All files", "*.*")],
        )
        if not filepath:
            return

        # Delegate all internal reset logic to the molecule itself
        mol_obj.change_line_list(filepath)

        msg = f"Line list for '{mol_name}' changed to: {os.path.basename(filepath)}"
        if self.data_field is not None:
            self.data_field.insert_text(msg)
        else:
            print(f"ControlPanel: {msg}")

        # Redraw
        if hasattr(self, 'plot') and self.plot is not None:
            try:
                self.plot.update_model_plot()
            except Exception as e:
                print(f"ControlPanel: plot update failed after line list change — {e}")

        # ── Refresh control panel display if this was the active molecule ─────
        try:
            active_mol = self._get_active_molecule_object()
            if active_mol is not None and getattr(active_mol, 'name', None) == mol_name:
                self._update_active_molecule_changes()
        except Exception as e:
            print(f"ControlPanel: panel refresh failed after line list change — {e}")

    def _edit_molecule_name(self, mol_name: str) -> None:
        """Prompt the user to rename a molecule (changes the dict key)."""
        new_name = simpledialog.askstring(
            "Edit Name",
            "Enter new molecule name:",
            initialvalue=mol_name,
            parent=self,
        )
        if not new_name or new_name.strip() == mol_name:
            return
        mol_dict = self.islat.molecules_dict
        ok = mol_dict.rename_molecule(mol_name, new_name.strip())
        if not ok:
            messagebox.showerror(
                "Rename Failed",
                f"Could not rename '{mol_name}' to '{new_name.strip()}'.\n"
                "The name may already be in use or is invalid.",
                parent=self,
            )
            return
        # If this was the active molecule, update the reference by name
        if hasattr(self.islat, 'active_molecule'):
            am = self.islat.active_molecule
            if hasattr(am, 'name') and am.name == new_name.strip():
                self.islat.active_molecule = new_name.strip()
        try:
            self.plot.update_model_plot()
            self._rebuild_color_and_vis_controls()
        except Exception as e:
            print(f"_edit_molecule_name: refresh warning — {e}")

    def _edit_molecule_label(self, mol_name: str) -> None:
        """Prompt the user to change a molecule's display label."""
        mol_dict = self.islat.molecules_dict
        if mol_name not in mol_dict:
            return
        mol_obj = mol_dict[mol_name]
        new_label = simpledialog.askstring(
            "Edit Label",
            "Enter new display label (supports LaTeX, e.g. $H_2O$):",
            initialvalue=getattr(mol_obj, 'displaylabel', mol_name),
            parent=self,
        )
        if not new_label or new_label == getattr(mol_obj, 'displaylabel', mol_name):
            return
        mol_obj.displaylabel = new_label
        try:
            self.plot.update_model_plot()
        except Exception as e:
            print(f"_edit_molecule_label: plot refresh warning — {e}")

    def _copy_molecule_parameters(self, mol_name: str) -> None:
        """Copy physics parameters from *mol_name* to the in-memory clipboard."""
        mol_dict = getattr(self.islat, 'molecules_dict', {})
        if mol_name not in mol_dict:
            return
        self._parameter_clipboard = mol_dict[mol_name].copy_parameters()
        print(f"Copied parameters from '{mol_name}'.")

    def _paste_molecule_parameters(self, mol_name: str) -> None:
        """Apply the clipboard parameters to *mol_name* and refresh the UI."""
        if self._parameter_clipboard is None:
            return
        mol_dict = getattr(self.islat, 'molecules_dict', {})
        if mol_name not in mol_dict:
            return
        mol_obj = mol_dict[mol_name]
        mol_obj.bulk_update_parameters(self._parameter_clipboard)
        # Refresh the parameter entry fields if this is (or becomes) the active mol
        self._update_molecule_parameter_fields()
        try:
            self.plot.update_model_plot()
        except Exception as e:
            print(f"_paste_molecule_parameters: plot refresh warning — {e}")
        print(f"Pasted parameters to '{mol_name}'.")

    def _set_active_molecule(self, mol_name):
        """Set *mol_name* as the active molecule.

        Looks up the molecule directly in ``islat.molecules_dict`` so that
        molecules added after construction (e.g. via duplicate) are found
        correctly.
        """
        live_dict = getattr(self.islat, 'molecules_dict', None)
        if not live_dict:
            return
        try:
            if mol_name in live_dict:
                self.islat.active_molecule = mol_name
                return
            # Fallback: set the first available molecule
            first_mol = next(iter(live_dict.keys()), None)
            if first_mol:
                self.islat.active_molecule = first_mol
        except Exception as e:
            print(f"Error setting active molecule: {e}")

    def refresh_from_molecules_dict(self):
        """Refresh all fields from current molecules_dict values"""
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        molecules_dict = self.islat.molecules_dict
        
        # Re-register callbacks in case molecules_dict was created after ControlPanel
        try:
            molecules_dict.add_global_parameter_change_callback(self._on_global_parameter_change)
        except:
            pass  # Callback might already be registered
        
        # Update global parameter fields
        self._update_global_parameter_fields()
        
        # Update wavelength range fields
        if (hasattr(self, 'min_wavelength_var') and hasattr(self, 'max_wavelength_var') 
            and hasattr(molecules_dict, 'global_wavelength_range')):
            min_val, max_val = molecules_dict.global_wavelength_range
            self._set_var(self.min_wavelength_var, f'{min_val:.6g}')
            self._set_var(self.max_wavelength_var, f'{max_val:.6g}')

        # Update display range fields
        self._update_display_range_fields()
        
        # Update molecule-specific parameter fields
        self._update_molecule_parameter_fields()
        
        # Rebuild color and visibility controls for new molecules
        self._rebuild_color_and_vis_controls()

    def _rebuild_color_and_vis_controls(self):
        """Rebuild color and visibility controls when molecules are added/removed"""
        if hasattr(self, 'color_vis_frame'):
            # Destroy all children (header + molecule rows) to avoid stale widgets
            for child in self.color_vis_frame.winfo_children():
                child.destroy()
            self.mol_frames.clear()
            self.mol_visibility.clear()
            self._build_color_and_vis_controls(self.color_vis_frame)

    def cleanup(self):
        try:
            if hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict:
                self.islat.molecules_dict.remove_active_molecule_change_callback(self._on_active_molecule_change)
        except Exception as e:
            print(f"Error during ControlPanel cleanup: {e}")

    def _update_molecule_parameter_fields(self, default = False):
        """Update all molecule-specific parameter fields with values from the active molecule"""
        if not hasattr(self, '_molecule_parameter_entries'):
            return
        
        if default:
            print("resetting to defaults")
            for field_key, field_config in self.MOLECULE_FIELDS.items():
                param_name = field_config['attribute']
                if hasattr(self, '_molecule_parameter_entries') and param_name in self._molecule_parameter_entries:
                    entry, var = self._molecule_parameter_entries[param_name]
                    new_value = float(field_config['default'])
                    if float(var.get()) != new_value:
                        self._set_var(var, new_value)
            return
        else:
            for param_name, (entry, var) in self._molecule_parameter_entries.items():
                new_value = self._get_active_molecule_parameter_value(param_name)
                # OptionMenu vars store display labels; convert the raw key first
                if hasattr(var, '_key_to_label'):
                    new_value = var._key_to_label.get(str(new_value), str(new_value))
                current_value = var.get()
                if current_value != new_value:
                    self._set_var(var, new_value)

    def _update_global_parameter_fields(self):
        """Update all global parameter fields with values from the molecules_dict"""
        if not hasattr(self, '_global_parameter_entries'):
            return
        
        if not (hasattr(self.islat, 'molecules_dict') and self.islat.molecules_dict):
            return
            
        for property_name, (entry, var) in self._global_parameter_entries.items():
            try:
                new_value = getattr(self.islat.molecules_dict, property_name, 0.0)
                # Format the value properly using the field configuration
                formatted_value = self._format_value(new_value, property_name)
                current_value = var.get()
                if str(current_value) != str(formatted_value):
                    self._set_var(var, formatted_value)
                    # Reset entry styling to normal (not gray/italic)
                    entry.configure(fg=self.fg_color, font=(self.font.cget("family"), self.font.cget("size"), "roman"))
            except (AttributeError, TypeError):
                pass
        
        # Also sync the match spectral sampling button state
        if hasattr(self, '_match_sampling_btn'):
            try:
                is_matched = self.islat.molecules_dict.match_spectral_sampling
                if is_matched:
                    self._match_sampling_btn.configure(text="✓ Match Pix. Sampling")
                else:
                    self._match_sampling_btn.configure(text="  Match Pix. Sampling")
            except (AttributeError, TypeError):
                pass

    def _update_display_range_fields(self):
        """Update display range fields with values from the iSLAT display_range property"""
        if not (hasattr(self, 'plot_start_var') and hasattr(self, 'plot_range_var')):
            return
            
        if not hasattr(self.islat, 'display_range') or not self.islat.display_range:
            return
            
        try:
            start, end = self.islat.display_range
            range_val = end - start
            
            # Update only if values are different to avoid unnecessary updates
            current_start = self.plot_start_var.get()
            current_range = self.plot_range_var.get()
            
            if str(current_start) != str(start):
                self._set_var(self.plot_start_var, self._format_value(start, "display_range_start"))
            if str(current_range) != str(range_val):
                self._set_var(self.plot_range_var, self._format_value(range_val, "display_range_range"))
        except (ValueError, AttributeError, TypeError):
            pass

    def _set_var(self, var, value):
        self.updating = True
        var.set(value)
        self.updating = False