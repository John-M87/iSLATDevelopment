import tkinter as tk
from tkinter import filedialog, ttk, font, simpledialog, messagebox
import os
import platform
import threading
import queue

from iSLAT.Modules.Debug.PerformanceLogger import get_performance_summary
from iSLAT.Modules.FileHandling.iSLATFileHandling import write_molecules_to_csv

# High-DPI / Retina display support — imported before figures are created so that matplotlib rcParams are configured early.
from iSLAT.Modules.GUI.DisplayConfig import apply_tk_scaling, display_config

from .Widgets.DataField import DataField
from .Widgets.ControlPanel import ControlPanel
from .Widgets.TopBar import TopBar
from .Widgets.FileInteractionPane import FileInteractionPane
from .Tooltips import set_tooltip_theme
from .GUIFunctions import configure_all_button_styles
from .ThemeFonts import apply_theme_fonts

#: Sizes in the Windows window / taskbar icon made from the logo PNG.
ICON_SIZES = ((16, 16), (32, 32), (48, 48), (64, 64))

def windows_icon_file(png_path) -> str:
    """Return a ``.ico`` of *png_path* (at :data:`ICON_SIZES`) for Windows, made once and kept."""
    import hashlib
    import tempfile

    stat = os.stat(png_path)
    key = f"{stat.st_size}:{stat.st_mtime_ns}:{ICON_SIZES}"
    digest = hashlib.sha1(key.encode("ascii")).hexdigest()[:12]
    ico_path = os.path.join(tempfile.gettempdir(), f"islat_icon_{digest}.ico")
    if os.path.isfile(ico_path) and os.path.getsize(ico_path) > 0:
        return ico_path

    from PIL import Image
    partial_path = f"{ico_path}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with Image.open(png_path) as image:
            image.save(partial_path, format="ICO", sizes=list(ICON_SIZES))
        try:
            os.replace(partial_path, ico_path)
        except OSError:
            # Windows refuses while another launch has the finished file
            # open; that launch already wrote it.
            if not os.path.isfile(ico_path):
                raise
    finally:
        if os.path.exists(partial_path):
            os.remove(partial_path)
    return ico_path

def _windows_user32():
    """A private ``user32`` handle with 64-bit-safe signatures (leaves ``ctypes.windll`` alone)."""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.LoadImageW.restype = wintypes.HANDLE
    user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.SendMessageW.restype = wintypes.LPARAM
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    # SetClassLongPtrW exists only in 64-bit user32; 32-bit Windows has SetClassLongW.
    set_class = getattr(user32, "SetClassLongPtrW", None) or user32.SetClassLongW
    set_class.restype = ctypes.c_size_t
    set_class.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_ssize_t]
    user32.set_class_icon = set_class
    user32.DestroyIcon.restype = wintypes.BOOL
    user32.DestroyIcon.argtypes = [wintypes.HICON]
    return user32

def set_windows_frame_icon(window, ico_path) -> None:
    """Put the ``.ico`` on *window*'s frame and on the Tk toplevel window class (Windows).

    ``wm iconbitmap`` gives the frame window its own icon, but the window
    class keeps Tk's feather.  When the taskbar's query for the window's
    icon is not answered in time, it shows the class icon instead (the
    feather) and keeps it.  That happens at startup: the window appears while
    the model calculation runs on another thread and the main thread is slow
    to answer.  So the logo is also put on the class, which every Tk toplevel
    of this process shares, and :meth:`GUI._refresh_window_icon` calls this
    again once startup is done: new icon handles sent with ``WM_SETICON``
    make the taskbar ask again, and then the main thread answers.

    Tk creates the frame (the window that owns the taskbar button) when the
    window is first mapped, so before that this waits for the first
    ``<Map>``.  ``winfo_id()`` is Tk's inner child window, not the frame.
    The previous call's icon handles are released once replaced.
    """
    WM_SETICON, ICON_SMALL, ICON_BIG = 0x0080, 0, 1
    IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010
    GCLP_HICON, GCLP_HICONSM = -14, -34

    user32 = _windows_user32()
    big = user32.LoadImageW(None, str(ico_path), IMAGE_ICON, 64, 64, LR_LOADFROMFILE)
    small = user32.LoadImageW(None, str(ico_path), IMAGE_ICON, 16, 16, LR_LOADFROMFILE)
    if not big and not small:
        return
    previous = getattr(window, "_islat_icon_handles", ())
    window._islat_icon_handles = (big, small)  # kept for the window's life

    def frame_hwnd():
        try:
            frame = int(window.wm_frame(), 16)
        except (tk.TclError, ValueError):
            return None
        # Before the first map the frame does not exist yet and Tk reports the window itself.
        return None if frame == window.winfo_id() else frame

    def apply() -> bool:
        frame = frame_hwnd()
        if frame is None:
            return False
        for which, class_index, hicon in ((ICON_BIG, GCLP_HICON, big), (ICON_SMALL, GCLP_HICONSM, small)):
            if hicon:
                user32.set_class_icon(frame, class_index, hicon)
                user32.SendMessageW(frame, WM_SETICON, which, hicon)
        for old in previous:
            if old and old not in (big, small):
                user32.DestroyIcon(old)
        return True

    if apply():
        return
    state = {"applied": False}

    def on_map(event):
        # The root's bindings see every widget's <Map>; only the root's own counts.
        if state["applied"] or event.widget is not window:
            return
        try:
            state["applied"] = apply()
        except Exception:
            state["applied"] = True  # icon is cosmetic; do not retry on every map

    window.bind("<Map>", on_map, add="+")

class GUI:
    def __init__(self, master, molecule_data, wave_data, flux_data, config, islat_class_ref,
                 model_calculation=None):
        """
        Parameters
        ----------
        model_calculation : queue.Queue, optional
            Result queue of a model calculation already running (from
            :meth:`start_model_calculation`).  The spectrum display uses it
            instead of starting its own calculation.
        """
        self._pending_model_calculation = model_calculation
        if master is None:
            self.master = tk.Tk()
            # Apply system DPI scaling for HiDPI / Retina displays
            apply_tk_scaling(self.master)
            self.style = ttk.Style()
            self._style_config()
            # self.master = ThemedTk()
            self.master.title("iSLAT - Interactive Spectral-Line Analysis Tool")
            self.master.resizable(True, True)
            # Set minimum size to maintain usability
            self.master.minsize(800, 600)
            #self.master.attributes('-topmost', config.get("start_on_top", True))
            # Configure initial window size based on screen dimensions
            self._configure_initial_size()
            # Set window / taskbar icon
        else:
            self.master = master

        self.molecule_data = molecule_data
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.config = config
        self.theme = config["theme"]
        self.islat_class = islat_class_ref
        # The theme's font sizes, before any widget is built.
        apply_theme_fonts(self.master, self.theme)
        self.default_font = font.nametofont("TkDefaultFont")

        # Set module-level tooltip theme so every tooltip picks up
        # the current theme's colors and font size automatically.
        set_tooltip_theme(self.theme)
        self._set_window_icon(self.master)
        
        # Apply theme to root window
        # self._apply_theme_to_widget(self.master)

    def _style_config(self):
        # On Windows, prefer a modern built-in ttk theme for cleaner widgets.
        if platform.system() == "Windows":
            available = self.style.theme_names()
            for preferred in ("vista", "winnative", "xpnative"):
                if preferred in available:
                    self.style.theme_use(preferred)
                    break

        self.style.configure("Small.TButton", padding=(0, 5))

    def _set_window_icon(self, window: "tk.Tk") -> None:
        """Set the window title-bar and taskbar icon from logo png."""
        try:
            from iSLAT.Modules.FileHandling import data_files_path
            icon_path = data_files_path / "CONFIG" / "Logos" / "Logo2a.png"
            if not icon_path.exists():
                return

            if platform.system() == "Windows":
                try:
                    import ctypes

                    # Without a distinct App User Model ID, Windows groups
                    # every Python script under python.exe and shows that
                    # exe's icon in the taskbar regardless of iconbitmap.
                    # Setting a unique AUMID makes Windows treat iSLAT as
                    # its own application so the window icon appears there.
                    try:
                        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                            "iSLAT.Application.1"
                        )
                    except Exception:
                        pass

                    ico_path = windows_icon_file(icon_path)
                    window.iconbitmap(ico_path)
                    set_windows_frame_icon(window, ico_path)
                    window._islat_icon_file = ico_path  # for _refresh_window_icon
                    return
                except Exception:
                    pass  # fall through to iconphoto

            # Non-Windows (and Windows fallback): use iconphoto.
            try:
                from PIL import Image, ImageTk
                pil_img = Image.open(str(icon_path)).resize((64, 64))
                photo = ImageTk.PhotoImage(pil_img)
            except ImportError:
                photo = tk.PhotoImage(file=str(icon_path))

            window.iconphoto(True, photo)
            window._icon_image = photo  # prevent GC
        except Exception:
            pass  # Icon is cosmetic; never block startup
    
    def _force_theme_update(self):
        """Force theme update on all widgets in the window."""
        if hasattr(self, 'plot') and hasattr(self.plot, 'apply_theme'):
            #print("applying theme to plot")
            self.plot.apply_theme(self.theme)

    # ------------------------------------------------------------------
    # Reactive system-theme synchronisation
    # ------------------------------------------------------------------
    def _bind_appearance_change(self):
        """Bind to OS appearance changes so the GUI theme updates reactively.

        On macOS (Tk ≥ 8.6.9) the ``<<AppearanceChanged>>`` virtual event
        is fired whenever the user switches between Light and Dark mode in
        System Preferences. sWe listen for that event and re-apply the
        theme across the entire GUI.
        """
        if platform.system() != "Darwin":
            return  # Only macOS fires <<AppearanceChanged>>

        # Only auto-sync when the user chose "auto" theme
        theme_key = self.config.get("_theme_key", "")
        if isinstance(theme_key, str) and theme_key.lower() != "auto":
            return

        try:
            self.master.bind("<<AppearanceChanged>>", self._on_appearance_changed)
        except tk.TclError:
            # Older Tk build that doesn't support the virtual event
            pass

    def _on_appearance_changed(self, _event=None):
        """Callback fired by Tk when the macOS appearance changes."""
        from iSLAT.Modules.Plotting.BasePlot import _detect_system_theme, BasePlot

        new_theme_name = _detect_system_theme()

        # Load the new theme dict
        new_theme = BasePlot.load_theme(new_theme_name)

        # Check whether the theme actually changed (avoid redundant redraws)
        old_bg = self.theme.get("background")
        new_bg = new_theme.get("background")
        if old_bg == new_bg:
            return

        # Store the new theme globally
        self.theme = new_theme
        self.config["theme"] = new_theme

        # --- Propagate to every component ---

        # 1. Tooltip colors and the GUI font sizes
        set_tooltip_theme(new_theme)
        apply_theme_fonts(self.master, new_theme)

        # 2. Custom ttk button / menubutton styles
        configure_all_button_styles(new_theme)

        # 3. Matplotlib-based plot (visual-only, no spectrum recalculation)
        if hasattr(self, 'plot') and self.plot is not None:
            self.plot.apply_theme(new_theme)

        # 4. Recurse through every tk/ttk widget in the window.
        self._apply_theme_to_all_widgets(new_theme)

    def _apply_theme_to_all_widgets(self, theme):
        """Walk the entire widget tree and apply *theme* to every widget.

        This mirrors the logic in
        :pymethod:`ResizableFrame._apply_theme_to_widget` but is invoked
        from the top-level ``GUI`` instance so that widgets that are *not*
        children of a ResizableFrame are also themed (e.g. raw ``tk.Frame``
        containers, ``ControlPanel``, ``DataField``, etc.).
        """
        for child in self.master.winfo_children():
            self._apply_theme_to_widget_recursive(child, theme)

    def _apply_theme_to_widget_recursive(self, widget, theme):
        """Apply *theme* colors to *widget* and recurse into children."""
        try:
            # Skip matplotlib canvas widget subtrees entirely – they are
            # already themed by MainPlot.apply_theme() and recursing into
            # their many internal tk children triggers expensive redraws.
            if 'FigureCanvasTkAgg' in type(widget).__name__:
                return
            if hasattr(widget, 'master') and 'FigureCanvasTkAgg' in type(widget.master).__name__:
                return

            widget_class = widget.winfo_class()

            if widget_class in ('Frame', 'LabelFrame'):
                try:
                    widget.configure(bg=theme.get("background", "#181A1B"))
                except tk.TclError:
                    pass
                if widget_class == 'LabelFrame':
                    try:
                        widget.configure(
                            fg=theme.get("foreground", "#F0F0F0"),
                            highlightbackground=theme.get("foreground", "#F0F0F0"),
                        )
                    except tk.TclError:
                        pass

            elif widget_class == 'Button':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Label':
                # Skip ColorButton instances – they use bg for the
                # molecule color and must not be overwritten.
                if not hasattr(widget, 'color'):
                    try:
                        widget.configure(
                            bg=theme.get("background", "#181A1B"),
                            fg=theme.get("foreground", "#F0F0F0"),
                        )
                    except tk.TclError:
                        pass

            elif widget_class == 'Entry':
                try:
                    widget.configure(
                        bg=theme.get("background_accent_color", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        insertbackground=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Text':
                try:
                    widget.configure(
                        bg=theme.get("data_field_background", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        insertbackground=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Listbox':
                try:
                    widget.configure(
                        bg=theme.get("background_accent_color", "#23272A"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        selectbackground=theme.get("selection_color", "#00FF99"),
                        selectforeground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Checkbutton':
                try:
                    widget.configure(
                        bg=theme.get("background", "#181A1B"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        selectcolor=theme.get("background_accent_color", "#23272A"),
                        activebackground=theme.get("background", "#181A1B"),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Menubutton':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                        highlightbackground=theme.get("background", "#181A1B"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Menu':
                btn_theme = theme.get("buttons", {}).get("DefaultBotton", {})
                try:
                    widget.configure(
                        bg=btn_theme.get("background", "lightgray"),
                        fg=theme.get("foreground", "#F0F0F0"),
                        activebackground=btn_theme.get("active_background",
                                                       theme.get("selection_color", "#00FF99")),
                        activeforeground=theme.get("foreground", "#F0F0F0"),
                    )
                except tk.TclError:
                    pass

            elif widget_class == 'Canvas':
                # Skip matplotlib canvas widgets (handled by plot.apply_theme)
                if 'NavigationToolbar2Tk' not in type(widget.master).__name__:
                    try:
                        widget.configure(bg=theme.get("background", "#181A1B"))
                    except tk.TclError:
                        pass

            # Recurse into children
            for child in widget.winfo_children():
                self._apply_theme_to_widget_recursive(child, theme)

        except tk.TclError:
            pass

    def _configure_initial_size(self):
        """Configure initial window size based on screen resolution."""
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        
        # Use 80% of screen width and 75% of screen height
        window_width = int(screen_width * 0.95)
        window_height = int(screen_height * 0.87)
        
        # Ensure minimum size constraints
        window_width = max(window_width, 800)
        window_height = max(window_height, 600)
        
        # Calculate position to center the window
        pos_x = int((screen_width - window_width) / 2)
        pos_y = int((screen_height - window_height) / 2)
        
        self.master.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")

    def build_left_panel(self, parent: tk.Frame):
        # Main data field - create this first so we can pass it to other components
        self.data_field = DataField("", parent, theme=self.theme)

        #Control panel for molecule controls
        self.control_panel = ControlPanel(parent, self.islat_class, self.plot, self.data_field, self.default_font)
        
        # Spectrum file selector
        self.file_interaction_pane = FileInteractionPane(parent, self.islat_class, self.theme, self.data_field)
        
        self.file_interaction_pane.grid(row=0, padx=(1,0), sticky="nsew", pady=0)
        self.control_panel.grid(row=1, padx=(1,0), sticky="nsew", pady=0)
        self.data_field.grid(row=2, padx=(1,0), sticky="nsew", pady=0)

        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(0, weight=1)

    def create_window(self):
        from iSLAT import __version__ as iSLAT_version
        from iSLAT.Modules.Plotting.MainPlot import iSLATPlot
        self.window = self.master
        self.window.title(f"iSLAT Version {iSLAT_version}")
        
        # Create a main container frame
        main_container = ttk.Frame(self.window)

        # Create frames for left panel and right panel (plot)
        left_main_frame = tk.Frame(main_container)
        left_main_frame.pack(side="left", fill="y", expand=False, padx=0, pady=0)

        right_main_frame = tk.Frame(main_container)
        right_main_frame.pack(side="right", fill="both", expand=True, padx=0, pady=0)
        
        # Create the plot directly in right_frame without extra container
        self.plot = iSLATPlot(right_main_frame, self.wave_data, self.flux_data, self.theme, self.islat_class)

        # Left side: all controls
        self.build_left_panel(left_main_frame)

        # Bottom function buttons
        self.top_bar = TopBar(self.window, self.islat_class, self.theme, self.plot, self.data_field, self.control_panel, self.config)
        self.top_bar.pack(side="top", fill="x", padx=0, pady=0)

        # Wire ControlBus surfaces now that all widgets are constructed.
        self.plot.control_bus.register_surface(
            "control_panel", self.control_panel._control_panel_surface
        )
        self.plot.control_bus.register_surface(
            "top_bar", self.top_bar._top_bar_surface
        )
        self.plot.control_bus.register_surface(
            "views_menu", self.top_bar._views_menu_surface
        )
        # The default view was activated inside iSLATPlot.__init__, before
        # these surfaces existed, so its control fields went nowhere.
        # Register them again now that the surfaces are wired.
        active_view = getattr(self.plot, 'active_view', None)
        if active_view is not None and hasattr(active_view, '_register_control_fields'):
            active_view._register_control_fields()

        main_container.pack(fill="both", expand=True, padx=0, pady=0)

        self._force_theme_update()

    def start(self, display_spectrum_async=True):
        """
        Start the GUI and enter the main event loop.
        
        Parameters
        ----------
        display_spectrum_async : bool, default True
            If True, display spectrum asynchronously after GUI is shown (faster startup).
            If False, display spectrum synchronously (blocks until complete).
        """
        self.create_window()

        # Hold the plot's draws until Tk has laid the window out (the left-hand
        # panels size themselves only after the window is mapped); every draw
        # before that is at a provisional canvas size.
        canvas = getattr(getattr(self, "plot", None), "canvas", None)
        if hasattr(canvas, "hold_draws_until_laid_out"):
            canvas.hold_draws_until_laid_out()

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Listen for OS appearance changes (macOS dark ↔ light mode)
        self._bind_appearance_change()

        # Check config for start-on-top behavior
        if self.config.get("start_on_top", True):
            self.master.attributes('-topmost', True)
            self.master.after(100, lambda: self.master.attributes('-topmost', False))
        
        self.master.lift()
        
        # Schedule async spectrum display after GUI is shown
        if display_spectrum_async:
            # Use after(1) to run after first event loop iteration (GUI visible first)
            self.master.after(1, self._start_async_spectrum_display)
        
        self.window.mainloop()

    def _start_async_spectrum_display(self):
        """Called after GUI is shown - starts async spectrum calculation."""
        if (hasattr(self, 'plot') and self.plot is not None and
            hasattr(self.islat_class, 'wave_data') and hasattr(self.islat_class, 'flux_data')):
            #print("Starting async spectrum display...")
            self.display_spectrum_async(callback=self._on_async_display_complete)
    
    def _on_async_display_complete(self, success):
        """Called when async spectrum display completes."""
        self._refresh_window_icon_after_next_draw()
        if success:
            # Update file label if available
            if (hasattr(self, "file_interaction_pane") and 
                hasattr(self.islat_class, 'loaded_spectrum_name')):
                self.file_interaction_pane.update_file_label(self.islat_class.loaded_spectrum_name)

    def _refresh_window_icon_after_next_draw(self):
        """Send the window icon again once the plot's startup draw is done (Windows)."""
        if not getattr(getattr(self, "master", None), "_islat_icon_file", None):
            return
        canvas = getattr(getattr(self, "plot", None), "canvas", None)
        if canvas is None or not hasattr(canvas, "mpl_connect"):
            self.master.after_idle(self._refresh_window_icon)
            return
        connection = {}

        def on_draw(_event):
            canvas.mpl_disconnect(connection["id"])
            self.master.after_idle(self._refresh_window_icon)

        connection["id"] = canvas.mpl_connect("draw_event", on_draw)

    def _refresh_window_icon(self):
        """Load the window icon again and send it to the frame (Windows; see :func:`set_windows_frame_icon`)."""
        ico_path = getattr(self.master, "_islat_icon_file", None)
        if not ico_path:
            return
        try:
            set_windows_frame_icon(self.master, ico_path)
        except Exception:
            pass  # Icon is cosmetic

    def on_closing(self):
        #if messagebox.askokcancel("Quit", "Do you want to save your work?"):
        if True:            
            try:
                # Save the current molecule parameters
                write_molecules_to_csv(
                    self.islat_class.molecules_dict, 
                )
            except Exception as e:
                print("Error", f"Failed to save molecule parameters: {str(e)}")

        # Close all matplotlib figures to release resources
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass

        # Destroy the stacked views' figures (non-pyplot managed)
        for view_attr in ('_full_spectrum_view', '_optical_depth_view'):
            try:
                if hasattr(self, 'plot') and hasattr(self.plot, view_attr):
                    getattr(self.plot, view_attr).destroy()
            except Exception:
                pass

        try:
            self.window.destroy()
        except Exception:
            pass

        # Force-exit the process so no background threads keep it alive
        import os
        os._exit(0)

    # ================================
    # Async Spectrum Display
    # ================================
    def display_spectrum_async(self, callback=None):
        """
        Display spectrum asynchronously to avoid blocking the GUI during startup.
        
        This method shows a loading indicator, runs the intensive spectrum calculations
        in a background thread, and updates the GUI when complete.
        
        When a calculation started before the window was built
        (``model_calculation`` in the constructor) has already finished, the
        models are put on the plot at once, without the loading indicator, so
        the first draw already shows them.  Otherwise the observed spectrum is
        shown with a loading indicator until the calculation finishes.

        Parameters
        ----------
        callback : callable, optional
            Function to call when display is complete
        """
        self._async_callback = callback
        pending = getattr(self, '_pending_model_calculation', None)
        self._pending_model_calculation = None

        if pending is not None and not pending.empty():
            self._async_result_queue = pending
            self._poll_async_result()
            return

        # Show loading state immediately
        if hasattr(self, 'plot') and self.plot is not None:
            self.plot.show_loading_indicator("Calculating molecule spectra...")

        if pending is not None:
            self._async_result_queue = pending
        else:
            self._async_result_queue = queue.Queue()
            calc_thread = threading.Thread(
                target=self._async_spectrum_calculation,
                daemon=True
            )
            calc_thread.start()

        # Start polling for completion (non-blocking)
        self._poll_async_result()

    @staticmethod
    def start_model_calculation(islat_class):
        """Start computing the visible molecules' summed model spectrum on a background thread.

        Called before the window is built, so the calculation runs while Tk
        builds the window.  Pass the returned queue to the constructor as
        ``model_calculation``; it receives ``('success', None)`` or
        ``('error', message)`` when the calculation ends.
        """
        result_queue = queue.Queue()
        threading.Thread(
            target=GUI._calculate_models,
            args=(islat_class, result_queue),
            daemon=True,
        ).start()
        return result_queue

    def _async_spectrum_calculation(self):
        """Background thread: perform spectrum calculations using parallel intensity calculation."""
        self._calculate_models(self.islat_class, self._async_result_queue)

    @staticmethod
    def _calculate_models(islat_class, result_queue):
        """Compute the visible molecules' summed flux (fills their caches) and report to *result_queue*."""
        try:
            # This triggers lazy intensity calculations for all visible molecules
            if (hasattr(islat_class, 'molecules_dict') and
                hasattr(islat_class, 'wave_data_original')):
                wave_data = islat_class.wave_data_original
                # Use parallel pre-calculation for significant speedup
                islat_class.molecules_dict.get_summed_flux_parallel(wave_data, visible_only=True)

            result_queue.put(('success', None))
        except Exception as e:
            result_queue.put(('error', str(e)))
    
    def _poll_async_result(self):
        """Poll for async calculation completion (runs on main thread)."""
        try:
            status, error = self._async_result_queue.get_nowait()
            
            # Calculation complete - update GUI
            if hasattr(self, 'plot') and self.plot is not None:
                self.plot.hide_loading_indicator()
                
                if status == 'success':
                    #print("Async spectrum calculation complete - updating display...")
                    # Initialize data-dependent plot elements now that calculations are done
                    self.plot.update_model_plot()
                    self.plot._set_initial_zoom_range()
                    # Both calls above already asked for a draw; draw_idle merges
                    # with that one (a draw() here rendered the figure twice).
                    if hasattr(self.plot, 'canvas'):
                        self.plot.canvas.draw_idle()
                    #print("Spectrum displayed successfully (async)")
                    
                    # Print final performance summary including all async operations
                    #print("\n" + "="*80)
                    #print("FINAL PERFORMANCE SUMMARY (including async operations)")
                    #print("="*80)
                    get_performance_summary()
                else:
                    print(f"Warning: Async spectrum calculation failed: {error}")
            
            # Call completion callback if provided
            if self._async_callback:
                self._async_callback(status == 'success')
                
        except queue.Empty:
            # Still calculating - poll again in 50ms
            self.master.after(50, self._poll_async_result)

    @staticmethod
    def file_selector(title : str = None, filetypes=None, initialdir=None, use_abspath=True, allow_multiple=False):
        """Ask for one file or several; see :func:`iSLAT.Modules.FileHandling.file_dialogs.file_selector`."""
        from iSLAT.Modules.FileHandling.file_dialogs import file_selector
        return file_selector(title=title, filetypes=filetypes, initialdir=initialdir,
                             use_abspath=use_abspath, allow_multiple=allow_multiple)

    @staticmethod
    def add_molecule_name_popup(title : str = "Assign label"):
        """Open a popup to add a new molecule name."""
        molecule_name = simpledialog.askstring(title, "Enter a label for this model (LaTeX and case sensitive):")

        return molecule_name