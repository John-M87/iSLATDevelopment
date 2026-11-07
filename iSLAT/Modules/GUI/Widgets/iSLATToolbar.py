"""
iSLATToolbar — custom matplotlib NavigationToolbar2Tk for iSLAT.

Provides ``iSLATNavigationToolbar``, which extends the standard matplotlib
toolbar by replacing the built-in "Configure Subplots" behaviour with a
custom dialog that:

1. Embeds matplotlib's :class:`~matplotlib.widgets.SubplotTool` for the
   standard subplot-margin sliders.
2. Adds a dynamic lower section managed by :class:`ConfigureSubplotsSurface`
   where views can register extra :class:`~iSLAT.Modules.GUI.ControlField`
   instances (e.g. "N Panels" for the full-spectrum view).
"""

from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
import weakref
from typing import Any, Dict, Optional, TYPE_CHECKING

from matplotlib import cbook
from matplotlib.backend_bases import _Mode
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk

from iSLAT.Modules.GUI.ControlSurface import ControlSurface

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------------------------------------------------------
# ConfigureSubplotsSurface
# ---------------------------------------------------------------------------
class ConfigureSubplotsSurface(ControlSurface):
    """Concrete :class:`ControlSurface` for the Configure Subplots dialog.

    This surface is persistent across dialog openings: fields can be
    registered before the dialog is ever opened.  When the dialog is opened,
    ``attach_dialog()`` is called and the surface renders its fields into
    *field_frame*.  When the dialog is closed, ``detach_dialog()`` is called
    and ``_rebuild()`` becomes a no-op until the next open.

    Parameters
    ----------
    No constructor parameters required.
    """
    def __init__(self) -> None:
        super().__init__()
        self._field_frame: Optional[tk.Widget] = None

    # ------------------------------------------------------------------
    # Dialog lifecycle
    # ------------------------------------------------------------------

    def attach_dialog(self, field_frame: tk.Widget) -> None:
        """Called when the dialog opens.  Supplies the frame to render into."""
        self._field_frame = field_frame
        self._rebuild()

    def detach_dialog(self) -> None:
        """Called when the dialog closes."""
        self._widget_refs.clear()
        self._field_frame = None

    # ------------------------------------------------------------------
    # Internal rebuild
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        if self._field_frame is None:
            return  # Dialog not currently open

        from iSLAT.Modules.GUI.ControlField import RenderContext

        try:
            for child in self._field_frame.winfo_children():
                child.destroy()
        except Exception:
            pass
        self._widget_refs.clear()

        row = 0
        for field in self._fields.values():
            try:
                widgets = field.build_widget(self._field_frame, RenderContext.DIALOG)
            except Exception:
                widgets = []

            self._widget_refs[field.key] = widgets

            if len(widgets) == 1:
                widgets[0].grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=3)
            elif len(widgets) >= 2:
                widgets[0].grid(row=row, column=0, sticky="w", padx=(6, 2), pady=3)
                widgets[1].grid(row=row, column=1, sticky="w", padx=(2, 6), pady=3)

            if widgets:
                row += 1

# ---------------------------------------------------------------------------
# iSLATConfigureSubplotsDialog
# ---------------------------------------------------------------------------
class iSLATConfigureSubplotsDialog(tk.Toplevel):
    """Custom "Configure Subplots" dialog for iSLAT.

    Top section: the standard matplotlib subplot-margin adjustment sliders
    (rendered via :class:`~matplotlib.widgets.SubplotTool`).

    Bottom section: dynamic :class:`ControlField` widgets managed by the
    supplied :class:`ConfigureSubplotsSurface`.

    Parameters
    ----------
    parent:
        The root Tk window (used as the dialog's *master*).
    figure:
        The matplotlib :class:`~matplotlib.figure.Figure` whose subplot
        parameters are being adjusted.
    configure_subplots_surface:
        The :class:`ConfigureSubplotsSurface` instance.  May be ``None``
        (the bottom section is then empty).
    """
    def __init__(
        self,
        parent: tk.Widget,
        figure: Any,
        configure_subplots_surface: Optional[ConfigureSubplotsSurface],
    ) -> None:
        super().__init__(parent)
        self.title("Configure Subplots")
        self.resizable(True, True)

        self._surface = configure_subplots_surface

        # ------------------------------------------------------------------
        # Top section: matplotlib SubplotTool
        # ------------------------------------------------------------------
        tool_frame = tk.Frame(self)
        tool_frame.pack(side="top", fill="both", expand=True)

        try:
            from matplotlib.figure import Figure as MplFigure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.widgets import SubplotTool

            tool_fig = MplFigure(figsize=(6, 3.5))
            tool_canvas = FigureCanvasTkAgg(tool_fig, master=tool_frame)
            tool_canvas.get_tk_widget().pack(fill="both", expand=True)
            self._subplot_tool = SubplotTool(figure, tool_fig)
            tool_canvas.draw()
        except Exception:
            # Graceful degradation: show a fallback label if SubplotTool fails
            tk.Label(
                tool_frame,
                text="Subplot adjustment unavailable.",
                anchor="center",
            ).pack(expand=True)

        # ------------------------------------------------------------------
        # Separator
        # ------------------------------------------------------------------
        ttk.Separator(self, orient="horizontal").pack(fill="x", padx=4, pady=(6, 0))

        # ------------------------------------------------------------------
        # Bottom section: dynamic ControlField widgets
        # ------------------------------------------------------------------
        self._field_frame = ttk.Frame(self)
        self._field_frame.pack(side="top", fill="x", padx=4, pady=4)

        if self._surface is not None:
            self._surface.attach_dialog(self._field_frame)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------

    def _on_close(self) -> None:
        if self._surface is not None:
            self._surface.detach_dialog()
        self.destroy()

# ---------------------------------------------------------------------------
# iSLATNavigationToolbar
# ---------------------------------------------------------------------------
class iSLATNavigationToolbar(NavigationToolbar2Tk):
    """matplotlib :class:`NavigationToolbar2Tk` subclass used in iSLAT.

    Replaces the built-in "Configure Subplots" behaviour with
    :class:`iSLATConfigureSubplotsDialog`, which embeds both the standard
    subplot-margin sliders *and* a dynamic :class:`ConfigureSubplotsSurface`
    field area.

    Parameters
    ----------
    canvas:
        The :class:`~matplotlib.backends.backend_tkagg.FigureCanvasTkAgg`
        to attach the toolbar to.
    window:
        The Tk parent widget for the toolbar.
    configure_subplots_surface:
        The :class:`ConfigureSubplotsSurface` instance managed by the
        :class:`~iSLAT.Modules.Plotting.MainPlot.iSLATPlot` controller.
        May be ``None`` (falls back to the default matplotlib dialog).
    """
    def __init__(
        self,
        canvas: "FigureCanvasTkAgg",
        window: tk.Widget,
        configure_subplots_surface: Optional[ConfigureSubplotsSurface] = None,
    ) -> None:
        super().__init__(canvas, window)
        self._configure_subplots_surface = configure_subplots_surface
        self._configure_subplots_dialog: Optional[iSLATConfigureSubplotsDialog] = None
        # Per-canvas view history, so Back / Forward / Home survive view switches.
        self._nav_stacks: "weakref.WeakKeyDictionary[Any, cbook._Stack]" = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------
    # Canvas retargeting
    # ------------------------------------------------------------------

    def set_canvas(self, canvas: Optional["FigureCanvasTkAgg"], reset_history: bool = False) -> None:
        """Drive *canvas* (e.g. the active plot view's) instead of the current one.

        The Pan / Zoom mode carries over; each canvas keeps its own view
        history unless *reset_history* is set (for redrawn content, where the
        old limits no longer apply).
        """
        if canvas is None:
            return
        if canvas is not self.canvas:
            mode = self.mode
            self._detach_canvas()
            self.canvas = canvas
            canvas.toolbar = self
            self._id_press = canvas.mpl_connect("button_press_event", self._zoom_pan_handler)
            self._id_release = canvas.mpl_connect("button_release_event", self._zoom_pan_handler)
            self._id_drag = canvas.mpl_connect("motion_notify_event", self.mouse_move)
            self._nav_stack = self._nav_stacks.setdefault(canvas, cbook._Stack())
            self._set_mode(mode)
        if reset_history:
            self._nav_stack.clear()
        self.sync_navigate_mode()
        self.set_history_buttons()

    def sync_navigate_mode(self) -> None:
        """Apply the toolbar's mode to every axes of the canvas (new axes start with none)."""
        self._set_axes_navigate_mode(self.canvas, getattr(self.mode, "_navigate_mode", None))

    @staticmethod
    def _set_axes_navigate_mode(canvas: Any, value: Optional[str]) -> None:
        # matplotlib < 3.11 mirrors the toolbar mode on each axes; 3.11 dropped
        # that (and deprecated Axes.set_navigate_mode), so there is nothing to do.
        if not hasattr(_Mode.NONE, "_navigate_mode"):
            return
        figure = getattr(canvas, "figure", None)
        if figure is None:
            return
        for ax in figure.get_axes():
            ax.set_navigate_mode(value)

    def _detach_canvas(self) -> None:
        """Disconnect from the current canvas and release its widget lock."""
        old = self.canvas
        if old is None:
            return
        self._nav_stacks[old] = self._nav_stack
        for cid in (self._id_press, self._id_release, self._id_drag):
            old.mpl_disconnect(cid)
        try:
            self.remove_rubberband()
        except (tk.TclError, AttributeError):
            pass  # the old canvas widget may already be destroyed
        if old.widgetlock.isowner(self):
            old.widgetlock.release(self)
        self._set_axes_navigate_mode(old, None)
        if getattr(old, "toolbar", None) is self:
            old.toolbar = None
        self._pan_info = None
        self._zoom_info = None

    def _set_mode(self, mode: "_Mode") -> None:
        """Enter *mode* on the current canvas (falls back to no mode if the canvas is locked)."""
        if mode != _Mode.NONE and not self.canvas.widgetlock.available(self):
            mode = _Mode.NONE
        self.mode = mode
        if mode != _Mode.NONE:
            self.canvas.widgetlock(self)
        self._update_buttons_checked()

    # ------------------------------------------------------------------

    def configure_subplots(self, *args: Any) -> None:
        """Open (or raise) the iSLAT Configure Subplots dialog."""
        # Raise existing window if still open
        if self._configure_subplots_dialog is not None:
            try:
                self._configure_subplots_dialog.lift()
                return
            except Exception:
                self._configure_subplots_dialog = None

        if self.canvas is None:
            return

        try:
            root = self.canvas.get_tk_widget().winfo_toplevel()
            dialog = iSLATConfigureSubplotsDialog(
                root,
                self.canvas.figure,
                self._configure_subplots_surface,
            )
            self._configure_subplots_dialog = dialog
        except Exception:
            # Fall back to default matplotlib dialog on any error
            super().configure_subplots(*args)