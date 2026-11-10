from tkinter import Toplevel, Label, LEFT, SOLID, TclError

from .ThemeFonts import tooltip_font

# Module-level theme reference for tooltip colors.
# Call set_tooltip_theme(theme_dict) once at GUI startup so every
# tooltip automatically uses the current theme colors.
_current_theme = None

_LIGHT_DEFAULTS = {"tooltip_background": "peachpuff", "tooltip_foreground": "#000000"}
_DARK_DEFAULTS  = {"tooltip_background": "#3C3F41",  "tooltip_foreground": "#F0F0F0"}

def set_tooltip_theme(theme: dict | None) -> None:
    """Set the module-level theme used by all future tooltips."""
    global _current_theme
    _current_theme = theme

def _resolve_tooltip_colors(explicit_bg=None, explicit_fg=None):
    """Return (bg, fg) for a tooltip, respecting explicit overrides,
    the current theme, and sensible fallback defaults."""
    theme = _current_theme or {}
    bg = explicit_bg or theme.get("tooltip_background") or _LIGHT_DEFAULTS["tooltip_background"]
    fg = explicit_fg or theme.get("tooltip_foreground") or _LIGHT_DEFAULTS["tooltip_foreground"]
    return bg, fg

class ToolTip(object):
    def __init__(self, widget, text, bg=None, fg=None):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.active = True
        self.x = self.y = 0
        # Store explicit overrides; resolved at show-time so theme
        # changes that happen after construction are picked up.
        self._explicit_bg = bg
        self._explicit_fg = fg

    def showtip(self):
        "Display text in tooltip window"
        if self.tipwindow or not self.text:
            return
        try:
            x, y, cx, cy = self.widget.bbox("insert")
        except Exception:
            # Fallback for widgets that don't support "insert" (e.g. Listbox)
            x, y, cx, cy = 0, 0, 0, 0
        x = x + self.widget.winfo_rootx() + 45
        y = y + cy + self.widget.winfo_rooty() + 27
        self.tipwindow = tw = Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        bg, fg = _resolve_tooltip_colors(self._explicit_bg, self._explicit_fg)
        label = Label(tw, text=self.text, justify=LEFT,
                       background=bg, foreground=fg,
                       relief=SOLID, borderwidth=1,
                       font=tooltip_font(_current_theme))
        label.pack(ipadx=1)

    def disable(self):
        self.active = False

    def enable(self):
        self.active = True

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

def CreateToolTip(widget, text, bg=None, fg=None):
    toolTip = ToolTip(widget, text, bg=bg, fg=fg)

    def enter(event):
        if toolTip.active:
            toolTip.showtip()

    def leave(event):
        if toolTip.active:
            toolTip.hidetip()

    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)

    return toolTip

class MenuToolTip:
    """Tooltip support for individual tk.Menu entries.

    Uses the ``<<MenuSelect>>`` virtual event, which Tk fires on all
    platforms (including native Windows / macOS menus) whenever the
    highlighted item changes.  Tooltip position follows the pointer so
    it works even when ``winfo_rootx`` returns 0 for native menus.

    Usage::

        tips = {
            "Label A": "Short description of Label A.",
            "Label B": "Short description of Label B.",
        }
        MenuToolTip(some_menu, tips)
    """

    def __init__(self, menu, tips: dict) -> None:
        self.menu = menu
        self.tips = tips
        self.tipwindow = None
        self._last_index = None

        menu.bind("<<MenuSelect>>", self._on_select)
        menu.bind("<Unmap>", self._on_leave)

    def _on_select(self, event) -> None:
        try:
            index = self.menu.index("active")
        except TclError:
            self._hide()
            return

        if index == self._last_index:
            return
        self._last_index = index
        self._hide()

        try:
            label = self.menu.entrycget(index, "label")
        except TclError:
            return  # separator or other non-labelled entry

        tip_text = self.tips.get(label)
        if not tip_text:
            return

        # Use the pointer position - reliable even for native OS menus
        x = self.menu.winfo_pointerx() + 20
        y = self.menu.winfo_pointery()
        self._show(tip_text, x, y)

    def _on_leave(self, event=None) -> None:
        self._last_index = None
        self._hide()

    def _show(self, text: str, x: int, y: int) -> None:
        if self.tipwindow:
            return
        bg, fg = _resolve_tooltip_colors()
        tw = Toplevel(self.menu)
        tw.wm_overrideredirect(True)
        tw.attributes("-topmost", True)
        tw.wm_geometry(f"+{x}+{y}")
        Label(
            tw, text=text, justify=LEFT,
            background=bg, foreground=fg,
            relief=SOLID, borderwidth=1,
            font=tooltip_font(_current_theme),
        ).pack(ipadx=1)
        self.tipwindow = tw

    def _hide(self) -> None:
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None