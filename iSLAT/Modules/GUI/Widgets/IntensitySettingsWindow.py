"""IntensitySettingsWindow — popout window for the intensity calculation settings.

Lets the user choose the method that turns each line's optical depth into its
intensity (curve of growth with or without line overlap, or the RADEX
``1 - e^-τ`` expression).  The choice applies to every molecule (through
:attr:`MoleculeDict.global_intensity_calculation_method`, which recalculates
the models and redraws the plots) and is saved in the user settings as the
default for future sessions.

Usage
-----
    from iSLAT.Modules.GUI.Widgets.IntensitySettingsWindow import IntensitySettingsWindow
    IntensitySettingsWindow.open(parent_widget, islat_instance, theme)
"""
from __future__ import annotations

import tkinter as tk
import tkinter.ttk as ttk
from typing import Any, Dict, List, Optional

DEFAULT_METHOD = "curve_growth"


# ----------------------------------------------------------------------
# Logic (no Tk) - also used by the tests
# ----------------------------------------------------------------------
def available_methods() -> List[str]:
    """The intensity calculation methods the user can choose from."""
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    return Intensity.available_strategies()


def describe_method(method: str):
    """``(display_name, description)`` of *method*."""
    from iSLAT.Modules.DataTypes.Intensity import Intensity
    return Intensity.describe_strategy(method)


def current_method(islat: Any) -> str:
    """The method currently in use (the molecules' global setting)."""
    mol_dict = getattr(islat, "molecules_dict", None)
    method = getattr(mol_dict, "global_intensity_calculation_method", None)
    if not method:
        settings = getattr(islat, "user_settings", None)
        if isinstance(settings, dict):
            method = settings.get("intensity_calculation_method")
    return str(method or DEFAULT_METHOD)


def apply_intensity_method(islat: Any, method: str, save: bool = True) -> bool:
    """Use *method* for every molecule's intensity calculation.

    Sets :attr:`MoleculeDict.global_intensity_calculation_method` (which
    recalculates the models and notifies the plots) when the global setting
    or any molecule differs from *method*, and stores *method* in the user
    settings (saved to disk when *save*).

    Returns ``True`` when the models were recalculated.

    Raises
    ------
    ValueError
        If *method* is not a registered intensity calculation method.
    """
    method = str(method)
    if method not in available_methods():
        raise ValueError(
            f"Unknown intensity calculation method {method!r}; "
            f"choose one of: {', '.join(available_methods())}."
        )

    changed = False
    mol_dict = getattr(islat, "molecules_dict", None)
    if mol_dict is not None:
        molecules = list(mol_dict.values()) if hasattr(mol_dict, "values") else []
        stale = (
            getattr(mol_dict, "global_intensity_calculation_method", None) != method
            or any(getattr(mol, "intensity_calculation_method", method) != method for mol in molecules)
        )
        if stale:
            mol_dict.global_intensity_calculation_method = method
            changed = True

    settings = getattr(islat, "user_settings", None)
    if isinstance(settings, dict):
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        settings[ifh.INTENSITY_METHOD_SETTING_KEY] = method
        settings["intensity_calculation_method"] = method
        if save:
            try:
                ifh.save_user_settings(settings)
            except Exception as exc:
                print(f"Could not save the intensity calculation method: {exc}")
    return changed


# ----------------------------------------------------------------------
# Window
# ----------------------------------------------------------------------
class IntensitySettingsWindow:
    """Singleton popout window for choosing the intensity calculation method."""

    # Class-level registry: one window per (islat instance id)
    _instances: Dict[int, "IntensitySettingsWindow"] = {}

    @classmethod
    def open(
        cls,
        parent: tk.Widget,
        islat: Any,
        theme: Optional[Dict[str, Any]] = None,
    ) -> "IntensitySettingsWindow":
        """Open (or raise) the intensity settings window for *islat*."""
        key = id(islat)
        existing = cls._instances.get(key)
        if existing is not None:
            try:
                if existing._win.winfo_exists():
                    existing._method_var.set(current_method(islat))
                    existing._win.lift()
                    existing._win.focus_force()
                    return existing
            except Exception:
                pass
            del cls._instances[key]

        instance = cls(parent, islat, theme or {})
        cls._instances[key] = instance
        return instance

    def __init__(self, parent: tk.Widget, islat: Any, theme: Dict[str, Any]) -> None:
        self._islat = islat
        self._theme = theme

        self._win = tk.Toplevel(parent)
        self._win.title("Intensity Settings")
        self._win.resizable(False, False)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._method_var = tk.StringVar(master=self._win, value=current_method(islat))
        self._build_ui()

        # Centre on the screen.
        self._win.update_idletasks()
        x = self._win.winfo_screenwidth() // 2 - self._win.winfo_width() // 2
        y = self._win.winfo_screenheight() // 2 - self._win.winfo_height() // 2
        self._win.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        content = ttk.Frame(self._win)
        content.pack(fill="both", expand=True, padx=12, pady=12)

        methods_frame = ttk.LabelFrame(content, text="Intensity calculation method")
        methods_frame.pack(fill="x")
        self._method_buttons: Dict[str, ttk.Radiobutton] = {}
        row = 0
        for method in available_methods():
            name, description = describe_method(method)
            button = ttk.Radiobutton(methods_frame, text=name, value=method, variable=self._method_var)
            button.grid(row=row, column=0, sticky="w", padx=6, pady=(6, 0))
            self._method_buttons[method] = button
            row += 1
            if description:
                ttk.Label(methods_frame, text=description, wraplength=380, foreground="gray").grid(
                    row=row, column=0, sticky="w", padx=(26, 6), pady=(0, 2),
                )
                row += 1

        ttk.Label(
            content,
            text="Applies to all molecules and is saved as the default for future sessions.",
            wraplength=400,
        ).pack(anchor="w", pady=(8, 8))

        buttons = ttk.Frame(content)
        buttons.pack(anchor="e")
        ttk.Button(buttons, text="Apply", command=self._on_apply).pack(side="right", padx=(4, 0))
        ttk.Button(buttons, text="Cancel", command=self._on_close).pack(side="right", padx=4)

    # ------------------------------------------------------------------
    def _on_apply(self) -> None:
        method = self._method_var.get()
        try:
            self._win.config(cursor="watch")
            self._win.update_idletasks()
            changed = apply_intensity_method(self._islat, method)
        except ValueError as exc:
            self._report(str(exc))
            return
        finally:
            try:
                self._win.config(cursor="")
            except tk.TclError:
                pass
        name = describe_method(method)[0]
        self._report(
            f"Intensity calculation method set to: {name}"
            + ("" if changed else " (unchanged)")
        )
        self._on_close()

    def _report(self, message: str) -> None:
        data_field = getattr(getattr(self._islat, "GUI", None), "data_field", None)
        if data_field is not None and hasattr(data_field, "insert_text"):
            data_field.insert_text(message)
        else:
            print(message)

    def _on_close(self) -> None:
        IntensitySettingsWindow._instances.pop(id(self._islat), None)
        try:
            self._win.destroy()
        except Exception:
            pass
