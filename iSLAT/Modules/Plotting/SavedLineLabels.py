"""Text labels for saved lines (the input line list) on the spectrum plots.

One app-wide setting chooses the label: by default ``"<species> <line>"``,
or the value of any column of the saved-line CSV, with an optional prefix and
suffix - and in which views (:data:`LABEL_VIEWS`) labels are shown at all.  Every plot that draws saved lines reads it through
:func:`saved_line_labels`, so the Three Panel view, the Full Spectrum view
and window, the line grid and the full-spectrum exports all agree.  The GUI
stores the setting in the user settings (see :data:`USER_SETTING_KEYS`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Dict, FrozenSet, Iterable, List, Optional

#: User-settings keys for (column, prefix, suffix, hidden views).
USER_SETTING_KEYS = {
    "column": "saved_line_label_column",
    "prefix": "saved_line_label_prefix",
    "suffix": "saved_line_label_suffix",
    "hidden_views": "saved_line_label_hidden_views",
}

#: Views (by their name in the Views menu) that draw saved lines.  Full-spectrum
#: exports follow "Full Spectrum".
LABEL_VIEWS = ("Three Panel", "Full Spectrum", "Line Grid")

#: Views without labels until the user turns them on.
DEFAULT_HIDDEN_VIEWS: "FrozenSet[str]" = frozenset({"Three Panel"})


@dataclass(frozen=True)
class SavedLineLabelSettings:
    """``column`` ``None`` means the default ``"<species> <line>"`` label.

    ``hidden_views`` names the views (see :data:`LABEL_VIEWS`) whose saved
    lines are drawn without labels.
    """

    column: Optional[str] = None
    prefix: str = ""
    suffix: str = ""
    hidden_views: FrozenSet[str] = DEFAULT_HIDDEN_VIEWS


_settings = SavedLineLabelSettings()


def get_saved_line_label_settings() -> SavedLineLabelSettings:
    return _settings


def set_saved_line_label_settings(
    column: Optional[str] = None,
    prefix: Optional[str] = "",
    suffix: Optional[str] = "",
    hidden_views: Optional[Iterable[str]] = None,
) -> SavedLineLabelSettings:
    """Set the app-wide label; a blank *column* restores the default label.

    *hidden_views* ``None`` keeps the current per-view visibility.
    """
    global _settings
    column = (str(column).strip() or None) if column is not None else None
    hidden = _settings.hidden_views if hidden_views is None else frozenset(map(str, hidden_views))
    _settings = SavedLineLabelSettings(column, str(prefix or ""), str(suffix or ""), hidden)
    return _settings


def reset_saved_line_label_settings() -> SavedLineLabelSettings:
    """Back to the default label and per-view visibility (hidden in :data:`DEFAULT_HIDDEN_VIEWS`)."""
    global _settings
    _settings = SavedLineLabelSettings()
    return _settings


def labels_visible(view: str) -> bool:
    """Whether saved-line labels are shown in *view* (a :data:`LABEL_VIEWS` name)."""
    return view not in _settings.hidden_views


def set_labels_visible(view: str, visible: bool) -> SavedLineLabelSettings:
    """Show or hide saved-line labels in *view*, keeping the rest of the setting."""
    global _settings
    hidden = set(_settings.hidden_views)
    if visible:
        hidden.discard(view)
    else:
        hidden.add(view)
    _settings = replace(_settings, hidden_views=frozenset(hidden))
    return _settings


def settings_from_user_settings(user_settings: Optional[Dict[str, Any]]) -> SavedLineLabelSettings:
    """Apply the label stored in *user_settings* (missing keys mean the default)."""
    us = user_settings if isinstance(user_settings, dict) else {}
    hidden = us.get(USER_SETTING_KEYS["hidden_views"], DEFAULT_HIDDEN_VIEWS)
    if not isinstance(hidden, (list, tuple, set, frozenset)):
        hidden = DEFAULT_HIDDEN_VIEWS
    return set_saved_line_label_settings(
        us.get(USER_SETTING_KEYS["column"]),
        us.get(USER_SETTING_KEYS["prefix"], ""),
        us.get(USER_SETTING_KEYS["suffix"], ""),
        hidden,
    )


def store_in_user_settings(user_settings: Dict[str, Any],
                           settings: Optional[SavedLineLabelSettings] = None) -> None:
    """Write *settings* (default: the current ones) into *user_settings*."""
    s = settings or _settings
    user_settings[USER_SETTING_KEYS["column"]] = s.column
    user_settings[USER_SETTING_KEYS["prefix"]] = s.prefix
    user_settings[USER_SETTING_KEYS["suffix"]] = s.suffix
    user_settings[USER_SETTING_KEYS["hidden_views"]] = sorted(s.hidden_views)


def _cell_text(value: Any) -> str:
    """Display text for one CSV cell: blanks for missing values, compact numbers."""
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return format(value, ".10g")
    text = str(value)
    return "" if text.lower() == "nan" else text


def _column(line_data: Any, name: str) -> Optional[List[Any]]:
    columns = getattr(line_data, "columns", ())
    return list(line_data[name].values) if name in columns else None


def default_label_column_text(line_data: Any, n: int) -> List[str]:
    """The built-in ``"<species> <line>"`` text for each of *n* rows."""
    species = _column(line_data, "species") or [""] * n
    line_ids = _column(line_data, "line") or [""] * n
    return [f"{_cell_text(s)} {_cell_text(l)}".strip() for s, l in zip(species, line_ids)]


def saved_line_labels(line_data: Any,
                      settings: Optional[SavedLineLabelSettings] = None) -> List[str]:
    """One label per row of *line_data* according to *settings* (default: the current ones).

    A chosen column the file does not have falls back to the default label,
    still wrapped in the prefix and suffix.  A line whose text is empty gets
    an empty label (no lone prefix / suffix).
    """
    if line_data is None:
        return []
    n = len(line_data)
    s = settings or _settings
    values = _column(line_data, s.column) if s.column else None
    texts = [_cell_text(v) for v in values] if values is not None else default_label_column_text(line_data, n)
    return [f"{s.prefix}{t}{s.suffix}" if t else "" for t in texts]


def with_overrides(settings: SavedLineLabelSettings, **changes: Any) -> SavedLineLabelSettings:
    """Copy of *settings* with *changes* (for previews that must not touch the app setting)."""
    return replace(settings, **changes)
