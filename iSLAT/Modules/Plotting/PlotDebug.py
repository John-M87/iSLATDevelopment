"""
PlotDebug - the ``debug_config`` logger shared by the plotting modules.

Re-exports :data:`iSLAT.Modules.Debug.debug_config`.  When the Debug module
cannot be imported (for example when the plotting classes are used on their
own), a small stand-in takes its place: it prints warnings and errors and
ignores the other levels.

Usage::

    from .PlotDebug import debug_config
    debug_config.warning("full_spectrum_view", "Something went wrong")
"""
from __future__ import annotations

from typing import Any

try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:  # pragma: no cover - the Debug module ships with iSLAT
    class _FallbackDebugConfig:
        """Minimal stand-in for :class:`iSLAT.Modules.Debug.DebugConfig`."""

        def verbose(self, component: str, message: Any = "", **kwargs: Any) -> None:
            pass

        def info(self, component: str, message: Any = "", **kwargs: Any) -> None:
            pass

        def trace(self, component: str, message: Any = "", **kwargs: Any) -> None:
            pass

        def warning(self, component: str, message: Any = "", **kwargs: Any) -> None:
            print(f"[{str(component).upper()}] WARNING: {message}")

        def error(self, component: str, message: Any = "", **kwargs: Any) -> None:
            print(f"[{str(component).upper()}] ERROR: {message}")

    debug_config = _FallbackDebugConfig()

__all__ = ["debug_config"]
