"""
ControlSurface and ControlBus — the registration backbone for iSLAT UI controls.

A :class:`ControlSurface` is a named UI destination (e.g. the ControlPanel
view-fields section, the TopBar dynamic area, the view-controls section of
the Views menu, the Configure Subplots dialog) that can host
:class:`~iSLAT.Modules.GUI.ControlField.ControlField` instances.

A :class:`ControlBus` owns a registry of surfaces and forwards
register / unregister calls from views to the correct surface(s).

Typical lifecycle
-----------------
1. ``MainPlot`` creates a ``ControlBus``.
2. ``ControlPanel``, ``TopBar``, and ``iSLATNavigationToolbar`` each create
   a concrete :class:`ControlSurface` subclass and register it with the bus::

       bus.register_surface("control_panel", ControlPanelSurface(frame))
       bus.register_surface("top_bar", TopBarSurface(frame))
       bus.register_surface("views_menu", ViewsMenuSurface(views_menu))
       bus.register_surface("configure_subplots", ConfigureSubplotsSurface())

3. When ``FullSpectrumView.activate()`` is called, it registers its fields::

       bus.register(EntryField("n_panels", ..., owner=self), "control_panel")
       bus.register(DisplayRangeField("fsv_range", ..., owner=self), "control_panel")
       bus.register(ToggleField("show_residuals", ..., owner=self), "views_menu")

4. When ``FullSpectrumView.deactivate()`` is called::

       bus.unregister_owner(self)

   which fans out to all surfaces and triggers a ``_rebuild()`` on each one
   that actually had fields removed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .ControlField import ControlField

# ---------------------------------------------------------------------------
# Abstract surface
# ---------------------------------------------------------------------------

class ControlSurface(ABC):
    """Abstract base for a UI area that hosts registered :class:`ControlField` objects.

    Subclasses provide the concrete ``_rebuild()`` implementation that
    translates the current ``_fields`` dict into actual widgets.
    """

    def __init__(self) -> None:
        # Ordered so that fields appear in registration order.
        self._fields: OrderedDict[str, ControlField] = OrderedDict()
        # widget_refs[key] = list of tk.Widget returned by field.build_widget()
        self._widget_refs: Dict[str, list] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, field: ControlField) -> None:
        """Add *field* to this surface and trigger a rebuild.

        If a field with the same *key* is already registered it is replaced.
        """
        self._fields[field.key] = field
        self._rebuild()

    def register_many(self, fields: List[ControlField]) -> None:
        """Add multiple fields in one operation and trigger a single rebuild.

        More efficient than calling :meth:`register` in a loop when adding
        several fields at once (e.g. during a view activation bridge).
        """
        for field in fields:
            self._fields[field.key] = field
        if fields:
            self._rebuild()

    def unregister(self, key: str) -> bool:
        """Remove the field identified by *key* and trigger a rebuild.

        Returns ``True`` if the key was found and removed, ``False`` otherwise.
        """
        if key in self._fields:
            del self._fields[key]
            self._widget_refs.pop(key, None)
            self._rebuild()
            return True
        return False

    def unregister_owner(self, owner: Any) -> bool:
        """Remove all fields whose *owner* attribute equals *owner*.

        Triggers a single rebuild if any fields were removed.  Returns
        ``True`` if at least one field was removed.
        """
        keys_to_remove = [k for k, f in self._fields.items() if f.owner is owner]
        if not keys_to_remove:
            return False
        for k in keys_to_remove:
            del self._fields[k]
            self._widget_refs.pop(k, None)
        self._rebuild()
        return True

    def refresh(self, key: str) -> None:
        """Re-read the current value for field *key* without a full rebuild.

        Calls :meth:`ControlField.refresh` with the stored widget references.
        Does nothing if *key* is not registered.
        """
        field = self._fields.get(key)
        refs = self._widget_refs.get(key)
        if field is not None and refs is not None:
            field.refresh(refs)

    def get_field(self, key: str) -> Optional[ControlField]:
        """Return the registered field for *key*, or ``None``."""
        return self._fields.get(key)

    def fields(self) -> List[ControlField]:
        """Return a snapshot list of all registered fields in order."""
        return list(self._fields.values())

    # ------------------------------------------------------------------
    # Abstract — implemented by concrete surfaces
    # ------------------------------------------------------------------

    @abstractmethod
    def _rebuild(self) -> None:
        """Destroy existing widgets and recreate them from ``_fields``.

        Implementations should populate ``self._widget_refs[field.key]``
        with the list returned by ``field.build_widget(...)``.
        """
        ...

# ---------------------------------------------------------------------------
# ControlBus
# ---------------------------------------------------------------------------

class ControlBus:
    """Registry of named :class:`ControlSurface` instances.

    Views interact with the bus rather than with individual surfaces, keeping
    them decoupled from the concrete UI implementations.
    """

    def __init__(self) -> None:
        self._surfaces: Dict[str, ControlSurface] = {}

    # ------------------------------------------------------------------
    # Surface management
    # ------------------------------------------------------------------

    def register_surface(self, name: str, surface: ControlSurface) -> None:
        """Register *surface* under the given *name*.

        If a surface was previously registered under *name* it is replaced.
        """
        self._surfaces[name] = surface

    def get_surface(self, name: str) -> Optional[ControlSurface]:
        """Return the surface registered as *name*, or ``None``."""
        return self._surfaces.get(name)

    def surface_names(self) -> List[str]:
        """Return the names of all registered surfaces."""
        return list(self._surfaces.keys())

    # ------------------------------------------------------------------
    # Field routing
    # ------------------------------------------------------------------

    def register(self, field: ControlField, surface_name: str) -> None:
        """Register *field* on the surface identified by *surface_name*.

        Silently does nothing if no surface with that name exists, so views
        can register fields speculatively (useful for surfaces that may not
        have been wired up yet in lightweight test environments).
        """
        surface = self._surfaces.get(surface_name)
        if surface is not None:
            surface.register(field)

    def register_many(self, fields: List[ControlField], surface_name: str) -> None:
        """Register multiple fields on a surface in one rebuild cycle."""
        surface = self._surfaces.get(surface_name)
        if surface is not None:
            surface.register_many(fields)

    def unregister(self, key: str, surface_name: str) -> bool:
        """Remove the field *key* from the named surface.

        Returns ``True`` if the field was found and removed.
        """
        surface = self._surfaces.get(surface_name)
        if surface is not None:
            return surface.unregister(key)
        return False

    def unregister_owner(self, owner: Any) -> None:
        """Remove all fields owned by *owner* from **every** surface.

        Each surface triggers its own rebuild only when it had matching
        fields, so surfaces with no matching fields are unaffected.
        """
        for surface in self._surfaces.values():
            surface.unregister_owner(owner)

    def refresh(self, key: str, surface_name: str) -> None:
        """Refresh the field *key* on the named surface without a full rebuild."""
        surface = self._surfaces.get(surface_name)
        if surface is not None:
            surface.refresh(key)