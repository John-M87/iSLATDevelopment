"""
HITRAN-specific quantum state schemas for iSLAT.

Implements :class:`~iSLAT.Modules.DataTypes.QuantumStateSchema.QuantumStateSchema`
subclasses for all molecule classes described in HITRAN2024 supplementary material, Tables S1 (global quanta) and S2 (local quanta).

**Label encoding recap** — every iSLAT ``.par`` upper/lower state label is
built from the HITRAN ``global_*_quanta`` and ``local_*_quanta`` API fields
via::

    "_".join(quanta_field.strip().split())  # collapse whitespace → underscores

followed by concatenation with a ``|`` separator::

    label = global_encoded + "|" + local_encoded

Because HITRAN uses fixed-width fields with no mandatory space between
adjacent columns, some values run together into a single token (e.g. CO₂'s
*v3* and ranking *r*, or H₂O's all-negative vibrational sentinel
``"-2-2-2"``).  Each schema's ``_fixup_*_tokens`` hooks resolve these cases.

Registration
------------
At the end of this module every concrete schema is registered with
:class:`~iSLAT.Modules.DataTypes.QuantumStateSchema.QuantumStateRegistry`.
Import this module once to activate all HITRAN schemas::

    import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # side-effect: registers

References
----------
HITRAN2024 supplementary material: Description of upper-state and lower-state quanta, v.2024.1 (Tables S1, S2).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .QuantumStateSchema import (
    QuantumField,
    QuantumStateRegistry,
    QuantumStateSchema,
    _DelimitedSchema,
    _coerce,
)

__all__: List[str] = []   # concrete classes are not part of the public API;
                           # use QuantumStateRegistry to look them up.

# HITRAN molecule number -> molecule name (the ``id:`` in a .par header).
HITRAN_MOLECULE_NUMBERS: Dict[int, str] = {
    1: 'H2O', 2: 'CO2', 3: 'O3', 4: 'N2O', 5: 'CO', 6: 'CH4', 7: 'O2', 8: 'NO',
    9: 'SO2', 10: 'NO2', 11: 'NH3', 12: 'HNO3', 13: 'OH', 14: 'HF', 15: 'HCl', 16: 'HBr',
    17: 'HI', 18: 'ClO', 19: 'OCS', 20: 'H2CO', 21: 'HOCl', 22: 'N2', 23: 'HCN', 24: 'CH3Cl',
    25: 'H2O2', 26: 'C2H2', 27: 'C2H6', 28: 'PH3', 29: 'COF2', 30: 'SF6', 31: 'H2S', 32: 'HCOOH',
    33: 'HO2', 34: 'O', 35: 'ClONO2', 36: 'NO+', 37: 'HOBr', 38: 'C2H4', 39: 'CH3OH', 40: 'CH3Br',
    41: 'CH3CN', 42: 'CF4', 43: 'C4H2', 44: 'HC3N', 45: 'H2', 46: 'CS', 47: 'SO3', 48: 'C2N2',
    49: 'COCl2', 50: 'SO', 51: 'CH3F', 52: 'GeH4', 53: 'CS2', 54: 'CH3I', 55: 'NF3', 56: 'H3+',
    57: 'CH3', 58: 'S2', 59: 'COFCl', 60: 'HONO', 61: 'ClNO2',
}


def species_for_hitran_number(number: Any) -> Optional[str]:
    """The molecule name for a HITRAN molecule number (``None`` if unknown)."""
    try:
        return HITRAN_MOLECULE_NUMBERS.get(int(number))
    except (TypeError, ValueError):
        return None

# short alias
_QF = QuantumField

# ═══════════════════════════════════════════════════════════════════
#  Shared local field sets  (Table S2 groups)
# ═══════════════════════════════════════════════════════════════════

# Group 1 — Asymmetric rotors (H₂O, O₃, SO₂, NO₂, H₂CO, …)
_LOCAL_G1 = (
    _QF('J',   'int',   'Total angular momentum J'),
    _QF('Ka',  'int',   'Projection on a-axis'),
    _QF('Kc',  'int',   'Projection on c-axis'),
    _QF('F',   'str',   'Hyperfine quantum number'),
    _QF('Sym', 'str',   'Symmetry / parity'),
)

# Group 2a — Closed-shell diatomics and linear molecules (CO, CO₂, HCN, …)
# Upper-state: m | 9X | F'           Lower-state: 5X | Br | J'' | Sym'' | F''
# In practice both upper and lower carry branch + J information (+ optional Sym/F).
_LOCAL_G2A = (
    _QF('Br',  'str',   'Rotational branch (P, Q, R, S, O)'),
    _QF('J',   'int',   'Rotational quantum number J'),
    _QF('Sym', 'str',   'Symmetry / parity (e, f, +, -)'),
    _QF('F',   'str',   'Hyperfine quantum number'),
)

# Group 3 — Spherical rotors (CH₄, SF₆, CF₄, GeH₄)
_LOCAL_G3 = (
    _QF('J',     'int',   'Rotational quantum number J'),
    _QF('C',     'str',   'Symmetry species'),
    _QF('alpha', 'int',   'Multiplicity index α'),
    _QF('F',     'str',   'Hyperfine quantum number'),
)

# Group 4a — Symmetric rotors (CH₃D, CH₃Cl, PH₃, CH₃OH, …)
_LOCAL_G4A = (
    _QF('J',   'int',   'Rotational quantum number J'),
    _QF('K',   'int',   'Projection of J on symmetry axis'),
    _QF('l',   'int',   'Vibrational angular momentum (for degenerate modes)'),
    _QF('C',   'str',   'Symmetry species C'),
    _QF('Sym', 'str',   'State symmetry'),
    _QF('F',   'str',   'Hyperfine quantum number'),
)

# Group 4b — Ammonia (NH₃)
_LOCAL_G4B = (
    _QF('J',         'int',   'Rotational quantum number J'),
    _QF('K',         'int',   'Projection of J'),
    _QF('l',         'str',   'Inversion symmetry (s/a) or vibrational l'),
    _QF('Gamma_rot', 'str',   'Rotational symmetry species'),
    _QF('Gamma_tot', 'str',   'Total symmetry species'),
)

# Group 5a — Planar symmetric molecules (SO₃, CH₃)
_LOCAL_G5A = (
    _QF('J',         'int',   'Rotational quantum number J'),
    _QF('K',         'int',   'Projection of J'),
    _QF('Gamma_rot', 'str',   'Rotational symmetry'),
    _QF('Gamma_tot', 'str',   'Total symmetry'),
)

# Group 6 — Open-shell diatomics with ³Σ ground states (O₂, SO, S₂)
_LOCAL_G6 = (
    _QF('Br_N', 'str',   'Branch for N'),
    _QF('N',    'int',   'Total orbital angular momentum N'),
    _QF('Br_J', 'str',   'Branch for J'),
    _QF('J',    'int',   'Total angular momentum J'),
    _QF('F',    'str',   'Hyperfine quantum number'),
    _QF('M',    'str',   'Magnetic dipole / electric quadrupole flag'),
)

# Group 7a — Open-shell diatomics with ²Π ground states (NO, ClO)
_LOCAL_G7A = (
    _QF('m',   'str',   'Magnetic dipole transition flag'),
    _QF('F',   'str',   'Upper-state hyperfine'),
    _QF('Br',  'str',   'Rotational branch'),
    _QF('J',   'float', 'Rotational quantum number J (half-integer)'),
    _QF('Sym', 'str',   'Symmetry / parity'),
    _QF('Fpp', 'str',   'Lower-state hyperfine'),
)

# Group 7b — OH (special two-char combined symmetry)
_LOCAL_G7B = (
    _QF('Br',      'str',   'Rotational branch (encodes both N and J branch)'),
    _QF('J',       'float', 'Rotational quantum number J (half-integer)'),
    _QF('Sym',     'str',   'Combined upper/lower-state symmetry (e.g. fe, ef)'),
    _QF('F',       'str',   'Hyperfine quantum number'),
)

# ═══════════════════════════════════════════════════════════════════
#  Shared local token fixup helpers
# ═══════════════════════════════════════════════════════════════════

def _split_j_sym(token: str) -> Tuple[str, str]:
    """Split a ``"JSym"`` merged token into ``(J_str, Sym_str)``.

    In HITRAN fixed-width format, the *J* (integer or float) and *Sym*
    (1–2 alpha characters) fields are adjacent with no space, so the
    whitespace-split in :func:`write_line_data` joins them.

    Examples
    --------
    >>> _split_j_sym("29f")
    ('29', 'f')
    >>> _split_j_sym("12.5fe")
    ('12.5', 'fe')
    >>> _split_j_sym("36")
    ('36', '')
    """
    for i in range(len(token) - 1, -1, -1):
        if token[i].isdigit():
            return token[:i + 1], token[i + 1:]
    return '', token

def _fixup_group2a_local(tokens: List[str]) -> List[str]:
    """Resolve *J+Sym* merging in Group 2a lower-state local quanta.

    Typical raw patterns:
    - ``["R", "2"]``        → ``["R", "2", "", ""]``
    - ``["P", "29f"]``      → ``["P", "29", "f", ""]``
    - ``["Q", "36f", "..."] → branch present, J+Sym in second token
    - ``[]``                → empty upper-state local (blank m / F')
    """
    if not tokens:
        return tokens  # upper-state local (empty); leave for sentinel fill

    # Ensure we have at least Br, J, Sym, F slots
    while len(tokens) < 4:
        tokens.append('')

    # Second token might be "JSym" merged (e.g. "29f", "12e")
    if len(tokens) >= 2 and tokens[1]:
        j_str, sym_str = _split_j_sym(tokens[1])
        if sym_str:
            tokens[1] = j_str
            # Insert Sym before any existing Sym slot
            tokens[2] = sym_str + tokens[2]  # prepend to existing (usually '')
    return tokens

def _fixup_group7b_local(tokens: List[str]) -> List[str]:
    """Resolve *J(float)+Sym* merging for OH (Group 7b) local quanta.

    The OH J quantum number is a half-integer (e.g. ``12.5``) and is
    followed by a 2-character symmetry string (e.g. ``"fe"``), making the
    merged token ``"12.5fe"``.
    """
    if not tokens:
        return tokens
    while len(tokens) < 4:
        tokens.append('')
    if len(tokens) >= 2 and tokens[1]:
        j_str, sym_str = _split_j_sym(tokens[1])
        if sym_str:
            tokens[1] = j_str
            tokens[2] = sym_str
    return tokens

# ═══════════════════════════════════════════════════════════════════
#  Class 1a — Simple diatomics
#  Molecules: CO, HF, HCl, HBr, HI, N₂, NO⁺, H₂, CS
#  Global Table S1 Class 1a: 13X  v I2
#  Local  Table S2 Group 2a
# ═══════════════════════════════════════════════════════════════════
class DiatomicSimpleSchema(_DelimitedSchema):
    """Schema for simple closed-shell diatomic molecules (HITRAN Class 1a).

    Molecules: CO, HF, HCl, HBr, HI, N₂, NO⁺, H₂, CS.

    Global field
    ------------
    ``v`` — vibrational quantum number.

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v', 'int', 'Vibrational quantum number'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 1b — Open-shell diatomics (except OH, which is Group 7b)
#  Molecules: O₂, NO, ClO, SO, S₂
#  Global Table S1 Class 1b: 6X X A2  Ω A3  2X  v I2
#  Local  Table S2 Group 6 (O₂, SO, S₂) or Group 7a (NO, ClO)
# ═══════════════════════════════════════════════════════════════════
class DiatomicOpenShell3SigmaSchema(_DelimitedSchema):
    """Schema for ³Σ open-shell diatomics (HITRAN Class 1b, Group 6).

    Molecules: O₂, SO, S₂.

    Global fields
    -------------
    ``X`` — electronic state (e.g. ``"X"``).
    ``Omega`` — projection of total angular momentum (``"1/2"`` or ``"3/2"``).
    ``v`` — vibrational quantum number.

    Local fields (Group 6)
    ----------------------
    ``Br_N``, ``N``, ``Br_J``, ``J``, ``F``, ``M``.
    """

    global_fields = (
        _QF('X',     'str', 'Electronic state designation'),
        _QF('Omega', 'str', 'Projection of total angular momentum Ω'),
        _QF('v',     'int', 'Vibrational quantum number'),
    )
    local_fields = _LOCAL_G6

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        # "X3/2_4" → tokens=["X3/2","4"] — split "X3/2" into X="X", Omega="3/2"
        if tokens and len(tokens[0]) > 1 and tokens[0][0].isalpha():
            first = tokens[0]
            i = 0
            while i < len(first) and first[i].isalpha():
                i += 1
            tokens = [first[:i], first[i:]] + tokens[1:]
        return tokens

class DiatomicOpenShell2PiSchema(_DelimitedSchema):
    """Schema for ²Π open-shell diatomics (HITRAN Class 1b, Group 7a).

    Molecules: NO, ClO.

    Global fields
    -------------
    ``X``, ``Omega``, ``v``.

    Local fields (Group 7a)
    -----------------------
    ``m``, ``F``, ``Br``, ``J``, ``Sym``, ``Fpp``.
    """

    global_fields = (
        _QF('X',     'str', 'Electronic state'),
        _QF('Omega', 'str', 'Projection of total angular momentum Ω'),
        _QF('v',     'int', 'Vibrational quantum number'),
    )
    local_fields = _LOCAL_G7A

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        if tokens and len(tokens[0]) > 1 and tokens[0][0].isalpha():
            first = tokens[0]
            i = 0
            while i < len(first) and first[i].isalpha():
                i += 1
            tokens = [first[:i], first[i:]] + tokens[1:]
        return tokens

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

class OHSchema(_DelimitedSchema):
    """Schema for OH (HITRAN Class 1b, Group 7b).

    OH uses a 2-character combined symmetry field in the lower-state local
    quanta (e.g. ``"fe"``), and the branch encodes both N and J transitions.

    Global fields
    -------------
    ``X``, ``Omega``, ``v``.

    Local fields (Group 7b)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('X',     'str', 'Electronic state'),
        _QF('Omega', 'str', 'Projection of total angular momentum Ω'),
        _QF('v',     'int', 'Vibrational quantum number'),
    )
    local_fields = _LOCAL_G7B

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        if tokens and len(tokens[0]) > 1 and tokens[0][0].isalpha():
            first = tokens[0]
            i = 0
            while i < len(first) and first[i].isalpha():
                i += 1
            tokens = [first[:i], first[i:]] + tokens[1:]
        return tokens

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group7b_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 2a — CO₂
#  Global: 6X v1 I2  v2 I2  l2 I2  v3 I2  r I1
#  Local:  Group 2a  (m + F' for upper; 5X Br J Sym F for lower)
# ═══════════════════════════════════════════════════════════════════
class CO2Schema(_DelimitedSchema):
    """Schema for CO₂ (HITRAN Class 2a).

    Global fields
    -------------
    ``v1``, ``v2``, ``l2``, ``v3``, ``r`` (Fermi resonance ranking).

    .. note::
       The *v3* (I2) and *r* (I1) HITRAN fields are adjacent with no mandatory
       separator.  When both appear without whitespace between them the
       ``_``-split produces a merged token (e.g. ``"03"`` for v3=0, r=3).
       :meth:`_fixup_global_tokens` handles this case.

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'Symmetric stretching mode'),
        _QF('v2', 'int', 'Bending mode'),
        _QF('l2', 'int', 'Vibrational angular momentum of bending'),
        _QF('v3', 'int', 'Anti-symmetric stretching mode'),
        _QF('r',  'int', 'Fermi resonance ranking index'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        # 5 fields expected. When v3 (I2, e.g. " 0") and r (I1, e.g. "3")
        # are both small and adjacent the whitespace-split produces "03" as
        # one token instead of two → 4 tokens total.
        if len(tokens) == 4 and len(tokens[-1]) >= 2:
            last = tokens[-1]
            tokens = tokens[:-1] + [last[:-1].lstrip('0') or '0', last[-1]]
        return tokens

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 2b — Other linear triatomics
#  Molecules: N₂O, OCS, HCN, CS₂
#  Global: 7X v1 I2  v2 I2  l2 I2  v3 I2
#  Local:  Group 2a
# ═══════════════════════════════════════════════════════════════════
class LinearTriatomicSchema(_DelimitedSchema):
    """Schema for linear triatomic molecules except CO₂ (HITRAN Class 2b).

    Molecules: N₂O, OCS, HCN, CS₂.

    Global fields
    -------------
    ``v1``, ``v2``, ``l2``, ``v3``.

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'Symmetric stretching mode'),
        _QF('v2', 'int', 'Bending mode'),
        _QF('l2', 'int', 'Vibrational angular momentum'),
        _QF('v3', 'int', 'Anti-symmetric stretching mode'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 3 — Non-linear triatomic molecules
#  Molecules: H₂O, O₃, SO₂, NO₂, HOCl, H₂S, HO₂, HOBr
#  Global: 9X v1 I2  v2 I2  v3 I2
#  Local:  Group 1 (J Ka Kc F Sym)
# ═══════════════════════════════════════════════════════════════════
class NonLinearTriatomicSchema(_DelimitedSchema):
    """Schema for non-linear triatomic molecules (HITRAN Class 3).

    Molecules: H₂O, O₃, SO₂, NO₂, HOCl, H₂S, HO₂, HOBr.

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``.

    .. note::
       When all three vibrational quanta are the missing-assignment sentinel
       ``-2``, the HITRAN fixed-width format produces the merged token
       ``"-2-2-2"`` (no spaces between adjacent 2-char fields).
       :meth:`_fixup_global_tokens` handles this.

    Local fields (Group 1)
    ----------------------
    ``J``, ``Ka``, ``Kc``, ``F``, ``Sym``.
    """

    global_fields = (
        _QF('v1', 'int', 'First normal mode'),
        _QF('v2', 'int', 'Second normal mode (bending)'),
        _QF('v3', 'int', 'Third normal mode'),
    )
    local_fields = _LOCAL_G1

    _PACKED_NEGATIVE_RE = re.compile(r'^[-\d]+$')

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        n = len(self.global_fields)  # 3
        if len(tokens) == 1:
            s = tokens[0]
            # Handle all-negative packed case, e.g. "-2-2-2" (3 × I2 = 6 chars)
            if len(s) == n * 2 and self._PACKED_NEGATIVE_RE.match(s):
                return [s[i * 2:(i + 1) * 2].strip() for i in range(n)]
        return tokens

# ═══════════════════════════════════════════════════════════════════
#  Class 4a — Pyramidal tetratomic molecules (simple variant)
#  Molecules: PH₃, NF₃
#  Global: 5X v1 I2  v2 I2  v3 I2  v4 I2  S A2
#  Local:  Group 4a
# ═══════════════════════════════════════════════════════════════════
class PyramidalTetraatomicSchema(_DelimitedSchema):
    """Schema for pyramidal tetratomic molecules (HITRAN Class 4a).

    Molecules: PH₃, NF₃.

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``v4``, ``S`` (symmetry).

    Local fields (Group 4a)
    -----------------------
    ``J``, ``K``, ``l``, ``C``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'First normal mode'),
        _QF('v2', 'int', 'Second normal mode'),
        _QF('v3', 'int', 'Third normal mode (degenerate)'),
        _QF('v4', 'int', 'Fourth normal mode (degenerate)'),
        _QF('S',  'str', 'Vibrational symmetry species'),
    )
    local_fields = _LOCAL_G4A

# ═══════════════════════════════════════════════════════════════════
#  Class 4b — Ammonia  (NH₃)
#  Global: 1X v1 I1 v2 I1 v3 I1 v4 I1  1X l3 I1 l4 I1  1X l I1  1X Γvib A4
#  Local:  Group 4b
# ═══════════════════════════════════════════════════════════════════
class NH3Schema(_DelimitedSchema):
    """Schema for ammonia isotopologues (HITRAN Class 4b).

    Molecules: ¹⁴NH₃, ¹⁵NH₃.

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``v4``, ``l3``, ``l4``, ``l``, ``Gamma_vib``.

    The HITRAN fixed-width format packs v1–v4 into 4 consecutive I1 (1-char)
    fields without separating spaces (e.g. ``"0001"`` for v4=1, v1=v2=v3=0).
    Similarly l3 and l4 share a 2-char block (e.g. ``"01"``).

    Local fields (Group 4b)
    -----------------------
    ``J``, ``K``, ``l``, ``Gamma_rot``, ``Gamma_tot``.
    """

    global_fields = (
        _QF('v1',        'int', 'Symmetric N–H stretching mode'),
        _QF('v2',        'int', 'Symmetric deformation (umbrella) mode'),
        _QF('v3',        'int', 'Anti-symmetric N–H stretching mode (degenerate)'),
        _QF('v4',        'int', 'Anti-symmetric deformation mode (degenerate)'),
        _QF('l3',        'int', 'Vibrational angular momentum of v3'),
        _QF('l4',        'int', 'Vibrational angular momentum of v4'),
        _QF('l',         'int', 'Total vibrational angular momentum |l3+l4|'),
        _QF('Gamma_vib', 'str', 'Vibrational symmetry species'),
    )
    local_fields = _LOCAL_G4B

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        # Expected: ["v1v2v3v4", "l3l4", "l", "Gamma_vib"]  (4 tokens, 8 fields)
        # Unpack first token (4 × I1 = 4 chars)
        if not tokens:
            return tokens
        result: List[str] = []
        first = tokens[0]
        if len(first) >= 4 and first.isdigit():
            result.extend(list(first[:4]))  # v1, v2, v3, v4 as single chars
        else:
            result.extend([first] + [''] * 3)
        # Unpack second token (2 × I1 = 2 chars)
        if len(tokens) > 1:
            second = tokens[1]
            if len(second) >= 2 and second.isdigit():
                result.extend(list(second[:2]))  # l3, l4
            else:
                result.extend([second, ''])
        else:
            result.extend(['', ''])
        # Remaining tokens: l and Gamma_vib
        result.extend(tokens[2:])
        return result

# ═══════════════════════════════════════════════════════════════════
#  Class 5a — Acetylene (C₂H₂)
#  Global: 1X v1 I1 v2 I1 v3 I1  v4 I2 v5 I2  l4 I2 l5 I2  ± A1  1X  S A1
#  Local:  Group 2a
# ═══════════════════════════════════════════════════════════════════

#: One (possibly negative) integer field in C₂H₂ global quanta.
_C2H2_SIGNED_INT_RE = re.compile(r'-?\d+')
class C2H2Schema(_DelimitedSchema):
    """Schema for acetylene C₂H₂ (HITRAN Class 5a).

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``v4``, ``v5``, ``l4``, ``l5``, ``pm``, ``S``.

    HITRAN packs v1, v2, v3 into three adjacent I1 fields without separators
    (e.g. ``"000"``).  The l4, l5, and ± (pm) fields can also merge (e.g.
    ``"1-1+"`` for l4=1, l5=-1, pm=+).

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'C–H symmetric stretching'),
        _QF('v2', 'int', 'C≡C stretching'),
        _QF('v3', 'int', 'C–H anti-symmetric stretching'),
        _QF('v4', 'int', 'Trans bending mode (degenerate)'),
        _QF('v5', 'int', 'Cis bending mode (degenerate)'),
        _QF('l4', 'int', 'Vibrational angular momentum l4'),
        _QF('l5', 'int', 'Vibrational angular momentum l5'),
        _QF('pm', 'str', 'Vibrational parity of Σ states (+ or -)'),
        _QF('S',  'str', 'Parity (u/g) of vibrational level'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        # Field order: v1 v2 v3 (packed "000"), v4 v5 l4 l5 (I2 each), then
        # an optional ± glued to l5, then S.  A negative I2 value has no
        # leading blank, so it merges with the field before it and the token
        # count varies: "000_1_2-1_2_g" is v4=1 v5=2 l4=-1 l5=2, and
        # "000_1_1_1-1+_u" is v4=1 v5=1 l4=1 l5=-1 ±=+.  Splitting the
        # remainder into signed integers recovers every field regardless.
        if not tokens:
            return tokens
        result: List[str] = []

        # First token: v1v2v3 packed as 3 × I1
        first = tokens[0]
        if len(first) == 3 and first.isdigit():
            result.extend(list(first))       # v1, v2, v3
        else:
            result.extend([first, '', ''])

        rest = tokens[1:]
        sym = rest.pop() if rest and rest[-1].isalpha() else ''
        body = '_'.join(rest)
        pm = ''
        if len(body) >= 2 and body[-1] in '+-' and body[-2].isdigit():
            body, pm = body[:-1], body[-1]
        numbers = _C2H2_SIGNED_INT_RE.findall(body)
        if len(numbers) == 4 and _C2H2_SIGNED_INT_RE.sub('', body).strip('_') == '':
            result.extend(numbers)           # v4, v5, l4, l5
        else:
            # Unrecognised layout: leave v4..l5 unparsed rather than guess.
            result.extend(['', '', '', ''])
        result.extend([pm, sym])
        return result

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 5b — C₄H₂
#  Global: 1X v1..v9 I1 (9 packed digits)  1X Sym A1  1X S A2
#  Local:  Group 2b (l6..l9 vibrational + Br J Sym)
# ═══════════════════════════════════════════════════════════════════
class C4H2Schema(QuantumStateSchema):
    """Schema for diacetylene C₄H₂ (HITRAN Class 5b, Group 2b).

    Unlike most molecules, C₄H₂ shares vibrational bending mode quantum
    numbers (l6, l7, l8, l9) between the global and local quanta fields.

    Global fields
    -------------
    ``v1``–``v9``, ``Sym``, ``S``.

    The nine vibrational modes are packed as nine single-character digits in
    the fixed-width HITRAN format.

    Local fields (Group 2b)
    -----------------------
    ``l6``, ``l7``, ``l8``, ``l9``, ``Br``, ``J``, ``Sym``.
    """

    global_fields = (
        _QF('v1',  'int', 'Normal mode 1'),
        _QF('v2',  'int', 'Normal mode 2'),
        _QF('v3',  'int', 'Normal mode 3'),
        _QF('v4',  'int', 'Normal mode 4'),
        _QF('v5',  'int', 'Normal mode 5'),
        _QF('v6',  'int', 'Normal mode 6'),
        _QF('v7',  'int', 'Normal mode 7'),
        _QF('v8',  'int', 'Normal mode 8'),
        _QF('v9',  'int', 'Normal mode 9'),
        _QF('Sym', 'str', 'e/f symmetry'),
        _QF('S',   'str', 'u/g parity'),
    )
    local_fields = (
        _QF('l6',  'str', 'Bending angular momentum l6'),
        _QF('l7',  'str', 'Bending angular momentum l7'),
        _QF('l8',  'str', 'Bending angular momentum l8'),
        _QF('l9',  'str', 'Bending angular momentum l9'),
        _QF('Br',  'str', 'Rotational branch'),
        _QF('J',   'int', 'Rotational quantum number J'),
        _QF('Sym', 'str', 'Symmetry / parity'),
    )

    def parse_label(self, label: str) -> Dict[str, Any]:
        parts = label.split('|', 1)
        global_str = parts[0] if parts else ''
        local_str = parts[1] if len(parts) > 1 else ''

        g_tokens = global_str.split('_') if global_str else []
        l_tokens = local_str.split('_') if local_str else []

        result: Dict[str, Any] = {}

        # Global: first token is 9 packed single-char vibrational quanta
        v_str = g_tokens[0] if g_tokens else ''
        for i in range(9):
            field = self.global_fields[i]
            ch = v_str[i] if i < len(v_str) else ''
            result[field.name] = _coerce(ch, field.dtype)
        # Sym and S
        result['Sym'] = _coerce(g_tokens[1] if len(g_tokens) > 1 else '', 'str')
        result['S']   = _coerce(g_tokens[2] if len(g_tokens) > 2 else '', 'str')

        # Local: l6, l7, l8, l9 then optional Br, J, Sym
        local_field_names = [f.name for f in self.local_fields]
        for i, fname in enumerate(local_field_names):
            dtype = self.local_fields[i].dtype
            raw = l_tokens[i] if i < len(l_tokens) else ''
            result[fname] = _coerce(raw, dtype)

        return result

# ═══════════════════════════════════════════════════════════════════
#  Class 5c — HC₃N
#  Global: 2X v1..v7 I1  l5 I2  l6 I2  l7 I2
#  Local:  Group 2a
# ═══════════════════════════════════════════════════════════════════
class HC3NSchema(_DelimitedSchema):
    """Schema for cyanoacetylene HC₃N (HITRAN Class 5c).

    Global fields
    -------------
    ``v1``–``v7``, ``l5``, ``l6``, ``l7``.

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'Normal mode 1'),
        _QF('v2', 'int', 'Normal mode 2'),
        _QF('v3', 'int', 'Normal mode 3'),
        _QF('v4', 'int', 'Normal mode 4'),
        _QF('v5', 'int', 'Normal mode 5 (degenerate bending)'),
        _QF('v6', 'int', 'Normal mode 6 (degenerate bending)'),
        _QF('v7', 'int', 'Normal mode 7 (degenerate bending)'),
        _QF('l5', 'int', 'Vibrational angular momentum l5'),
        _QF('l6', 'int', 'Vibrational angular momentum l6'),
        _QF('l7', 'int', 'Vibrational angular momentum l7'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 5d — C₂N₂ (using Plíva notation)
#  Global: v1 I2 v2 I2 v3 I2 v4 I2 v5 I2 l I2 ± A1 r I1 S A1
#  Local:  Group 2a
# ═══════════════════════════════════════════════════════════════════
class C2N2Schema(_DelimitedSchema):
    """Schema for cyanogen C₂N₂ (HITRAN Class 5d, Plíva notation).

    Global fields
    -------------
    ``v1``–``v5``, ``l``, ``pm``, ``r``, ``S``.

    Local fields (Group 2a)
    -----------------------
    ``Br``, ``J``, ``Sym``, ``F``.
    """

    global_fields = (
        _QF('v1', 'int', 'Normal mode 1'),
        _QF('v2', 'int', 'Normal mode 2'),
        _QF('v3', 'int', 'Normal mode 3'),
        _QF('v4', 'int', 'Normal mode 4 (degenerate)'),
        _QF('v5', 'int', 'Normal mode 5 (degenerate)'),
        _QF('l',  'int', 'Vibrational angular momentum |l3+l4|'),
        _QF('pm', 'str', '± symmetry of Σ states'),
        _QF('r',  'int', 'Ranking number'),
        _QF('S',  'str', 'Vibrational symmetry'),
    )
    local_fields = _LOCAL_G2A

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        return _fixup_group2a_local(tokens)

# ═══════════════════════════════════════════════════════════════════
#  Class 6a — Asymmetric top (6 normal modes)
#  Molecules: H₂CO, COF₂, COCl₂
#  Global: 3X v1 I2  v2 I2  v3 I2  v4 I2  v5 I2  v6 I2
#  Local:  Group 1
# ═══════════════════════════════════════════════════════════════════
class AsymTopSixModeSchema(_DelimitedSchema):
    """Schema for asymmetric-top molecules with six normal modes (Class 6a).

    Molecules: H₂CO, COF₂, COCl₂.

    Global fields
    -------------
    ``v1``–``v6``.

    Local fields (Group 1)
    ----------------------
    ``J``, ``Ka``, ``Kc``, ``F``, ``Sym``.
    """

    global_fields = (
        _QF('v1', 'int', 'Normal mode 1'),
        _QF('v2', 'int', 'Normal mode 2'),
        _QF('v3', 'int', 'Normal mode 3'),
        _QF('v4', 'int', 'Normal mode 4'),
        _QF('v5', 'int', 'Normal mode 5'),
        _QF('v6', 'int', 'Normal mode 6'),
    )
    local_fields = _LOCAL_G1

# ═══════════════════════════════════════════════════════════════════
#  Class 6b — H₂O₂
#  Global: 3X v1 I2 v2 I2 v3 I2 n I1 τ I1 v5 I2 v6 I2
#  Local:  Group 1
# ═══════════════════════════════════════════════════════════════════
class H2O2Schema(_DelimitedSchema):
    """Schema for hydrogen peroxide H₂O₂ (HITRAN Class 6b).

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``n``, ``tau``, ``v5``, ``v6``.

    Local fields (Group 1)
    ----------------------
    ``J``, ``Ka``, ``Kc``, ``F``, ``Sym``.
    """

    global_fields = (
        _QF('v1',  'int', 'O–H stretching (symmetric)'),
        _QF('v2',  'int', 'O–O stretching'),
        _QF('v3',  'int', 'O–H stretching (anti-symmetric)'),
        _QF('n',   'int', 'Torsional quantum number n'),
        _QF('tau', 'int', 'Torsional symmetry τ'),
        _QF('v5',  'int', 'H–O–O bending (symmetric)'),
        _QF('v6',  'int', 'H–O–O bending (anti-symmetric)'),
    )
    local_fields = _LOCAL_G1

# ═══════════════════════════════════════════════════════════════════
#  Class 7a — SO₃, CH₃ (planar symmetric)
#  Global: v1 I2 v2 I2 v3 I2 l3 I2 v4 I2 l4 I2 Γvib A3
#  Local:  Group 5a
# ═══════════════════════════════════════════════════════════════════
class SO3Schema(_DelimitedSchema):
    """Schema for SO₃ and CH₃ (HITRAN Class 7a, Group 5a).

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``l3``, ``v4``, ``l4``, ``Gamma_vib``.

    Local fields (Group 5a)
    -----------------------
    ``J``, ``K``, ``Gamma_rot``, ``Gamma_tot``.
    """

    global_fields = (
        _QF('v1',        'int', 'Normal mode 1 (A1, symmetric stretching)'),
        _QF('v2',        'int', 'Normal mode 2 (A2, out-of-plane)'),
        _QF('v3',        'int', 'Normal mode 3 (E, degenerate stretching)'),
        _QF('l3',        'int', 'Vibrational angular momentum l3'),
        _QF('v4',        'int', 'Normal mode 4 (E, degenerate bending)'),
        _QF('l4',        'int', 'Vibrational angular momentum l4'),
        _QF('Gamma_vib', 'str', 'Vibrational symmetry species'),
    )
    local_fields = _LOCAL_G5A

# ═══════════════════════════════════════════════════════════════════
#  Class 8 — Spherical top molecules
#  Molecules: ¹²CH₄, ¹³CH₄, CF₄, GeH₄
#  Global: 3X v1 I2 v2 I2 v3 I2 v4 I2 n A2 C A2
#  Local:  Group 3
# ═══════════════════════════════════════════════════════════════════
class SphericalTopSchema(_DelimitedSchema):
    """Schema for spherical-top molecules (HITRAN Class 8, Group 3).

    Molecules: ¹²CH₄, ¹³CH₄, CF₄, GeH₄.

    Global fields
    -------------
    ``v1``, ``v2``, ``v3``, ``v4``, ``n``, ``C``.

    .. note::
       The *n* (A2, 2-char) and *C* (A2, 2-char) HITRAN fields may merge into
       a single token when *n* is a digit (e.g. ``"1E"`` for n=1, C=E).
       :meth:`_fixup_global_tokens` handles this.

    Local fields (Group 3)
    ----------------------
    ``J``, ``C``, ``alpha``, ``F``.

    .. note::
       In the local quanta, *J* and *C* are often merged (e.g. ``"13F1"``
       for J=13, C=F1).  :meth:`_fixup_local_tokens` handles this.
    """

    global_fields = (
        _QF('v1', 'int', 'Normal mode 1 (A1)'),
        _QF('v2', 'int', 'Normal mode 2 (E)'),
        _QF('v3', 'int', 'Normal mode 3 (F2, triply degenerate)'),
        _QF('v4', 'int', 'Normal mode 4 (F2, triply degenerate)'),
        _QF('n',  'str', 'Multiplicity index n'),
        _QF('C',  'str', 'Symmetry species C'),
    )
    local_fields = _LOCAL_G3

    def _fixup_global_tokens(self, tokens: List[str]) -> List[str]:
        # 6 fields expected. If last token is "nC" merged (digit(s) + letters):
        # e.g. "1E" → n="1", C="E";  "2F2" → n="2", C="F2"
        if len(tokens) == 5:
            last = tokens[-1]
            # Find transition from digits to letters
            i = 0
            while i < len(last) and (last[i].isdigit() or last[i] in '-+'):
                i += 1
            if 0 < i < len(last):
                tokens = tokens[:-1] + [last[:i], last[i:]]
        return tokens

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        # "JC" merged: e.g. "13F1" → J="13", C="F1"; then alpha, then F
        if not tokens:
            return tokens
        while len(tokens) < 4:
            tokens.append('')
        first = tokens[0]
        # Find transition from leading digits to trailing alpha
        i = 0
        while i < len(first) and first[i].isdigit():
            i += 1
        if 0 < i < len(first):
            tokens = [first[:i], first[i:]] + tokens[1:]
        return tokens

# ═══════════════════════════════════════════════════════════════════
#  Class 9 — Explicit band molecules
#  Molecules: CH₃D, HNO₃, CH₃Cl, C₂H₆, SF₆, HCOOH, ClONO₂,
#             C₂H₄, CH₃OH, CH₃Br, CH₃CN, CH₃F, CH₃I, COFCl, HONO, ClNO₂
#  Global: free-text vibrational band label (up to 15 chars right-aligned)
#  Local:  molecule-specific (asymmetric rotor or symmetric rotor)
# ═══════════════════════════════════════════════════════════════════
class ExplicitBandSchema(_DelimitedSchema):
    """Base schema for Class 9 molecules that use explicit band labels.

    The *global* quanta field contains a free-text vibrational band
    description (e.g. ``"V7"``, ``"GROUND"``, ``"3V4"``, ``"V3+V4"``).  It
    is stored verbatim in the single ``"band"`` field.

    The *local* quanta defaults to asymmetric-rotor (J, Ka, Kc) format,
    which covers the majority of Class 9 molecules.  Subclasses can override
    :attr:`local_fields` for molecules with different rotational structures.

    Global fields
    -------------
    ``band`` — raw vibrational band label.

    Local fields (Group 1, asymmetric rotor)
    -----------------------------------------
    ``J``, ``Ka``, ``Kc``, ``F``, ``Sym``.
    """

    global_fields = (
        _QF('band', 'str', 'Vibrational band label (free text)'),
    )
    local_fields = _LOCAL_G1

class ExplicitBandSymRotorSchema(ExplicitBandSchema):
    """Class 9 molecules with symmetric-rotor local quanta (C₂H₆, C₂H₆-like).

    Local fields (Group 4a)
    -----------------------
    ``J``, ``K``, ``l``, ``C``, ``Sym``, ``F``.
    """

    local_fields = _LOCAL_G4A

    def _fixup_local_tokens(self, tokens: List[str]) -> List[str]:
        # "0A34" → l=0, Sym="A34" (packed l I2 + Sym A3 with leading space stripped)
        if not tokens:
            return tokens
        while len(tokens) < 6:
            tokens.append('')
        # Third token may be "lSym" merged (digit prefix + alpha suffix)
        if len(tokens) >= 3 and tokens[2]:
            tok = tokens[2]
            i = 0
            while i < len(tok) and tok[i].isdigit():
                i += 1
            if 0 < i < len(tok):
                tokens = tokens[:2] + [tok[:i], tok[i:]] + tokens[3:]
        return tokens

# ═══════════════════════════════════════════════════════════════════
#  Additional asymmetric-top molecules not in Class 3 or 6a
#  (larger asymmetric tops with Group 1 local quanta)
#  Molecules: HCOOH, HOCl, H₂O₂ (already has dedicated schema),
#             ClONO₂, HOBr, C₂H₄, COFCl, HONO, ClNO₂, HNO₃
# ═══════════════════════════════════════════════════════════════════
class LargeAsymTopSchema(_DelimitedSchema):
    """Schema for larger asymmetric-top molecules (Group 1 local quanta).

    Used for molecules not covered by a more specific schema but that use
    the standard asymmetric-rotor local format (J, Ka, Kc, F, Sym).

    Global fields
    -------------
    ``band`` — raw global quanta string (stored verbatim).

    Local fields (Group 1)
    ----------------------
    ``J``, ``Ka``, ``Kc``, ``F``, ``Sym``.
    """

    global_fields = (
        _QF('band', 'str', 'Global quanta (vibrational / band label)'),
    )
    local_fields = _LOCAL_G1

# ═══════════════════════════════════════════════════════════════════
#  Registration block
# ═══════════════════════════════════════════════════════════════════
# This block runs exactly once when the module is first imported and registers all HITRAN molecule schemas with QuantumStateRegistry.
def _register_all() -> None:
    """Register all HITRAN schemas with :class:`QuantumStateRegistry`."""
    reg = QuantumStateRegistry.register

    # ── Class 1a: Simple diatomics ───────────────────────────────────
    _dia = DiatomicSimpleSchema()
    for mol in ('CO', 'HF', 'HCl', 'HBr', 'HI', 'N2', 'NO+', 'H2', 'CS'):
        reg(mol, _dia)

    # ── Class 1b: Open-shell diatomics ───────────────────────────────
    _3sig = DiatomicOpenShell3SigmaSchema()
    for mol in ('O2', 'SO', 'S2'):
        reg(mol, _3sig)

    _2pi = DiatomicOpenShell2PiSchema()
    for mol in ('NO', 'ClO'):
        reg(mol, _2pi)

    reg('OH', OHSchema())

    # ── Class 2a: CO₂ ────────────────────────────────────────────────
    reg('CO2', CO2Schema())

    # ── Class 2b: Other linear triatomics ────────────────────────────
    _lin3 = LinearTriatomicSchema()
    for mol in ('N2O', 'OCS', 'HCN', 'CS2'):
        reg(mol, _lin3)

    # ── Class 3: Non-linear triatomics ───────────────────────────────
    _tri3 = NonLinearTriatomicSchema()
    for mol in ('H2O', 'O3', 'SO2', 'NO2', 'HOCl', 'H2S', 'HO2', 'HOBr'):
        reg(mol, _tri3)

    # ── Class 4a: Pyramidal tetratomics (simple) ─────────────────────
    _pyr = PyramidalTetraatomicSchema()
    for mol in ('PH3', 'NF3'):
        reg(mol, _pyr)

    # ── Class 4b: Ammonia ────────────────────────────────────────────
    _nh3 = NH3Schema()
    for mol in ('NH3', '14NH3', '15NH3'):
        reg(mol, _nh3)

    # ── Class 5a: Acetylene ──────────────────────────────────────────
    reg('C2H2', C2H2Schema())

    # ── Class 5b: C₄H₂ ──────────────────────────────────────────────
    reg('C4H2', C4H2Schema())

    # ── Class 5c: HC₃N ──────────────────────────────────────────────
    reg('HC3N', HC3NSchema())

    # ── Class 5d: C₂N₂ ──────────────────────────────────────────────
    reg('C2N2', C2N2Schema())

    # ── Class 6a: Asymmetric tops (6 modes) ──────────────────────────
    _asym6 = AsymTopSixModeSchema()
    for mol in ('H2CO', 'COF2', 'COCl2'):
        reg(mol, _asym6)

    # ── Class 6b: H₂O₂ ───────────────────────────────────────────────
    reg('H2O2', H2O2Schema())

    # ── Class 7a: SO₃, CH₃ ──────────────────────────────────────────
    _so3 = SO3Schema()
    for mol in ('SO3', 'CH3'):
        reg(mol, _so3)

    # ── Class 8: Spherical tops ──────────────────────────────────────
    _sph = SphericalTopSchema()
    for mol in ('CH4', '12CH4', '13CH4', 'CF4', 'GeH4', 'SF6'):
        reg(mol, _sph)

    # ── Class 9: Explicit-band asymmetric tops ───────────────────────
    _exp = ExplicitBandSchema()
    for mol in ('CH3D', 'HNO3', 'HCOOH', 'ClONO2', 'CH3OH',
                'CH3CN', 'CH3F', 'CH3I', 'COFCl', 'HONO', 'ClNO2'):
        reg(mol, _exp)

    # Class 9 asymmetric-top molecules that appear in the iSLAT HITRAN files:
    _c2h4 = ExplicitBandSchema()   # local: J Ka Kc (Group 1 asymmetric rotor)
    reg('C2H4', _c2h4)

    # C₂H₆ and CH₃Cl use symmetric-rotor local quanta
    _sym_rot = ExplicitBandSymRotorSchema()
    for mol in ('C2H6', 'CH3Cl', 'CH3Br'):
        reg(mol, _sym_rot)

_register_all()