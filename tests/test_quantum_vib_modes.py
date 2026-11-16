"""Vibrational-mode extraction from the quantum-state schemas.

The vibrational-band property and filter read mode quantum numbers through
``QuantumStateSchema.vib_modes``; these tests pin which fields count as modes
and the C2H2 global-quanta parsing those modes depend on.
"""
import pytest

import iSLAT.Modules.DataTypes.HITRANQuantumSchemas  # noqa: F401  (registers schemas)
from iSLAT.Modules.DataTypes.QuantumStateSchema import QuantumStateRegistry


def _schema(species):
    return QuantumStateRegistry.get_schema(species)


@pytest.mark.parametrize("label, expected", [
    # Positive l5: every field its own token.
    ("000_0_1_0_1_u|", dict(v4=0, v5=1, l4=0, l5=1, pm="", S="u")),
    # Negative l5 merges with l4, and the +/- tag sticks to l5.
    ("000_1_1_1-1+_u|", dict(v4=1, v5=1, l4=1, l5=-1, pm="+", S="u")),
    ("000_1_1_1-1-_g|", dict(v4=1, v5=1, l4=1, l5=-1, pm="-", S="g")),
    # Negative l4 merges with v5.
    ("000_1_2-1_2_g|P_3e", dict(v4=1, v5=2, l4=-1, l5=2, pm="", S="g")),
    # +/- tag on a positive l5.
    ("000_2_0_0_0+_g|", dict(v4=2, v5=0, l4=0, l5=0, pm="+", S="g")),
    ("201_0_1_0_1_u|", dict(v1=2, v2=0, v3=1, v4=0, v5=1)),
])
def test_c2h2_global_quanta(label, expected):
    parsed = _schema("C2H2").parse_label(label)
    assert {k: parsed[k] for k in expected} == expected


def test_c2h2_unrecognised_layout_leaves_modes_unparsed():
    parsed = _schema("C2H2").parse_label("000_1_2_g|")   # only two integers
    assert parsed["v4"] == -999 and parsed["l5"] == -999
    assert _schema("C2H2").vib_modes("000_1_2_g") is None


@pytest.mark.parametrize("species, modes", [
    ("C2H2", ("v1", "v2", "v3", "v4", "v5")),
    ("CO2", ("v1", "v2", "v3")),          # not l2 or the ranking r
    ("HCN", ("v1", "v2", "v3")),          # not l2
    ("H2O", ("v1", "v2", "v3")),
    ("CH4", ("v1", "v2", "v3", "v4")),    # not n / C
    ("CO", ("v",)),
    ("OH", ("v",)),
    ("CH3OH", ()),                        # explicit free-text band label
])
def test_vib_mode_fields(species, modes):
    assert _schema(species).vib_mode_fields == modes


@pytest.mark.parametrize("species, label, modes", [
    ("C2H2", "200_0_1_0_1_u|", (2, 0, 0, 0, 1)),
    ("CO2", "2_1_1_03|", (2, 1, 0)),
    ("H2O", "0_2_0|39_7_33", (0, 2, 0)),
    ("H2O", "-2-2-2|1_0_1", None),          # HITRAN "unassigned" sentinel
    ("CO", "4|R_2", (4,)),
    ("CH3OH", "GROUND|1_0_1", None),
    (None, "0_1_0|5", (0, 1, 0)),            # generic fallback
    (None, "0_1_0", (0, 1, 0)),              # global part alone
    (None, "000_1_1_1-1+_u|", None),
])
def test_vib_modes(species, label, modes):
    assert _schema(species).vib_modes(label) == modes
