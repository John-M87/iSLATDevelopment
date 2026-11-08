# -*- coding: utf-8 -*-
"""Starting the GUI must not import scipy or lmfit.

Together they add ~2 s to startup.  The code that needs them (line fits,
slab fits, line finding, writing HITRAN line files) imports them on first
use instead.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_STARTUP = """
import sys
import iSLAT.iSLATClass
import iSLAT.Modules.GUI.GUI
import iSLAT.Modules.Plotting.MainPlot
from iSLAT.Modules.DataProcessing.FittingEngine import FittingEngine
from iSLAT.Modules.DataProcessing.LineAnalyzer import LineAnalyzer
FittingEngine(), LineAnalyzer()  # both created by the main plot at startup
print("HEAVY:", sorted({name.split(".")[0] for name in sys.modules} & {"scipy", "lmfit"}))
"""


def test_gui_startup_does_not_import_scipy_or_lmfit():
    # A fresh interpreter: the test process has already imported scipy.
    result = subprocess.run(
        [sys.executable, "-c", _STARTUP],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "HEAVY: []" in result.stdout.splitlines(), result.stdout[-2000:]
