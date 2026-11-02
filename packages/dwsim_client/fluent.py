"""Load DWSIM 10's Fluent API in-process via pythonnet (plan B to the MCP server).

DWSIM is GPL-3.0. This module only *uses* its public API from a separate Python process;
nothing from DWSIM is copied into this repo.

Env:
    DWSIM_BIN    folder with DWSIM.Automation.FluentAPI.dll (default ~/Applications/DWSIM10)
    DOTNET_ROOT  .NET 10 runtime root (default ~/.dotnet)
"""
import json
import os
import sys
import tempfile
from pathlib import Path

DWSIM_BIN = Path(os.environ.get("DWSIM_BIN", Path.home() / "Applications/DWSIM10"))
DOTNET_ROOT = Path(os.environ.get("DOTNET_ROOT", Path.home() / ".dotnet"))

_loaded = False


def load_dwsim():
    """Start coreclr (.NET 10) and reference the Fluent API. Idempotent."""
    global _loaded
    if _loaded:
        return
    from pythonnet import load

    cfg = {
        "runtimeOptions": {
            "tfm": "net10.0",
            "framework": {"name": "Microsoft.NETCore.App", "version": "10.0.0"},
            "rollForward": "LatestMajor",
        }
    }
    cfg_path = Path(tempfile.gettempdir()) / "proceso_ia.runtimeconfig.json"
    cfg_path.write_text(json.dumps(cfg))
    os.environ.setdefault("DOTNET_ROOT", str(DOTNET_ROOT))
    load("coreclr", runtime_config=str(cfg_path), dotnet_root=str(DOTNET_ROOT))

    import clr  # noqa: F401  (available after load)

    sys.path.append(str(DWSIM_BIN))
    clr.AddReference("DWSIM.Automation.FluentAPI")
    _loaded = True


def fluent():
    """Return (Flowsheet, PropertyPackages, Q) from the Fluent API."""
    load_dwsim()
    from DWSIM.Automation.FluentAPI import Flowsheet, PropertyPackages, Q

    return Flowsheet, PropertyPackages, Q
