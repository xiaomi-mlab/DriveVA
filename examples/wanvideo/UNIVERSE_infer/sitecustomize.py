from __future__ import annotations

import sys
from pathlib import Path


# Let script-mode launches from examples/wanvideo/UNIVERSE_infer resolve
# package imports such as examples.wanvideo.UNIVERSE_infer.*.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_REPO_ROOT_STR = str(_REPO_ROOT)

if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

_THIRD_PARTY = _REPO_ROOT / "third_party"
if _THIRD_PARTY.exists():
    _THIRD_PARTY_STR = str(_THIRD_PARTY)
    if _THIRD_PARTY_STR not in sys.path:
        sys.path.insert(0, _THIRD_PARTY_STR)

_NUSCENES_DEVKIT = _REPO_ROOT / "third_party" / "nuscenes-devkit" / "python-sdk"
if _NUSCENES_DEVKIT.exists():
    _NUSCENES_DEVKIT_STR = str(_NUSCENES_DEVKIT)
    if _NUSCENES_DEVKIT_STR not in sys.path:
        sys.path.insert(0, _NUSCENES_DEVKIT_STR)
