# plato-midi-bridge Recovery Manifest

## Damage Summary
**22 orphan .pyc files found** — Python source code was deleted from the main branch.
These files exist only as compiled bytecode artifacts (`.pyc` files).

## Deleted Files (Not Recoverable from Source — only .pyc bytecode remains)

### Module: `plato_midi_bridge/autopilot/` (5 files)
- `__init__.py`
- `train.py`
- `wheel.py`
- `room.py`
- `dataset.py`

### Module: `plato_midi_bridge/jepa/` (6 files)
- `__init__.py`
- `loss.py`
- `predictor.py`
- `train.py`
- `equilibrium.py`
- `room.py`

### Module: `flux_modules/` (7 files)
- `flux_adaptive.py`
- `flux_eigenstyle.py`
- `flux_penrose.py`
- `flux_encoder.py`
- `flux_coupling.py`
- `flux_provenance.py`
- `test_flux_modules.py`

### Tests (4 files)
- `test_jepa.py`
- `test_autopilot.py`

## Recovered Files

### From published source tarball (RECOVERED-FROM-TARBALL/)
- `test_decompose.py` — ✅ Full source recovered
- `test_torch_bridge.py` — ✅ Full source recovered

## Still Missing (20 files)
The following source files are **lost from GitHub** and only exist as .pyc bytecode:
- 5 autopilot source files
- 6 jepa source files
- 7 flux_modules source files
- 2 test files (test_jepa, test_autopilot)

These were NOT included in any published PyPI package. They were likely deleted during a force push that rewrote git history.

## Action Items
1. ✅ `.gitignore` updated to block `__pycache__/`, `*.pyc`, `dist/`, `*.whl`
2. ✅ CI workflow added to fail on blocked files
3. These source files need to be recreated from scratch

## Decompilation Attempt
All 22 orphan .pyc files are **Python 3.10 bytecode** — standard decompilers (uncompyle6)
cannot decompile 3.10. The bytecode files contain:

| File | Size (bytes) |
|------|-------------|
| plato_midi_bridge/autopilot/train.py | 15,843 |
| plato_midi_bridge/autopilot/__init__.py | small module init |
| plato_midi_bridge/autopilot/wheel.py | moderate |
| plato_midi_bridge/autopilot/room.py | moderate |
| plato_midi_bridge/autopilot/dataset.py | moderate |
| plato_midi_bridge/jepa/loss.py | 6,519 |
| plato_midi_bridge/jepa/predictor.py | moderate |
| plato_midi_bridge/jepa/train.py | moderate |
| plato_midi_bridge/jepa/__init__.py | 982 |
| plato_midi_bridge/jepa/equilibrium.py | moderate |
| plato_midi_bridge/jepa/room.py | moderate |
| flux_modules/flux_adaptive.py | 3,921 |
| flux_modules/flux_eigenstyle.py | moderate |
| flux_modules/flux_penrose.py | moderate |
| flux_modules/flux_encoder.py | moderate |
| flux_modules/flux_coupling.py | moderate |
| flux_modules/flux_provenance.py | moderate |
| tests/test_jepa.py | pytest file |
| tests/test_autopilot.py | pytest file |

Note: Original files compiled from `/tmp/plato-midi-bridge/` on 2026-05-14.
