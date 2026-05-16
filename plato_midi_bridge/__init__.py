"""
plato-midi-bridge — Tensor MIDI engine for PLATO rooms.
Multi-dimensional tensor operations on room state → MIDI.
Embeds t-minus-event predictions as temporal coupling vectors.
"""

__version__ = "0.1.0"

from .tensor import RoomTensor, CouplingTensor, TMinusTensor
from .midi import MIDIStream, MIDINote, MIDIControl
from .engine import PlatoMIDIEngine
from .web import serve_web_interface
