"""
MIDI stream generation from PLATO tensors.

Takes RoomTensor, CouplingTensor, TMinusTensor → MIDI events.
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from .tensor import RoomTensor, CouplingTensor, TMinusTensor, EISENSTEIN_CHAMBERS


@dataclass
class MIDINote:
    note: int   # 0-127
    velocity: int  # 0-127
    start: float  # in beats
    duration: float  # in beats
    channel: int = 0  # 0-15
    source: str = ""

    def to_bytes(self) -> bytes:
        """Convert to raw MIDI bytes (note on + note off)."""
        on = bytes([0x90 | self.channel, self.note, self.velocity])
        vel_off = max(0, min(127, self.velocity - 20))
        off = bytes([0x80 | self.channel, self.note, vel_off])
        return on + off


@dataclass
class MIDIControl:
    channel: int
    controller: int  # 0-127 (CC number)
    value: int       # 0-127
    time: float      # in beats

    def to_bytes(self) -> bytes:
        return bytes([0xB0 | self.channel, self.controller, self.value])


@dataclass
class MIDIStream:
    """Full MIDI stream from a PLATO snapshot."""
    notes: List[MIDINote] = field(default_factory=list)
    controls: List[MIDIControl] = field(default_factory=list)
    tempo: int = 120  # BPM
    time_signature: Tuple[int, int] = (4, 4)

    def to_bytes(self, include_header: bool = True) -> bytes:
        """Serialize to standard MIDI file bytes."""
        if not include_header:
            # Just raw events
            data = b''
            for n in self.notes:
                data += n.to_bytes()
            for c in self.controls:
                data += c.to_bytes()
            return data

        # Standard MIDI file format
        ticks_per_beat = 480
        track_data = b''
        # Tempo event
        us_per_beat = int(60_000_000 / self.tempo)
        track_data += bytes([0x00, 0xFF, 0x51, 0x03,
                            (us_per_beat >> 16) & 0xFF,
                            (us_per_beat >> 8) & 0xFF,
                            us_per_beat & 0xFF])
        # Time signature
        track_data += bytes([0x00, 0xFF, 0x58, 0x04,
                            self.time_signature[0], 2, 24, 8])
        # Notes
        for n in self.notes:
            delta = int(n.start * ticks_per_beat)
            track_data += self._encode_vlq(delta)
            track_data += n.to_bytes()
        # End of track
        track_data += bytes([0x00, 0xFF, 0x2F, 0x00])

        track_len = len(track_data)
        header = b'MThd' + bytes([0, 0, 0, 6, 0, 1, 0, 1])
        header += (ticks_per_beat >> 8).to_bytes(1, 'big')
        header += (ticks_per_beat & 0xFF).to_bytes(1, 'big')
        track_header = b'MTrk' + track_len.to_bytes(4, 'big')

        return header + track_header + track_data

    def _encode_vlq(self, value: int) -> bytes:
        """Variable-length quantity encoding."""
        if value == 0:
            return bytes([0])
        result = []
        while value > 0:
            result.insert(0, value & 0x7F)
            value >>= 7
        for i in range(len(result) - 1):
            result[i] |= 0x80
        return bytes(result)


class MIDIGenerator:
    """Generates MIDI from PLATO tensors."""

    def __init__(self, bpm: int = 120):
        self.bpm = bpm
        self.beat_duration = 60.0 / bpm

    def generate(self, rooms: List[RoomTensor], 
                 coupling: CouplingTensor,
                 tminus: TMinusTensor,
                 measures: int = 4) -> MIDIStream:
        """Generate a MIDI stream from current PLATO state."""
        stream = MIDIStream(tempo=self.bpm)
        n_rooms = len(rooms)

        # ── Each room is an instrument channel ──
        for i, room in enumerate(rooms):
            channel = i % 16
            root = room.root_note

            # Room activity as arpeggiated chord
            for beat in range(measures * 4):
                t = beat * 0.25  # quarter notes
                vel = room.velocity
                # Arpeggiate the chamber's chord
                for offset in [0, 4, 7, 12]:  # root, third, fifth, octave
                    note = root + offset
                    dur = 0.2 + 0.1 * room.tile_count / 100
                    stream.notes.append(MIDINote(
                        note=note % 128,
                        velocity=max(20, min(127, vel)),
                        start=t + offset * 0.05,
                        duration=dur,
                        channel=channel,
                        source=room.name,
                    ))

            # Coupling as harmonic intervals
            for j, other_room in enumerate(rooms):
                if i >= j:
                    continue
                weight = float(coupling.matrix[i][j])
                semitones, _ = coupling_to_interval_weight(weight)
                # Play interval at measure boundaries
                for m in range(measures):
                    t = m * self.beat_duration * 4
                    stream.notes.append(MIDINote(
                        note=(root + semitones) % 128,
                        velocity=max(20, min(127, int(weight * 100))),
                        start=t + 0.5,
                        duration=1.0,
                        channel=channel,
                        source=f"{room.name}↔{other_room.name}",
                    ))

        # ── T-minus events as temporal resolution ──
        for event in tminus.events:
            predicted = event["predicted"]
            actual = event.get("actual")
            confidence = event["confidence"]
            
            # Musical representation of the prediction gap
            if actual is not None:
                gap = actual - predicted
                # Resolved — the gap determines the harmony
                gap_hours = abs(gap)
                if gap_hours < 1:
                    semitones = 7  # perfect fifth — well predicted
                elif gap_hours < 6:
                    semitones = 4  # major third — close but not perfect
                else:
                    semitones = 6  # tritone — big gap, dissonance
                
                for channel in range(min(n_rooms, 4)):
                    stream.notes.append(MIDINote(
                        note=60 + semitones + channel * 12,
                        velocity=int(confidence * 100),
                        start=measures * 4 - 0.5,  # near end of phrase
                        duration=1.5,
                        channel=channel,
                        source=f"t-minus:{event.get('name', '?')}",
                    ))
            else:
                # Still pending — suspension
                semitones = 11  # major seventh — wants resolution
                for channel in range(min(n_rooms, 3)):
                    stream.notes.append(MIDINote(
                        note=60 + semitones,
                        velocity=int(confidence * 80),
                        start=0.5,  # early in phrase
                        duration=0.5,
                        channel=channel,
                        source=f"t-minus-pending:{event.get('name', '?')}",
                    ))

        # ── Overall tension as CC messages ──
        tension = tminus.musical_tension()
        stream.controls.append(MIDIControl(
            channel=0, controller=1,  # modulation wheel = tension
            value=int(tension * 127), time=0
        ))

        return stream


def coupling_to_interval_weight(weight: float) -> Tuple[int, str]:
    """Map coupling weight to semitone interval."""
    if weight < 0.1: return (0, "unison")
    if weight < 0.2: return (1, "m2")
    if weight < 0.3: return (2, "M2")
    if weight < 0.4: return (3, "m3")
    if weight < 0.5: return (4, "M3")
    if weight < 0.6: return (5, "P4")
    if weight < 0.7: return (6, "TT")
    if weight < 0.8: return (7, "P5")
    if weight < 0.9: return (8, "m6")
    return (9, "M6")
