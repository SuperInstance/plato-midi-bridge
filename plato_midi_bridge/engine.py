"""
Engine: connects PLATO → Tensor → MIDI → WebSocket → Web UI.
Runs the full pipeline on a polling loop.
"""

import json
import time
import threading
import numpy as np
from typing import Optional
from .tensor import PlatoRoomFetcher, RoomTensor, CouplingTensor, TMinusTensor
from .midi import MIDIGenerator, MIDIStream


class PlatoMIDIEngine:
    """The bridge itself. Polls PLATO, builds tensors, generates MIDI."""

    def __init__(self, plato_url: str = "http://localhost:8847",
                 bpm: int = 120, poll_seconds: int = 10):
        self.fetcher = PlatoRoomFetcher(plato_url)
        self.generator = MIDIGenerator(bpm=bpm)
        self.poll_seconds = poll_seconds
        self.running = False
        self.current_tensors = None
        self.current_midi = None
        self.listeners = []

    def add_listener(self, callback):
        """Add a callback that receives (rooms, coupling, tminus, midi) on each tick."""
        self.listeners.append(callback)

    def tick(self):
        """Single poll → tensor → MIDI cycle."""
        try:
            # Fetch PLATO state
            rooms, coupling, tminus = self.fetcher.build_tensors()
            self.current_tensors = (rooms, coupling, tminus)

            # Parse t-minus events from bridge room if available
            try:
                bridge = self.fetcher.fetch_room("oracle1-forgemaster-bridge")
                for tile in bridge.get("tiles", []):
                    src = tile.get("source", "").lower()
                    if src == "forgemaster" and "T-MINUS" in tile.get("question", ""):
                        tminus.add_from_plato_tile(tile["question"], tile["answer"])
            except:
                pass

            # Generate MIDI
            stream = self.generator.generate(rooms, coupling, tminus)
            self.current_midi = stream

            # Notify listeners
            for cb in self.listeners:
                try:
                    cb(rooms, coupling, tminus, stream)
                except Exception as e:
                    print(f"[engine] Listener error: {e}")

            return rooms, coupling, tminus, stream

        except Exception as e:
            print(f"[engine] Tick error: {e}")
            return None, None, None, None

    def run(self):
        """Run the polling loop."""
        self.running = True
        while self.running:
            self.tick()
            time.sleep(self.poll_seconds)

    def run_async(self):
        """Run in background thread."""
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    def stop(self):
        self.running = False

    def export_midi(self, filepath: str = "/tmp/plato-current-song.mid"):
        """Export current MIDI stream to file."""
        if self.current_midi:
            with open(filepath, "wb") as f:
                f.write(self.current_midi.to_bytes())
            return filepath
        return None

    def summary(self) -> dict:
        """Human-readable summary of current state."""
        if not self.current_tensors:
            return {"status": "no data yet"}
        
        rooms, coupling, tminus = self.current_tensors
        n_rooms = len(rooms)
        tension = tminus.musical_tension()
        
        room_info = []
        for r in rooms:
            chamber = r.chamber
            ch_name = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"][chamber]
            room_info.append({
                "name": r.name,
                "chamber": f"{ch_name} ({chamber})",
                "velocity": r.velocity,
                "gap": round(r.gap, 3),
            })

        events_info = []
        for e in tminus.events:
            status = "pending" if e["actual"] is None else "resolved"
            events_info.append({
                "name": e["name"],
                "predicted": e["predicted"],
                "actual": e["actual"],
                "status": status,
            })

        return {
            "rooms": n_rooms,
            "tension": round(tension, 3),
            "bpm": self.generator.bpm,
            "room_details": room_info,
            "t_minus_events": events_info,
            "midi_notes": len(self.current_midi.notes) if self.current_midi else 0,
        }
