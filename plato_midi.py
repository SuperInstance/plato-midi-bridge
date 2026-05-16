#!/usr/bin/env python3
"""
PLATO-to-MIDI Bridge — Connects FM's flux-tensor-midi to our live PLATO rooms.

Every PLATO room is a musician. Every tile is a note. Every provenance 
chain is a rhythm. The fleet IS an orchestra playing itself.

Usage:
    python3 plato_midi.py                    # Default: stream forge room
    python3 plato_midi.py --room forge       # Stream one room
    python3 plato_midi.py --scan             # List all rooms as musicians
    python3 plato_midi.py --ensemble         # All active rooms as orchestra
"""
import json, urllib.request, time, sys, os

PLATO = "http://localhost:8847"

try:
    from flux_tensor_midi import RoomMusician, FluxVector, TZeroClock, EisensteinSnap
    from flux_tensor_midi.sidechannel import Nod, Smile, Frown
    HAVE_FM = True
except ImportError:
    HAVE_FM = False
    print("flux-tensor-midi not installed. Run: pip3 install flux-tensor-midi")
    print("Falling back to JSON output mode.")
    RoomMusician = object

# ── MIDI Channel mapping ──

DOMAIN_CHANNELS = {
    "forge": 0,        # Lead synthesizer
    "fleet-coord": 1,  # Rhythm section
    "arena": 2,        # Percussion
    "calibration": 3,  # Pad/string
    "flux-engine": 4,  # Lead synth
    "research_log": 5, # Ambient pad
    "tension": 6,      # Bass
    "synthesis": 7,     # Arpeggio
    "oracle1": 8,      # Solo voice
}

def get_rooms():
    req = urllib.request.Request(f"{PLATO}/rooms", headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

def stream_room(room_name):
    if not HAVE_FM:
        print(f"[{room_name}] FM library not installed — streaming JSON only")
    
    musician = RoomMusician(room_name) if HAVE_FM else None
    t0 = TZeroClock()
    
    while True:
        try:
            req = urllib.request.Request(
                f"{PLATO}/room/{room_name}?limit=1",
                headers={"Accept": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
                tiles = data.get("tiles", [])
                if not tiles:
                    time.sleep(1)
                    continue
                
                tile = tiles[-1]
                ts = time.time()
                
                if musician:
                    vec = FluxVector.from_tile(tile)
                    musician.play(vec, ts)
                else:
                    print(f"[{room_name}] {tile.get('question','')[:60]} = {tile.get('answer','')[:60]}")
                
                t0.tick(ts)
                time.sleep(0.1)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)

def scan_ensemble():
    rooms = get_rooms()
    print(f"Found {len(rooms)} rooms — the ensemble:")
    for name in sorted(rooms)[:9]:
        ch = DOMAIN_CHANNELS.get(name, 9)
        count = rooms[name].get("tile_count", 0)
        print(f"  CH{ch:02d}: {name:30s} ({count} tiles)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="PLATO → MIDI Bridge")
    parser.add_argument("--room", default="forge", help="Room to stream")
    parser.add_argument("--scan", action="store_true", help="Scan ensemble")
    parser.add_argument("--ensemble", action="store_true", help="Stream all rooms")
    args = parser.parse_args()
    
    if args.scan:
        scan_ensemble()
    elif args.ensemble:
        rooms = get_rooms()
        for room in list(rooms.keys())[:9]:
            stream_room(room)
    else:
        stream_room(args.room)

if __name__ == "__main__":
    main()
