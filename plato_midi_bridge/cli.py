#!/usr/bin/env python3
"""CLI entry point for plato-midi-bridge."""
import argparse
import time
from .engine import PlatoMIDIEngine
from .web import serve_web_interface


def main():
    parser = argparse.ArgumentParser(description="Plato-MIDI Bridge")
    parser.add_argument("--plato", default="http://localhost:8847", help="PLATO server URL")
    parser.add_argument("--bpm", type=int, default=120, help="Tempo in BPM")
    parser.add_argument("--poll", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--web-port", type=int, default=9710, help="Web UI port")
    parser.add_argument("--export", type=str, default=None, 
                       help="Export MIDI to file and exit")
    parser.add_argument("--oneshot", action="store_true",
                       help="Single tick then exit")
    args = parser.parse_args()

    engine = PlatoMIDIEngine(
        plato_url=args.plato,
        bpm=args.bpm,
        poll_seconds=args.poll,
    )

    # Web interface
    if not args.oneshot and not args.export:
        serve_web_interface(engine, port=args.web_port)
        print(f"[plato-midi] Bridge running. Polling {args.plato} every {args.poll}s")
        print(f"[plato-midi] Web UI: http://localhost:{args.web_port}")
        engine.run()
        return

    # Single tick
    rooms, coupling, tminus, midi = engine.tick()
    
    if args.export:
        path = engine.export_midi(args.export)
        if path:
            print(f"Exported MIDI to {path}")
        else:
            print("No MIDI data to export")

    if rooms:
        print(f"Bridge: {len(rooms)} rooms, {len(tminus.events)} t-minus events")
        summary = engine.summary()
        print(f"Tension: {summary.get('tension', '?')}")
        for r in rooms:
            print(f"  {r.name}: chamber={r.chamber}, velocity={r.velocity}")
        for e in tminus.events:
            s = "resolved" if e.get("actual") else "pending"
            print(f"  TM: {e['name']} T-{e['predicted']}h [{s}]")


if __name__ == "__main__":
    main()
