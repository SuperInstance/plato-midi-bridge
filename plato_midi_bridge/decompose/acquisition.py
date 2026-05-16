"""
MIDI Acquisition — download public domain MIDI from multiple sources.
"""
import urllib.request, json, time, os
from pathlib import Path
from typing import List, Tuple

SOURCES = {
    "bitmidi": {
        "url": "https://bitmidi.com/uploads/{id}.mid",
        "type": "sequential_id",
        "start": 1,
    },
}

def download_midi(url: str, dest: Path) -> Tuple[bool, int]:
    """Download a MIDI file from URL. Returns (success, size)."""
    try:
        data = urllib.request.urlopen(url, timeout=10).read()
        if len(data) < 10:
            return False, 0
        if data[:4] != b'MThd':
            return False, 0
        dest.write_bytes(data)
        return True, len(data)
    except:
        return False, 0

def acquire_bitmidi(dest_dir: Path, start: int = 1, count: int = 100):
    """Download MIDI files from bitmidi.com using sequential IDs."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    for i in range(start, start + count):
        url = SOURCES["bitmidi"]["url"].format(id=i)
        filepath = dest_dir / f"{i}.mid"
        ok, size = download_midi(url, filepath)
        if ok:
            success += 1
        if i % 10 == 0:
            time.sleep(0.1)
    return success

def acquire_batch(dest_dir: str = "/tmp/midi-library/real", 
                   ids: List[int] = None,
                   count: int = 50) -> int:
    """Download a batch of MIDI files. Returns count of valid files."""
    dest = Path(dest_dir)
    if ids:
        total = 0
        for midi_id in ids:
            url = SOURCES["bitmidi"]["url"].format(id=midi_id)
            ok, size = download_midi(url, dest / f"{midi_id}.mid")
            if ok:
                total += 1
            time.sleep(0.05)
        return total
    else:
        return acquire_bitmidi(dest, count=count)

def pipeline_status():
    """Print summary of current MIDI library."""
    for label, d in [("generated", Path("/tmp/midi-library/classical")),
                     ("real (bitmidi)", Path("/tmp/midi-library/real"))]:
        mids = list(d.glob("*.mid")) if d.exists() else []
        valid = sum(1 for f in mids if f.read_bytes()[:4] == b'MThd')
        total_size = sum(f.stat().st_size for f in mids if f.read_bytes()[:4] == b'MThd')
        if valid:
            print(f"{label}: {valid} files, {total_size//1024}KB total")
