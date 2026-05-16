# 🎵 PLATO → MIDI Bridge

Connects FM's `flux-tensor-midi` library to our live PLATO rooms.
Every room is a musician. Every tile is a note. The fleet is an orchestra.

## Setup
```bash
pip3 install flux-tensor-midi
```

## Usage
```bash
# Stream one room
python3 plato_midi.py --room forge

# Scan the ensemble
python3 plato_midi.py --scan

# All rooms as an orchestra
python3 plato_midi.py --ensemble
```

## Musical Mapping
- Room → Musician (MIDI channel)
- Tile → Note (velocity = confidence)
- 9-channel FluxVector → Harmonic spectrum
- TZeroClock → Rhythmic grid
- EisensteinSnap → Rhythmic quantization
- Sidechannels (Nod/Smile/Frown) → Agreement/disagreement
