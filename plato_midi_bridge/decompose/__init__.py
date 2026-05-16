"""
Style Decomposer — decomposes multi-track MIDI into PLATO rooms.

Each track becomes a room.
Each note becomes a tile with pitch, velocity, onset, duration, articulation.
Cross-track couplings encode the musician's timing/voicing fingerprint.

Algorithm:
  1. Parse MIDI → per-track note arrays
  2. Per-track: extract style dimensions (pitch profile, velocity curve, timing deviation, release shape)
  3. Cross-track: build coupling matrix from co-occurrence + call/response patterns
  4. Per-piece: construct style tensor
  5. Cross-piece (multiple files): aggregate into musician fingerprint
"""

"""
Exports:
  - parse_midi(filepath) -> List[List[MIDINoteEvent]]
  - TempoMap: tick-to-second converter with tempo change support
  - MIDINoteEvent: single note event dataclass
  - TrackStyle: per-track style fingerprint dataclass
  - extract_track_style(notes, ...) -> TrackStyle
  - PieceCoupling: cross-track coupling dataclass
  - compute_coupling(styles, notes) -> PieceCoupling
  - MusicianFingerprint: aggregation across pieces
  - aggregate_fingerprint(styles, couplings) -> MusicianFingerprint
  - decompose_to_plato_tiles(midi_path, ...) -> dict
  - decompose_directory(dir_path, ...) -> List[dict]
  - beat_grid(notes, duration) -> Tuple[np.ndarray, List[float]]
  - _compute_rest_ratio(notes, duration) -> float (internal)
"""

import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import struct
import gzip


__all__ = [
    'MIDINoteEvent', 'TempoMap', 'TempoChange',
    'TimeSignature', 'TrackStyle', 'PieceCoupling', 'MusicianFingerprint',
    'ScaleLevel', 'ScaleCoupling', 'MultiScaleAnalyzer', 'MultiScaleFingerprint',
    'PenroseEncoder', 'PenroseTiling', 'EncodingExperiment', 'EncodingResult',
    'StylePCA', 'snap_to_pythagorean', 'validate_on_real_files',
    'encode_penrose', 'decompose_real',
    'parse_midi', 'extract_track_style', 'compute_coupling',
    'aggregate_fingerprint', 'decompose_to_plato_tiles',
    'decompose_directory', 'beat_grid', 'main',
    '_compute_rest_ratio',
]

CHUNK_SECONDS = 0.05  # 50ms time slices for micro-timing resolution


# ── Tempo Map ───────────────────────────────────────

@dataclass
class TempoChange:
    """A tempo change event at a specific tick position."""
    tick: int          # absolute tick position of this change
    us_per_beat: int   # microseconds per quarter note
    bpm: float         # beats per minute

    def __post_init__(self):
        self.bpm = 60_000_000 / self.us_per_beat if self.us_per_beat > 0 else 120.0


class TempoMap:
    """Maps absolute tick positions to seconds, accounting for tempo changes.
    
    Builds a piecewise-linear mapping: each segment between tempo changes
    has a constant microseconds-per-beat rate.
    """
    
    def __init__(self, ticks_per_beat: int, 
                 changes: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            ticks_per_beat: MIDI ticks per quarter note
            changes: list of (tick, us_per_beat) tempo changes, 
                     NOT including the default at tick 0
        """
        self.ticks_per_beat = ticks_per_beat
        
        # Normalize: always include tick 0 with default or first tempo
        tempo_points = [(0, 500000)]  # default 120 BPM at tick 0
        if changes:
            # Merge, keeping first occurrence of each tick
            seen_ticks = {0}
            for tick, us in sorted(changes, key=lambda x: x[0]):
                if tick not in seen_ticks:
                    tempo_points.append((tick, us))
                    seen_ticks.add(tick)
        
        self.tempo_points = tempo_points
        
        # Build mapping: for each segment, compute the seconds at the start of the segment
        # and the seconds-per-tick rate within the segment
        self.segment_seconds = []  # cumulative seconds at start of each segment
        self.segment_rates = []    # seconds per tick for each segment
        cum_sec = 0.0
        for i, (tick, us_per_beat) in enumerate(tempo_points):
            if i > 0:
                # How many ticks are in the previous segment?
                prev_tick = tempo_points[i - 1][0]
                tick_count = tick - prev_tick
                prev_us = tempo_points[i - 1][1]
                sec_per_tick = prev_us / (ticks_per_beat * 1_000_000)
                cum_sec += tick_count * sec_per_tick
            
            self.segment_seconds.append(cum_sec)
            sec_per_tick = us_per_beat / (ticks_per_beat * 1_000_000)
            self.segment_rates.append(sec_per_tick)
    
    def ticks_to_seconds(self, tick: int) -> float:
        """Convert an absolute tick position to seconds."""
        if tick < 0:
            return 0.0
        
        # Find the segment containing this tick
        # Binary search for last tempo_point.tick <= tick
        lo, hi = 0, len(self.tempo_points) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.tempo_points[mid][0] <= tick:
                lo = mid
            else:
                hi = mid - 1
        
        seg_idx = lo
        seg_start_tick = self.tempo_points[seg_idx][0]
        seg_start_sec = self.segment_seconds[seg_idx]
        sec_per_tick = self.segment_rates[seg_idx]
        
        return seg_start_sec + (tick - seg_start_tick) * sec_per_tick
    
    def ticks_to_seconds_array(self, ticks: np.ndarray) -> np.ndarray:
        """Convert an array of tick positions to seconds (vectorized)."""
        if len(ticks) == 0:
            return np.array([], dtype=float)
        
        result = np.zeros_like(ticks, dtype=float)
        
        for i in range(len(self.tempo_points)):
            seg_start_tick = self.tempo_points[i][0]
            seg_start_sec = self.segment_seconds[i]
            sec_per_tick = self.segment_rates[i]
            
            if i < len(self.tempo_points) - 1:
                seg_end_tick = self.tempo_points[i + 1][0]
                mask = (ticks >= seg_start_tick) & (ticks < seg_end_tick)
            else:
                mask = ticks >= seg_start_tick
            
            result[mask] = seg_start_sec + (ticks[mask] - seg_start_tick) * sec_per_tick
        
        return result


# ── MIDI Parsing ────────────────────────────────────

@dataclass
class MIDINoteEvent:
    pitch: int        # 0-127
    velocity: int     # 0-127
    start_sec: float  # absolute start time in seconds
    duration_sec: float  # note-off - note-on in seconds
    channel: int = 0
    track: int = 0

    @property
    def release_ratio(self) -> float:
        """Duration relative to beat. < 0.5 = staccato, > 0.8 = legato."""
        return min(1.0, self.duration_sec / max(0.1, self.start_sec % 1.0 + 0.01))


def _read_var_len(data: bytes, pos: int) -> Tuple[int, int]:
    """Read a variable-length value from data starting at pos.
    MIDI VLQ is MSB-first: first byte has the most significant 7 bits.
    Returns (value, new_pos)."""
    value = 0
    while pos < len(data):
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, pos


def parse_midi(filepath: str) -> List[List[MIDINoteEvent]]:
    """Parse a MIDI file into per-track note arrays.
    Handles Format 0, 1, and 2 MIDI files.
    Properly handles mid-piece tempo changes via TempoMap.
    Handles running status, note-off (0x8_ and note-on vel=0).
    Skips empty tracks (meta-only tracks with 0 notes).
    Uses raw byte parsing (no external deps)."""
    
    with open(filepath, "rb") as f:
        data = f.read()

    if data[:4] != b'MThd':
        raise ValueError("Not a valid MIDI file")

    # Parse header
    header_len = struct.unpack(">I", data[4:8])[0]
    format_type = struct.unpack(">H", data[8:10])[0]
    num_tracks = struct.unpack(">H", data[10:12])[0]
    time_division_raw = struct.unpack(">H", data[12:14])[0]

    # Time division
    if time_division_raw & 0x8000:
        fps = -(time_division_raw >> 8)  # negative signed
        ticks_per_frame = time_division_raw & 0xFF
        ticks_per_beat = ticks_per_frame * fps
    else:
        ticks_per_beat = time_division_raw & 0x7FFF

    # For Format 2: each track is an independent sequence.
    # We still parse them into separate track lists.
    # For Format 0: single multi-channel track.
    # For Format 1: synchronized multi-track.

    # Parse all tracks
    all_raw_tracks = []
    pos = 14 + header_len - 6  # skip to first track chunk
    for track_idx in range(num_tracks):
        if pos >= len(data):
            break
        if data[pos:pos+4] != b'MTrk':
            # Some files have non-track chunks; skip them
            pos += 4
            continue
            
        track_len = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        track_data = data[pos + 8:pos + 8 + track_len]
        pos += 8 + track_len
        all_raw_tracks.append(track_data)

    if format_type == 2:
        # Format 2: Each track is independent. Parse each with its own tempo map.
        return _parse_format2_tracks(all_raw_tracks, ticks_per_beat)
    else:
        # Format 0 or 1: All tracks share the same tempo map.
        # We need to scan all tracks for tempo events first, 
        # then parse each track using the merged tempo map.
        return _parse_multi_track(all_raw_tracks, ticks_per_beat)


def _collect_tempo_changes(track_data: bytes) -> List[Tuple[int, int]]:
    """Scan a track for tempo (0x51) and time signature (0x58) events.
    Returns list of (tick, us_per_beat) tempo changes."""
    changes = []
    i = 0
    abs_time = 0
    running_status = 0
    
    while i < len(track_data):
        delta, i = _read_var_len(track_data, i)
        abs_time += delta
        if i >= len(track_data):
            break
        
        event_byte = track_data[i]
        if event_byte & 0x80:
            running_status = event_byte
            i += 1
        
        event_type = running_status >> 4
        
        if event_type == 0x9 and i + 1 < len(track_data):
            i += 2
        elif event_type == 0x8 and i + 1 < len(track_data):
            i += 2
        elif event_type in (0xA, 0xB, 0xE) and i + 1 < len(track_data):
            i += 2
        elif event_type in (0xC, 0xD) and i < len(track_data):
            i += 1
        elif event_type == 0xF:
            if running_status == 0xFF:
                if i >= len(track_data):
                    break
                meta_type = track_data[i]
                i += 1
                meta_len, i = _read_var_len(track_data, i)
                if meta_type == 0x51 and meta_len == 3:  # Tempo
                    if i + 2 < len(track_data):
                        us = (track_data[i] << 16) | (track_data[i+1] << 8) | track_data[i+2]
                        changes.append((abs_time, us))
                i += meta_len
            else:
                # System exclusive — skip length + data
                if i < len(track_data):
                    sys_len, i = _read_var_len(track_data, i)
                    i += sys_len
        else:
            i += 1
    
    return changes


def _parse_format2_tracks(all_raw_tracks: List[bytes], 
                           ticks_per_beat: int) -> List[List[MIDINoteEvent]]:
    """Parse Format 2 MIDI: each track is an independent sequence.
    Each track gets its own tempo map (independent timelines)."""
    all_notes = []
    for track_idx, track_data in enumerate(all_raw_tracks):
        # Collect tempo changes for this track
        tempo_changes = _collect_tempo_changes(track_data)
        tempo_map = TempoMap(ticks_per_beat, tempo_changes)
        
        notes = _parse_track_events(track_data, track_idx, tempo_map, ticks_per_beat)
        if notes:  # skip empty tracks
            all_notes.append(notes)
    
    return all_notes


def _parse_multi_track(all_raw_tracks: List[bytes],
                        ticks_per_beat: int) -> List[List[MIDINoteEvent]]:
    """Parse Format 0/1 MIDI: all tracks share merged tempo info."""
    # Merge tempo changes from all tracks
    all_changes = []
    for track_data in all_raw_tracks:
        all_changes.extend(_collect_tempo_changes(track_data))
    
    # Deduplicate by tick (keep first occurrence)
    seen_ticks = set()
    merged_changes = []
    for tick, us in sorted(all_changes, key=lambda x: x[0]):
        if tick not in seen_ticks:
            merged_changes.append((tick, us))
            seen_ticks.add(tick)
    
    tempo_map = TempoMap(ticks_per_beat, merged_changes)
    
    all_notes = []
    for track_idx, track_data in enumerate(all_raw_tracks):
        notes = _parse_track_events(track_data, track_idx, tempo_map, ticks_per_beat)
        if notes:  # skip empty tracks
            all_notes.append(notes)
    
    return all_notes


def _parse_track_events(track_data: bytes, track_idx: int,
                         tempo_map: TempoMap,
                         ticks_per_beat: int) -> List[MIDINoteEvent]:
    """Parse a single track's events, returning a list of MIDINoteEvent.
    Skips tracks with no note events (meta-only tracks)."""
    
    notes = []
    running_status = 0
    abs_time = 0  # absolute ticks
    track_name = None
    
    # Pending note-ons: pitch -> (start_tick, velocity)
    pending_notes = {}
    
    i = 0
    while i < len(track_data):
        delta, i = _read_var_len(track_data, i)
        abs_time += delta
        if i >= len(track_data):
            break
        
        event_byte = track_data[i]
        if event_byte & 0x80:
            running_status = event_byte
            i += 1
        
        event_type = running_status >> 4
        channel = running_status & 0x0F
        
        if event_type == 0x9 and i + 1 < len(track_data):  # Note On
            pitch = track_data[i]
            velocity = track_data[i + 1]
            i += 2
            if velocity > 0:
                pending_notes[pitch] = (abs_time, velocity)
            else:
                # Note On with velocity 0 = Note Off
                if pitch in pending_notes:
                    start_tick, vel = pending_notes.pop(pitch)
                    dur_ticks = abs_time - start_tick
                    start_sec = tempo_map.ticks_to_seconds(start_tick)
                    dur_sec = tempo_map.ticks_to_seconds(abs_time) - start_sec
                    notes.append(MIDINoteEvent(
                        pitch=pitch, velocity=vel,
                        start_sec=start_sec, duration_sec=max(0.0, dur_sec),
                        channel=channel, track=track_idx
                    ))
                    
        elif event_type == 0x8 and i + 1 < len(track_data):  # Note Off
            pitch = track_data[i]
            i += 2
            if pitch in pending_notes:
                start_tick, vel = pending_notes.pop(pitch)
                dur_ticks = abs_time - start_tick
                start_sec = tempo_map.ticks_to_seconds(start_tick)
                dur_sec = tempo_map.ticks_to_seconds(abs_time) - start_sec
                notes.append(MIDINoteEvent(
                    pitch=pitch, velocity=vel,
                    start_sec=start_sec, duration_sec=max(0.0, dur_sec),
                    channel=channel, track=track_idx
                ))
                
        elif event_type == 0xA and i + 1 < len(track_data):  # Key Aftertouch
            i += 2
        elif event_type == 0xB and i + 1 < len(track_data):  # Control Change
            i += 2
        elif event_type == 0xC and i < len(track_data):  # Program Change
            i += 1
        elif event_type == 0xD and i < len(track_data):  # Channel Aftertouch
            i += 1
        elif event_type == 0xE and i + 1 < len(track_data):  # Pitch Bend
            i += 2
        elif event_type == 0xF and running_status == 0xFF:  # Meta events
            if i >= len(track_data):
                break
            meta_type = track_data[i]
            i += 1
            meta_len, i = _read_var_len(track_data, i)
            
            if meta_type == 0x03:  # Track name
                if meta_len > 0 and i + meta_len <= len(track_data):
                    track_name = track_data[i:i+meta_len].decode('latin1', errors='replace')
                    # Clean up null bytes and whitespace
                    track_name = track_name.replace('\x00', '').strip()
                    if not track_name:
                        track_name = None
            elif meta_type == 0x58 and meta_len >= 4:  # Time signature
                pass  # Tracked but not used directly in decomposition
            elif meta_type == 0x51:  # Tempo — already handled by TempoMap
                pass
            elif meta_type == 0x2F:  # End of Track
                pass
            
            i += meta_len
        elif event_type == 0xF:  # System Exclusive
            if i < len(track_data):
                sys_len, i = _read_var_len(track_data, i)
                i += sys_len
        else:
            i += 1
    
    return notes


# ── Time Signature / Grid Utilities ─────────────────

@dataclass
class TimeSignature:
    """Time signature context for a piece."""
    numerator: int = 4
    denominator: int = 4
    ticks_per_beat: int = 480
    
    @property
    def beats_per_bar(self) -> int:
        return self.numerator


def beat_grid(notes: List[MIDINoteEvent], 
              total_duration: float,
              beats_per_bar: int = 4) -> Tuple[np.ndarray, List[float]]:
    """Compute a quantized beat grid from note onsets.
    Returns (grid_times, beat_divisions) where grid_times are the quantized
    beat boundaries and beat_divisions give the subdivision at each level."""
    
    if len(notes) < 4 or total_duration <= 0:
        # Fallback: evenly spaced grid
        grid = np.arange(0, total_duration, 0.5)
        return grid, [0.5]
    
    starts = np.array([n.start_sec for n in notes])
    
    # Estimate tempo from median inter-onset interval of clustered notes
    sorted_starts = np.sort(starts)
    io_is = np.diff(sorted_starts)
    
    # Filter out very small IOIs (ornaments) and very large (rests)
    main_iois = io_is[(io_is > 0.05) & (io_is < 5.0)]
    
    if len(main_iois) == 0:
        grid = np.arange(0, total_duration, 0.5)
        return grid, [0.5]
    
    # Estimate beat duration from median IOI
    beat_dur = float(np.median(main_iois))
    
    # Constrain to reasonable range
    beat_dur = max(0.15, min(3.0, beat_dur))
    
    # Build grid from 0 to total_duration
    n_beats = max(1, int(total_duration / beat_dur))
    grid = np.arange(start=0, stop=total_duration, step=beat_dur)
    
    return grid, [beat_dur]


# ── Per-track Style Dimensions ──────────────────────

@dataclass
class TrackStyle:
    """A single track's style fingerprint."""
    track_index: int
    channel: int
    
    # Pitch profile
    pitch_histogram: np.ndarray  # (128,) — normalized note usage
    pitch_range: Tuple[int, int]  # lowest, highest
    avg_interval: float  # average interval between consecutive notes
    
    # Velocity profile
    velocity_histogram: np.ndarray  # (128,) — dynamics distribution
    velocity_curve: np.ndarray  # velocity over time (resampled)
    
    # Timing profile
    onset_deviation: np.ndarray  # deviation from grid in seconds
    timing_consistency: float  # std of onset deviation (lower = more mechanical)
    
    # Articulation profile
    duration_histogram: np.ndarray  # (100,) normalized duration distribution
    release_curve: np.ndarray  # note-off shape over time
    staccato_ratio: float  # fraction of notes with duration < 0.5 of beat
    
    # Silence (negative space)
    silence_profile: np.ndarray  # gaps between notes
    rest_ratio: float  # fraction of time silent (capped at 1.0)
    
    # New style dimensions (Phase 1, default values for backward compat)
    dynamic_range: float = 0.0      # max - min velocity
    note_density: float = 0.0       # notes per second
    syncopation_index: float = 0.0  # ratio of off-beat notes
    harmonic_complexity: float = 0.0  # avg unique pitch-classes per 2-second window
    register_breadth: float = 0.0   # span between lowest and highest note

    def to_vector(self) -> np.ndarray:
        """Flatten to a style vector for comparison."""
        return np.concatenate([
            self.pitch_histogram[:48],  # most used pitches
            self.velocity_histogram[:32],
            [self.timing_consistency, self.staccato_ratio, min(1.0, self.rest_ratio)],
            self.duration_histogram[:20],
            [self.avg_interval, 
             self.dynamic_range, self.note_density,
             self.syncopation_index, self.harmonic_complexity,
             self.register_breadth],
        ])


def extract_track_style(notes: List[MIDINoteEvent], 
                        total_duration: float,
                        beat_grid_times: Optional[np.ndarray] = None,
                        beats_per_bar: int = 4) -> TrackStyle:
    """Extract style dimensions from a single track's notes."""
    if not notes:
        return TrackStyle(
            track_index=0, channel=0,
            pitch_histogram=np.zeros(128),
            pitch_range=(60, 72),
            avg_interval=0,
            velocity_histogram=np.zeros(128),
            velocity_curve=np.zeros(100),
            onset_deviation=np.zeros(50),
            timing_consistency=0,
            duration_histogram=np.zeros(100),
            release_curve=np.zeros(100),
            staccato_ratio=0,
            silence_profile=np.zeros(50),
            rest_ratio=1.0,
            dynamic_range=0.0,
            note_density=0.0,
            syncopation_index=0.0,
            harmonic_complexity=0.0,
            register_breadth=0.0,
        )
    
    notes_arr = np.array([(n.pitch, n.velocity, n.start_sec, n.duration_sec) 
                          for n in notes])
    pitches = notes_arr[:, 0].astype(int)
    velocities = notes_arr[:, 1].astype(int)
    starts = notes_arr[:, 2]
    durations = notes_arr[:, 3]
    
    # Pitch histogram
    pitch_hist = np.zeros(128)
    for p in pitches:
        pitch_hist[p] += 1
    pitch_hist = pitch_hist / (pitch_hist.sum() + 1e-8)
    
    # Pitch range
    pitch_min = int(pitches.min()) if len(pitches) > 0 else 60
    pitch_max = int(pitches.max()) if len(pitches) > 0 else 72
    
    # Average interval
    sorted_notes = sorted(notes, key=lambda n: n.start_sec)
    intervals = []
    for i in range(1, len(sorted_notes)):
        intervals.append(abs(sorted_notes[i].pitch - sorted_notes[i-1].pitch))
    avg_interval = float(np.mean(intervals)) if intervals else 0
    
    # Velocity histogram
    vel_hist = np.zeros(128)
    for v in velocities:
        vel_hist[v] += 1
    vel_hist = vel_hist / (vel_hist.sum() + 1e-8)
    
    # Velocity curve over time (resampled)
    if total_duration > 0:
        time_bins = np.linspace(0, total_duration, 100)
        vel_curve = np.zeros(100)
        for v, s in zip(velocities, starts):
            bin_idx = int(np.clip((s / total_duration) * 99, 0, 99))
            vel_curve[bin_idx] = max(vel_curve[bin_idx], v)
    else:
        vel_curve = np.zeros(100)
    
    # Better onset deviation: measure from quantized beat grid
    if beat_grid_times is not None and len(beat_grid_times) > 1 and len(starts) > 1:
        deviations = []
        for s in starts:
            nearest_idx = np.argmin(np.abs(beat_grid_times - s))
            nearest = beat_grid_times[nearest_idx]
            deviations.append(s - nearest)
        onset_dev = np.array(deviations)
    elif total_duration > 0 and len(starts) > 1:
        # Fallback: compute beat grid from median IOI
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        beat_dur = float(np.median(main_iois)) if len(main_iois) > 0 else 0.5
        beat_dur = max(0.15, min(3.0, beat_dur))
        grid = np.arange(0, total_duration, beat_dur)
        deviations = []
        for s in starts:
            nearest = grid[np.argmin(np.abs(grid - s))]
            deviations.append(s - nearest)
        onset_dev = np.array(deviations)
    else:
        onset_dev = np.zeros(50)
    
    timing_consistency = float(np.std(onset_dev)) if len(onset_dev) > 1 else 0
    
    # Duration histogram
    max_dur = max(1.0, durations.max()) if len(durations) > 0 else 1.0
    dur_hist, _ = np.histogram(durations, bins=100, range=(0, min(4, max_dur)))
    dur_hist = dur_hist / (dur_hist.sum() + 1e-8)
    
    # Release curve — how note durations change over time
    release = np.zeros(100)
    if len(durations) > 1:
        for i, d in enumerate(durations):
            idx = int(np.clip((i / len(durations)) * 99, 0, 99))
            release[idx] = d
    
    # Staccato ratio: notes with duration < 0.5 of estimated beat
    if len(starts) > 1:
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        beat_dur = float(np.median(main_iois)) if len(main_iois) > 0 else 0.5
    else:
        beat_dur = 0.5
    beat_dur = max(0.15, min(3.0, beat_dur))
    staccato = float(np.mean((durations / max(beat_dur, 0.01)) < 0.5))
    
    # Silence profile — gaps between notes
    rest_ratio = _compute_rest_ratio(notes, total_duration) if total_duration > 0 else 1.0
    if len(starts) > 1:
        sorted_starts = np.sort(starts)
        gaps = sorted_starts[1:] - sorted_starts[:-1]
        max_gap = max(gaps) if len(gaps) > 0 else 1.0
        silence, _ = np.histogram(gaps, bins=50, range=(0, min(2, max_gap)))
        silence = silence / (silence.sum() + 1e-8)
    else:
        silence = np.zeros(50)
    
    # -- New style dimensions --
    
    # Dynamic range
    dyn_range = float(velocities.max() - velocities.min()) if len(velocities) > 0 else 0.0
    
    # Note density
    note_density = len(notes) / max(total_duration, 0.01)
    
    # Syncopation index: ratio of notes that start off the beat grid
    if beat_grid_times is None or len(beat_grid_times) <= 1:
        # Compute fallback beat grid
        if len(starts) > 1:
            sorted_starts = np.sort(starts)
            iois = np.diff(sorted_starts)
            main_iois = iois[(iois > 0.05) & (iois < 5.0)]
            beat_dur = float(np.median(main_iois)) if len(main_iois) > 0 else 0.5
        else:
            beat_dur = 0.5
        beat_dur = max(0.15, min(3.0, beat_dur))
        beat_grid_times = np.arange(0, total_duration, beat_dur) if total_duration > 0 else np.array([0.0])
    
    beat_dur = beat_grid_times[1] - beat_grid_times[0] if len(beat_grid_times) > 1 else 0.5
    off_beat_count = 0
    for s in starts:
        nearest_idx = np.argmin(np.abs(beat_grid_times - s))
        nearest = beat_grid_times[nearest_idx]
        if abs(s - nearest) > beat_dur * 0.25:
            off_beat_count += 1
    syncopation = off_beat_count / max(len(starts), 1)
    
    # Harmonic complexity: average unique pitch classes per 2-second window
    if total_duration > 0 and len(starts) > 0:
        window_count = max(1, int(total_duration / 2.0))
        complexity_sum = 0.0
        for w in range(window_count):
            win_start = w * 2.0
            win_end = (w + 1) * 2.0
            window_pitches = pitches[(starts >= win_start) & (starts < win_end)]
            if len(window_pitches) > 0:
                unique_pcs = len(set(window_pitches % 12))
                complexity_sum += unique_pcs
        harmonic_complexity = complexity_sum / window_count
    else:
        harmonic_complexity = 0.0
    
    # Register breadth
    register_breadth = float(pitch_max - pitch_min) if len(pitches) > 0 else 0.0
    
    return TrackStyle(
        track_index=0, channel=0,  # filled in later
        pitch_histogram=pitch_hist,
        pitch_range=(pitch_min, pitch_max),
        avg_interval=avg_interval,
        velocity_histogram=vel_hist,
        velocity_curve=vel_curve,
        onset_deviation=onset_dev[:50] if len(onset_dev) >= 50 else np.pad(onset_dev, (0, 50 - len(onset_dev))),
        timing_consistency=timing_consistency,
        duration_histogram=dur_hist,
        release_curve=release,
        staccato_ratio=staccato,
        silence_profile=silence[:50] if len(silence) >= 50 else np.pad(silence, (0, 50 - len(silence))),
        rest_ratio=min(1.0, rest_ratio),
        dynamic_range=dyn_range,
        note_density=note_density,
        syncopation_index=syncopation,
        harmonic_complexity=harmonic_complexity,
        register_breadth=register_breadth,
    )


def _compute_rest_ratio(notes: List[MIDINoteEvent], total_duration: float) -> float:
    """Compute the fraction of total_duration where no note is playing.
    Uses interval merging to handle polyphony correctly and efficiently."""
    if total_duration <= 0 or not notes:
        return 1.0
    
    # Collect all note intervals (start, end) sorted by start
    intervals = sorted([(n.start_sec, n.start_sec + n.duration_sec) for n in notes])
    
    # Merge overlapping intervals
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            # Overlapping: extend the last merged interval
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    
    # Sum active time from merged intervals
    active_time = sum(end - start for start, end in merged)
    
    return 1.0 - min(1.0, active_time / max(total_duration, 0.01))


# ── Cross-track Coupling ────────────────────────────

@dataclass
class PieceCoupling:
    """Coupling matrix between tracks in a single piece."""
    n_tracks: int
    track_names: List[str]
    
    # Time alignment: which tracks lead/follow
    onset_coupling: np.ndarray  # (n, n) — co-occurrence within CHUNK_SECONDS window
    
    # Velocity interaction
    velocity_coupling: np.ndarray  # (n, n) — velocity correlation
    
    # Call/response pattern
    alternation_matrix: np.ndarray  # (n, n) — who follows whom
    
    # Harmonic coherence
    harmonic_coupling: np.ndarray  # (n, n) — pitch class correlation
    
    # Coupling entropy (Phase 1)
    coupling_entropy: np.ndarray = None  # (n, n) — entropy of onset coupling distribution


def compute_coupling(tracks_styles: List[TrackStyle], 
                     all_notes: List[List[MIDINoteEvent]]) -> PieceCoupling:
    """Compute coupling matrix between tracks."""
    n = len(tracks_styles)
    names = [f"track_{i}" for i in range(n)]
    
    onset_coupling = np.zeros((n, n))
    velocity_coupling = np.zeros((n, n))
    alt_matrix = np.zeros((n, n))
    harmonic_coupling = np.zeros((n, n))
    coupling_entropy = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                onset_coupling[i][j] = 1.0
                velocity_coupling[i][j] = 1.0
                harmonic_coupling[i][j] = 1.0
                coupling_entropy[i][j] = 0.0
                continue
            
            notes_i = all_notes[i] if i < len(all_notes) else []
            notes_j = all_notes[j] if j < len(all_notes) else []
            
            if not notes_i or not notes_j:
                continue
            
            # Onset coupling: co-occurrence within time window
            starts_i = np.array([n.start_sec for n in notes_i])
            starts_j = np.array([n.start_sec for n in notes_j])
            co_occurrences = 0
            for si in starts_i:
                matches = np.sum(np.abs(starts_j - si) < CHUNK_SECONDS)
                co_occurrences += matches
            max_possible = min(len(starts_i), len(starts_j))
            onset_coupling[i][j] = min(1.0, co_occurrences / max_possible) if max_possible > 0 else 0
            
            # Velocity coupling
            vels_i = np.array([n.velocity for n in notes_i])
            vels_j = np.array([n.velocity for n in notes_j])
            if len(vels_i) > 1 and len(vels_j) > 1:
                min_len = min(len(vels_i), len(vels_j))
                vel_corr = float(np.corrcoef(vels_i[:min_len], vels_j[:min_len])[0, 1])
                velocity_coupling[i][j] = max(0, (vel_corr + 1) / 2)
            
            # Alternation: who follows whom
            if len(starts_i) > 0 and len(starts_j) > 0:
                alternations = 0
                all_starts = sorted([(s, 'i') for s in starts_i] + [(s, 'j') for s in starts_j])
                for k in range(1, len(all_starts)):
                    if all_starts[k][1] != all_starts[k-1][1]:
                        alternations += 1
                alt_matrix[i][j] = min(1.0, alternations / max(1, len(all_starts)))
            
            # Harmonic coupling: pitch class correlation
            pitches_i = np.array([n.pitch % 12 for n in notes_i])
            pitches_j = np.array([n.pitch % 12 for n in notes_j])
            hist_i = np.bincount(pitches_i, minlength=12) / max(1, len(pitches_i))
            hist_j = np.bincount(pitches_j, minlength=12) / max(1, len(pitches_j))
            harmonic_coupling[i][j] = float(np.dot(hist_i, hist_j))
            
            # Coupling entropy: entropy of time-aligned onset distribution
            # Measures how unpredictable the onset coupling is between tracks
            coupling_entropy[i][j] = _compute_coupling_entropy(notes_i, notes_j)
    
    # Try to use track names from MIDI data (if we can detect them)
    # We don't have track names from notes directly, so leave as numeric
    for i, style in enumerate(tracks_styles):
        if hasattr(style, 'track_index') and style.track_index < len(names):
            pass  # keep default name
    
    return PieceCoupling(
        n_tracks=n,
        track_names=names,
        onset_coupling=onset_coupling,
        velocity_coupling=velocity_coupling,
        alternation_matrix=alt_matrix,
        harmonic_coupling=harmonic_coupling,
        coupling_entropy=coupling_entropy,
    )


def _compute_coupling_entropy(notes_i: List[MIDINoteEvent],
                               notes_j: List[MIDINoteEvent]) -> float:
    """Compute the entropy of the co-occurrence time distribution between two tracks.
    
    Higher entropy = more random coupling (loosely coupled).
    Lower entropy = very structured interaction (tightly coupled).
    """
    if not notes_i or not notes_j:
        return 0.0
    
    starts_i = np.array([n.start_sec for n in notes_i])
    starts_j = np.array([n.start_sec for n in notes_j])
    
    # Find co-occurrences and bin their time positions
    co_occur_times = []
    for si in starts_i:
        near = np.abs(starts_j - si) < CHUNK_SECONDS
        if near.any():
            co_occur_times.append(si)
    
    if len(co_occur_times) < 2:
        return 0.0
    
    # Bin co-occurrence times into 10 uniform bins
    min_t = min(co_occur_times)
    max_t = max(co_occur_times)
    if max_t == min_t:
        return 0.0
    
    bins = np.linspace(min_t, max_t, 11)
    hist, _ = np.histogram(co_occur_times, bins=bins)
    hist = hist / max(hist.sum(), 1e-8)
    
    # Compute Shannon entropy
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    
    # Normalize by log2(n_bins) to get [0, 1] range
    max_entropy = np.log2(10)
    normalized = entropy / max_entropy
    
    return float(min(1.0, normalized))


# ── Musician Fingerprint ────────────────────────────

@dataclass
class MusicianFingerprint:
    """Statistical profile across all pieces by one musician."""
    name: str
    pieces: List[str]  # piece identifiers
    n_pieces: int
    
    # Aggregate style vectors
    mean_style_vector: np.ndarray  # average across all pieces
    style_covariance: np.ndarray   # variance — how much the musician varies
    
    # Aggregate coupling
    mean_coupling: np.ndarray  # (12, 12) average coupling across pieces
    coupling_variance: np.ndarray
    
    # Characteristic patterns
    timing_signature: float   # how far ahead/behind the beat
    dynamic_range: Tuple[float, float]  # (softest, loudest)
    articulation_bias: float  # 0 = staccato, 1 = legato
    
    def to_plato_tile(self) -> str:
        """Format as a PLATO tile for posting to fleet."""
        lines = [
            f"Musician: {self.name}",
            f"Pieces analyzed: {self.n_pieces}",
            f"Timing signature: {self.timing_signature:.3f}s deviation",
            f"Dynamic range: pp={self.dynamic_range[0]:.0f} ff={self.dynamic_range[1]:.0f}",
            f"Articulation bias: {'legato' if self.articulation_bias > 0.5 else 'staccato'} ({self.articulation_bias:.2f})",
        ]
        return "\n".join(lines)


def aggregate_fingerprint(styles: List[List[TrackStyle]], 
                          couplings: List[PieceCoupling],
                          name: str = "unknown") -> MusicianFingerprint:
    """Aggregate multiple pieces into a musician fingerprint."""
    if not styles:
        return MusicianFingerprint(
            name=name, pieces=[], n_pieces=0,
            mean_style_vector=np.zeros(106),  # updated vector length
            style_covariance=np.zeros((106, 106)),
            mean_coupling=np.zeros((12, 12)),
            coupling_variance=np.zeros((12, 12)),
            timing_signature=0, dynamic_range=(0, 127),
            articulation_bias=0.5,
        )
    
    # Collect all style vectors
    style_vectors = []
    for style_list in styles:
        for s in style_list:
            style_vectors.append(s.to_vector())
    
    style_arr = np.array(style_vectors)
    vec_len = style_arr.shape[1] if style_arr.ndim > 1 else 106
    mean_style = np.mean(style_arr, axis=0) if len(style_arr) > 0 else np.zeros(vec_len)
    cov = np.cov(style_arr.T) if style_arr.shape[0] > 1 else np.identity(vec_len)
    
    # Aggregate coupling
    coupling_mats = [c.onset_coupling for c in couplings]
    if coupling_mats:
        max_dim = max(m.shape[0] for m in coupling_mats)
        c_arr = np.array([np.pad(m[:12, :12], ((0, max(0, 12-m.shape[0])), (0, max(0, 12-m.shape[0]))))[:12, :12] for m in coupling_mats])
        mean_c = np.mean(c_arr, axis=0) if len(c_arr) > 0 else np.zeros((12, 12))
        var_c = np.var(c_arr, axis=0) if len(c_arr) > 1 else np.zeros((12, 12))
    else:
        mean_c = np.zeros((12, 12))
        var_c = np.zeros((12, 12))
    
    # Timing signature
    all_deviations = []
    for style_list in styles:
        for s in style_list:
            all_deviations.extend(s.onset_deviation[:5])
    timing = float(np.mean(all_deviations)) if all_deviations else 0
    
    # Dynamic range
    all_vels = []
    for s in styles:
        for st in s:
            v = st.velocity_histogram
            active = np.where(v > 0.01)[0]
            if len(active) > 0:
                all_vels.extend(active.tolist())
    dyn_range = (min(all_vels), max(all_vels)) if all_vels else (0, 127)
    
    # Articulation bias
    all_staccato = [s.staccato_ratio for styles_list in styles for s in styles_list]
    articulation = 1.0 - (float(np.mean(all_staccato)) if all_staccato else 0.5)
    
    return MusicianFingerprint(
        name=name,
        pieces=[],
        n_pieces=len(styles),
        mean_style_vector=mean_style,
        style_covariance=cov,
        mean_coupling=mean_c,
        coupling_variance=var_c,
        timing_signature=timing,
        dynamic_range=dyn_range,
        articulation_bias=articulation,
    )


# ── Decomposition to PLATO ──────────────────────────

def decompose_to_plato_tiles(midi_path: str, piece_name: str = "",
                              plato_url: str = "http://localhost:8847",
                              base_room: str = "style-decomposed") -> dict:
    """Decompose a MIDI file and post to PLATO rooms.
    
    Creates one room per track, populates with style tiles.
    Creates a coupling room with cross-track relationships.
    Posts musician fingerprint to style-decomposed room.
    """
    import urllib.request
    
    notes_by_track = parse_midi(midi_path)
    total_dur = max((n.start_sec + n.duration_sec for track in notes_by_track for n in track), default=60)
    
    # Compute beat grid from combined onsets
    all_starts = []
    for track in notes_by_track:
        all_starts.extend(n.start_sec for n in track)
    if all_starts:
        sorted_starts = np.sort(all_starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        beat_dur = float(np.median(main_iois)) if len(main_iois) > 0 else 0.5
        beat_dur = max(0.15, min(3.0, beat_dur))
        grid = np.arange(0, total_dur, beat_dur)
    else:
        grid = np.arange(0, total_dur, 0.5)
    
    # Per-track decomposition
    track_styles = []
    for i, notes in enumerate(notes_by_track):
        style = extract_track_style(notes, total_dur, beat_grid_times=grid)
        style.track_index = i
        track_styles.append(style)
    
    # Cross-track coupling
    coupling = compute_coupling(track_styles, notes_by_track)
    
    # Multi-scale analysis
    analyzer = MultiScaleAnalyzer()
    all_scales = []
    all_scale_couplings = []
    for notes in notes_by_track:
        scales = analyzer.analyze(notes, total_dur)
        sc = analyzer.compute_scale_coupling(scales)
        all_scales.append(scales)
        all_scale_couplings.append(sc)
    
    # Aggregate multi-scale fingerprint
    multi_fp = analyzer.aggregate_multi_scale(
        all_scales, all_scale_couplings, composer=piece_name or "unknown"
    )
    
    # Post to PLATO
    piece_room = f"{base_room}/{piece_name.replace(' ', '-')}" if piece_name else base_room
    
    results = []
    for i, (style, notes, scales, scale_couple) in enumerate(zip(
        track_styles, notes_by_track, all_scales, all_scale_couplings
    )):
        track_room = f"{piece_room}/track-{i}"
        
        # Style tile
        style_tile = {
            "question": f"Track {i} style — {piece_name}",
            "answer": json.dumps({
                "pitch_range": list(style.pitch_range),
                "avg_interval": style.avg_interval,
                "timing_consistency": float(style.timing_consistency),
                "staccato_ratio": float(style.staccato_ratio),
                "rest_ratio": float(style.rest_ratio),
                "n_notes": len(notes),
                "dynamic_range": float(style.dynamic_range),
                "note_density": float(style.note_density),
                "syncopation_index": float(style.syncopation_index),
                "harmonic_complexity": float(style.harmonic_complexity),
                "register_breadth": float(style.register_breadth),
            }),
            "source": "style-decomposer",
            "confidence": 0.9,
        }
        try:
            req = urllib.request.Request(
                f"{plato_url}/room/{track_room}/submit",
                data=json.dumps(style_tile).encode(),
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=5)
            results.append(resp.read().decode())
        except:
            pass
        
        # Post scale tiles for each level
        for scale_name, scale_level in scales.items():
            scale_tile = {
                "question": f"Track {i} — {scale_name} scale — {piece_name}",
                "answer": json.dumps(scale_level_to_tile_dict(scale_level)),
                "source": "style-decomposer",
                "confidence": 0.85,
            }
            try:
                req = urllib.request.Request(
                    f"{plato_url}/room/{track_room}/submit",
                    data=json.dumps(scale_tile).encode(),
                    headers={"Content-Type": "application/json"}
                )
                resp = urllib.request.urlopen(req, timeout=5)
                results.append(resp.read().decode())
            except:
                pass
        
        # Post scale coupling tile
        coupling_tile_data = scale_coupling_to_dict(scale_couple)
        coupling_tile_data["track_index"] = i
        scale_coupling_post = {
            "question": f"Track {i} scale coupling — {piece_name}",
            "answer": json.dumps(coupling_tile_data),
            "source": "style-decomposer",
            "confidence": 0.85,
        }
        try:
            req = urllib.request.Request(
                f"{plato_url}/room/{track_room}/submit",
                data=json.dumps(scale_coupling_post).encode(),
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req, timeout=5)
            results.append(resp.read().decode())
        except:
            pass
    
    # Coupling tile
    coupling_tile = {
        "question": f"Coupling matrix — {piece_name}",
        "answer": json.dumps({
            "onset": coupling.onset_coupling.tolist(),
            "velocity": coupling.velocity_coupling.tolist(),
            "alternation": coupling.alternation_matrix.tolist(),
            "harmonic": coupling.harmonic_coupling.tolist(),
            "coupling_entropy": coupling.coupling_entropy.tolist(),
            "n_tracks": coupling.n_tracks,
        }),
        "source": "style-decomposer",
        "confidence": 0.9,
    }
    try:
        req = urllib.request.Request(
            f"{plato_url}/room/{piece_room}/submit",
            data=json.dumps(coupling_tile).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        results.append(resp.read().decode())
    except:
        pass
    
    # Post multi-scale fingerprint
    fp_tile = {
        "question": f"Multi-scale fingerprint — {piece_name}",
        "answer": json.dumps({
            "composer": multi_fp.composer,
            "piece_count": multi_fp.piece_count,
            "features_per_scale": {
                k: v for k, v in multi_fp.features.items()
            },
            "scale_coupling": {
                "pairs": {f"{a}→{b}": c for (a, b), c in multi_fp.scale_coupling.scale_pairs.items()},
                "inflation_ratios": {f"{a}→{b}": r for (a, b), r in multi_fp.scale_coupling.inflation_ratios.items()},
            },
        }),
        "source": "style-decomposer",
        "confidence": 0.85,
    }
    try:
        req = urllib.request.Request(
            f"{plato_url}/room/{piece_room}/submit",
            data=json.dumps(fp_tile).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        results.append(resp.read().decode())
    except:
        pass
    
    return {
        "piece": piece_name,
        "tracks": len(notes_by_track),
        "total_notes": sum(len(n) for n in notes_by_track),
        "duration_sec": total_dur,
        "styles": track_styles,
        "coupling": coupling,
        "multi_scale_analysis": all_scales,
        "scale_coupling": all_scale_couplings,
        "multi_scale_fingerprint": multi_fp,
        "plato_results": results,
    }


def decompose_directory(dir_path: str, plato_url: str = "http://localhost:8847",
                        base_room: str = "style-decomposed") -> List[dict]:
    """Decompose all MIDI files in a directory."""
    from pathlib import Path
    p = Path(dir_path)
    results = []
    for f in sorted(p.glob("*.mid")):
        name = f.stem
        print(f"Decomposing: {name}...")
        result = decompose_to_plato_tiles(str(f), name, plato_url, base_room)
        results.append(result)
        print(f"  {result['tracks']} tracks, {result['total_notes']} notes")
    return results


def decompose_real(real_dir: str, plato_url: str = "http://localhost:8847",
                   base_room: str = "style-library") -> Dict:
    """Decompose real MIDI files and post analysis to PLATO.
    
    Runs the full pipeline:
    1. Parse all real MIDI files
    2. Extract styles and run Penrose encoding
    3. PCA reduce and cluster
    4. Post results to PLATO style-library room
    
    Args:
        real_dir: Directory containing real MIDI files
        plato_url: PLATO server URL (default http://localhost:8847)
        base_room: Base PLATO room name (default style-library)
        
    Returns:
        Dict with analysis results
    """
    from pathlib import Path
    import json
    import urllib.request
    
    p = Path(real_dir)
    if not p.exists():
        return {"error": f"Directory not found: {real_dir}"}
    
    midi_files = sorted(p.glob("*.mid"))
    if not midi_files:
        return {"error": "No MIDI files found", "n_files": 0}
    
    print(f"Processing {len(midi_files)} real MIDI files...")
    
    # 1. Parse all files
    all_tracks = []
    all_notes_flat = []
    all_paths = []
    
    for f in midi_files:
        try:
            tracks = parse_midi(str(f))
            if tracks:
                flat = [n for track in tracks for n in track]
                if flat:
                    all_tracks.append(tracks)
                    all_notes_flat.append(flat)
                    all_paths.append(f.name)
        except:
            pass
    
    if len(all_tracks) < 2:
        return {"error": f"Too few valid files ({len(all_tracks)})", "n_files": len(all_tracks)}
    
    print(f"  Parsed {len(all_tracks)} valid files")
    
    # 2. Extract 109D style vectors
    all_styles = []
    for i, notes in enumerate(all_notes_flat):
        total_dur = max((n.start_sec + n.duration_sec for n in notes), default=60.0)
        style = extract_track_style(notes, total_dur)
        all_styles.append(style)
    
    style_arr = np.array([s.to_vector() for s in all_styles])
    
    # 3. PCA reduction
    pca = StylePCA()
    pca_vectors = pca.fit_transform(style_arr, n_components=12)
    
    # 4. Penrose encoding on PCA-reduced vectors
    # Map the 12 PCA components to a 5D style vector for Penrose
    from .penrose import _extract_style_vector_from_notes
    
    style_5d_vectors = np.array([_extract_style_vector_from_notes(n) for n in all_notes_flat])
    
    penrose_encoder = PenroseEncoder()
    penrose_sigs = []
    for sv in style_5d_vectors:
        tiling = penrose_encoder.encode(sv)
        penrose_sigs.append(tiling.to_dict())
    
    # 5. Cluster to find natural groups
    from .penrose import _kmeans, _silhouette_score
    
    max_k = min(6, len(all_tracks) // 2)
    if max_k < 2:
        max_k = min(2, len(all_tracks) - 1)
    
    best_k = 1
    best_sil = -1.0
    best_labels = None
    
    for k in range(2, max_k + 1):
        labels, centroids = _kmeans(style_arr, k)
        if labels is None:
            continue
        sil = _silhouette_score(style_arr, labels)
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels
    
    if best_labels is None:
        best_labels, _ = _kmeans(style_arr, 2)
        best_sil = _silhouette_score(style_arr, best_labels)
    
    # 6. Penrose vs Eisenstein comparison
    experiment = EncodingExperiment()
    pieces = []
    for i in range(len(all_tracks)):
        pieces.append({
            'composer': f'Cluster_{best_labels[i]}',
            'name': all_paths[i],
            'style_5d': style_5d_vectors[i],
        })
    encoding_result = experiment.compare(pieces)
    
    # 7. Build summary
    summary = {
        "real_files_processed": len(all_tracks),
        "natural_clusters_found": int(best_k),
        "cluster_silhouette": float(best_sil),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_[:5]],
        "pca_cumulative_variance_12": float(np.sum(pca.explained_variance_ratio_[:12])),
        "penrose_vs_eisenstein": encoding_result.winner,
        "penrose_silhouette": encoding_result.penrose['silhouette'],
        "eisenstein_silhouette": encoding_result.eisenstein['silhouette'],
    }
    
    # 8. Post to PLATO
    room_name = f"{base_room}/real-library"
    tile = {
        "question": "Real MIDI Library — Decomposition Summary",
        "answer": json.dumps(summary),
        "source": "plato-midi-bridge",
        "confidence": 0.85,
    }
    try:
        req = urllib.request.Request(
            f"{plato_url}/room/{room_name}/submit",
            data=json.dumps(tile).encode(),
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req, timeout=5)
        summary["plato_result"] = resp.read().decode()
    except Exception as e:
        summary["plato_result"] = f"Post failed: {e}"
    
    print(f"\nReal Library Analysis:")
    print(f"  Files processed: {len(all_tracks)}")
    print(f"  Natural clusters: {int(best_k)} (silhouette: {best_sil:.3f})")
    print(f"  PCA (12 dims): {np.sum(pca.explained_variance_ratio_[:12]):.1%} variance explained")
    print(f"  Encoding winner: {encoding_result.winner}")
    
    return summary


# ── CLI ──────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Style Decomposer — MIDI → PLATO rooms")
    parser.add_argument("input", help="MIDI file or directory")
    parser.add_argument("--plato", default="http://localhost:8847", help="PLATO server URL")
    parser.add_argument("--room", default="style-library", help="Base PLATO room name")
    parser.add_argument("--analyze-only", action="store_true", help="Don't post to PLATO")
    args = parser.parse_args()
    
    from pathlib import Path
    p = Path(args.input)
    
    if p.is_dir():
        results = decompose_directory(str(p), args.plato, args.room)
        print(f"\nDecomposed {len(results)} pieces")
        
        # Aggregate fingerprint
        all_styles = [r["styles"] for r in results]
        all_couplings = [r["coupling"] for r in results]
        fp = aggregate_fingerprint(all_styles, all_couplings, name=p.name)
        print(f"\nMusician fingerprint:")
        print(f"  Timing: {fp.timing_signature:.3f}s deviation")
        print(f"  Range: pp={fp.dynamic_range[0]:.0f} ff={fp.dynamic_range[1]:.0f}")
        print(f"  Articulation: {'legato' if fp.articulation_bias > 0.5 else 'staccato'} ({fp.articulation_bias:.2f})")
        
    elif p.suffix.lower() in ('.mid', '.midi'):
        result = decompose_to_plato_tiles(str(p), p.stem, args.plato, args.room)
        print(f"Decomposed: {p.stem}")
        print(f"  {result['tracks']} tracks, {result['total_notes']} notes, {result['duration_sec']:.1f}s")
    else:
        print(f"Unsupported: {p.suffix}")

# ── Multi-Scale Analysis (deferred import to avoid circular dependency) ──

# These re-export the scale types at the package level for convenience.
# The actual implementation is in scale.py, which imports from this module.
# To break the circular dependency, we add these to __all__ at the top
# but do the actual import here at the bottom after all types are defined.
# Importing at the bottom works because by the time this is reached,
# MIDINoteEvent and other types are fully defined.
from plato_midi_bridge.decompose.scale import (
    ScaleLevel, ScaleCoupling, MultiScaleAnalyzer, MultiScaleFingerprint,
    analyze_notes_multi_scale, compute_scales_coupling,
    scale_level_to_tile_dict, scale_coupling_to_dict,
)

# Penrose encoding is independent (no circular dependency)
from plato_midi_bridge.decompose.penrose import (
    PenroseEncoder, PenroseTiling, EncodingExperiment, EncodingResult,
    StylePCA, snap_to_pythagorean, validate_on_real_files,
    encode_penrose,
)


if __name__ == "__main__":
    main()
