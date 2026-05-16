"""
Multi-Scale Analysis — Penrose-Inspired Scale Hierarchy.

Decomposes a track's notes into structural levels:
  micro → note → phrase → section → piece

Each scale's features predict the next scale's features,
mirroring Penrose tiling inflation/deflation.

The key insight: a 2-bar crescendo and a 16-bar crescendo are
the SAME pattern at different scales.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ── Scale Level ─────────────────────────────────────

@dataclass
class ScaleLevel:
    """Features extracted at a single scale level."""
    name: str  # "micro", "note", "phrase", "section", "piece"
    window_seconds: float
    features: Dict[str, float]  # dimension → value at this scale


# ── Scale Coupling ──────────────────────────────────

@dataclass
class ScaleCoupling:
    """How patterns at one scale relate to adjacent scales.
    Directly maps to Penrose tiling inflation/deflation."""
    scale_pairs: Dict[Tuple[str, str], float]  # ("micro","note") = correlation
    inflation_ratios: Dict[Tuple[str, str], float]  # ("phrase","section") = ~4.0


# ── Multi-Scale Fingerprint ─────────────────────────

@dataclass
class MultiScaleFingerprint:
    """Aggregated fingerprint across ALL scales, not just piece-level."""
    composer: str
    piece_count: int
    features: Dict[str, Dict[str, float]]  # scale_name -> {dimension: mean_value}
    scale_coupling: ScaleCoupling


# ── Multi-Scale Analyzer ────────────────────────────

# Window sizes for each scale level (in seconds)
MICRO_WINDOW = 0.025       # ~25ms windows (40 per second)
NOTE_WINDOW = 0.25         # for per-note features that need windowing
PHRASE_MIN_SEC = 1.0        # min 2-bar phrase at 120 BPM
PHRASE_MAX_SEC = 4.0        # max 8-bar phrase
SECTION_MIN_SEC = 4.0       # min 8-bar section
SECTION_MAX_SEC = 16.0      # max 32-bar section

# Scale names
SCALE_MICRO = "micro"
SCALE_NOTE = "note"
SCALE_PHRASE = "phrase"
SCALE_SECTION = "section"
SCALE_PIECE = "piece"

SCALE_ORDER = [SCALE_MICRO, SCALE_NOTE, SCALE_PHRASE, SCALE_SECTION, SCALE_PIECE]


class MultiScaleAnalyzer:
    """Decompose a track's notes into structural scale levels."""

    def analyze(self, notes: List[MIDINoteEvent], total_duration: float) -> Dict[str, ScaleLevel]:
        """Analyze notes at all scale levels.
        
        Args:
            notes: List of MIDI note events from a single track.
            total_duration: Total duration of the piece in seconds.
            
        Returns:
            Dict mapping scale name to ScaleLevel with features.
        """
        if not notes or total_duration <= 0:
            return self._empty_scales()
        
        return {
            SCALE_MICRO: self._analyze_micro(notes, total_duration),
            SCALE_NOTE: self._analyze_note(notes, total_duration),
            SCALE_PHRASE: self._analyze_phrase(notes, total_duration),
            SCALE_SECTION: self._analyze_section(notes, total_duration),
            SCALE_PIECE: self._analyze_piece(notes, total_duration),
        }

    def compute_scale_coupling(self, scales: Dict[str, ScaleLevel]) -> ScaleCoupling:
        """Compute coupling between adjacent scales.
        
        Measures:
        - Correlation between adjacent scales' feature vectors.
        - Inflation ratio: time window size ratio (should approximate φ=1.618 for musical structures).
        
        Args:
            scales: Dict from analyze() output.
            
        Returns:
            ScaleCoupling with correlations and inflation ratios.
        """
        scale_pairs = {}
        inflation_ratios = {}
        
        for i in range(len(SCALE_ORDER) - 1):
            lo_name = SCALE_ORDER[i]
            hi_name = SCALE_ORDER[i + 1]
            
            if lo_name not in scales or hi_name not in scales:
                continue
            
            lo = scales[lo_name]
            hi = scales[hi_name]
            
            # Feature correlation
            lo_vec = self._features_to_vector(lo.features)
            hi_vec = self._features_to_vector(hi.features)
            
            if len(lo_vec) > 0 and len(hi_vec) > 0:
                min_len = min(len(lo_vec), len(hi_vec))
                if min_len > 1 and np.std(lo_vec[:min_len]) > 0 and np.std(hi_vec[:min_len]) > 0:
                    corr = float(np.corrcoef(lo_vec[:min_len], hi_vec[:min_len])[0, 1])
                else:
                    corr = 0.0
            else:
                corr = 0.0
            
            scale_pairs[(lo_name, hi_name)] = max(-1.0, min(1.0, corr))
            
            # Inflation ratio: how many lo-windows fit in one hi-window
            if hi.window_seconds > 0 and lo.window_seconds > 0:
                ratio = hi.window_seconds / lo.window_seconds
            else:
                ratio = 1.0
            
            inflation_ratios[(lo_name, hi_name)] = ratio
        
        return ScaleCoupling(
            scale_pairs=scale_pairs,
            inflation_ratios=inflation_ratios,
        )

    def aggregate_multi_scale(self, all_scales: List[Dict[str, ScaleLevel]],
                               all_couplings: List[ScaleCoupling],
                               composer: str = "unknown") -> MultiScaleFingerprint:
        """Aggregate multiple pieces' scale analyses into a multi-scale fingerprint.
        
        Args:
            all_scales: List of analyze() outputs from multiple pieces.
            all_couplings: List of compute_scale_coupling() outputs.
            composer: Name of the composer/musician.
            
        Returns:
            MultiScaleFingerprint with mean features at each scale.
        """
        if not all_scales:
            empty_features = {name: {} for name in SCALE_ORDER}
            empty_coupling = ScaleCoupling(scale_pairs={}, inflation_ratios={})
            return MultiScaleFingerprint(
                composer=composer, piece_count=0,
                features=empty_features,
                scale_coupling=empty_coupling,
            )
        
        # Aggregate features per scale
        aggregated = {}
        for name in SCALE_ORDER:
            # Collect all feature dicts for this scale across all pieces
            feature_dicts = []
            for scales in all_scales:
                if name in scales:
                    feature_dicts.append(scales[name].features)
            
            if feature_dicts:
                # Get union of all dimension names
                all_dims = set()
                for d in feature_dicts:
                    all_dims.update(d.keys())
                
                mean_features = {}
                for dim in all_dims:
                    values = [d.get(dim, 0.0) for d in feature_dicts]
                    mean_features[dim] = float(np.mean(values)) if values else 0.0
                
                aggregated[name] = mean_features
            else:
                aggregated[name] = {}
        
        # Aggregate coupling
        if all_couplings:
            all_pairs = set()
            for c in all_couplings:
                all_pairs.update(c.scale_pairs.keys())
            
            mean_couple_pairs = {}
            mean_inflation = {}
            for pair in all_pairs:
                corr_vals = [c.scale_pairs.get(pair, 0.0) for c in all_couplings if pair in c.scale_pairs]
                inf_vals = [c.inflation_ratios.get(pair, 1.0) for c in all_couplings if pair in c.inflation_ratios]
                mean_couple_pairs[pair] = float(np.mean(corr_vals)) if corr_vals else 0.0
                mean_inflation[pair] = float(np.mean(inf_vals)) if inf_vals else 1.0
            
            mean_coupling = ScaleCoupling(
                scale_pairs=mean_couple_pairs,
                inflation_ratios=mean_inflation,
            )
        else:
            mean_coupling = ScaleCoupling(scale_pairs={}, inflation_ratios={})
        
        return MultiScaleFingerprint(
            composer=composer,
            piece_count=len(all_scales),
            features=aggregated,
            scale_coupling=mean_coupling,
        )

    # ── Internal helpers ────────────────────────────

    def _empty_scales(self) -> Dict[str, ScaleLevel]:
        """Return zeroed-out scales for empty input."""
        return {
            name: ScaleLevel(name=name, window_seconds=self._window_for_scale(name), features={})
            for name in SCALE_ORDER
        }

    def _window_for_scale(self, name: str) -> float:
        """Get the analysis window size for a scale level."""
        windows = {
            SCALE_MICRO: MICRO_WINDOW,
            SCALE_NOTE: NOTE_WINDOW,
            SCALE_PHRASE: PHRASE_MIN_SEC,
            SCALE_SECTION: SECTION_MIN_SEC,
            SCALE_PIECE: 0.0,  # full duration
        }
        return windows.get(name, 0.0)

    def _features_to_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert a feature dict to a sorted vector for correlation."""
        if not features:
            return np.array([])
        # Sort by key for deterministic ordering
        keys = sorted(features.keys())
        return np.array([features[k] for k in keys])

    # ── Micro Scale ─────────────────────────────────

    def _analyze_micro(self, notes: List[MIDINoteEvent],
                        total_duration: float) -> ScaleLevel:
        """Analyze at micro scale (10-50ms windows).
        
        Features:
        - onset_jitter: std of onset timing within each micro window
        - velocity_micro_variation: std of velocity within micro windows
        - articulation_micro_timing: mean note duration within micro windows
        - micro_note_density: notes per micro window
        - micro_onset_cluster: fraction of windows with multiple onsets
        """
        if total_duration <= 0:
            return ScaleLevel(SCALE_MICRO, MICRO_WINDOW, {})
        
        # Create micro windows
        n_windows = max(1, int(total_duration / MICRO_WINDOW))
        
        # Initialize per-window accumulators
        window_onsets = [[] for _ in range(n_windows)]
        window_velocities = [[] for _ in range(n_windows)]
        window_durations = [[] for _ in range(n_windows)]
        
        for n in notes:
            idx = min(int(n.start_sec / MICRO_WINDOW), n_windows - 1)
            window_onsets[idx].append(n.start_sec)
            window_velocities[idx].append(n.velocity)
            window_durations[idx].append(n.duration_sec)
        
        # Compute features
        onset_jitter_values = []
        velocity_variation_values = []
        articulation_timing_values = []
        density_values = []
        multi_onset_count = 0
        
        for i in range(n_windows):
            n_onsets = len(window_onsets[i])
            density_values.append(n_onsets)
            
            if n_onsets > 0:
                # Onset jitter within window
                if n_onsets > 1:
                    jitter = float(np.std(window_onsets[i]))
                else:
                    jitter = 0.0
                onset_jitter_values.append(jitter)
                
                # Velocity micro-variation
                if n_onsets > 1:
                    vel_var = float(np.std(window_velocities[i]))
                else:
                    vel_var = 0.0
                velocity_variation_values.append(vel_var)
                
                # Articulation micro-timing
                art_timing = float(np.mean(window_durations[i]))
                articulation_timing_values.append(art_timing)
                
                # Multi-onset clusters
                if n_onsets > 1:
                    multi_onset_count += 1
        
        return ScaleLevel(
            name=SCALE_MICRO,
            window_seconds=MICRO_WINDOW,
            features={
                "onset_jitter": float(np.mean(onset_jitter_values)) if onset_jitter_values else 0.0,
                "velocity_micro_variation": float(np.mean(velocity_variation_values)) if velocity_variation_values else 0.0,
                "articulation_micro_timing": float(np.mean(articulation_timing_values)) if articulation_timing_values else 0.0,
                "micro_note_density": float(np.mean(density_values)) if density_values else 0.0,
                "micro_onset_cluster_ratio": multi_onset_count / max(1, n_windows),
                "micro_peak_density": float(np.max(density_values)) if density_values else 0.0,
            }
        )

    # ── Note Scale ──────────────────────────────────

    def _analyze_note(self, notes: List[MIDINoteEvent],
                       total_duration: float) -> ScaleLevel:
        """Analyze at note scale (individual notes aggregated).
        
        Features:
        - mean_pitch: average pitch (MIDI number)
        - mean_velocity: average velocity
        - mean_duration: average note duration
        - pitch_variety: unique pitches / total notes
        - interval_variety: unique intervals
        - note_range: max - min pitch
        """
        if not notes:
            return ScaleLevel(SCALE_NOTE, NOTE_WINDOW, {})
        
        pitches = np.array([n.pitch for n in notes])
        velocities = np.array([n.velocity for n in notes])
        durations = np.array([n.duration_sec for n in notes])
        
        # Compute intervals between consecutive notes (chronological)
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)
        intervals = [abs(sorted_notes[i].pitch - sorted_notes[i-1].pitch) for i in range(1, len(sorted_notes))]
        
        return ScaleLevel(
            name=SCALE_NOTE,
            window_seconds=NOTE_WINDOW,
            features={
                "mean_pitch": float(np.mean(pitches)) if len(pitches) > 0 else 0.0,
                "pitch_std": float(np.std(pitches)) if len(pitches) > 1 else 0.0,
                "mean_velocity": float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
                "velocity_std": float(np.std(velocities)) if len(velocities) > 1 else 0.0,
                "mean_duration": float(np.mean(durations)) if len(durations) > 0 else 0.0,
                "duration_std": float(np.std(durations)) if len(durations) > 1 else 0.0,
                "pitch_variety": len(set(pitches)) / max(1, len(pitches)),
                "interval_variety": len(set(intervals)) / max(1, len(intervals)) if intervals else 0.0,
                "note_range": float(np.max(pitches) - np.min(pitches)) if len(pitches) > 1 else 0.0,
                "mean_interval": float(np.mean(intervals)) if intervals else 0.0,
            }
        )

    # ── Phrase Scale ────────────────────────────────

    def _detect_melodic_contour(self, pitches_window: List[int]) -> int:
        """Detect melodic contour direction.
        
        Returns:
            +1 = rising, -1 = falling, 0 = flat/neutral
        """
        if len(pitches_window) < 2:
            return 0
        
        # Linear regression slope
        x = np.arange(len(pitches_window))
        y = np.array(pitches_window)
        
        if np.std(y) == 0:
            return 0
        
        slope = float(np.polyfit(x, y, 1)[0])
        
        if slope > 2.0:
            return 1
        elif slope < -2.0:
            return -1
        else:
            return 0

    def _estimate_phrase_windows(self, notes: List[MIDINoteEvent],
                                   total_duration: float) -> int:
        """Estimate how many phrase-level windows to use.
        
        Uses IOI-based bar estimation: approx beats = duration / median_IOI
        Approx bars = beats / 4 → phrase ≈ 2-8 bars.
        """
        if len(notes) < 4:
            return max(1, int(total_duration / 2.0))
        
        starts = np.array([n.start_sec for n in notes])
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        
        if len(main_iois) == 0:
            return max(1, int(total_duration / PHRASE_MIN_SEC))
        
        beat_dur = float(np.median(main_iois))
        beat_dur = max(0.15, min(3.0, beat_dur))
        
        # Approx bars (4 beats per bar)
        total_beats = total_duration / beat_dur
        total_bars = total_beats / 4.0
        
        # Target 2-8 bars per phrase window
        # Aim for 4-bar phrases
        phrase_bar_target = 4.0
        n_phrases = max(1, int(total_bars / phrase_bar_target))
        
        # Cap at reasonable number
        return min(n_phrases, 50)

    def _analyze_phrase(self, notes: List[MIDINoteEvent],
                         total_duration: float) -> ScaleLevel:
        """Analyze at phrase scale (2-8 bar windows).
        
        Features:
        - melodic_contour_dir: avg contour (+1 up, -1 down, 0 flat)
        - dynamic_arc: how dynamics change (positive = crescendo, negative = diminuendo)
        - phrase_rhythmic_density: notes per second within phrase windows
        - register_center: median pitch across phrases
        - phrase_consistency: std of contour directions across phrases
        """
        if not notes or total_duration <= 0:
            return ScaleLevel(SCALE_PHRASE, PHRASE_MIN_SEC, {})
        
        n_windows = self._estimate_phrase_windows(notes, total_duration)
        window_len = total_duration / max(1, n_windows)
        
        if window_len < PHRASE_MIN_SEC:
            window_len = PHRASE_MIN_SEC
            n_windows = max(1, int(total_duration / window_len))
            # Recompute
            window_len = total_duration / max(1, n_windows)
        
        # Sort notes by time
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)
        
        # Per-window analysis
        contours = []
        dynamic_arcs = []
        densities = []
        register_centers = []
        
        for w in range(n_windows):
            win_start = w * window_len
            win_end = (w + 1) * window_len
            
            win_notes = [n for n in sorted_notes if win_start <= n.start_sec < win_end]
            
            if len(win_notes) < 2:
                if len(win_notes) == 1:
                    contours.append(0)
                    densities.append(1.0 / max(0.001, window_len))
                    register_centers.append(win_notes[0].pitch)
                    # Can't compute arc from 1 note
                continue
            
            # Melodic contour
            win_pitches = [n.pitch for n in win_notes]
            contour = self._detect_melodic_contour(win_pitches)
            contours.append(contour)
            
            # Dynamic arc (first half avg vel vs second half avg vel)
            mid = len(win_notes) // 2
            first_half_vel = np.mean([n.velocity for n in win_notes[:mid]]) if mid > 0 else 64
            second_half_vel = np.mean([n.velocity for n in win_notes[mid:]]) if len(win_notes) - mid > 0 else 64
            dynamic_arc = second_half_vel - first_half_vel
            dynamic_arcs.append(float(dynamic_arc))
            
            # Rhythmic density
            density = len(win_notes) / max(0.001, window_len)
            densities.append(density)
            
            # Register center (median pitch)
            register_centers.append(float(np.median(win_pitches)))
        
        return ScaleLevel(
            name=SCALE_PHRASE,
            window_seconds=window_len,
            features={
                "melodic_contour_mean": float(np.mean(contours)) if contours else 0.0,
                "melodic_contour_std": float(np.std(contours)) if len(contours) > 1 else 0.0,
                "dynamic_arc_mean": float(np.mean(dynamic_arcs)) if dynamic_arcs else 0.0,
                "dynamic_arc_std": float(np.std(dynamic_arcs)) if len(dynamic_arcs) > 1 else 0.0,
                "phrase_rhythmic_density": float(np.mean(densities)) if densities else 0.0,
                "register_center": float(np.mean(register_centers)) if register_centers else 0.0,
                "register_center_std": float(np.std(register_centers)) if len(register_centers) > 1 else 0.0,
                "phrase_count": n_windows,
            }
        )

    # ── Section Scale ───────────────────────────────

    def _analyze_section(self, notes: List[MIDINoteEvent],
                          total_duration: float) -> ScaleLevel:
        """Analyze at section scale (8-32 bar windows).
        
        Features:
        - harmonic_rhythm: pitch class change rate (unique PCs per second)
        - section_dynamic_range: velocity range within section
        - texture_density: avg simultaneous notes
        - structural_boundary_strength: how distinct adjacent sections are
        - section_count: number of sections detected
        """
        if not notes or total_duration <= 0:
            return ScaleLevel(SCALE_SECTION, SECTION_MIN_SEC, {})
        
        sorted_notes = sorted(notes, key=lambda n: n.start_sec)
        
        # Determine section windows: target 8-32 bars
        # Estimate from IOI
        starts = np.array([n.start_sec for n in sorted_notes])
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        
        if len(main_iois) > 0:
            beat_dur = float(np.median(main_iois))
            beat_dur = max(0.15, min(3.0, beat_dur))
            total_bars = total_duration / (beat_dur * 4.0)
        else:
            total_bars = total_duration / 2.0  # assume ~2s per bar
        
        # Target 16-bar sections
        section_bar_target = 16.0
        n_sections = max(1, int(total_bars / section_bar_target))
        # Cap at reasonable
        n_sections = min(n_sections, 20)
        
        window_len = total_duration / max(1, n_sections)
        
        if window_len < SECTION_MIN_SEC:
            window_len = SECTION_MIN_SEC
            n_sections = max(1, int(total_duration / window_len))
            window_len = total_duration / max(1, n_sections)
        
        # Per-section analysis
        harmonic_rhythms = []
        section_dyn_ranges = []
        texture_densities = []
        section_features = []
        
        for w in range(n_sections):
            win_start = w * window_len
            win_end = (w + 1) * window_len
            
            win_notes = [n for n in sorted_notes if win_start <= n.start_sec < win_end]
            
            if not win_notes:
                harmonic_rhythms.append(0.0)
                section_dyn_ranges.append(0.0)
                texture_densities.append(0.0)
                continue
            
            # Harmonic rhythm: unique pitch classes per second
            if window_len > 0:
                pc_set = set(n.pitch % 12 for n in win_notes)
                harm_rhythm = len(pc_set) / window_len
                harmonic_rhythms.append(harm_rhythm)
            else:
                harmonic_rhythms.append(0.0)
            
            # Dynamic range within section
            win_vels = np.array([n.velocity for n in win_notes])
            section_dyn_ranges.append(float(np.max(win_vels) - np.min(win_vels)))
            
            # Texture density: average simultaneous notes
            # Sample at multiple points within the window
            n_samples = max(2, int(window_len / 0.25))  # sample every 250ms
            overlap_counts = []
            for s in range(n_samples):
                t = win_start + (s / max(1, n_samples)) * window_len
                active = sum(1 for n in win_notes if n.start_sec <= t < n.start_sec + n.duration_sec)
                overlap_counts.append(active)
            texture_densities.append(float(np.mean(overlap_counts)) if overlap_counts else 0.0)
            
            # Store aggregate features for boundary detection
            if win_notes:
                win_pitches = [n.pitch for n in win_notes]
                win_vels_arr = np.array([n.velocity for n in win_notes])
                section_features.append({
                    "mean_pitch": float(np.mean(win_pitches)),
                    "mean_vel": float(np.mean(win_vels_arr)),
                    "density": len(win_notes) / max(0.001, window_len),
                })
            else:
                section_features.append({})
        
        # Structural boundary strength: how distinct adjacent sections are
        boundary_strengths = []
        for s in range(1, len(section_features)):
            a = section_features[s - 1]
            b = section_features[s]
            if a and b:
                # Euclidean distance between section feature vectors
                keys = set(a.keys()) & set(b.keys())
                if keys:
                    dist = np.sqrt(sum((a[k] - b[k]) ** 2 for k in keys))
                    boundary_strengths.append(dist)
        
        return ScaleLevel(
            name=SCALE_SECTION,
            window_seconds=window_len,
            features={
                "harmonic_rhythm": float(np.mean(harmonic_rhythms)) if harmonic_rhythms else 0.0,
                "harmonic_rhythm_std": float(np.std(harmonic_rhythms)) if len(harmonic_rhythms) > 1 else 0.0,
                "section_dynamic_range": float(np.mean(section_dyn_ranges)) if section_dyn_ranges else 0.0,
                "section_dynamic_range_std": float(np.std(section_dyn_ranges)) if len(section_dyn_ranges) > 1 else 0.0,
                "texture_density": float(np.mean(texture_densities)) if texture_densities else 0.0,
                "texture_density_std": float(np.std(texture_densities)) if len(texture_densities) > 1 else 0.0,
                "structural_boundary_strength": float(np.mean(boundary_strengths)) if boundary_strengths else 0.0,
                "structural_boundary_max": float(np.max(boundary_strengths)) if boundary_strengths else 0.0,
                "section_count": n_sections,
            }
        )

    # ── Piece Scale ─────────────────────────────────

    def _analyze_piece(self, notes: List[MIDINoteEvent],
                        total_duration: float) -> ScaleLevel:
        """Analyze at piece scale (full duration).
        
        Features are derived from the existing TrackStyle dimensions
        that operate at the piece level plus compositional attributes.
        
        Features:
        - rest_ratio: fraction of time silent
        - note_density: notes per second
        - timing_consistency: std of onset deviation
        - staccato_ratio: fraction of short notes
        - dynamic_range: max - min velocity
        - syncopation_index: off-beat ratio
        - harmonic_complexity: avg unique PCs per 2s window
        - register_breadth: pitch span
        - n_notes: total note count
        - duration: total duration in seconds
        """
        if not notes:
            return ScaleLevel(SCALE_PIECE, total_duration, {})
        
        # Extract style to get the piece-level dimensions
        from plato_midi_bridge.decompose import extract_track_style
        style = extract_track_style(notes, total_duration)
        
        # Compute additional piece-level metrics
        starts = np.array([n.start_sec for n in notes])
        pitches = np.array([n.pitch for n in notes])
        
        # Total active note-seconds (sum of all durations)
        total_active = sum(n.duration_sec for n in notes)
        
        # Density variability: std of notes per second in 1s chunks
        one_sec_chunks = max(1, int(total_duration))
        chunk_counts = []
        for c in range(one_sec_chunks):
            c_start = float(c)
            c_end = c_start + 1.0
            count = np.sum((starts >= c_start) & (starts < c_end))
            chunk_counts.append(count)
        
        return ScaleLevel(
            name=SCALE_PIECE,
            window_seconds=total_duration,
            features={
                "rest_ratio": float(style.rest_ratio),
                "note_density": float(style.note_density),
                "timing_consistency": float(style.timing_consistency),
                "staccato_ratio": float(style.staccato_ratio),
                "dynamic_range": float(style.dynamic_range),
                "syncopation_index": float(style.syncopation_index),
                "harmonic_complexity": float(style.harmonic_complexity),
                "register_breadth": float(style.register_breadth),
                "avg_interval": float(style.avg_interval),
                "total_active_seconds": float(total_active),
                "density_variability": float(np.std(chunk_counts)) if len(chunk_counts) > 1 else 0.0,
                "n_notes": len(notes),
                "duration": total_duration,
            }
        )


# ── Integration helper ──────────────────────────────

def analyze_notes_multi_scale(notes: List[MIDINoteEvent],
                               total_duration: float) -> Dict[str, ScaleLevel]:
    """Convenience function: analyze a track's notes at all scales."""
    analyzer = MultiScaleAnalyzer()
    return analyzer.analyze(notes, total_duration)


def compute_scales_coupling(scales: Dict[str, ScaleLevel]) -> ScaleCoupling:
    """Convenience function: compute coupling between adjacent scales."""
    analyzer = MultiScaleAnalyzer()
    return analyzer.compute_scale_coupling(scales)


def scale_level_to_tile_dict(level: ScaleLevel) -> dict:
    """Convert a ScaleLevel to a PLATO-compatible dictionary."""
    return {
        "scale": level.name,
        "window_seconds": level.window_seconds,
        "features": level.features,
    }


def scale_coupling_to_dict(coupling: ScaleCoupling) -> dict:
    """Convert a ScaleCoupling to a PLATO-compatible dictionary."""
    return {
        "scale_pairs": {f"{a}→{b}": v for (a, b), v in coupling.scale_pairs.items()},
        "inflation_ratios": {f"{a}→{b}": v for (a, b), v in coupling.inflation_ratios.items()},
    }
