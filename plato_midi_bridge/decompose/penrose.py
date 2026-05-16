"""
Penrose Encoding Experiment — 5D Cut-and-Project Style Encoder.

Implements Penrose tiling encoding of music style vectors using the
cut-and-project method from the FLUX Penrose expertise module.

Two encoding schemes compared:
  1. Eisenstein (12-chamber) — harmonic relationships
  2. Penrose (5D→2D cut-and-project) — expressive relationships
  3. Combined (17D) — benefits of both
"""

from __future__ import annotations
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ── Constants ───────────────────────────────────────

PHI = (1.0 + math.sqrt(5.0)) / 2.0  # ~1.618

# 5th roots of unity angles
THETA = 2.0 * math.pi / 5.0
_C1 = math.cos(THETA)
_C2 = math.cos(2.0 * THETA)
_C3 = math.cos(3.0 * THETA)
_C4 = math.cos(4.0 * THETA)
_S1 = math.sin(THETA)
_S2 = math.sin(2.0 * THETA)
_S3 = math.sin(3.0 * THETA)
_S4 = math.sin(4.0 * THETA)

# Projection matrix: 5D → 2D using 5th roots of unity
# Unnormalized (matches spec)
PROJECTION = np.array([
    [1.0, _C1, _C2, _C3, _C4],
    [0.0, _S1, _S2, _S3, _S4],
], dtype=np.float64)

# Perpendicular projection: 5D → 2D (for decagon acceptance check)
# Uses doubled angles: 2*THETA, 4*THETA, 6*THETA, 8*THETA
PERP_PROJECTION = np.array([
    [1.0, _C2, _C4, _C1, _C3],
    [0.0, _S2, _S4, _S1, _S3],
], dtype=np.float64)

# Decagon vertices for a unit-radius decagon
_DECAGON_ANGLES = [math.pi / 10.0 + k * math.pi / 5.0 for k in range(10)]
_DECAGON_VERTICES = np.array([
    [math.cos(a), math.sin(a)] for a in _DECAGON_ANGLES
], dtype=np.float64)

# Style dimension names
STYLE_DIMENSIONS = [
    "pitch_complexity",
    "timing_expressiveness",
    "velocity_energy",
    "articulation_clarity",
    "timbral_breadth",
]


# ── Helper: Decagon containment test ────────────────

def _point_in_decagon(point: np.ndarray, radius: float = 2.0) -> bool:
    """Test if a 2D point lies inside a regular decagon.
    
    Uses the half-plane method: for each edge of the decagon,
    check that the point is on the correct side.
    
    Args:
        point: 2D point (x, y)
        radius: Circumscribed circle radius
        
    Returns:
        True if point is inside or on the decagon edge
    """
    vertices = _DECAGON_VERTICES * radius
    n = len(vertices)
    
    for i in range(n):
        v0 = vertices[i]
        v1 = vertices[(i + 1) % n]
        edge = v1 - v0
        # Cross product: edge × (point - v0)
        cross = edge[0] * (point[1] - v0[1]) - edge[1] * (point[0] - v0[0])
        if cross < -1e-10:  # negative = right of edge = outside for CCW winding
            return False
    
    return True


def _generate_decagon_vertices(radius: float) -> np.ndarray:
    """Generate vertices of a regular decagon."""
    return _DECAGON_VERTICES * radius


# ── Part A: Penrose Encoder ─────────────────────────

@dataclass
class PenroseTiling:
    """A Penrose tiling generated from a style vector."""
    points: np.ndarray  # (n_points, 2) array of 2D accepted points
    style_vector: np.ndarray  # original 5D style vector
    acceptance_threshold: float  # radius of decagon window
    
    # Derived metrics
    point_density: float = 0.0
    radial_distribution: float = 0.0
    symmetry_score: float = 0.0
    closest_pair_distance: float = 0.0
    vertex_count: int = 0
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary (PLATO tile compatible)."""
        return {
            "encoding": "penrose_5d_cut_and_project",
            "style_vector": self.style_vector.tolist(),
            "acceptance_threshold": self.acceptance_threshold,
            "n_points": len(self.points),
            "point_density": float(self.point_density),
            "radial_distribution": float(self.radial_distribution),
            "symmetry_score": float(self.symmetry_score),
            "closest_pair_distance": float(self.closest_pair_distance),
            "vertex_count": self.vertex_count,
            "centroid": self.points.mean(axis=0).tolist() if len(self.points) > 0 else [0.0, 0.0],
        }


class PenroseEncoder:
    """
    5D → 2D cut-and-project encoding for style vectors.
    
    Maps 5 musical dimensions (pitch, time, velocity, articulation, timbre)
    to a 2D Penrose tiling via 5D Z^5 lattice projection.
    
    The 5D → 2D projection uses the 5th roots of unity:
      proj[i] = sum_{j=0}^{4} v[j] * cos(2*pi*j*(i+1)/5) for i in 0,1
    
    Acceptance window: regular decagon with radius = threshold
    """
    
    def __init__(self, threshold: float = 2.0, lattice_scale: float = 3.0):
        """
        Args:
            threshold: Radius of the decagon acceptance window (default 2.0)
            lattice_scale: Controls how many Z^5 lattice points to sample around the style vector.
                           Higher = more points = finer tiling (default 3.0)
        """
        self.threshold = threshold
        self.lattice_scale = lattice_scale
        self._projection = PROJECTION
        self._perp_projection = PERP_PROJECTION
    
    def encode(self, style_vector: np.ndarray) -> PenroseTiling:
        """Encode 5D style vector as 2D Penrose tiling points.
        
        Uses the standard cut-and-project method:
        1. Z^5 lattice points are generated around a center determined by the style vector
        2. Each point is projected to physical 2D space (P @ z) and perpendicular 2D space (P_perp @ z)
        3. Points whose PERPENDICULAR projection falls within the decagon acceptance window
           are accepted — their PHYSICAL projection becomes a tiling vertex
        
        The 5D vector dimensions map to:
        0: pitch_complexity (from note range and pitch variety)
        1: timing_expressiveness (onset_jitter, syncopation)
        2: velocity_energy (dynamic_range, mean_velocity)
        3: articulation_clarity (staccato_ratio, duration_std)
        4: timbral_breadth (harmonic_complexity, register_breadth)
        
        Returns:
            PenroseTiling with (n_points, 2) array of accepted points
        """
        vec = np.asarray(style_vector, dtype=np.float64).flatten()
        if vec.shape[0] != 5:
            raise ValueError(f"Style vector must be 5D, got {vec.shape[0]}D")
        
        # Clamp to [0, 1] range
        vec = np.clip(vec, 0.0, 1.0)
        
        # Determine Z^5 lattice center from style vector
        center = vec * self.lattice_scale
        
        # Generate Z^5 lattice point ranges around the center
        # Use a fixed range [-lat_scale, lat_scale] shifted by center
        ranges = []
        for i in range(5):
            low = int(np.floor(center[i] - self.lattice_scale))
            high = int(np.ceil(center[i] + self.lattice_scale))
            ranges.append(list(range(low, high + 1)))
        
        # Estimate total combinations
        total_combinations = 1
        for r in ranges:
            total_combinations *= len(r)
        
        # Pre-compute projection columns for efficiency
        proj_cols = [self._projection[:, i] for i in range(5)]
        perp_cols = [self._perp_projection[:, i] for i in range(5)]
        
        accepted = []
        
        # Use stride sampling if grid is too large
        if total_combinations > 100000:
            strides = [max(1, len(r) // max(1, int(total_combinations ** 0.2))) for r in ranges]
            coords = [np.array(r, dtype=np.int64)[::s] for r, s in zip(ranges, strides)]
        else:
            coords = [np.array(r, dtype=np.int64) for r in ranges]
        
        # Iterate over 5D lattice points using nested loops
        # Cut-and-project: accept point iff perpendicular projection is inside decagon
        for z0 in coords[0]:
            c0_phys = proj_cols[0] * z0
            c0_perp = perp_cols[0] * z0
            for z1 in coords[1]:
                c1_phys = c0_phys + proj_cols[1] * z1
                c1_perp = c0_perp + perp_cols[1] * z1
                for z2 in coords[2]:
                    c2_phys = c1_phys + proj_cols[2] * z2
                    c2_perp = c1_perp + perp_cols[2] * z2
                    for z3 in coords[3]:
                        c3_phys = c2_phys + proj_cols[3] * z3
                        c3_perp = c2_perp + perp_cols[3] * z3
                        for z4 in coords[4]:
                            # Perpendicular projection — used for acceptance
                            x_perp = c3_perp + perp_cols[4] * z4
                            if _point_in_decagon(x_perp, self.threshold):
                                # Physical projection — this is the tiling vertex
                                x_phys = c3_phys + proj_cols[4] * z4
                                accepted.append(x_phys.copy())
        
        accepted = np.array(accepted, dtype=np.float64) if accepted else np.zeros((0, 2), dtype=np.float64)
        
        # Compute derived metrics
        return self._compute_tiling(accepted, vec)
    
    def _compute_tiling(self, points: np.ndarray, style_vector: np.ndarray) -> PenroseTiling:
        """Compute all derived metrics for a set of accepted points."""
        n = len(points)
        
        if n == 0:
            return PenroseTiling(
                points=points,
                style_vector=style_vector,
                acceptance_threshold=self.threshold,
                point_density=0.0,
                radial_distribution=0.0,
                symmetry_score=0.0,
                closest_pair_distance=0.0,
                vertex_count=0,
            )
        
        # Point density
        area = math.pi * self.threshold ** 2
        point_density = n / max(area, 0.01)
        
        # Radial distribution: how points spread from center
        # Lower = more concentrated near center
        centroid = points.mean(axis=0)
        radii = np.linalg.norm(points - centroid, axis=1)
        radial_distribution = float(np.std(radii)) if len(radii) > 1 else 0.0
        
        # Symmetry score: measure 5-fold rotational symmetry
        symmetry_score = self._compute_symmetry(points, centroid)
        
        # Closest pair distance
        closest_pair_distance = self._compute_closest_pair(points)
        
        # Vertex count (from Delaunay triangulation if available, else n)
        vertex_count = self._delaunay_vertex_count(points)
        
        return PenroseTiling(
            points=points,
            style_vector=style_vector,
            acceptance_threshold=self.threshold,
            point_density=point_density,
            radial_distribution=radial_distribution,
            symmetry_score=symmetry_score,
            closest_pair_distance=closest_pair_distance,
            vertex_count=vertex_count,
        )
    
    def _compute_symmetry(self, points: np.ndarray, centroid: np.ndarray) -> float:
        """Compute 5-fold rotational symmetry score.
        
        Rotates points by 72° around centroid and measures overlap
        with original set. Returns [0, 1] where 1 = perfect 5-fold symmetry.
        """
        if len(points) < 5:
            return 0.0
        
        centered = points - centroid
        
        # Put points in polar coordinates and bin by angle
        angles = np.arctan2(centered[:, 1], centered[:, 0])
        radii = np.linalg.norm(centered, axis=1)
        
        # 5-fold symmetry: after 72° rotation, points should map near themselves
        # Compare angle distribution modulo 72°
        angles_mod = angles % (2 * math.pi / 5.0)
        
        # If perfectly symmetric, angles_mod should be tightly clustered
        # near specific values (the fundamental domain boundaries)
        std_angles = float(np.std(angles_mod))
        
        # Normalize: 0 std = perfect symmetry → score 1.0
        # Max practical std for uniform = ~0.9 rad → score 0.0
        symmetry = max(0.0, 1.0 - std_angles / 0.9)
        return symmetry
    
    def _compute_closest_pair(self, points: np.ndarray) -> float:
        """Compute minimum distance between any two points."""
        if len(points) < 2:
            return 0.0
        
        # Brute-force for small sets, sample for large
        if len(points) <= 500:
            min_dist = float('inf')
            for i in range(len(points)):
                dists = np.linalg.norm(points[i+1:] - points[i], axis=1)
                if len(dists) > 0:
                    d = float(dists.min())
                    if d < min_dist:
                        min_dist = d
            return float(min_dist) if min_dist < float('inf') else 0.0
        else:
            # Randomly sample 500 pairs
            n = len(points)
            indices = np.random.choice(n, min(500, n), replace=False)
            min_dist = float('inf')
            for i in indices:
                dists = np.linalg.norm(points[i+1:] - points[i], axis=1)
                if len(dists) > 0:
                    d = float(dists.min())
                    if d < min_dist:
                        min_dist = d
            return float(min_dist)
    
    def _delaunay_vertex_count(self, points: np.ndarray) -> int:
        """Count vertices in Delaunay triangulation.
        
        Returns number of vertices (simplices) of the triangulation.
        Uses scipy.spatial.Delaunay if available, else approximate.
        """
        if len(points) < 3:
            return len(points)
        
        try:
            from scipy.spatial import Delaunay
            tri = Delaunay(points)
            return tri.nsimplex
        except ImportError:
            # Fallback: approximate vertex count
            # Number of simplices in Delaunay ≈ 2 * n - 2 for convex sets
            return 2 * len(points) - 2
    
    def inflation(self, points: np.ndarray, factor: float = PHI) -> np.ndarray:
        """Inflate a Penrose tiling by factor φ.
        
        Each point's position is scaled → finer subdivisions.
        Inflation maps a coarse tiling to a finer one by
        scaling all point coordinates by factor.
        
        Args:
            points: (n, 2) array of 2D points
            factor: Inflation factor (default φ ≈ 1.618)
            
        Returns:
            (n, 2) inflated points
        """
        return points * factor
    
    def deflation(self, points: np.ndarray) -> Tuple[np.ndarray, float]:
        """Deflate: coarser structure.
        
        Inverse of inflation: shrinks points by factor 1/φ.
        
        Args:
            points: (n, 2) array of 2D points
            
        Returns:
            (deflated_points, deflation_factor)
            deflation_factor = 1/φ ≈ 0.618
        """
        deflation_factor = 1.0 / PHI
        return points * deflation_factor, deflation_factor
    
    def tiling_signature(self, points: np.ndarray) -> Dict:
        """Statistical fingerprint of the tiling.
        
        Args:
            points: (n, 2) array of 2D Penrose points
            
        Returns:
            Dict with signature metrics
        """
        # Compute tiling metrics
        tiling = self._compute_tiling(points, np.zeros(5))
        return {
            "point_density": float(tiling.point_density),
            "radial_distribution": float(tiling.radial_distribution),
            "symmetry_score": float(tiling.symmetry_score),
            "closest_pair_distance": float(tiling.closest_pair_distance),
            "vertex_count": tiling.vertex_count,
        }


# ── Part B: Encoding Comparison Experiment ─────────

@dataclass
class EncodingResult:
    """Results of comparing encoding schemes for a set of pieces."""
    eisenstein: Dict[str, float]  # intra, inter, silhouette
    penrose: Dict[str, float]
    combined: Dict[str, float]
    winner: str  # 'eisenstein' | 'penrose' | 'combined'
    
    def summary(self) -> str:
        return (
            f"Encoding Comparison:\n"
            f"  Eisenstein: intra={self.eisenstein['intra']:.4f}, "
            f"inter={self.eisenstein['inter']:.4f}, "
            f"silhouette={self.eisenstein['silhouette']:.4f}\n"
            f"  Penrose:    intra={self.penrose['intra']:.4f}, "
            f"inter={self.penrose['inter']:.4f}, "
            f"silhouette={self.penrose['silhouette']:.4f}\n"
            f"  Combined:   intra={self.combined['intra']:.4f}, "
            f"inter={self.combined['inter']:.4f}, "
            f"silhouette={self.combined['silhouette']:.4f}\n"
            f"  Winner: {self.winner}"
        )


class EncodingExperiment:
    """
    Compare Eisenstein (12-chamber) vs Penrose (5D cut-and-project) encoding.
    
    Hypothesis:
    - Eisenstein better captures harmonic relationships (which notes/rooms)
    - Penrose better captures expressive relationships (how they feel)
    - Combined (17D) best of both
    """
    
    def __init__(self):
        self.penrose = PenroseEncoder()
    
    def compare(self, pieces: List[Dict]) -> EncodingResult:
        """
        For each piece, encode its style vector in both:
        - Eisenstein: 12-chamber assignment
        - Penrose: 5D→2D projection signature
        - Combined: 12 Eisenstein + 5 Penrose = 17D
        
        Each piece dict should have:
            'composer': str (same composer = intra-class)
            'style': np.ndarray (TrackStyle.to_vector() or similar)
            'name': str (optional)
        
        Metrics:
        - Same-composer intra-class distance (lower = better)
        - Different-composer inter-class distance (higher = better)
        - Silhouette score (ranges -1 to 1, higher = better clustering)
        
        Returns:
            EncodingResult with metrics for each scheme
        """
        if len(pieces) < 2:
            return EncodingResult(
                eisenstein={'intra': 0, 'inter': 0, 'silhouette': 0},
                penrose={'intra': 0, 'inter': 0, 'silhouette': 0},
                combined={'intra': 0, 'inter': 0, 'silhouette': 0},
                winner='eisenstein'
            )
        
        # Assign composer labels
        composers = [p.get('composer', 'unknown') for p in pieces]
        unique_composers = list(set(composers))
        
        # Extract style vectors
        style_vectors = []
        for p in pieces:
            style = p.get('style_5d', self._to_5d_style(p.get('style', None)))
            style_vectors.append(np.asarray(style, dtype=np.float64).flatten())
        
        # ── 1. Eisenstein encoding ──
        # Map 5D style to 12-chamber vector
        eisenstein_vectors = [self._to_eisenstein(sv) for sv in style_vectors]
        eisenstein_metrics = self._compute_encoding_metrics(
            eisenstein_vectors, composers, unique_composers
        )
        
        # ── 2. Penrose encoding ──
        # Encode to 2D Penrose projection signature
        penrose_vectors = [self._to_penrose_signature(sv) for sv in style_vectors]
        penrose_metrics = self._compute_encoding_metrics(
            penrose_vectors, composers, unique_composers
        )
        
        # ── 3. Combined encoding ──
        # Concatenate: 12 Eisenstein + 5 Penrose signature dimensions = 17D
        combined_vectors = []
        for ev, pv in zip(eisenstein_vectors, penrose_vectors):
            combined_vectors.append(np.concatenate([ev, pv]))
        combined_metrics = self._compute_encoding_metrics(
            combined_vectors, composers, unique_composers
        )
        
        # Determine winner: best silhouette score
        scores = {
            'eisenstein': eisenstein_metrics['silhouette'],
            'penrose': penrose_metrics['silhouette'],
            'combined': combined_metrics['silhouette'],
        }
        winner = max(scores, key=scores.get)
        
        return EncodingResult(
            eisenstein=eisenstein_metrics,
            penrose=penrose_metrics,
            combined=combined_metrics,
            winner=winner,
        )
    
    def _to_5d_style(self, style_vector: Optional[np.ndarray]) -> np.ndarray:
        """Convert any style vector to 5D by selecting relevant dimensions.
        
        If the vector is long (109D from TrackStyle.to_vector()), extract
        the relevant indices. Otherwise, assume it's already 5D or pad.
        """
        if style_vector is None:
            return np.zeros(5)
        
        vec = np.asarray(style_vector, dtype=np.float64).flatten()
        
        if len(vec) == 5:
            return vec
        elif len(vec) >= 109:
            # TrackStyle.to_vector(): 109D
            # Indices for our 5 dimensions (approximate mapping):
            # pitch_complexity: from pitch features (indices 0-47) + avg_interval (index 103)
            # timing_expressiveness: timing_consistency (index 80)
            # velocity_energy: from velocity features (indices 48-79) 
            # articulation_clarity: staccato_ratio (index 81)
            # timbral_breadth: harmonic_complexity (index 107)
            idx_pitch = 103  # avg_interval, but normalized
            idx_timing = 80   # timing_consistency
            idx_velocity = 81 # repurpose: use staccato complement + dynamic_range
            idx_articulation = 82  # rest_ratio complement
            idx_timbre = 107  # harmonic_complexity
            
            # Build 5D vector with sensible defaults
            pitch = float(vec[103]) / 12.0 if len(vec) > 103 else 0.5  # avg_interval normalized
            timing = float(vec[80]) * 10.0 if len(vec) > 80 else 0.5   # timing_consistency
            velocity = float(vec[104]) / 127.0 if len(vec) > 104 else 0.5  # dynamic_range
            articulation = float(vec[81]) if len(vec) > 81 else 0.5     # staccato_ratio
            timbre = float(vec[107]) / 12.0 if len(vec) > 107 else 0.5  # harmonic_complexity
            
            return np.clip([pitch, timing, velocity, articulation, timbre], 0.0, 1.0)
        else:
            # Unknown vector; extract first 5 or pad
            if len(vec) >= 5:
                return vec[:5]
            return np.pad(vec, (0, 5 - len(vec)))
    
    def _to_eisenstein(self, style_5d: np.ndarray) -> np.ndarray:
        """Convert 5D style to 12-chamber Eisenstein vector.
        
        Each chamber C, C#, D, ... B gets an activation level based on
        how the style vector projects onto that chamber's characteristic
        emotional quality.
        
        Returns: 12D vector of chamber activations
        """
        from plato_midi_bridge.tensor import EISENSTEIN_CHAMBERS
        
        chamber_activations = np.zeros(12, dtype=np.float64)
        
        for i, chamber in enumerate(EISENSTEIN_CHAMBERS):
            # Map style dimensions to chamber quality/emotion
            root = chamber["root"]
            
            # Activation based on style vector projection onto chamber
            # pitch_complexity → chamber root spacing
            pitch_contrib = style_5d[0] * math.cos(2 * math.pi * root / 12.0)
            
            # timing_expressiveness → minor/major quality
            timing_contrib = style_5d[1] * (1.0 if root % 2 == 0 else -1.0)
            
            # velocity_energy → fifth/fourth relationship
            vel_contrib = style_5d[2] * math.sin(2 * math.pi * (root % 7) / 7.0)
            
            # articulation_clarity → staccato/legato quality
            art_contrib = style_5d[3] * (1.0 if root < 6 else -1.0)
            
            # timbral_breadth → harmonic richness
            timbre_contrib = style_5d[4] * (1.0 - abs(root - 6) / 6.0)
            
            activation = (pitch_contrib + timing_contrib + vel_contrib 
                         + art_contrib + timbre_contrib) / 5.0
            
            chamber_activations[i] = max(0.0, activation)
        
        # Normalize
        total = chamber_activations.sum()
        if total > 0:
            chamber_activations = chamber_activations / total
        
        return chamber_activations
    
    def _to_penrose_signature(self, style_5d: np.ndarray) -> np.ndarray:
        """Convert 5D style to Penrose tiling signature vector.
        
        Encodes the style vector and extracts signature metrics:
        [point_density, radial_distribution, symmetry_score, 
         closest_pair_distance, vertex_count_normalized]
        
        Returns: 5D signature vector
        """
        tiling = self.penrose.encode(style_5d)
        
        # Normalize vertex count by max possible
        max_vertices = 100.0  # arbitrary normalization cap
        vertex_norm = min(1.0, tiling.vertex_count / max_vertices)
        
        return np.array([
            min(1.0, tiling.point_density / 10.0),
            min(1.0, tiling.radial_distribution / 5.0),
            tiling.symmetry_score,
            min(1.0, tiling.closest_pair_distance * 5.0),
            vertex_norm,
        ], dtype=np.float64)
    
    def _compute_encoding_metrics(
        self,
        vectors: List[np.ndarray],
        composers: List[str],
        unique_composers: List[str],
    ) -> Dict[str, float]:
        """Compute intra/inter-class distances and silhouette score."""
        n = len(vectors)
        if n < 2 or len(unique_composers) < 1:
            return {'intra': 0, 'inter': 0, 'silhouette': 0}
        
        # Compute pairwise distance matrix
        vec_arr = np.array(vectors)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(vec_arr[i] - vec_arr[j]))
                dist_matrix[i][j] = d
                dist_matrix[j][i] = d
        
        # Intra-class: mean distance between same-composer pairs
        intra_distances = []
        for i in range(n):
            for j in range(i + 1, n):
                if composers[i] == composers[j]:
                    intra_distances.append(dist_matrix[i][j])
        
        # Inter-class: mean distance between different-composer pairs
        inter_distances = []
        for i in range(n):
            for j in range(n):
                if i != j and composers[i] != composers[j]:
                    inter_distances.append(dist_matrix[i][j])
        
        intra = float(np.mean(intra_distances)) if intra_distances else 0.0
        inter = float(np.mean(inter_distances)) if inter_distances else 0.0
        
        # Silhouette score: (inter - intra) / max(inter, intra)
        denom = max(inter, intra, 1e-10)
        silhouette = (inter - intra) / denom if denom > 0 else 0.0
        
        return {'intra': intra, 'inter': inter, 'silhouette': silhouette}


# ── Part C: StylePCA — Manual Numpy PCA ──────────────────

class StylePCA:
    """PCA reducer for style vectors. Same architecture as plato-soul-fingerprint.
    
    Uses manual numpy PCA (eigendecomposition of covariance matrix) —
    no sklearn dependency.
    """
    
    def __init__(self):
        self.components_ = None  # (n_components, n_dims)
        self.mean_ = None  # (n_dims,)
        self.explained_variance_ratio_ = None  # (n_components,)
        self.n_components_ = 0
        self.n_features_ = 0
    
    def fit(self, vectors: np.ndarray) -> 'StylePCA':
        """Fit PCA to the data.
        
        Args:
            vectors: (n_pieces, n_dims) array of style vectors
            
        Returns:
            self (fitted)
        """
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        n, d = vectors.shape
        self.n_features_ = d
        
        # Center the data
        self.mean_ = np.mean(vectors, axis=0)
        centered = vectors - self.mean_
        
        # Covariance matrix: (d, d)
        cov = np.cov(centered, rowvar=False)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Keep all components (user can truncate later)
        self.components_ = eigenvectors.T  # (d, d) → components as rows
        
        # Explained variance ratio
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = eigenvalues / max(total_var, 1e-10)
        self.n_components_ = d
        
        return self
    
    def transform(self, vectors: np.ndarray, n_components: int = None) -> np.ndarray:
        """Project vectors into PCA space.
        
        Args:
            vectors: (n_pieces, n_dims) or (n_dims,) array
            n_components: Number of dimensions to keep. Defaults to all.
            
        Returns:
            (n_pieces, n_components) or (n_components,) projected vectors
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        vectors = np.asarray(vectors, dtype=np.float64)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        centered = vectors - self.mean_
        
        n_components = n_components or self.n_components_
        n_components = min(n_components, self.n_components_)
        
        # Project using first k components
        projected = centered @ self.components_[:n_components].T
        
        if projected.shape[0] == 1:
            return projected.flatten()
        return projected
    
    def fit_transform(self, vectors: np.ndarray, n_components: int = None) -> np.ndarray:
        """Fit PCA and transform in one step."""
        self.fit(vectors)
        return self.transform(vectors, n_components)
    
    def cumulative_variance(self, threshold: float = 0.95) -> int:
        """Return the minimum number of components needed to reach threshold variance."""
        if self.explained_variance_ratio_ is None:
            return 0
        cumulative = np.cumsum(self.explained_variance_ratio_)
        return int(np.searchsorted(cumulative, threshold) + 1)


def snap_to_pythagorean(vector: np.ndarray, density: int = 100) -> np.ndarray:
    """Snap to nearest Pythagorean grid point.
    
    Same as FM's plato-soul-fingerprint module.
    Maps each vector component to the nearest multiple of 1/density.
    
    Args:
        vector: (n,) array of values
        density: Grid density (default 100 → grid points at 0.01 intervals)
        
    Returns:
        (n,) array snapped to grid
    """
    return np.round(np.asarray(vector, dtype=np.float64) * density) / density


# ── Part D: Real MIDI Validation ────────────────────

def _extract_style_vector_from_notes(notes: List) -> np.ndarray:
    """Extract a 5D style vector from a list of MIDI notes for Penrose encoding.
    
    Computes pitch_complexity, timing_expressiveness, velocity_energy,
    articulation_clarity, and timbral_breadth.
    """
    if not notes:
        return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    
    pitches = np.array([n.pitch for n in notes])
    velocities = np.array([n.velocity for n in notes])
    starts = np.array([n.start_sec for n in notes])
    durations = np.array([n.duration_sec for n in notes])
    
    # Pitch complexity: note range + unique pitch classes
    pitch_range = float(np.max(pitches) - np.min(pitches)) / 120.0 if len(pitches) > 1 else 0.5
    unique_pcs = len(set(pitches % 12)) / 12.0
    pitch_complexity = min(1.0, (pitch_range + unique_pcs) / 2.0)
    
    # Timing expressiveness: onset jitter
    if len(starts) > 1:
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        onset_jitter = float(np.std(iois)) if len(iois) > 1 else 0.0
        timing_expr = min(1.0, onset_jitter * 5.0)
    else:
        timing_expr = 0.0
    
    # Velocity energy
    mean_vel = float(np.mean(velocities)) if len(velocities) > 0 else 64.0
    vel_range = float(np.max(velocities) - np.min(velocities)) / 127.0 if len(velocities) > 1 else 0.5
    velocity_energy = min(1.0, (mean_vel / 127.0 + vel_range) / 2.0)
    
    # Articulation clarity: 1 - staccato_ratio (short notes)
    if len(starts) > 1:
        sorted_starts = np.sort(starts)
        iois = np.diff(sorted_starts)
        main_iois = iois[(iois > 0.05) & (iois < 5.0)]
        beat_dur = float(np.median(main_iois)) if len(main_iois) > 0 else 0.5
        beat_dur = max(0.15, min(3.0, beat_dur))
        staccato = float(np.mean((durations / max(beat_dur, 0.01)) < 0.5))
    else:
        staccato = 0.5
    articulation_clarity = 1.0 - min(1.0, staccato)
    
    # Timbral breadth: register span + pitch class variety
    register = float(np.max(pitches) - np.min(pitches)) / 120.0 if len(pitches) > 1 else 0.5
    timbral = min(1.0, (register + unique_pcs) / 2.0)
    
    return np.clip(np.array([
        pitch_complexity, timing_expr, velocity_energy,
        articulation_clarity, timbral
    ]), 0.0, 1.0)


def _compute_style_vector_from_track(notes: List) -> np.ndarray:
    """Extract a full style vector (109D) from track notes for clustering."""
    if not notes:
        return None
    
    # Lazy import to avoid circular dependency
    from plato_midi_bridge.decompose import extract_track_style
    total_dur = max((n.start_sec + n.duration_sec for n in notes), default=10.0)
    style = extract_track_style(notes, total_dur)
    return style.to_vector()


def validate_on_real_files(midi_dir: str) -> Dict:
    """Validate Penrose encoding on real MIDI files.
    
    Real MIDI has:
    - No ground-truth composer labels (unlike generated files)
    - Natural performance variance
    - Mixed instrumentations
    
    Groups files by acoustic similarity using k-means (from silhouette analysis),
    then compares Penrose vs Eisenstein on discovered clusters.
    
    Returns:
        Dict with validation results
    """
    from pathlib import Path
    
    midi_dir = Path(midi_dir)
    midi_files = sorted(midi_dir.glob("*.mid"))
    
    if not midi_files:
        return {"error": f"No MIDI files found in {midi_dir}", "n_files": 0}
    
    # Parse all files
    all_notes = []
    valid_files = []
    
    for f in midi_files:
        try:
            # Lazy import to avoid circular dependency
            from plato_midi_bridge.decompose import parse_midi
            tracks = parse_midi(str(f))
            if not tracks:
                continue
            # Flatten all tracks into one note list for per-piece comparison
            flat_notes = [n for track in tracks for n in track]
            if not flat_notes:
                continue
            all_notes.append(flat_notes)
            valid_files.append(f.name)
        except Exception as e:
            print(f"  Skipping {f.name}: {e}")
            continue
    
    if len(all_notes) < 4:
        return {"error": f"Too few valid files ({len(all_notes)}), need at least 4", "n_files": len(all_notes)}
    
    # Extract 5D style vectors for each piece
    style_5d_vectors = np.array([_extract_style_vector_from_notes(n) for n in all_notes])
    
    # Extract full 109D vectors for clustering
    full_vectors = []
    for notes in all_notes:
        sv = _compute_style_vector_from_track(notes)
        if sv is not None:
            full_vectors.append(sv)
    full_vectors = np.array(full_vectors) if full_vectors else np.zeros((0, 109))
    
    n_pieces = len(all_notes)
    
    # ── Find natural clusters via k-means + silhouette ──
    # Try k from 2 to min(6, n_pieces//2)
    max_k = min(6, n_pieces // 2)
    if max_k < 2:
        max_k = min(2, n_pieces - 1)
    
    best_k = 1
    best_silhouette = -1.0
    best_labels = None
    best_centroids = None
    
    for k in range(2, max_k + 1):
        labels, centroids = _kmeans(full_vectors, k)
        if labels is None:
            continue
        sil = _silhouette_score(full_vectors, labels)
        if sil > best_silhouette:
            best_silhouette = sil
            best_k = k
            best_labels = labels
            best_centroids = centroids
    
    # If no good clustering found, just use 2 groups based on a simple split
    if best_labels is None:
        best_k = 2
        best_labels, best_centroids = _kmeans(full_vectors, 2)
        best_silhouette = _silhouette_score(full_vectors, best_labels)
    
    # ── Penrose vs Eisenstein on discovered clusters ──
    experiment = EncodingExperiment()
    
    # Build pieces dict with cluster ID as "composer"
    pieces = []
    for i in range(n_pieces):
        pieces.append({
            'composer': f'Cluster_{best_labels[i]}',
            'name': valid_files[i],
            'style_5d': style_5d_vectors[i],
        })
    
    # Run encoding comparison on discovered clusters
    result = experiment.compare(pieces)
    
    # ── Also compute: do clusters separate better in Penrose? ──
    # Run Penrose encoding on each piece
    penrose_encoder = PenroseEncoder()
    penrose_sigs = []
    for sv in style_5d_vectors:
        tiling = penrose_encoder.encode(sv)
        penrose_sigs.append(np.array([
            tiling.point_density,
            tiling.radial_distribution,
            tiling.symmetry_score,
            tiling.closest_pair_distance,
            tiling.vertex_count / 50.0,
        ]))
    penrose_sigs = np.array(penrose_sigs)
    
    # Eisenstein vectors
    eisenstein_vectors = np.array([experiment._to_eisenstein(sv) for sv in style_5d_vectors])
    
    # Silhouette for each encoding on same labels
    penrose_cluster_sil = _silhouette_score(penrose_sigs, best_labels) if len(penrose_sigs) > 0 else -1.0
    eisenstein_cluster_sil = _silhouette_score(eisenstein_vectors, best_labels) if len(eisenstein_vectors) > 0 else -1.0
    
    return {
        "n_files": n_pieces,
        "valid_files": valid_files,
        "natural_clusters_found": best_k,
        "cluster_silhouette": float(best_silhouette),
        "cluster_labels": best_labels.tolist() if best_labels is not None else [],
        "penrose_vs_eisenstein": {
            "cluster_separation_penrose": float(penrose_cluster_sil),
            "cluster_separation_eisenstein": float(eisenstein_cluster_sil),
        },
        "encoding_comparison_winner": result.winner if isinstance(result, EncodingResult) else "unknown",
        "encoding_comparison": {
            "eisenstein": result.eisenstein if isinstance(result, EncodingResult) else {},
            "penrose": result.penrose if isinstance(result, EncodingResult) else {},
            "combined": result.combined if isinstance(result, EncodingResult) else {},
        },
    }


def _kmeans(data: np.ndarray, k: int, max_iters: int = 50, 
            n_init: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Simple k-means clustering (no sklearn dependency).
    
    Args:
        data: (n_samples, n_features) array
        k: Number of clusters
        max_iters: Max iterations per initialization
        n_init: Number of random initializations
        
    Returns:
        (labels, centroids) or (None, None) on failure
    """
    n = data.shape[0]
    if k >= n:
        return None, None
    
    best_inertia = float('inf')
    best_labels = None
    best_centroids = None
    
    for _ in range(n_init):
        # Random initialization (k-means++ style: pick distant centroids)
        rng = np.random.RandomState(None)
        indices = [rng.randint(0, n)]
        for _ in range(1, k):
            dists = np.min([np.linalg.norm(data - data[idx], axis=1) for idx in indices], axis=0)
            probs = dists / max(np.sum(dists), 1e-10)
            indices.append(rng.choice(n, p=probs))
        
        centroids = data[indices]
        
        for _ in range(max_iters):
            # Assign labels
            dists = np.array([np.linalg.norm(data - c, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    new_centroids[j] = np.mean(data[mask], axis=0)
                else:
                    new_centroids[j] = data[rng.randint(0, n)]
            
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids
        
        # Compute inertia
        inertia = 0.0
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                inertia += float(np.sum(np.linalg.norm(data[mask] - centroids[j], axis=1) ** 2))
        
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centroids = centroids
    
    return best_labels, best_centroids


def _silhouette_score(data: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean silhouette score for a clustering.
    
    Higher = better separated clusters. Range [-1, 1].
    """
    n = data.shape[0]
    unique_labels = np.unique(labels)
    
    if len(unique_labels) < 2:
        return 0.0
    
    silhouette_values = []
    
    for i in range(n):
        same_cluster = labels == labels[i]
        same_count = np.sum(same_cluster)
        
        if same_count <= 1:
            silhouette_values.append(0.0)
            continue
        
        # Average intra-cluster distance
        intra_dists = np.linalg.norm(data[same_cluster] - data[i], axis=1)
        intra = float(np.sum(intra_dists)) / max(same_count - 1, 1)
        
        # Nearest-cluster distance
        min_inter = float('inf')
        for label in unique_labels:
            if label == labels[i]:
                continue
            other_cluster = labels == label
            inter_dists = np.linalg.norm(data[other_cluster] - data[i], axis=1)
            inter = float(np.mean(inter_dists))
            if inter < min_inter:
                min_inter = inter
        
        s = (min_inter - intra) / max(intra, min_inter, 1e-10)
        silhouette_values.append(s)
    
    return float(np.mean(silhouette_values)) if silhouette_values else 0.0


# ── Part E: Integration with decompose pipeline ─────

def encode_penrose(notes_by_track: List, styles: List) -> Dict:
    """Encode a parsed piece's style as a Penrose tiling signature tile.
    
    Takes a parsed piece (notes_by_track + styles) and returns a
    Penrose tiling signature tile that can be posted to PLATO.
    
    The tile includes: point_density, radial_distribution, symmetry_score,
    closest_pair_distance, vertex_count.
    
    Args:
        notes_by_track: List of per-track MIDI note lists (from parse_midi)
        styles: List of TrackStyle objects (from extract_track_style)
        
    Returns:
        Dict with Penrose tiling signature (PLATO tile compatible)
    """
    if not notes_by_track or not styles:
        return {
            "encoding": "penrose_5d_cut_and_project",
            "error": "No notes or styles provided",
            "signature": None,
        }
    
    # Build 5D style vector from aggregate piece features
    # Average per-track style features
    n_tracks = len(styles)
    
    # pitch_complexity: avg of note_range and pitch_variety across tracks
    pitch_values = []
    timing_values = []
    velocity_values = []
    articulation_values = []
    timbre_values = []
    
    for style in styles:
        # pitch_complexity (from TrackStyle and its pitch_range)
        pitch_range = style.pitch_range[1] - style.pitch_range[0]
        harmonic = getattr(style, 'harmonic_complexity', 0.0)
        pitch_complexity = min(1.0, (pitch_range / 120.0 + harmonic / 12.0) / 2.0)
        pitch_values.append(pitch_complexity)
        
        # timing_expressiveness
        timing = getattr(style, 'timing_consistency', 0.0)
        syncopation = getattr(style, 'syncopation_index', 0.0)
        timing_expr = min(1.0, timing * 10.0 + syncopation)
        timing_values.append(timing_expr)
        
        # velocity_energy
        dyn_range = getattr(style, 'dynamic_range', 0.0)
        mean_vel = np.mean([n.velocity for n in notes_by_track[styles.index(style)]]) if styles.index(style) < len(notes_by_track) and notes_by_track[styles.index(style)] else 64
        vel_energy = min(1.0, (dyn_range / 127.0 + mean_vel / 127.0) / 2.0)
        velocity_values.append(vel_energy)
        
        # articulation_clarity
        staccato = getattr(style, 'staccato_ratio', 0.0)
        articulation_clarity = max(0.0, 1.0 - staccato)  # 1 - staccato_ratio = legato clarity
        articulation_values.append(articulation_clarity)
        
        # timbral_breadth
        register = getattr(style, 'register_breadth', 0.0)
        harmonic = getattr(style, 'harmonic_complexity', 0.0)
        timbral = min(1.0, (register / 120.0 + harmonic / 12.0) / 2.0)
        timbre_values.append(timbral)
    
    # Average across tracks
    style_5d = np.array([
        float(np.mean(pitch_values)) if pitch_values else 0.5,
        float(np.mean(timing_values)) if timing_values else 0.5,
        float(np.mean(velocity_values)) if velocity_values else 0.5,
        float(np.mean(articulation_values)) if articulation_values else 0.5,
        float(np.mean(timbre_values)) if timbre_values else 0.5,
    ])
    
    # Encode
    encoder = PenroseEncoder()
    tiling = encoder.encode(style_5d)
    
    # Also extract scale-level features for multi-scale Penrose encoding
    scale_encodings = {}
    try:
        from plato_midi_bridge.decompose.scale import MultiScaleAnalyzer, SCALE_ORDER
        
        analyzer = MultiScaleAnalyzer()
        for i, notes in enumerate(notes_by_track):
            if i >= len(styles):
                continue
            total_dur = max((n.start_sec + n.duration_sec for n in notes), default=10.0)
            scales = analyzer.analyze(notes, total_duration=total_dur)
            
            # Encode features from each scale as Penrose points
            for scale_name in SCALE_ORDER:
                if scale_name in scales and scales[scale_name].features:
                    features = scales[scale_name].features
                    # Build 5D style from scale features
                    scale_5d = np.array([
                        features.get('pitch_variety', 0.0) 
                            or features.get('note_range', 60) / 120.0,
                        features.get('timing_consistency', 0.0)
                            or features.get('onset_jitter', 0.0),
                        features.get('dynamic_range', 0.0) / 127.0
                            or features.get('mean_velocity', 64) / 127.0,
                        features.get('staccato_ratio', 0.5),
                        features.get('harmonic_complexity', 0.0) / 12.0
                            or features.get('harmonic_rhythm', 0.0),
                    ])
                    scale_tiling = encoder.encode(np.clip(scale_5d, 0.0, 1.0))
                    sig = encoder.tiling_signature(scale_tiling.points)
                    if scale_name not in scale_encodings:
                        scale_encodings[scale_name] = []
                    scale_encodings[scale_name].append(sig)
    except Exception:
        pass
    
    # Build result
    sig = tiling.to_dict()
    sig["style_5d"] = style_5d.tolist()
    sig["n_tracks"] = n_tracks
    sig["total_notes"] = sum(len(notes) for notes in notes_by_track)
    
    # Add scale-level signatures
    if scale_encodings:
        sig["scale_encodings"] = {
            scale: {
                "mean_density": float(np.mean([s["point_density"] for s in sigs])),
                "mean_symmetry": float(np.mean([s["symmetry_score"] for s in sigs])),
                "n_encodings": len(sigs),
            }
            for scale, sigs in scale_encodings.items()
        }
    
    return sig
