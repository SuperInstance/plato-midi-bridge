"""
Tensor operations on PLATO room state.

Each room is a vector in a high-dimensional space:
  [tile_count, coupling_weights[12], gap, focus_depth, presence, provenance_length]

Room state tensor: (n_rooms, n_features)
Coupling tensor: (n_rooms, n_rooms) — weighted edges
T-minus tensor: (n_events, 3) — [predicted_time, actual_time, confidence]
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json
import urllib.request
from datetime import datetime, timedelta

PLATO_BASE = "http://localhost:8847"

# ── Eisenstein Lattice — 12 Chambers ─────────────────
EISENSTEIN_CHAMBERS = [
    {"name": "C", "root": 0,   "quality": "unison",     "emotion": "stillness"},
    {"name": "C#", "root": 1,  "quality": "minor_second","emotion": "tension"},
    {"name": "D",  "root": 2,  "quality": "major_second","emotion": "movement"},
    {"name": "D#", "root": 3,  "quality": "minor_third", "emotion": "melancholy"},
    {"name": "E",  "root": 4,  "quality": "major_third", "emotion": "hope"},
    {"name": "F",  "root": 5,  "quality": "fourth",      "emotion": "stability"},
    {"name": "F#", "root": 6,  "quality": "tritone",     "emotion": "question"},
    {"name": "G",  "root": 7,  "quality": "fifth",       "emotion": "resolution"},
    {"name": "G#", "root": 8,  "quality": "minor_sixth", "emotion": "depth"},
    {"name": "A",  "root": 9,  "quality": "major_sixth", "emotion": "joy"},
    {"name": "A#", "root": 10, "quality": "minor_seventh","emotion": "longing"},
    {"name": "B",  "root": 11, "quality": "major_seventh","emotion": "anticipation"},
]

# ── Coupling Weight → Musical Interval ────────────────
COUPLING_TO_INTERVAL = [
    (0.0, 0.1, 0,    "unison"),
    (0.1, 0.2, 1,    "minor_second"),
    (0.2, 0.3, 2,    "major_second"),
    (0.3, 0.4, 3,    "minor_third"),
    (0.4, 0.5, 4,    "major_third"),
    (0.5, 0.6, 5,    "fourth"),
    (0.6, 0.7, 6,    "tritone"),
    (0.7, 0.8, 7,    "fifth"),
    (0.8, 0.9, 8,    "minor_sixth"),
    (0.9, 1.0, 9,    "major_sixth"),
]

def coupling_to_interval(coupling: float) -> Tuple[int, str]:
    """Map coupling weight [0,1] to semitone interval + name."""
    for low, high, semitones, name in COUPLING_TO_INTERVAL:
        if low <= coupling < high:
            return semitones, name
    return 0, "unison"


@dataclass
class RoomTensor:
    """A room's state as a vector in the Eisenstein lattice."""
    name: str
    tile_count: int
    coupling_vector: List[float]  # length 12, one per chamber
    gap: float                    # FLUX gap — prediction vs observation
    focus_depth: int              # number of unanswered questions
    presence: int                 # agents in this room
    provenance_length: int        # chain of tile history
    source: str = "unknown"
    last_updated: float = 0.0

    @property
    def chamber(self) -> int:
        """Which Eisenstein chamber this room snaps to.
        Determined by dominant coupling direction."""
        if not self.coupling_vector:
            return 0
        return int(np.argmax(self.coupling_vector))

    @property
    def root_note(self) -> int:
        """MIDI root note based on chamber (C4 = 60)."""
        return 60 + EISENSTEIN_CHAMBERS[self.chamber]["root"]

    @property
    def velocity(self) -> int:
        """Velocity based on activity level."""
        base = min(127, max(20, int(self.tile_count * 0.5)))
        gap_mod = max(-20, min(20, int(self.gap * 40)))
        return max(20, min(127, base + gap_mod))

    def to_vector(self) -> np.ndarray:
        return np.array([
            self.tile_count / 1000.0,           # normalize
            *self.coupling_vector[:6],           # first 6 couplings
            self.gap,
            self.focus_depth / 10.0,
            self.presence / 5.0,
            self.provenance_length / 100.0,
        ], dtype=np.float32)


@dataclass
class CouplingTensor:
    """The coupling matrix between rooms.
    Shape: (n_rooms, n_rooms) — weighted directed edges."""
    rooms: List[str]
    matrix: np.ndarray  # (n, n)

    @classmethod
    def from_room_tensors(cls, tensors: List[RoomTensor]) -> "CouplingTensor":
        n = len(tensors)
        matrix = np.zeros((n, n), dtype=np.float32)
        for i, t1 in enumerate(tensors):
            for j, t2 in enumerate(tensors):
                if i == j:
                    matrix[i][j] = 1.0  # self-coupling
                else:
                    # Coupling = similarity of chamber + proximity of vectors
                    chamber_sim = 1.0 - abs(t1.chamber - t2.chamber) / 12.0
                    vec_sim = float(np.dot(t1.to_vector(), t2.to_vector()))
                    matrix[i][j] = float(np.clip((chamber_sim + vec_sim) / 2, 0, 1))
        names = [t.name for t in tensors]
        return cls(rooms=names, matrix=matrix)

    def interval_matrix(self) -> np.ndarray:
        """Convert coupling weights to semitone intervals."""
        intervals = np.zeros_like(self.matrix, dtype=int)
        for i in range(len(self.rooms)):
            for j in range(len(self.rooms)):
                sems, _ = coupling_to_interval(float(self.matrix[i][j]))
                intervals[i][j] = sems
        return intervals


@dataclass
class TMinusTensor:
    """T-minus event predictions as temporal coupling vectors.
    
    Each event is a vector:
        [predicted_delta_hours, actual_delta_hours, confidence]
    
    The gap (actual - predicted) is the musical tension.
    When gap = 0, the prediction was perfect → resolution (fifth).
    When gap != 0, the gap creates dissonance → tritone or second.
    """
    events: List[Dict] = field(default_factory=list)

    def add_event(self, name: str, predicted_hours: float, 
                  actual_hours: Optional[float] = None,
                  confidence: float = 0.5, source: str = "forgemaster"):
        self.events.append({
            "name": name,
            "predicted": predicted_hours,
            "actual": actual_hours,
            "confidence": confidence,
            "source": source,
            "created": datetime.utcnow().isoformat(),
        })

    def add_from_plato_tile(self, question: str, answer: str):
        """Parse a T-minus prediction tile from FM and add to events."""
        import re
        # Pattern: TASK N: ... → T-Xh or T-X.Yh
        matches = re.findall(r'TASK (\d+).*?T-?(\d+(?:\.\d+)?)h', 
                            question + answer, re.IGNORECASE)
        for task_id, hours_str in matches:
            hours = float(hours_str)
            name = f"TASK {task_id}"
            self.add_event(name, hours)

    def to_tensor(self) -> np.ndarray:
        """Shape: (n_events, 3) — [predicted, actual_or_nan, confidence]"""
        rows = []
        for e in self.events:
            actual = e["actual"] if e["actual"] is not None else float('nan')
            rows.append([e["predicted"], actual, e["confidence"]])
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 3))

    def gap_vector(self) -> np.ndarray:
        """The gap between prediction and actuality.
        Positive = event took longer than predicted (tension).
        Negative = event finished early (release).
        NaN = still pending (suspension)."""
        t = self.to_tensor()
        if len(t) == 0:
            return np.array([])
        return t[:, 1] - t[:, 0]  # actual - predicted

    def musical_tension(self) -> float:
        """Overall tension from unresolved predictions.
        0 = all resolved perfectly. 1 = maximum tension."""
        gaps = self.gap_vector()
        if len(gaps) == 0:
            return 0.0
        # Count unresolved as max tension
        resolved = gaps[~np.isnan(gaps)]
        unresolved = len(gaps) - len(resolved)
        if len(resolved) == 0 and unresolved > 0:
            return min(1.0, unresolved * 0.2)
        avg_gap = float(np.mean(np.abs(resolved))) if len(resolved) > 0 else 0
        return min(1.0, avg_gap / 24.0 + unresolved * 0.15)

    def next_scheduled_events(self, within_hours: float = 48) -> List[Dict]:
        """Events predicted to happen within the window."""
        now = datetime.utcnow()
        upcoming = []
        for e in self.events:
            if e["actual"] is not None:
                continue  # already resolved
            if e["predicted"] <= within_hours:
                upcoming.append(e)
        return upcoming


class PlatoRoomFetcher:
    """Fetches room state from PLATO and builds tensors."""

    def __init__(self, base_url: str = PLATO_BASE):
        self.base = base_url

    def fetch_status(self) -> dict:
        req = urllib.request.Request(f"{self.base}/status")
        return json.loads(urllib.request.urlopen(req, timeout=5).read())

    def fetch_room(self, name: str) -> dict:
        req = urllib.request.Request(f"{self.base}/room/{name}/history")
        return json.loads(urllib.request.urlopen(req, timeout=5).read())

    def build_tensors(self, room_names: Optional[List[str]] = None) -> Tuple[
            List[RoomTensor], CouplingTensor, TMinusTensor]:
        """Fetch PLATO state and build all tensors."""
        status = self.fetch_status()
        rooms_data = status.get("rooms", {})
        
        # If no room list, use all fleet rooms
        if room_names is None:
            room_names = [r for r in rooms_data.keys() 
                         if r.startswith(("fleet-", "oracle1-", "forge")) 
                         or r == "fleet-registry"]
        
        room_tensors = []
        for name in room_names:
            rd = rooms_data.get(name, {})
            tc = rd.get("tile_count", 0) if isinstance(rd, dict) else 0
            # Generate coupling vector from room's tile history
            coupling = np.random.uniform(0, 0.3, 12)  # default low
            try:
                room_data = self.fetch_room(name)
                tiles = room_data.get("tiles", [])
                if tiles:
                    # Each tile source contributes to coupling direction
                    sources = [t.get("source", "?") for t in tiles[-20:]]
                    for j, s in enumerate(sources[:12]):
                        coupling[j] = min(1.0, coupling[j] + 0.1 * (1 - j/12))
                    coupling = coupling / (np.sum(coupling) + 0.001)
            except:
                pass
            
            tensor = RoomTensor(
                name=name,
                tile_count=tc,
                coupling_vector=coupling.tolist(),
                gap=np.random.uniform(0, 0.3),  # placeholder
                focus_depth=len(rooms_data),
                presence=5,  # 5 known agents
                provenance_length=tc,
            )
            room_tensors.append(tensor)
        
        coupling_tensor = CouplingTensor.from_room_tensors(room_tensors)
        tminus = TMinusTensor()
        
        return room_tensors, coupling_tensor, tminus
