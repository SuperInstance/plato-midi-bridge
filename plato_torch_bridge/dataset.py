"""MIDI Style Dataset — wraps decomposed style vectors as PyTorch Dataset.

Provides two main loading paths:
1. from_plato: load style vectors from PLATO room tiles
2. from_parsed_midi: parse MIDI files directly and extract style vectors

Each sample is a dict with:
  'style_vector': torch.Tensor (109-dim style vector)
  'penrose_tiling': torch.Tensor (n_points, 2) or empty (0, 2)
  'eisenstein_chamber': torch.Tensor (12,) one-hot
  'scale_features': Dict[str, torch.Tensor] of features at each scale
  'label': int (composer index, -1 for unlabeled)
"""

from __future__ import annotations

import json
import struct
import tempfile
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    from torch.utils.data._utils.collate import default_collate
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    # Stub for type-checking
    class Dataset:  # type: ignore
        pass
    default_collate = None


# ── Forward imports from the main decomposer ──

def _lazy_import_decomposer():
    """Lazy-import the main decomposer only when needed."""
    import sys
    from pathlib import Path
    # Add project root to path if not already
    root = Path(__file__).parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from plato_midi_bridge.decompose import (
        parse_midi, extract_track_style
    )
    return parse_midi, extract_track_style


class MIDIStyleDataset(Dataset):
    """PyTorch dataset from decomposed MIDI style vectors.

    Each sample is a dict:
        'style_vector': torch.Tensor (109-dim style vector)
        'penrose_tiling': torch.Tensor (n_points, 2) or empty (0, 2)
        'eisenstein_chamber': torch.Tensor (12-dim one-hot, -1 if unlabeled)
        'scale_features': Dict[str, torch.Tensor] features at each scale
        'label': int (composer index, -1 for unlabeled)
        'source': str (file path or plato tile source)
    """

    def __init__(self, samples: Optional[List[Dict]] = None):
        """Create dataset from pre-computed samples list.

        Args:
            samples: list of sample dicts, each with at least 'style_vector' key.
        """
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required. Install with: pip install plato-midi-bridge[torch]"
            )
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample dict with tensors."""
        raw = self.samples[idx]
        result = {}

        # Ensure style_vector is a tensor
        sv = raw.get("style_vector")
        if sv is None:
            raise KeyError(f"Sample {idx} has no 'style_vector'")
        if isinstance(sv, np.ndarray):
            result["style_vector"] = torch.from_numpy(sv).float()
        elif isinstance(sv, torch.Tensor):
            result["style_vector"] = sv.float()
        else:
            result["style_vector"] = torch.tensor(sv, dtype=torch.float32)

        # Penrose tiling
        pt = raw.get("penrose_tiling", [])
        if isinstance(pt, np.ndarray):
            result["penrose_tiling"] = torch.from_numpy(pt).float()
        elif isinstance(pt, torch.Tensor):
            result["penrose_tiling"] = pt.float()
        elif isinstance(pt, (list, tuple)):
            result["penrose_tiling"] = torch.tensor(pt, dtype=torch.float32)
        else:
            result["penrose_tiling"] = torch.empty((0, 2), dtype=torch.float32)

        # Eisenstein chamber (12-dim one-hot)
        ec = raw.get("eisenstein_chamber", -1)
        if isinstance(ec, (int, np.integer)) and ec >= 0 and ec < 12:
            result["eisenstein_chamber"] = torch.nn.functional.one_hot(
                torch.tensor(ec, dtype=torch.long), num_classes=12
            ).float()
        elif isinstance(ec, (list, tuple, np.ndarray)):
            result["eisenstein_chamber"] = torch.tensor(ec, dtype=torch.float32)
        else:
            result["eisenstein_chamber"] = torch.full((12,), -1.0, dtype=torch.float32)

        # Scale features (dict of str -> tensor)
        sf = raw.get("scale_features", {})
        if isinstance(sf, dict):
            parsed = {}
            for k, v in sf.items():
                if isinstance(v, np.ndarray):
                    parsed[k] = torch.from_numpy(v).float()
                elif isinstance(v, torch.Tensor):
                    parsed[k] = v.float()
                elif isinstance(v, (list, tuple)):
                    parsed[k] = torch.tensor(v, dtype=torch.float32)
                elif isinstance(v, (int, float)):
                    parsed[k] = torch.tensor([v], dtype=torch.float32)
                else:
                    parsed[k] = torch.empty(0, dtype=torch.float32)
            result["scale_features"] = parsed
        else:
            result["scale_features"] = {}

        # Label
        label = raw.get("label", -1)
        result["label"] = torch.tensor(label, dtype=torch.long)

        # Source
        result["source"] = raw.get("source", str(idx))

        return result

    # ── Factory Methods ─────────────────────────────────

    @classmethod
    def from_plato(cls, plato_url: str = "http://localhost:8847",
                   room: str = "style-library",
                   composer_labels: Optional[Dict[str, int]] = None) -> "MIDIStyleDataset":
        """Load from PLATO room tiles.

        Args:
            plato_url: PLATO server URL (default http://localhost:8847)
            room: Room name to fetch tiles from (default style-library)
            composer_labels: Mapping from composer name -> label index.
                             Auto-assigned if not provided.

        Returns:
            MIDIStyleDataset with samples from PLATO tiles.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MIDIStyleDataset")

        import urllib.request

        samples = []
        auto_labels: Dict[str, int] = {}
        next_label = 0

        try:
            url = f"{plato_url}/room/{room}/history"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                tiles = json.loads(resp.read().decode())

            for tile in tiles:
                answer = tile.get("answer", "")
                if isinstance(answer, str):
                    try:
                        data = json.loads(answer)
                    except json.JSONDecodeError:
                        continue
                elif isinstance(answer, dict):
                    data = answer
                else:
                    continue

                # Check for style vector
                style_vec = data.get("style_vector") or data.get("style")
                if style_vec is None:
                    continue

                source = tile.get("question", "")
                composer = data.get("composer", tile.get("source", "unknown"))

                if composer_labels is not None:
                    label = composer_labels.get(composer, -1)
                else:
                    if composer not in auto_labels:
                        auto_labels[composer] = next_label
                        next_label += 1
                    label = auto_labels[composer]

                sample = {
                    "style_vector": style_vec,
                    "label": label,
                    "source": source,
                    "penrose_tiling": data.get("penrose_tiling", []),
                    "eisenstein_chamber": data.get("eisenstein_chamber", -1),
                    "scale_features": data.get("scale_features", {}),
                }
                samples.append(sample)

        except Exception as e:
            print(f"Warning: Could not load from PLATO ({plato_url}/room/{room}): {e}")

        return cls(samples)

    @classmethod
    def from_parsed_midi(cls, midi_dir: str,
                         composer_labels: Optional[Dict[str, int]] = None,
                         max_files: Optional[int] = None) -> "MIDIStyleDataset":
        """Load from parsed MIDI files.

        Organizes files into subdirectories (one per composer) or
        treats each file individually.

        Args:
            midi_dir: Directory containing MIDI files.
                      Subdirectories are treated as composer groups.
            composer_labels: Mapping from composer name -> label index.
            max_files: Maximum number of files to process (None = all).

        Returns:
            MIDIStyleDataset with parsed style vectors.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MIDIStyleDataset")

        parse_midi_fn, extract_track_style_fn = _lazy_import_decomposer()

        midi_path = Path(midi_dir)
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")

        # Collect files organized by composer (subdirectory = composer)
        composer_files: Dict[str, List[Path]] = {}
        for child in sorted(midi_path.iterdir()):
            if child.is_dir():
                # Subdirectory named after composer
                composer = child.name
                midi_files = sorted(child.glob("*.mid")) + sorted(child.glob("*.midi"))
                if midi_files:
                    composer_files[composer] = midi_files
            elif child.suffix.lower() in (".mid", ".midi"):
                # Top-level file: use parent dir or "unknown"
                composer_files.setdefault(midi_path.name, []).append(child)

        if not composer_files:
            raise ValueError(f"No MIDI files found in {midi_dir}")

        auto_labels: Dict[str, int] = {} if composer_labels is None else composer_labels
        next_label = max(auto_labels.values()) + 1 if auto_labels else 0

        samples = []
        total_processed = 0

        for composer in sorted(composer_files.keys()):
            if composer not in auto_labels:
                auto_labels[composer] = next_label
                next_label += 1
            label = auto_labels[composer]

            files = composer_files[composer]
            if max_files is not None:
                files = files[:max_files]

            for f in files:
                try:
                    tracks = parse_midi_fn(str(f))
                    if not tracks:
                        continue
                    all_notes = [n for track in tracks for n in track]
                    if not all_notes:
                        continue

                    total_dur = max(
                        (n.start_sec + n.duration_sec for n in all_notes),
                        default=60.0
                    )
                    style = extract_track_style_fn(all_notes, total_dur)
                    style_vec = style.to_vector()

                    sample = {
                        "style_vector": style_vec,
                        "label": label,
                        "source": str(f.relative_to(midi_path)),
                        "penrose_tiling": [],
                        "eisenstein_chamber": -1,
                        "scale_features": {},
                    }
                    samples.append(sample)
                    total_processed += 1

                    if max_files is not None and total_processed >= max_files:
                        break
                except Exception as e:
                    print(f"  Warning: Could not parse {f.name}: {e}")

            if max_files is not None and total_processed >= max_files:
                break

        print(f"Loaded {len(samples)} samples from {len(composer_files)} composer groups")
        return cls(samples)

    # ── Utility Methods ─────────────────────────────────

    def add_sample(self, sample: Dict) -> None:
        """Add a single sample to the dataset."""
        self.samples.append(sample)

    def get_labels(self) -> torch.Tensor:
        """Return all labels as a tensor (N,). Unlabeled samples get -1."""
        labels = []
        for s in self.samples:
            lbl = s.get("label", -1)
            labels.append(lbl if isinstance(lbl, int) else -1)
        return torch.tensor(labels, dtype=torch.long)

    def get_style_vectors(self) -> torch.Tensor:
        """Return all style vectors as a (N, 109) tensor.

        Returns empty (0, 109) tensor if dataset is empty.
        """
        vecs = []
        for s in self.samples:
            sv = s.get("style_vector", None)
            if sv is None:
                vecs.append(torch.zeros(109))
            elif isinstance(sv, torch.Tensor):
                vecs.append(sv.flatten().float())
            elif isinstance(sv, np.ndarray):
                vecs.append(torch.from_numpy(sv).float())
            else:
                vecs.append(torch.tensor(sv, dtype=torch.float32).flatten())
        if not vecs:
            return torch.empty((0, 109), dtype=torch.float32)
        return torch.stack(vecs)

    def filter_by_label(self, label: int) -> "MIDIStyleDataset":
        """Return a new dataset containing only samples with the given label."""
        filtered = [s for s in self.samples if s.get("label", -1) == label]
        return MIDIStyleDataset(filtered)

    def split(self, frac: float = 0.8,
              seed: int = 42) -> Tuple["MIDIStyleDataset", "MIDIStyleDataset"]:
        """Split into train/val datasets by fraction.

        Args:
            frac: Fraction for training set (default 0.8)
            seed: Random seed for reproducibility

        Returns:
            (train_dataset, val_dataset)
        """
        rng = np.random.RandomState(seed)
        indices = list(range(len(self.samples)))
        rng.shuffle(indices)
        split_idx = int(len(indices) * frac)
        train_samples = [self.samples[i] for i in indices[:split_idx]]
        val_samples = [self.samples[i] for i in indices[split_idx:]]
        return MIDIStyleDataset(train_samples), MIDIStyleDataset(val_samples)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
        """Custom collate function for DataLoader.

        Handles variable-size fields (penrose_tiling) that default_collate
        cannot stack. Pads penrose_tiling to the max N in the batch.

        Args:
            batch: List of sample dicts from __getitem__

        Returns:
            Batched dict with padded tensors
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for MIDIStyleDataset")

        # Determine max penrose points
        max_pt = max(s.get("penrose_tiling", torch.empty((0, 2))).shape[0]
                     for s in batch)

        processed = []
        for s in batch:
            pt = s.get("penrose_tiling", torch.empty((0, 2), dtype=torch.float32))
            if pt.shape[0] < max_pt:
                # Pad with zeros
                padded = torch.zeros(max_pt, 2, dtype=torch.float32)
                padded[:pt.shape[0]] = pt
                s = dict(s)
                s["penrose_tiling"] = padded
            processed.append(s)

        return default_collate(processed)

    def to_plato_tile(self, room_url: str = "http://localhost:8847",
                      room_name: str = "style-library") -> Dict:
        """Convert dataset summary to a PLATO tile for posting."""
        n = len(self.samples)
        labels = self.get_labels()
        n_labeled = int((labels >= 0).sum().item())
        n_unlabeled = n - n_labeled

        sv = self.get_style_vectors()
        mean_vec = sv.mean(dim=0).tolist()
        var_vec = sv.var(dim=0).tolist()

        return {
            "question": f"Torch Bridge — Dataset Summary ({room_name})",
            "answer": json.dumps({
                "n_samples": n,
                "n_labeled": n_labeled,
                "n_unlabeled": n_unlabeled,
                "style_dim": 109,
                "mean_style_vector": mean_vec,
                "style_variance_vector": var_vec,
                "unique_labels": int((labels >= 0).unique().shape[0]),
                "source": "plato_torch_bridge",
            }),
            "source": "plato-torch-bridge",
            "confidence": 0.95,
        }
