"""Training scripts for the Plato-PyTorch bridge.

Provides training loops for:
1. train_dataset: Encoding/style vectors -> MIDIStyleDataset
2. train_contrastive: Train ContrastiveStyleEncoder on MIDIStyleDataset
3. train_lora: Train StyleLoRAAdapter for LM conditioning
4. generate_midi: Generate MIDI from style prompts (conceptual)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required. Install with: pip install plato-midi-bridge[torch]"
        )


def train_contrastive(
    dataset: "MIDIStyleDataset",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    margin: float = 0.5,
    temperature: float = 0.1,
    loss_type: str = "triplet",
    val_split: float = 0.15,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> "ContrastiveStyleEncoder":
    """Train the ContrastiveStyleEncoder on a MIDIStyleDataset.

    Uses triplet loss (default) or NT-Xent loss to learn style embeddings
    where same-composer pieces are close and different-composer pieces are far.

    Args:
        dataset: MIDIStyleDataset with labeled samples
        epochs: Number of training epochs (default 50)
        batch_size: Batch size for training (default 32)
        learning_rate: Adam learning rate (default 1e-3)
        margin: Triplet loss margin (default 0.5)
        temperature: NT-Xent temperature (default 0.1)
        loss_type: 'triplet' or 'nt_xent' (default 'triplet')
        val_split: Fraction of data for validation (default 0.15, 0 = no val)
        seed: Random seed (default 42)
        device: Torch device (default auto-detect)
        verbose: Print progress (default True)

    Returns:
        Trained ContrastiveStyleEncoder
    """
    _check_torch()
    from .dataset import MIDIStyleDataset
    from .encoder import ContrastiveStyleEncoder

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)

    # Split dataset
    if val_split > 0:
        train_ds, val_ds = dataset.split(1.0 - val_split, seed=seed)
    else:
        train_ds = dataset
        val_ds = None

    # Filter to labeled samples only
    labeled_train_samples = [s for s in train_ds.samples if s.get("label", -1) >= 0]
    if len(labeled_train_samples) < 3:
        raise ValueError(
            f"Need at least 3 labeled samples for contrastive training, "
            f"got {len(labeled_train_samples)}"
        )
    train_ds_labeled = MIDIStyleDataset(labeled_train_samples)

    if verbose:
        n_labels = len(set(s["label"] for s in labeled_train_samples))
        print(f"Training contrastive encoder on {len(labeled_train_samples)} samples "
              f"({n_labels} unique labels) using {device}")

    model = ContrastiveStyleEncoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = DataLoader(
        train_ds_labeled, batch_size=batch_size, shuffle=True,
        collate_fn=dataset.collate_fn
    )

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            labels = batch["label"].to(device)
            style_vectors = batch["style_vector"].to(device)

            optimizer.zero_grad()

            if loss_type == "nt_xent":
                # NT-Xent: encode all samples, compute loss over the batch
                embeddings = model(style_vectors)
                loss = model.nt_xent_loss(embeddings, labels, temperature=temperature)
            else:
                # Triplet: need anchor, positive, negative per sample
                loss = _triplet_loss_from_batch(model, style_vectors, labels, margin)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(avg_loss)

        # Validation
        val_loss = None
        if val_ds is not None:
            val_loss = _evaluate_contrastive(
                model, val_ds, loss_type, margin, temperature, device
            )
            history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            val_str = f", val={val_loss:.6f}" if val_loss is not None else ""
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.6f}{val_str}")

    if verbose:
        print(f"Training complete: final loss={history['train_loss'][-1]:.6f}")

    return model


def _triplet_loss_from_batch(
    model: "ContrastiveStyleEncoder",
    style_vectors: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Compute triplet loss from a single batch by mining triplets.

    For each sample, finds a same-label positive and a different-label negative.
    Falls back to random pairings when exact matches aren't available.
    """
    B = style_vectors.shape[0]
    if B < 3:
        return torch.tensor(0.0, requires_grad=True, device=style_vectors.device)

    embeddings = model(style_vectors)

    losses = []
    for i in range(B):
        # Find positive: same label, different index
        pos_indices = (labels == labels[i]).nonzero(as_tuple=True)[0]
        pos_indices = pos_indices[pos_indices != i]
        if len(pos_indices) == 0:
            continue
        pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,))].item()

        # Find negative: different label
        neg_indices = (labels != labels[i]).nonzero(as_tuple=True)[0]
        if len(neg_indices) == 0:
            continue
        neg_idx = neg_indices[torch.randint(0, len(neg_indices), (1,))].item()

        loss = model.triplet_loss(
            embeddings[i:i+1],
            embeddings[pos_idx:pos_idx+1],
            embeddings[neg_idx:neg_idx+1],
            margin=margin,
        )
        losses.append(loss)

    if not losses:
        return torch.tensor(0.0, requires_grad=True, device=style_vectors.device)

    return torch.stack(losses).mean()


def _evaluate_contrastive(
    model: "ContrastiveStyleEncoder",
    dataset: "MIDIStyleDataset",
    loss_type: str,
    margin: float,
    temperature: float,
    device: torch.device,
) -> float:
    """Evaluate contrastive model on validation dataset."""
    from .dataset import MIDIStyleDataset
    model.eval()

    labeled_samples = [s for s in dataset.samples if s.get("label", -1) >= 0]
    if len(labeled_samples) < 3:
        return 0.0

    val_ds = MIDIStyleDataset(labeled_samples)
    loader = DataLoader(val_ds, batch_size=len(val_ds), shuffle=False,
                        collate_fn=val_ds.collate_fn)

    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device)
            style_vectors = batch["style_vector"].to(device)

            if loss_type == "nt_xent":
                embeddings = model(style_vectors)
                loss = model.nt_xent_loss(embeddings, labels, temperature=temperature)
            else:
                loss = _triplet_loss_from_batch(model, style_vectors, labels, margin)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train_lora(
    encoder: "ContrastiveStyleEncoder",
    style_vector: np.ndarray,
    lm_embedding_dim: int = 4096,
    rank: int = 4,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> "StyleLoRAAdapter":
    """Train a StyleLoRAAdapter for a single style vector.

    In actual usage, you'd have a target LM embedding to match.
    Here we use a random target for demonstration; real usage would
    optimize the adapter to match the LM's embedding distribution.

    Args:
        encoder: Trained ContrastiveStyleEncoder
        style_vector: (109,) style vector to condition on
        lm_embedding_dim: LM embedding dimension (default 4096)
        rank: LoRA rank (default 4)
        epochs: Number of training epochs (default 100)
        learning_rate: Adam learning rate (default 1e-3)
        device: Torch device (default auto-detect)
        verbose: Print progress (default True)

    Returns:
        Trained StyleLoRAAdapter
    """
    _check_torch()
    from .encoder import ContrastiveStyleEncoder
    from .lora import StyleLoRAAdapter

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StyleLoRAAdapter(
        embedding_dim=encoder.embedding_dim,
        rank=rank,
        d_model=lm_embedding_dim,
    ).to(device)

    # Encode the style vector
    encoder.eval().to(device)
    with torch.no_grad():
        sv_tensor = torch.from_numpy(style_vector).float().unsqueeze(0).to(device)
        style_embedding = encoder(sv_tensor)  # (1, 32)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Random target embedding for demonstration
    target = torch.randn(1, lm_embedding_dim, device=device)
    target = target / target.norm(dim=1, keepdim=True)

    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        conditioning = model(style_embedding)  # (1, d_model)
        loss = torch.nn.functional.mse_loss(conditioning, target)

        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={loss.item():.6f}")

    if verbose:
        print(f"LoRA adapter trained: {model.get_lora_parameters()} parameters")
        print(f"  Final loss: {history[-1]:.6f}")

    return model


def generate_midi(
    style_prompt: str,
    encoder: "ContrastiveStyleEncoder",
    lora: "StyleLoRAAdapter",
    output_path: str,
    n_notes: int = 32,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
) -> Dict:
    """Generate a MIDI sequence conditioned on a style prompt.

    NOTE: This is a conceptual/skeletal implementation. Real generation
    requires a full language model to decode the conditioning vectors
    into MIDI tokens. Here we generate random MIDI events based on the
    style vector as a placeholder.

    Args:
        style_prompt: Text description of desired style
        encoder: Trained ContrastiveStyleEncoder
        lora: Trained StyleLoRAAdapter
        output_path: Path to write generated MIDI file (.mid)
        n_notes: Number of notes to generate (default 32)
        temperature: Sampling temperature (default 1.0)
        device: Torch device (default auto-detect)

    Returns:
        Dict with generation metadata
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required for generation")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder.eval().to(device)
    lora.eval().to(device)

    # Parse style prompt (simple heuristic)
    # For real implementation, use a text encoder to embed the prompt
    style_vector = _prompt_to_style_vector(style_prompt)
    sv_tensor = torch.from_numpy(style_vector).float().unsqueeze(0).to(device)

    with torch.no_grad():
        style_embedding = encoder(sv_tensor)
        conditioning = lora(style_embedding)

    # Generate random note sequence (placeholder)
    # Real impl would decode conditioning through a language model
    np.random.seed(hash(style_prompt) % (2**31))
    notes = []
    current_time = 0.0
    for i in range(n_notes):
        pitch = int(np.clip(
            np.random.normal(64, 12) + conditioning[0, 0].item() * 10,
            21, 108
        ))
        velocity = int(np.clip(
            np.random.normal(80, 20) + conditioning[0, 1].item() * 20,
            10, 127
        ))
        duration = max(0.05, np.random.exponential(0.3) * temperature)
        onset = current_time
        current_time += max(0.05, np.random.exponential(0.4) * temperature)

        notes.append({
            "pitch": pitch,
            "velocity": velocity,
            "start_sec": onset,
            "duration_sec": duration,
        })

    # Write MIDI file
    _write_midi(notes, output_path)

    result = {
        "output_path": output_path,
        "n_notes": n_notes,
        "duration_sec": current_time,
        "style_prompt": style_prompt,
        "conditioning_norm": float(conditioning.norm().item()),
    }

    print(f"Generated {n_notes} notes → {output_path}")
    return result


def _prompt_to_style_vector(prompt: str) -> np.ndarray:
    """Convert a text prompt to a 109D style vector.

    Uses simple keyword heuristics. Real implementation would use a
    text encoder (e.g., CLAP, MusicBERT, or a fine-tuned LLM).

    Args:
        prompt: Text description (e.g., "dark, staccato, mechanical")

    Returns:
        (109,) style vector
    """
    vec = np.zeros(109)

    # Simple keyword → dimension mapping (pitch, velocity, timing regions)
    prompt_lower = prompt.lower()

    # Pitch-related
    if "high" in prompt_lower or "bright" in prompt_lower or "treble" in prompt_lower:
        vec[30:48] = 0.02  # higher pitch histogram
    if "low" in prompt_lower or "dark" in prompt_lower or "bass" in prompt_lower:
        vec[0:18] = 0.02  # lower pitch histogram
    if "wide" in prompt_lower or "span" in prompt_lower:
        vec[108] = 1.0  # register_breadth

    # Velocity/dynamics
    if "loud" in prompt_lower or "strong" in prompt_lower or "aggressive" in prompt_lower:
        vec[48:64] = 0.02  # higher velocity
        vec[104] = 0.5  # dynamic_range
    if "soft" in prompt_lower or "gentle" in prompt_lower or "quiet" in prompt_lower:
        vec[48:56] = 0.02  # lower-mid velocity
        vec[104] = 0.1  # small dynamic_range

    # Articulation
    if "staccato" in prompt_lower or "short" in prompt_lower or "mechanical" in prompt_lower:
        vec[81] = 0.8  # staccato_ratio
    if "legato" in prompt_lower or "smooth" in prompt_lower or "flowing" in prompt_lower:
        vec[81] = 0.1  # legato (low staccato_ratio)

    # Timing
    if "fast" in prompt_lower or "quick" in prompt_lower:
        vec[105] = 8.0  # note_density
    if "slow" in prompt_lower or "relaxed" in prompt_lower or "calm" in prompt_lower:
        vec[105] = 1.0  # low note_density
    if "syncopated" in prompt_lower or "jazz" in prompt_lower:
        vec[106] = 0.6  # syncopation_index

    # Harmonic complexity
    if "complex" in prompt_lower or "chromatic" in prompt_lower or "jazz" in prompt_lower:
        vec[107] = 8.0  # harmonic_complexity
    if "simple" in prompt_lower or "diatonic" in prompt_lower:
        vec[107] = 2.0  # low harmonic_complexity

    # Avg interval
    if "leaping" in prompt_lower or "angular" in prompt_lower:
        vec[103] = 8.0  # avg_interval
    if "stepwise" in prompt_lower or "conjunct" in prompt_lower:
        vec[103] = 2.0  # small avg_interval

    return vec


def _write_midi(notes: List[Dict], output_path: str) -> None:
    """Write a minimal MIDI file from note dicts.

    Creates a single-track Format 0 file with tempo = 120 BPM.
    """
    import struct

    ticks_per_beat = 480
    us_per_beat = 500000  # 120 BPM

    def write_var_len(value):
        result = []
        while True:
            result.append(value & 0x7F)
            value >>= 7
            if value == 0:
                break
        result.reverse()
        for i in range(len(result) - 1):
            result[i] |= 0x80
        return bytes(result)

    # Sort notes by onset
    notes_sorted = sorted(notes, key=lambda n: n["start_sec"])

    # Convert seconds to ticks at 120 BPM
    sec_per_tick = us_per_beat / (ticks_per_beat * 1_000_000)

    # Build track events
    events = b""
    # Tempo event at tick 0
    events += write_var_len(0)
    events += b'\xFF\x51\x03' + struct.pack(">I", us_per_beat)[1:]

    abs_tick = 0
    for n in notes_sorted:
        tick = int(n["start_sec"] / sec_per_tick)
        delta = tick - abs_tick
        dur_ticks = max(1, int(n["duration_sec"] / sec_per_tick))

        # Note On
        events += write_var_len(max(0, delta))
        events += bytes([0x90, n["pitch"], n["velocity"]])
        abs_tick = tick

        # Note Off
        events += write_var_len(dur_ticks)
        events += bytes([0x80, n["pitch"], 64])
        abs_tick = tick + dur_ticks

    # End of track
    events += write_var_len(0) + b'\xFF\x2F\x00'

    # Header
    header = b'MThd' + struct.pack(">IHH", 6, 0, 1)
    header += struct.pack(">H", ticks_per_beat)

    # Track
    track = b'MTrk' + struct.pack(">I", len(events)) + events

    with open(output_path, "wb") as f:
        f.write(header + track)
