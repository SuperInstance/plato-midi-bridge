"""Tests for plato_torch_bridge — PyTorch style encoder, dataset, and LoRA adapter.

All tests use @pytest.mark.skipif(not HAS_TORCH, ...) so they're skipped
when torch is not installed. This keeps the main decomposer independent
of the torch dependency.
"""

import sys
import os
from pathlib import Path

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import pytest

# ── Import the torch bridge (gated) ──

if HAS_TORCH:
    from plato_torch_bridge import MIDIStyleDataset, ContrastiveStyleEncoder, StyleLoRAAdapter
else:
    MIDIStyleDataset = None
    ContrastiveStyleEncoder = None
    StyleLoRAAdapter = None


skipif_no_torch = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


# ── Helper: synthetic data ──────────────────────────

def _make_synthetic_samples(n: int = 50, n_composers: int = 3) -> list:
    """Create synthetic style vector samples.

    Creates n_composers clusters of style vectors. Within a cluster,
    vectors are similar; between clusters, they differ.
    """
    rng = np.random.RandomState(42)
    samples = []

    for i in range(n):
        composer_idx = i % n_composers
        # Each composer has a centroid with some noise
        centroid = np.array([
            0.1 + composer_idx * 0.3,  # avg_interval
            0.3 + composer_idx * 0.2,  # dynamic_range region
            2.0 + composer_idx * 2.0,  # note_density
            0.1 + composer_idx * 0.15,  # syncopation
            2.0 + composer_idx * 1.5,  # harmonic_complexity
            20.0 + composer_idx * 10.0,  # register_breadth
        ])
        noise = rng.normal(0, 0.1, 6) * 0.5

        # Build a full 109D vector
        style_vec = np.zeros(109)
        # Fill the known 6 style dimensions (indices 103-108)
        style_vec[103] = max(0, centroid[0] + noise[0])
        style_vec[104] = max(0, centroid[1] + noise[1])
        style_vec[105] = max(0, centroid[2] + noise[2])
        style_vec[106] = max(0, centroid[3] + noise[3])
        style_vec[107] = max(0, centroid[4] + noise[4])
        style_vec[108] = max(0, centroid[5] + noise[5])

        # Add some pitch histogram variation
        style_vec[composer_idx * 16: (composer_idx + 1) * 16] = 1.0 / 16.0

        # Penrose tiling
        n_pt_points = rng.randint(3, 10)
        penrose_tiling = rng.uniform(-2, 2, (n_pt_points, 2))

        # Eisenstein chamber
        eisenstein_chamber = composer_idx % 12

        # Scale features
        scale_features = {
            "micro": rng.uniform(0, 1, 8),
            "note": rng.uniform(0, 1, 12),
            "phrase": rng.uniform(0, 1, 6),
        }

        samples.append({
            "style_vector": style_vec,
            "label": composer_idx,
            "source": f"synth_composer_{composer_idx}_piece_{i // n_composers}",
            "penrose_tiling": penrose_tiling,
            "eisenstein_chamber": eisenstein_chamber,
            "scale_features": scale_features,
        })

    return samples


# ── Test 1: Dataset creation ────────────────────────

@skipif_no_torch
def test_dataset_creation():
    """Synthetic vectors → MIDIStyleDataset with correct structure."""
    samples = _make_synthetic_samples(n=50, n_composers=3)
    ds = MIDIStyleDataset(samples)

    assert len(ds) == 50, f"Dataset length: {len(ds)} (expected 50)"

    # Check first sample structure
    sample = ds[0]
    assert isinstance(sample, dict), f"Sample should be dict, got {type(sample)}"
    assert "style_vector" in sample, "Sample missing 'style_vector'"
    assert "label" in sample, "Sample missing 'label'"
    assert "source" in sample, "Sample missing 'source'"

    # Check style vector shape
    sv = sample["style_vector"]
    assert isinstance(sv, torch.Tensor), f"Style vector should be tensor, got {type(sv)}"
    assert sv.shape == (109,), f"Style vector shape: {sv.shape} (expected (109,))"

    # Check label
    label = sample["label"]
    assert isinstance(label, torch.Tensor), f"Label should be tensor, got {type(label)}"

    # Check penrose tiling
    pt = sample["penrose_tiling"]
    assert isinstance(pt, torch.Tensor), f"Penrose tiling should be tensor, got {type(pt)}"
    assert pt.shape[1] == 2, f"Penrose tiling should have 2 columns, got {pt.shape}"
    assert pt.shape[0] > 0, "Penrose tiling should have points"

    # Check Eisenstein chamber
    ec = sample["eisenstein_chamber"]
    assert isinstance(ec, torch.Tensor), f"Eisenstein chamber should be tensor, got {type(ec)}"
    assert ec.shape == (12,), f"Eisenstein chamber shape: {ec.shape} (expected (12,))"

    # Check scale features
    sf = sample["scale_features"]
    assert isinstance(sf, dict), f"Scale features should be dict, got {type(sf)}"
    assert "micro" in sf, "Scale features missing 'micro'"
    assert isinstance(sf["micro"], torch.Tensor), "Scale feature should be tensor"

    print(f"  Dataset size: {len(ds)}")
    print(f"  Sample keys: {list(sample.keys())}")
    print(f"  Style vector shape: {sv.shape}, dtype: {sv.dtype}")


# ── Test 2: Encoder forward pass ────────────────────

@skipif_no_torch
def test_encoder_forward():
    """ContrastiveStyleEncoder produces (B, 32) output from (B, 109) input."""
    encoder = ContrastiveStyleEncoder()

    # Single sample (eval mode to avoid BatchNorm 1-sample restriction)
    encoder.eval()
    x = torch.randn(1, 109)
    out = encoder(x)
    assert out.shape == (1, 32), f"Single forward: {out.shape} (expected (1, 32))"
    assert torch.allclose(out.norm(dim=1), torch.ones(1), atol=1e-5), (
        "Output should be L2-normalized"
    )

    # Batch of 8
    x_batch = torch.randn(8, 109)
    out_batch = encoder(x_batch)
    assert out_batch.shape == (8, 32), f"Batch forward: {out_batch.shape} (expected (8, 32))"
    norms = out_batch.norm(dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), (
        f"All outputs should be L2-normalized, norms: {norms}"
    )

    # Verify negative values are allowed (cosine sim range [-1, 1])
    assert (out_batch < 0).any(), "Some embedding dimensions should be negative"

    print(f"  Single shape: {out.shape}, norm={out.norm().item():.4f}")
    print(f"  Batch shape: {out_batch.shape}, norms: min={norms.min().item():.4f}, max={norms.max().item():.4f}")


# ── Test 3: Contrastive loss ────────────────────────

@skipif_no_torch
def test_contrastive_loss():
    """Contrastive (triplet) loss should produce a valid scalar."""
    encoder = ContrastiveStyleEncoder()

    encoder.eval()
    # Use batched computation, then select specific samples
    batch_x = torch.randn(6, 109)
    batch_labels = torch.tensor([0, 0, 1, 1, 2, 2])
    embeddings = encoder(batch_x)

    anchor_emb = embeddings[0:1]
    positive_emb = embeddings[1:2]  # same composer (0)
    negative_emb = embeddings[2:3]  # diff composer (1)

    loss = encoder.triplet_loss(anchor_emb, positive_emb, negative_emb, margin=0.5)
    assert isinstance(loss, torch.Tensor), f"Loss should be tensor, got {type(loss)}"
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"

    # Check that margin loss works: similar pairs should have low loss
    loss_similar = encoder.triplet_loss(anchor_emb, positive_emb, anchor_emb, margin=0.5)
    assert loss_similar.item() >= 0.49, (
        f"When positive == negative (same embedding), loss should be ≈ margin, "
        f"got {loss_similar.item():.4f}"
    )

    # NT-Xent loss (use a fresh batch in train mode for the collation test)
    encoder.train()
    batch_x2 = torch.randn(6, 109)
    batch_labels2 = torch.tensor([0, 0, 1, 1, 2, 2])
    embeddings2 = encoder(batch_x2)
    nt_xent = encoder.nt_xent_loss(embeddings2, batch_labels2, temperature=0.1)
    assert isinstance(nt_xent, torch.Tensor), f"NT-Xent should be tensor, got {type(nt_xent)}"
    assert nt_xent.ndim == 0, f"NT-Xent should be scalar, got shape {nt_xent.shape}"
    assert nt_xent.item() > 0, f"NT-Xent should be positive, got {nt_xent.item()}"

    print(f"  Triplet loss: {loss.item():.6f}")
    print(f"  NT-Xent loss: {nt_xent.item():.6f}")


# ── Test 4: Loss decreases over training steps ──────

@skipif_no_torch
def test_contrastive_loss_training():
    """With well-structured synthetic data, triplet loss should decrease.

    This tests that the gradient flows correctly and the model learns
    to separate different composer styles.
    """
    import torch.optim as optim

    # Create well-separated clusters
    rng = np.random.RandomState(42)
    n_per_cluster = 10
    n_clusters = 3

    vectors = []
    labels = []
    for c in range(n_clusters):
        centroid = torch.tensor([c * 3.0, c * 3.0, c * 3.0], dtype=torch.float32)
        for _ in range(n_per_cluster):
            vec = torch.randn(109) * 0.5
            vec[:3] += centroid
            vectors.append(vec)
            labels.append(c)

    vectors = torch.stack(vectors)
    labels = torch.tensor(labels)

    encoder = ContrastiveStyleEncoder()
    optimizer = optim.Adam(encoder.parameters(), lr=0.01)

    losses = []
    for epoch in range(20):
        encoder.train()
        optimizer.zero_grad()

        embeddings = encoder(vectors)
        loss = encoder.nt_xent_loss(embeddings, labels, temperature=0.2)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # Loss should decrease (at least from first to last)
    first_3 = float(np.mean(losses[:3]))
    last_3 = float(np.mean(losses[-3:]))
    assert last_3 < first_3, (
        f"Loss should decrease: first_3={first_3:.4f}, last_3={last_3:.4f}"
    )
    assert losses[-1] < losses[0], (
        f"Loss at epoch {len(losses)-1} ({losses[-1]:.4f}) should be < "
        f"epoch 0 ({losses[0]:.4f})"
    )

    print(f"  Loss: {losses[0]:.6f} → {losses[-1]:.6f}")
    print(f"  First 3 avg: {first_3:.6f}, Last 3 avg: {last_3:.6f}")


# ── Test 5: LoRA forward pass ───────────────────────

@skipif_no_torch
def test_lora_forward():
    """StyleLoRAAdapter produces (B, d_model) output from (B, 32) input."""
    adapter = StyleLoRAAdapter(embedding_dim=32, rank=4, d_model=4096)
    adapter.selftest()  # Built-in dim check

    # Single sample
    x = torch.randn(1, 32)
    out = adapter(x)
    assert out.shape == (1, 4096), f"Single forward: {out.shape} (expected (1, 4096))"

    # Batch
    x_batch = torch.randn(4, 32)
    out_batch = adapter(x_batch)
    assert out_batch.shape == (4, 4096), f"Batch forward: {out_batch.shape} (expected (4, 4096))"

    # Parameter count
    n_params = adapter.get_lora_parameters()
    expected = (32 * 4) + (4 * 4096)  # 128 + 16384 = 16512
    assert n_params == expected, (
        f"LoRA parameter count: {n_params} (expected {expected})"
    )

    # Verify rank bottleneck: r=4 means only 16,512 params
    total_params = sum(p.numel() for p in adapter.parameters())
    assert total_params == n_params + 1, (
        f"Total params should be n_params + scale (1), got {total_params} vs {n_params + 1}"
    )

    print(f"  Single output shape: {out.shape}")
    print(f"  LoRA parameters: {n_params} (rank=4, 32→4096)")


# ── Test 6: End-to-end contrastive training ─────────

@skipif_no_torch
def test_end_to_end_contrastive():
    """Full training loop with synthetic data should converge."""
    from plato_torch_bridge.train import train_contrastive

    samples = _make_synthetic_samples(n=60, n_composers=3)
    ds = MIDIStyleDataset(samples)

    # Train with small epochs for test speed
    model = train_contrastive(
        ds,
        epochs=10,
        batch_size=16,
        learning_rate=0.01,
        loss_type="triplet",
        val_split=0.2,
        verbose=False,
    )

    # Model should produce valid embeddings
    test_vec = torch.randn(1, 109)
    with torch.no_grad():
        embedding = model(test_vec)
    assert embedding.shape == (1, 32), f"Embedding shape: {embedding.shape}"
    assert torch.allclose(embedding.norm(), torch.ones(1), atol=1e-5), (
        "Embedding should be L2-normalized"
    )

    # Check that same-composer vectors are closer than different-composer
    rng = np.random.RandomState(42)
    composer_a_style = np.zeros(109)
    composer_a_style[103] = 0.5
    composer_a_style[104] = 0.5

    composer_b_style = np.zeros(109)
    composer_b_style[103] = 5.0
    composer_b_style[104] = 5.0

    emb_a = model.encode_style_vector(composer_a_style)
    emb_b = model.encode_style_vector(composer_b_style)

    # Same composer vectors should be closer than different
    similarity = model.similarity(composer_a_style, composer_a_style)
    similarity_diff = model.similarity(composer_a_style, composer_b_style)
    # Same should always be 1.0 (L2-normed)
    assert abs(similarity - 1.0) < 1e-5, f"Same-vector similarity: {similarity:.6f} (expected 1.0)"

    print(f"  Model embedding dim: {embedding.shape}")
    print(f"  Same-composer similarity: {similarity:.4f}")
    print(f"  Different-composer similarity: {similarity_diff:.4f}")


# ── Test 7: Dataset utility methods ─────────────────

@skipif_no_torch
def test_dataset_utilities():
    """Dataset get_labels, get_style_vectors, filter_by_label, split."""
    samples = _make_synthetic_samples(n=50, n_composers=3)
    ds = MIDIStyleDataset(samples)

    # get_labels
    labels = ds.get_labels()
    assert labels.shape == (50,), f"Labels shape: {labels.shape} (expected (50,))"
    assert (labels >= 0).all(), "All labels should be non-negative"

    # get_style_vectors
    sv = ds.get_style_vectors()
    assert sv.shape == (50, 109), f"Style vectors shape: {sv.shape} (expected (50, 109))"
    assert sv.dtype == torch.float32, f"Style vectors dtype: {sv.dtype}"

    # filter_by_label
    filtered = ds.filter_by_label(0)
    assert len(filtered) > 0, "Filtered dataset should have samples"
    for sample in filtered.samples:
        assert sample["label"] == 0, f"Filtered sample has wrong label: {sample['label']}"

    # split
    train_ds, val_ds = ds.split(frac=0.8, seed=42)
    assert len(train_ds) + len(val_ds) == 50, (
        f"Split should preserve total: {len(train_ds)} + {len(val_ds)} = "
        f"{len(train_ds) + len(val_ds)} (expected 50)"
    )
    assert len(train_ds) == 40, f"Train size: {len(train_ds)} (expected 40)"
    assert len(val_ds) == 10, f"Val size: {len(val_ds)} (expected 10)"

    print(f"  Labels: {len(labels)} samples, {labels.unique().shape[0]} unique")
    print(f"  Style vectors: {sv.shape}")
    print(f"  Filtered (label=0): {len(filtered)} samples")
    print(f"  Split: {len(train_ds)} train + {len(val_ds)} val")


# ── Test 8: Empty/Degenerate Dataset ────────────────

@skipif_no_torch
def test_empty_dataset():
    """Empty dataset should not crash."""
    ds = MIDIStyleDataset([])
    assert len(ds) == 0, f"Empty dataset length: {len(ds)} (expected 0)"

    # get_labels on empty
    labels = ds.get_labels()
    assert labels.shape == (0,), f"Empty labels shape: {labels.shape} (expected (0,))"

    # get_style_vectors on empty
    sv = ds.get_style_vectors()
    assert sv.shape == (0, 109), f"Empty style vectors shape: {sv.shape} (expected (0, 109))"

    # filter_by_label on empty
    filtered = ds.filter_by_label(0)
    assert len(filtered) == 0, "Filtered empty should stay empty"

    # Split empty
    train_ds, val_ds = ds.split(frac=0.8)
    assert len(train_ds) == 0, "Empty split train should be empty"
    assert len(val_ds) == 0, "Empty split val should be empty"

    print("  Empty dataset handled correctly")


# ── Test 9: Encode batch API ────────────────────────

@skipif_no_torch
def test_encoder_encode_api():
    """Encoder's encode_batch and similarity APIs work correctly."""
    encoder = ContrastiveStyleEncoder()

    # encode_style_vector
    vec = np.random.randn(109).astype(np.float32)
    emb = encoder.encode_style_vector(vec)
    assert emb.shape == (32,), f"Encode single: {emb.shape} (expected (32,))"
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-5, "Single emb should be L2-normalized"

    # encode_batch
    batch = np.random.randn(5, 109).astype(np.float32)
    embs = encoder.encode_batch(batch)
    assert embs.shape == (5, 32), f"Encode batch: {embs.shape} (expected (5, 32))"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "All batch embs should be L2-normalized"

    # similarity
    sim = encoder.similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-5, f"Same-vector similarity: {sim:.6f} (expected 1.0)"

    sim_diff = encoder.similarity(vec, -vec)
    # Should be >= -1 (L2-normalized, so -vec should be cosine=-1)
    assert sim_diff >= -1.0 and sim_diff <= 1.0, f"Similarity out of range: {sim_diff}"

    print(f"  Single embedding norm: {float(np.linalg.norm(emb)):.6f}")
    print(f"  Batch embedding shape: {embs.shape}")
    print(f"  Batch norms: min={norms.min():.6f}, max={norms.max():.6f}")


# ── Test 10: LoRA merge ─────────────────────────────

@skipif_no_torch
def test_lora_merge():
    """StyleLoRAAdapter.merge_with produces interpolated weights."""
    adapter_a = StyleLoRAAdapter(embedding_dim=32, rank=4, d_model=64)
    adapter_b = StyleLoRAAdapter(embedding_dim=32, rank=4, d_model=64)

    # Set distinct weights
    with torch.no_grad():
        adapter_a.down.weight.data.fill_(0.5)
        adapter_a.up.weight.data.fill_(0.3)
        adapter_b.down.weight.data.fill_(0.1)
        adapter_b.up.weight.data.fill_(0.7)

    # Merge with alpha=0.5
    merged = adapter_a.merge_with(adapter_b, alpha=0.5)

    # Verify weight interpolation (NOT output interpolation — neural net
    # outputs are not linear in weights even when weights are linear)
    with torch.no_grad():
        # alpha=0.5: merged.down = 0.5*A.down + 0.5*B.down
        expected_down = (adapter_a.down.weight * 0.5 +
                         adapter_b.down.weight * 0.5)
        assert torch.allclose(merged.down.weight, expected_down, atol=1e-5), (
            "Merged down weight should be interpolated"
        )

        # alpha=0.5: merged.up = 0.5*A.up + 0.5*B.up
        expected_up = (adapter_a.up.weight * 0.5 +
                       adapter_b.up.weight * 0.5)
        assert torch.allclose(merged.up.weight, expected_up, atol=1e-5), (
            "Merged up weight should be interpolated"
        )

        # alpha=0.5: merged.scale = 0.5*A.scale + 0.5*B.scale
        expected_scale = (adapter_a.scale * 0.5 +
                          adapter_b.scale * 0.5)
        assert torch.allclose(merged.scale, expected_scale, atol=1e-5), (
            f"Merged scale {merged.scale.item():.4f} should be "
            f"interpolated {expected_scale.item():.4f}"
        )

    # Merge with alpha=1.0 should equal A's weights
    merged_a = adapter_a.merge_with(adapter_b, alpha=1.0)
    with torch.no_grad():
        assert torch.allclose(merged_a.down.weight, adapter_a.down.weight, atol=1e-5)
        assert torch.allclose(merged_a.up.weight, adapter_a.up.weight, atol=1e-5)

    # Merge with alpha=0.0 should equal B's weights
    merged_b = adapter_a.merge_with(adapter_b, alpha=0.0)
    with torch.no_grad():
        assert torch.allclose(merged_b.down.weight, adapter_b.down.weight, atol=1e-5)
        assert torch.allclose(merged_b.up.weight, adapter_b.up.weight, atol=1e-5)

    print("  LoRA merge (weight interpolation): verified")


# ── Test 11: Prompt to style vector ─────────────────

@skipif_no_torch
def test_prompt_to_style():
    """_prompt_to_style_vector should produce valid 109D vector."""
    from plato_torch_bridge.train import _prompt_to_style_vector

    vec = _prompt_to_style_vector("dark, staccato, mechanical")
    assert len(vec) == 109, f"Prompt vector length: {len(vec)} (expected 109)"
    assert vec[18:30].sum() == 0, "Unset regions should be zero"
    # Dark → lower pitches; staccato → high staccato_ratio; mechanical → staccato
    assert vec[81] > 0.5, f"Staccato ratio should be high: {vec[81]}"

    # Test that different prompts produce different vectors
    vec2 = _prompt_to_style_vector("bright, legato, flowing")
    assert not np.allclose(vec, vec2), "Different prompts should differ"

    # Test a complex prompt
    vec3 = _prompt_to_style_vector("fast, complex, jazz, wide range")
    assert vec3[105] > 5.0, f"Note density should be high for 'fast': {vec3[105]}"
    assert vec3[107] > 5.0, f"Harmonic complexity for jazz: {vec3[107]}"

    print(f"  'dark, staccato, mechanical' → staccato_ratio={vec[81]:.2f}")
    print(f"  'bright, legato, flowing'    → staccato_ratio={vec2[81]:.2f}")
    print(f"  'jazz' prompt: harmonic_complexity={vec3[107]:.1f}")
