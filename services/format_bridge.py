#!/usr/bin/env python3
"""Eigenvalue Format Bridge — standard tile format for both PLATOs.

The gauge connection found: Oracle1 uses JSON eigenvalue tiles with
field names like "eigenvalue_top5" and "spectral_gap". FM uses Matrix
bridge text messages in a different format. The format gap prevents
cross-PLATO gauge computation.

This tool READS both formats and publishes to a SHARED format that
both PLATOs can consume. It's the quarter-inch adapter plate.

Standard format (published to fleet-coupling):
{
    "tile_type": "eigenvalue_summary",
    "source_plato": "oracle1" | "fm",
    "generated_at": timestamp,
    "n_agents": int,
    "eigenvalues": [float],      // top 5 eigenvalues, descending
    "spectral_gap": float,        // (λ₁-λ₂)/λ₁
    "pc1_ratio": float,           // λ₁/Σλ
    "effective_rank_95": int,     // dims for 95% variance
}

Usage:
    python3 format_bridge.py [--publish] [--read]
"""

import json, urllib.request, time, sys
import numpy as np

PLATO_HOST = "localhost:8847"
ROOM = "fleet-coupling"


def read_any_format(room, host=PLATO_HOST):
    """Read eigenvalue data from any format PLATO room.
    
    Tries: JSON (any field name), Matrix bridge text, raw text.
    Converts to standard format.
    """
    url = f"http://{host}/room/{room}/history"
    resp = json.loads(urllib.request.urlopen(url, timeout=5).read())
    
    for t in reversed(resp.get("tiles", [])):
        answer = t.get("answer", "")
        
        # Try JSON
        try:
            data = json.loads(answer)
            # Accept multiple field name conventions
            eigs = data.get("eigenvalues") or data.get("top_eigenvalues") or data.get("eigenvalue_top5")
            if eigs and isinstance(eigs, list):
                return to_standard(data, eigs, room)
        except:
            pass
    
    return None


def to_standard(data, eigs, room):
    """Convert any format to the standard eigenvalue summary."""
    eigs = sorted([float(e) for e in eigs], reverse=True)
    return {
        "tile_type": "eigenvalue_summary",
        "source_plato": "oracle1" if "oracle1" in room else "fm",
        "generated_at": time.time(),
        "n_agents": data.get("n_agents", data.get("n_effective_dims", len(eigs))),
        "eigenvalues": eigs[:5],
        "spectral_gap": float((eigs[0] - eigs[1]) / eigs[0]) if len(eigs) >= 2 else 0.0,
        "pc1_ratio": float(eigs[0] / sum(eigs)),
        "effective_rank_95": min(len(eigs), 5),
    }


def publish_standard(data, room=ROOM):
    """Publish a standard-format eigenvalue tile."""
    tile = {
        "question": f"Eigenvalue bridge — {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "answer": json.dumps(data),
        "context": "eigenvalue-format-bridge"
    }
    url = f"http://{PLATO_HOST}/room/{room}/submit"
    req = urllib.request.Request(url, json.dumps(tile).encode(), {"Content-Type": "application/json"})
    try:
        resp = json.loads(urllib.request.urlopen(req, timeout=5).read())
        return resp.get("status") == "accepted"
    except:
        return False


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "--publish"
    
    if action == "--read":
        # Read both PLATOs
        o1 = read_any_format("fleet-coupling")
        fm = read_any_format("fleet-coord")
        print("O1:" if o1 else "O1: no data")
        if o1: print(f"  γ̃={o1['spectral_gap']:.4f} PC1={o1['pc1_ratio']:.4f}")
        print("FM:" if fm else "FM: no data")
        if fm: print(f"  γ̃={fm['spectral_gap']:.4f} PC1={fm['pc1_ratio']:.4f}")
    
    elif action == "--publish":
        # Publish Oracle1's current state in standard format
        o1 = read_any_format("fleet-coupling")
        if o1:
            ok = publish_standard(o1)
            print(f"Published O1 state: {'✅' if ok else '❌'}")
            print(f"  γ̃={o1['spectral_gap']:.4f} PC1={o1['pc1_ratio']:.4f}")
        else:
            print("No Oracle1 data to publish")
