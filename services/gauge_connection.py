#!/usr/bin/env python3
"""Gauge Connection — parallel transport of eigenvalue data across PLATO rooms.
From FM's constraint-theory-core: parallel transport across tile networks using holonomy matrices.
"""
import numpy as np, json, urllib.request, time

def fetch_eigenvalues(room, host="localhost:8847"):
    resp = json.loads(urllib.request.urlopen(f"http://{host}/room/{room}/history").read())
    eigs = []
    for t in resp.get("tiles", []):
        try:
            a = json.loads(t.get("answer", "{}"))
            top5 = a.get("eigenvalue_top5", a.get("eigenvalues", []))
            if top5: eigs.append(np.array([float(e) for e in top5]))
        except: continue
    if not eigs: return None
    min_len = min(len(e) for e in eigs)
    trimmed = np.array([e[:min_len] for e in eigs])
    return {"mean": np.mean(trimmed, axis=0), "count": len(eigs)}

def compute_gauge(o1, fm):
    o1e, fme = o1["mean"][:3], fm["mean"][:3]
    gauge = np.diag(fme / np.maximum(o1e, 1e-10))
    return {"gauge": gauge, "holonomy": float(np.linalg.norm(gauge - np.eye(3)))}

o1 = fetch_eigenvalues("fleet-coupling")
fm = fetch_eigenvalues("fleet-coord")
print(f"O1: {o1['count']} tiles" if o1 else "O1: no data")
print(f"FM: {fm['count']} tiles" if fm else "FM: no data")
if o1 and fm:
    g = compute_gauge(o1, fm)
    print(f"Holonomy: {g['holonomy']:.6f}")
    print("Alignment:", "IDENTITY" if g['holonomy'] < 0.3 else "MISALIGNED")
