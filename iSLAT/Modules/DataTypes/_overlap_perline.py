"""Per-line overlap windows for the curve-of-growth integral.

Every line is integrated on its own quadrature window (centred on its own
rest frequency) with the opacity of all neighbours within
``NEIGHBOR_RADIUS_X`` Doppler parameters summed in at each node:

    fint_i    = tau_i * sum_q w_q exp(-x_q^2) * esc(tau_tot,q)
    tau_tot,q = sum_j tau_j exp(-(x_q - d_ij)^2),   esc(t) = (1 - exp(-t)) / t
    d_ij      = c (nu_j - nu_i) / nu_i / b,          b = dv / (2 sqrt(ln 2))

Neighbours are contiguous in frequency order, so the tables hold only a
[lo, hi) range per line (O(n_lines) memory); each job builds its own
padded neighbour block.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np

import iSLAT.Constants as c

NEIGHBOR_RADIUS_X = 7.0          # neighbour radius, in Doppler parameters b
CORE_TAU_MAX = 1.0e4             # max tau of a call up to which only the 12 central nodes are used
CORE_NODES = slice(4, 16)        # the 12 central nodes of the 20-node rule (outer 8: exp(-x^2) <= 1e-11)
CHUNK_ELEMS = 1_000_000          # rows * nodes * width per job (bounds the float32 temporaries)
PARALLEL_MIN_WORK = 1_000_000    # pair-node evaluations below which a call runs inline
WIDTH_STEPS = 4                  # padded neighbour widths per octave
_TWO_SQRT_LN2 = 2.0 * np.sqrt(np.log(2.0))

_POOL: Optional[ThreadPoolExecutor] = None
_POOL_LOCK = threading.Lock()


def _n_threads() -> int:
    value = os.environ.get("ISLAT_INTENSITY_THREADS")
    if value:
        return max(1, int(value))
    return min(8, os.cpu_count() or 1)


def _pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(_n_threads(), thread_name_prefix="islat-overlap")
        return _POOL


def _reset_pool_after_fork() -> None:
    """A forked child has none of its parent's threads: start over with no pool."""
    global _POOL, _POOL_LOCK
    _POOL = None
    _POOL_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):     # POSIX only
    # Without this, a child forked after the parent used the pool (the
    # default start method of process pools on Linux before Python 3.14)
    # would queue its jobs on threads that do not exist, and never return.
    os.register_at_fork(after_in_child=_reset_pool_after_fork)


def _slot_widths(counts: np.ndarray, steps: int = WIDTH_STEPS) -> np.ndarray:
    """Round neighbour counts (>= 2) up to a ladder with ``steps`` widths per octave."""
    k = np.maximum(np.asarray(counts, dtype=np.int64), 2)
    _, exponent = np.frexp((k - 1).astype(np.float64))
    pow2 = np.left_shift(np.int64(1), exponent.astype(np.int64))
    stride = np.maximum(pow2 // (2 * steps), 1)
    return -(-k // stride) * stride


def build_tables(freq: np.ndarray, dv_max: float,
                 radius_x: float = NEIGHBOR_RADIUS_X) -> Optional[dict]:
    """Neighbour ranges of every line for intrinsic FWHM ``dv_max`` (km/s).

    Valid (merely over-complete) for any dv <= dv_max.  Returns ``None``
    when no line has a neighbour, so callers can keep their isolated-line
    fast path.
    """
    freq = np.asarray(freq, dtype=np.float64)
    if freq.size < 2:
        return None
    order = np.argsort(freq, kind="stable")
    fs = freq[order]
    vel = c.SPEED_OF_LIGHT_KMS * np.log(fs)       # window search only
    radius_kms = float(radius_x) * float(dv_max) / _TWO_SQRT_LN2
    lo = np.searchsorted(vel, vel - radius_kms, side="left")
    hi = np.searchsorted(vel, vel + radius_kms, side="right")
    counts = hi - lo                                # the line itself included
    blended = np.flatnonzero(counts > 1)
    if blended.size == 0:
        return None
    widths = _slot_widths(counts[blended])
    return {
        "dv_max": float(dv_max),
        "n_lines": int(freq.size),
        "order": order, "fs": fs, "lo": lo, "hi": hi,
        "isolated": order[counts == 1],
        "buckets": [(int(w), blended[widths == w]) for w in np.unique(widths)],
    }


def perline_fint(center_tau: np.ndarray, dv: np.ndarray, tables: dict,
                 x_quad: np.ndarray, w_quad: np.ndarray) -> np.ndarray:
    """Overlap-aware curve-of-growth factor, ``(n_cond, n_lines)`` float64.

    ``center_tau`` (n_cond, n_lines) line-centre opacities in line order,
    ``dv`` (n_cond,) intrinsic FWHM in km/s (each <= ``tables['dv_max']``),
    ``x_quad`` / ``w_quad`` the 20-node rule on [-6, 6].
    """
    tau = np.atleast_2d(np.asarray(center_tau, dtype=np.float64))
    n_cond, n_lines = tau.shape
    dv = np.broadcast_to(np.asarray(dv, dtype=np.float64), (n_cond,))

    # NaN fails the comparison and takes all nodes
    nodes = CORE_NODES if tau.max() <= CORE_TAU_MAX else slice(None)
    xq = np.asarray(x_quad, dtype=np.float64)[nodes]
    wq = np.asarray(w_quad, dtype=np.float64)[nodes]
    exq = np.exp(-xq * xq)
    w_self = wq * exq
    xq32 = xq.astype(np.float32)

    out = np.empty((n_cond, n_lines), dtype=np.float64)

    # Lines without a neighbour: plain isolated integral on the SAME nodes and
    # without the thin shortcut, so a line gets the same value on either side
    # of the classification.
    iso = tables["isolated"]
    if iso.size:
        out[:, iso] = (-np.expm1(-tau[:, iso, None] * exq)) @ wq

    order, fs, lo, hi = tables["order"], tables["fs"], tables["lo"], tables["hi"]
    tau_sorted = tau[:, order]                      # neighbours are contiguous here
    tau_sorted32 = tau_sorted.astype(np.float32)

    jobs = []
    for width, sel in tables["buckets"]:
        rows = max(1, CHUNK_ELEMS // (width * xq.size))
        jobs.extend((width, sel[s:s + rows]) for s in range(0, sel.size, rows))

    def run(job):
        width, sel = job                                        # sel: sorted positions
        pos = lo[sel, None] + np.arange(width)[None, :]
        valid = pos < hi[sel, None]
        pos = np.where(valid, pos, sel[:, None])                # padding -> self, zero opacity below
        f_self = fs[sel, None]
        # offsets differenced in float64, relative to each line, then float32
        dvel = (c.SPEED_OF_LIGHT_KMS * (fs[pos] - f_self) / f_self).astype(np.float32)
        lines = order[sel]
        for ic in range(n_cond):
            t_nb = np.where(valid, tau_sorted32[ic, pos], np.float32(0.0))       # (R, W)
            u = xq32[None, :, None] - (dvel * np.float32(_TWO_SQRT_LN2 / dv[ic]))[:, None, :]
            np.multiply(u, u, out=u)
            np.negative(u, out=u)
            np.exp(u, out=u)                                                     # (R, Q, W)
            tau_tot = np.matmul(u, t_nb[:, :, None])[:, :, 0].astype(np.float64)  # (R, Q)
            esc = np.ones_like(tau_tot)
            pos_t = tau_tot > 0.0
            esc[pos_t] = -np.expm1(-tau_tot[pos_t]) / tau_tot[pos_t]
            out[ic, lines] = tau_sorted[ic, sel] * (esc @ w_self)

    work = sum(w * s.size for w, s in jobs) * xq.size * n_cond
    if work >= PARALLEL_MIN_WORK and _n_threads() > 1:
        list(_pool().map(run, jobs))                # jobs write disjoint columns of out
    else:
        for job in jobs:
            run(job)
    return out
