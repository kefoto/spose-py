"""Triplet-softmax similarity: the port of ``embedding2sim.m`` plus ``squareformq``.

``embedding2sim`` is the one genuinely expensive computation in the whole paper
(O(N^3) in the number of objects; the MATLAB triple ``for`` loop takes ~10-15 min
per matrix). It is provided with three interchangeable backends selected at
runtime by :mod:`things_spose.backend`:

* ``"gpu"``   - batched torch tensor ops on CUDA/MPS (fast path on a cluster GPU).
* ``"numba"`` - a ``parallel=True`` fused CPU kernel (no N*N temporaries).
* ``"numpy"`` - a vectorized reference used for correctness checks.

All three reproduce the shipped ``spose_sim`` to the documented ~1e-3 tolerance
and agree with each other. The similarity for a pair (i, j) is

    cp[i, j] = mean_{k != i, j}  e_ij / (e_ij + e_ik + e_jk),   e = exp(E @ E.T)

which is symmetric in i, j, with unit diagonal.
"""
from __future__ import annotations

import numpy as np

from . import backend


# --------------------------------------------------------------------------- #
# squareformq: matrix <-> lower-triangular vector (MATLAB column-major order)
# --------------------------------------------------------------------------- #
def squareformq(x: np.ndarray) -> np.ndarray:
    """Generalized ``squareform`` matching MATLAB ``squareformq.m``.

    Uses **column-major** lower-triangle order (``tril(...,-1)`` linear indexing)
    so that a pair-vector produced here lines up with MATLAB's, which matters when
    the same random indices resample two vectors in bootstrap CIs.

    * (n, n) square matrix  -> (n*(n-1)/2,) vector of below-diagonal entries.
    * (m,)  vector          -> (n, n) symmetric matrix with zero diagonal.
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[0] == x.shape[1] and x.shape[0] > 1:
        n = x.shape[0]
        # Column-major lower triangle: iterate columns, take rows below diagonal.
        rows, cols = np.tril_indices(n, k=-1)
        order = np.argsort(cols, kind="stable")  # group by column (column-major)
        return x[rows[order], cols[order]]
    if x.ndim == 1 or (x.ndim == 2 and 1 in x.shape):
        v = x.ravel()
        m = v.size
        n = int(round(0.5 * (np.sqrt(8 * m + 1) + 1)))
        if n * (n - 1) // 2 != m:
            raise ValueError("input is not a valid dissimilarity vector length")
        out = np.zeros((n, n), dtype=v.dtype)
        rows, cols = np.tril_indices(n, k=-1)
        order = np.argsort(cols, kind="stable")
        out[rows[order], cols[order]] = v
        out = out + out.T
        return out
    raise ValueError("squareformq expects a vector or a square symmetric matrix")


# --------------------------------------------------------------------------- #
# Backend implementations
# --------------------------------------------------------------------------- #
def _emb2sim_numpy(esim: np.ndarray) -> np.ndarray:
    """Vectorized NumPy reference (loops over i, full j*k plane per i)."""
    n = esim.shape[0]
    cp = np.empty((n, n), dtype=esim.dtype)
    diag = np.diagonal(esim)
    for i in range(n):
        a = esim[i]                                  # e_ik over k / e_ij over j
        denom = a[:, None] + a[None, :] + esim       # a[j] + a[k] + e_jk  -> (n, n)
        row = a[:, None] / denom                     # term[j, k]
        # sum over all k, then remove the k==i and k==j contributions
        cp[i] = (row.sum(axis=1) - row[:, i] - np.diagonal(row))
    cp /= (n - 2)
    np.fill_diagonal(cp, 1.0)
    return cp


def _emb2sim_numba(esim: np.ndarray) -> np.ndarray:
    from ._numba_kernels import emb2sim_kernel  # compiled on first use

    esim = np.ascontiguousarray(esim)
    cp = emb2sim_kernel(esim)
    np.fill_diagonal(cp, 1.0)
    return cp


def _emb2sim_gpu(esim_host: np.ndarray, dtype=np.float32) -> np.ndarray:
    import torch

    dev = backend.select_device().torch_device
    n = esim_host.shape[0]
    tdtype = torch.float32 if np.dtype(dtype) == np.float32 else torch.float64
    esim = torch.as_tensor(esim_host, dtype=tdtype, device=dev)
    diag = torch.diagonal(esim)                       # (n,) e_ii
    cp = torch.empty((n, n), dtype=tdtype, device=dev)

    tile = _gpu_tile_size(n, tdtype)
    for start in range(0, n, tile):
        stop = min(start + tile, n)
        eik = esim[start:stop]                         # (B, n): e_ik over k, e_ij over j
        # denom[b, j, k] = e_ij + e_ik + e_jk
        denom = eik[:, :, None] + eik[:, None, :] + esim[None, :, :]
        full = (eik[:, :, None] / denom).sum(dim=2)    # (B, n): sum over all k
        # remove k==j and k==i self/other contributions
        term_kj = eik / (2.0 * eik + diag[None, :])    # k==j: e_ij/(2 e_ij + e_jj)
        term_ki = eik / (2.0 * eik + diag[start:stop][:, None])  # k==i
        cp[start:stop] = full - term_kj - term_ki
        del denom, full
    cp /= (n - 2)
    idx = torch.arange(n, device=dev)
    cp[idx, idx] = 1.0
    return cp.to("cpu").numpy()


def _gpu_tile_size(n: int, tdtype) -> int:
    """Rows per tile so the (B, n, n) working tensor fits in device memory."""
    import torch

    bytes_per = 4 if tdtype == torch.float32 else 8
    per_row = n * n * bytes_per * 3  # denom + intermediates headroom
    try:
        dev = backend.select_device()
        if dev.kind == "cuda":
            free, _total = torch.cuda.mem_get_info()
            budget = int(free * 0.5)
        else:  # mps / unknown: be conservative
            budget = 512 * 1024 * 1024
    except Exception:
        budget = 512 * 1024 * 1024
    return max(1, min(n, budget // max(per_row, 1)))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def embedding2sim(
    embedding: np.ndarray,
    backend_name: str = "auto",
    dtype=np.float64,
) -> np.ndarray:
    """Convert an ``(N, D)`` embedding to an ``(N, N)`` similarity matrix.

    Parameters
    ----------
    embedding : (N, D) array
    backend_name : ``"auto" | "gpu" | "numba" | "numpy"``
        ``"auto"`` picks GPU if available, else the Numba kernel, else NumPy.
    dtype : compute dtype. ``float64`` for parity; ``float32`` is ~2x faster and
        used by the ablation where the last digits are irrelevant.
    """
    E = np.asarray(embedding, dtype=dtype)
    esim = np.exp(E @ E.T)

    chosen = backend.resolve_backend(backend_name)
    if chosen == "gpu":
        return _emb2sim_gpu(esim, dtype=dtype)
    if chosen == "numba":
        return _emb2sim_numba(esim)
    return _emb2sim_numpy(esim)


def embedding2sim48(
    embedding: np.ndarray,
    wordposition48: np.ndarray,
    dtype=np.float64,
) -> np.ndarray:
    """The Fig-2 variant: similarity of the 48 objects, summing the softmax only
    over the 48 reference columns and dividing by 48 (``make_figures`` lines
    297-317). Returns the ``(48, 48)`` block directly.
    """
    E = np.asarray(embedding, dtype=dtype)
    W = np.asarray(wordposition48, dtype=np.int64)
    S = np.exp(E[W] @ E[W].T)                     # (48, 48) esim restricted to W
    n = W.size
    Sab = S[:, :, None]
    Sac = S[:, None, :]
    Sbc = S[None, :, :]
    term = Sab / (Sab + Sac + Sbc)                # (48, 48, 48) over (a, b, c=k)
    # exclude c==a and c==b (MATLAB `continue`), but still divide by 48
    a_idx = np.arange(n)
    term[a_idx, :, a_idx] = 0.0                   # c == a
    term[:, a_idx, a_idx] = 0.0                   # c == b
    cp = term.sum(axis=2) / n                     # divide by 48 (not 46), per MATLAB
    cp = 0.5 * (cp + cp.T)                         # symmetric (already is up to fp)
    np.fill_diagonal(cp, 1.0)
    return cp
