"""Runtime backend selection: device (GPU/CPU) and worker/thread counts.

Centralizes every decision that depends on *where* the code runs so the rest of
the package stays hardware-agnostic. On a GPU cluster node this reports a CUDA
(or Apple MPS) device; on a fat CPU node it reports the core count from the
batch scheduler. Nothing here imports torch unless a GPU is actually requested.

Environment overrides
---------------------
THINGS_DEVICE     : force ``"cpu"``, ``"cuda"``, ``"cuda:1"``, ``"mps"`` ...
THINGS_NUM_WORKERS: force the CPU worker/thread count.
CUDA_VISIBLE_DEVICES / SLURM_CPUS_PER_TASK are honored automatically.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Device:
    kind: str          # "cuda", "mps", or "cpu"
    torch_device: str  # e.g. "cuda:0", "mps", "cpu"

    @property
    def is_gpu(self) -> bool:
        return self.kind in ("cuda", "mps")

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.torch_device


@functools.cache
def _torch():
    """Import torch lazily; return None if unavailable."""
    try:
        import torch  # noqa: PLC0415

        return torch
    except Exception:  # pragma: no cover - torch optional
        return None


@functools.cache
def num_workers() -> int:
    """CPU worker/thread count, respecting batch schedulers.

    Priority: THINGS_NUM_WORKERS > SLURM_CPUS_PER_TASK > os.cpu_count().
    """
    for var in ("THINGS_NUM_WORKERS", "SLURM_CPUS_PER_TASK"):
        val = os.environ.get(var)
        if val and val.isdigit() and int(val) > 0:
            return int(val)
    return os.cpu_count() or 1


@functools.cache
def select_device(prefer: str | None = None) -> Device:
    """Pick a compute device.

    ``prefer`` (or ``THINGS_DEVICE``) may be ``"cpu"``, ``"cuda"``, ``"cuda:N"``,
    ``"mps"``, or ``"auto"``. ``"auto"`` chooses a GPU when one is visible, else
    CPU. Requests that cannot be satisfied fall back to CPU with no error.
    """
    request = (prefer or os.environ.get("THINGS_DEVICE") or "auto").lower()
    torch = _torch()

    if request in ("cpu", "numpy", "numba"):
        return Device("cpu", "cpu")

    if torch is None:
        return Device("cpu", "cpu")

    def cuda_ok() -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    def mps_ok() -> bool:
        try:
            return torch.backends.mps.is_available()
        except Exception:
            return False

    if request.startswith("cuda"):
        return Device("cuda", request) if cuda_ok() else Device("cpu", "cpu")
    if request == "mps":
        return Device("mps", "mps") if mps_ok() else Device("cpu", "cpu")

    # auto
    if cuda_ok():
        return Device("cuda", "cuda")
    if mps_ok():
        return Device("mps", "mps")
    return Device("cpu", "cpu")


def resolve_backend(backend: str = "auto") -> str:
    """Map a requested similarity backend to a concrete one available here.

    Returns one of ``"gpu"``, ``"numba"``, ``"numpy"``.
    """
    backend = (backend or "auto").lower()
    if backend == "gpu":
        return "gpu" if select_device().is_gpu else "numba"
    if backend in ("numba", "numpy"):
        return backend
    # auto: GPU if present, else numba if importable, else numpy.
    if select_device().is_gpu:
        return "gpu"
    try:
        import numba  # noqa: F401,PLC0415

        return "numba"
    except Exception:
        return "numpy"


def configure_threads() -> None:
    """Set BLAS/OpenMP thread counts to ``num_workers`` to avoid oversubscription.

    Call once at process start (before heavy NumPy work). No-op if variables are
    already set by the user.
    """
    n = str(num_workers())
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(var, n)


def describe() -> str:
    """Human-readable summary for logs/notebooks."""
    dev = select_device()
    return (
        f"device={dev.torch_device} (kind={dev.kind}), "
        f"num_workers={num_workers()}, "
        f"similarity_backend={resolve_backend('auto')}"
    )
