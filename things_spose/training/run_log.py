"""Per-run logging for training.

Named ``run_log`` rather than ``logging`` so it cannot be confused with — or
shadow — the standard library module it wraps.

:func:`setup` attaches a file handler (and, by default, a console handler) to a
dedicated logger, so a long fit leaves a durable record next to its checkpoints
instead of only scrolling past on stdout. :func:`log_epoch` renders the same
per-epoch record that :mod:`things_spose.training.train` collects into its
history, so the log file and the returned ``TrainResult.history`` never disagree.
"""
from __future__ import annotations

import logging
from pathlib import Path

LOGGER_NAME = "things_spose.training"


def setup(log_file: str | Path | None = None, console: bool = True,
          level: int = logging.INFO, append: bool = False) -> logging.Logger:
    """Return the training logger, writing to ``log_file`` and/or the console.

    Safe to call repeatedly: handlers are replaced, not stacked, so a second
    call in the same process does not double every line.

    ``append`` keeps an existing log file rather than truncating it — set it when
    resuming, so the record of the earlier epochs survives.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    fmt = logging.Formatter(fmt="%(asctime)s - [%(levelname)s] - %(message)s",
                            datefmt="%d/%m/%Y %H:%M:%S")
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a" if append else "w")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(ch)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def format_epoch(rec: dict) -> str:
    """One-line rendering of a training history record."""
    msg = (f"epoch {rec['epoch']:4d}  loss {rec['loss']:12.1f}"
           f"  active_dims {rec['active_dims']:3d}")
    if rec.get("val_acc") is not None:
        msg += f"  val_acc {rec['val_acc'] * 100:5.2f}%"
    return msg


def log_epoch(logger: logging.Logger, rec: dict) -> None:
    logger.info(format_epoch(rec))
