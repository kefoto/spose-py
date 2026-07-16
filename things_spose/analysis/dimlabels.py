"""Word-cloud data for the SPoSE dimensions.

Ports ``make_figures_behavsim.m`` lines 543-558: the free-text answers that 20
participants gave for each of the 49 dimensions (``dimlabel_answers``, a 20x49
cell array of comma-separated strings) are split on commas, trimmed, and counted
into a ``(word, occurrences)`` frequency table per dimension. Consumed by the
Figure 3 and Extended Data Figure 2 word clouds in :mod:`things_spose.analysis.viz`.

This module is pure-numeric (no plotting), so it is safe to import on a headless
cluster node.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _coerce_str(cell) -> str:
    """Turn one ``dimlabel_answers`` cell (a MATLAB char loaded by scipy) into str."""
    if isinstance(cell, str):
        return cell
    arr = np.asarray(cell).ravel()
    if arr.size == 0:
        return ""
    if arr.size == 1:
        return str(arr.item())
    return "".join(str(x) for x in arr)


@dataclass
class DimensionWords:
    """Frequency table for one dimension's free-text labels."""

    words: list[str]      # unique response words (alphabetical, matching MATLAB `unique`)
    counts: np.ndarray    # (len(words),) int occurrence counts

    def as_frequencies(self) -> dict[str, int]:
        """``{word: count}`` mapping, the form ``wordcloud`` expects."""
        return {w: int(c) for w, c in zip(self.words, self.counts)}


def dimension_words(dimlabel_answers: np.ndarray, dim: int) -> DimensionWords:
    """Word/count table for a single 0-based dimension index."""
    responses: list[str] = []
    n_sub = dimlabel_answers.shape[0]
    for i_sub in range(n_sub):
        raw = _coerce_str(dimlabel_answers[i_sub, dim])
        for token in raw.split(","):
            token = token.strip()
            if token:  # drop empties from trailing/double commas
                responses.append(token)

    if not responses:
        return DimensionWords(words=[], counts=np.zeros(0, dtype=np.int64))

    words, counts = np.unique(np.array(responses, dtype=object), return_counts=True)
    return DimensionWords(words=[str(w) for w in words], counts=counts.astype(np.int64))


def all_dimension_words(dimlabel_answers: np.ndarray) -> list[DimensionWords]:
    """Word/count tables for all 49 dimensions (list indexed by 0-based dim)."""
    n_dim = dimlabel_answers.shape[1]
    return [dimension_words(dimlabel_answers, d) for d in range(n_dim)]
