"""Two-level (grouped) shuffle sampler for study-grouped volumetric samples.

The stock ``DataLoader(shuffle=True)`` shuffles sample indices globally, so
consecutive batches jump across studies / raw volumes at random.  For RAW
reconstruction data each study maps to its own memmap files, and random
jumping destroys access locality (TLB misses, cache-line thrash, and seek
storms on a cold cache).  This sampler does a *two-level* shuffle instead:

1. shuffle the **group order** (one group == one study / volume index), then
2. shuffle the **samples within each group** (their window ``start`` /
   mapping / residual combinations).

The loader therefore yields every sample of one study (in random order)
before moving on to the next study -- no cross-file jumping within a local
window of time.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class GroupedShuffleSampler(Sampler[int]):
    """Yield sample indices grouped by ``group_ids`` with a two-level shuffle.

    Parameters
    ----------
    group_ids:
        One id per sample; samples sharing an id belong to the same group
        (e.g. the same study / volume index).  ``len(group_ids)`` must equal
        the dataset length.
    """

    def __init__(self, group_ids: Sequence[int]) -> None:
        self.group_ids: list[int] = list(group_ids)
        # group id -> ascending sample indices belonging to that group.
        self._groups: dict[int, list[int]] = {}
        for i, g in enumerate(self.group_ids):
            self._groups.setdefault(g, []).append(i)

    def __iter__(self) -> Iterator[int]:
        # Uses the global RNG (seeded once by ``seed_everything``) so the
        # whole run stays reproducible while each epoch gets a fresh order.
        order = list(self._groups)
        random.shuffle(order)
        for g in order:
            indices = list(self._groups[g])
            random.shuffle(indices)
            yield from indices

    def __len__(self) -> int:
        return len(self.group_ids)


__all__ = ["GroupedShuffleSampler"]
