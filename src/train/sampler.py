"""CollisionAwareSampler — prevents false negatives from same-PDF samples in a batch.

Problem: ColPali in-batch contrastive loss treats every other doc in the batch as a
negative. If two samples come from the same PDF (different pages), one may be a true
positive for the other's query → false negative → reversed gradient.

Solution: Build batches from a shuffled buffer, deferring any sample whose PDF is
already represented in the current batch. Flush deferred items into subsequent batches.
A fallback (retry_limit) prevents infinite loops at the end of the epoch.

Usage:
    sampler = CollisionAwareSampler(
        dataset,
        batch_size=8,
        buffer_size=1000,
        retry_limit=50,
        drop_last=True,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)
"""

import random
from collections import deque
from typing import Iterator, List

from torch.utils.data import Sampler


class CollisionAwareSampler(Sampler):
    """Yields batches where no two samples share the same PDF (pdf_hash).

    Requires the underlying dataset to have a `records` list where each record
    has a `pdf_hash` key (produced by LlamaIndexDataset).

    Args:
        dataset:      Dataset with `records[i]["pdf_hash"]`.
        batch_size:   Number of samples per batch.
        buffer_size:  How many shuffled indices to keep in the sliding window.
                      Larger = better collision avoidance, more memory.
        retry_limit:  Max deferred-queue sweeps before forcing a collision.
                      Prevents infinite loops when remaining items share one PDF.
        drop_last:    Drop the final incomplete batch.
    """

    def __init__(
        self,
        dataset,
        batch_size:   int = 8,
        buffer_size:  int = 1000,
        retry_limit:  int = 50,
        drop_last:    bool = True,
    ):
        self.dataset      = dataset
        self.batch_size   = batch_size
        self.buffer_size  = buffer_size
        self.retry_limit  = retry_limit
        self.drop_last    = drop_last

    def _pdf_hash(self, idx: int) -> str:
        return self.dataset.records[idx]["pdf_hash"]

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[List[int]]:
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)

        # Sliding buffer — filled from the shuffled index list
        buf: deque = deque()
        src = iter(indices)

        def _fill_buf():
            while len(buf) < self.buffer_size:
                try:
                    buf.append(next(src))
                except StopIteration:
                    break

        _fill_buf()

        n_collisions = 0

        while True:
            _fill_buf()
            if not buf:
                break

            batch:    List[int] = []
            seen_pdf: set       = set()
            deferred: List[int] = []

            # ── Build one batch from buffer ───────────────────────────────
            while len(batch) < self.batch_size and (buf or deferred):
                # Try buffer first
                candidate = None
                retries   = 0

                while buf and retries < self.retry_limit:
                    idx  = buf.popleft()
                    phash = self._pdf_hash(idx)
                    if phash not in seen_pdf:
                        candidate = idx
                        seen_pdf.add(phash)
                        break
                    else:
                        deferred.append(idx)
                        retries += 1

                if candidate is None:
                    # Buffer exhausted or retry limit hit — try deferred queue
                    found = False
                    for i, idx in enumerate(deferred):
                        if self._pdf_hash(idx) not in seen_pdf:
                            candidate = deferred.pop(i)
                            seen_pdf.add(self._pdf_hash(candidate))
                            found = True
                            break

                    if not found:
                        # Fallback: accept collision (end-of-epoch data exhaustion)
                        if deferred:
                            candidate = deferred.pop(0)
                            n_collisions += 1
                        elif buf:
                            candidate = buf.popleft()
                            n_collisions += 1
                        else:
                            break

                if candidate is not None:
                    batch.append(candidate)

            # Put deferred items back into buffer for next batch
            buf.extendleft(reversed(deferred))

            if len(batch) == self.batch_size:
                yield batch
            elif not self.drop_last and batch:
                yield batch

        if n_collisions > 0:
            print(f"[CollisionAwareSampler] Accepted {n_collisions} forced collisions (end-of-epoch)")
