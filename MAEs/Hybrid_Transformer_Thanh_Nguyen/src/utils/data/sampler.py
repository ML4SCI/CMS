from typing import List, NamedTuple, Iterator, Optional

import torch
from torch.utils.data import Sampler


class SampleKey(NamedTuple):
    file_idx: int  # index into your dataset's file list
    event_idx: int  # 0 .. events_per_file-1


def _coprime_step(n: int, gen: torch.Generator) -> int:
    # n = 100_000 = 2^5 * 5^5, so choose step that is odd and not divisible by 5
    # This guarantees gcd(step, n) == 1.
    while True:
        k = int(torch.randint(0, n // 2, (1,), generator=gen).item())
        step = 2 * k + 1
        if step % 5 != 0:
            return step
        

class JetClassDistributedSampler(Sampler[List[SampleKey]]):
    """
    Deterministic, class-balanced distributed batch sampler.

    Each batch draws evenly from exactly one file per class (10 files),
    and across all steps in an epoch it visits *every event of every file*
    exactly once, disjoint across ranks.

    Parameters
    ----------
    files_by_class: List[List[int]]
        10 lists of file indices (equal length per class).
    events_per_file: int
        Events per file (e.g., 100_000).
    batch_size: int
        Global batch size (sum over ranks). Must be divisible by world_size*10.
    rank: int
        Global DDP rank.
    world_size: int
        Total DDP processes.
    seed: Optional[int]
        Base seed for deterministic shuffles.
    shuffle_files: bool
        Shuffle file order per epoch.

    Returns
    -------
    Iterator[List[SampleKey]]
        Batches of `SampleKey(file_idx, event_idx)` of length local_batch_size.
    """
    def __init__(
        self,
        files_by_class: List[List[int]],
        events_per_file: int = 100_000,
        batch_size: int = 1000,  # global batch size
        rank: int = 0,
        world_size: int = 1,  # pass world_size here
        seed: Optional[int] = None,
        shuffle_files: bool = True,
    ):
        assert len(files_by_class) == 10, "Expect exactly 10 classes."
        n_per_class = [len(x) for x in files_by_class]
        assert len(set(n_per_class)) == 1, "All classes must have the same file count."

        self.num_groups = n_per_class[0]  # e.g., 100
        self.files_by_class = files_by_class
        self.events_per_file = int(events_per_file)
        self.shuffle_files = bool(shuffle_files)

        # Match DDP world_size when replicas_per_rank = 4
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.batch_size_global = int(batch_size)

        # Stronger guarantee if using world_size > 1
        if self.world_size > 1:
            assert self.batch_size_global % (self.world_size * 10) == 0, \
            f"with world_size > 1, batch_size must be divisible by {self.world_size * 10}."

        # Per-(virtual)rank batch size and per-file local draw
        self.local_batch_size = self.batch_size_global // self.world_size
        assert self.local_batch_size % 10 == 0, "local_batch_size must be divisible by 10."
        self.per_file_local = self.local_batch_size // 10  # per rank, per file, per step
        self.per_file_global = self.per_file_local * self.world_size  # across all ranks, per file, per step

        # Cover exactly all events in each file per epoch
        assert self.events_per_file % self.per_file_global == 0, \
        f"events_per_file ({self.events_per_file}) must be divisible by per_file_global ({self.per_file_global})."
        self.steps_per_file = self.events_per_file // self.per_file_global

        self.seed = int(seed) if seed is not None else int(torch.initial_seed() % 2**32)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_groups * self.steps_per_file

    def _file_orders_for_epoch(self) -> List[List[int]]:
        # Optionally shuffle the files per class; deterministic per epoch
        if not self.shuffle_files:
            return [list(range(len(cls_files))) for cls_files in self.files_by_class]
        
        orders = []
        for c in range(10):
            n = len(self.files_by_class[c])
            g = torch.Generator()
            g.manual_seed(self.seed * 1009 + self.epoch * 7919 + c * 271)
            orders.append(torch.randperm(n, generator=g).tolist())

        return orders

    def __iter__(self) -> Iterator[List[SampleKey]]:
        orders = self._file_orders_for_epoch()

        # Pick one file per class for each group
        for group_idx in range(self.num_groups):
            selected_files = [
                self.files_by_class[c][orders[c][group_idx]] for c in range(10)
            ]

            # Precompute a permutation (start, step) per file that is constant across steps
            per_file_rng = {}
            for fid in selected_files:
                g = torch.Generator()
                # Deterministic per (epoch, file); not dependent on pass index
                g.manual_seed(
                    self.seed ^ \
                    (self.epoch * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFF) ^ \
                    (fid * 0x94D049BB133111EB & 0xFFFFFFFFFFFF)
                )
                start = int(torch.randint(0, self.events_per_file, (1,), generator=g).item())
                step = _coprime_step(self.events_per_file, g)
                per_file_rng[fid] = (start, step)

            # Walk through all steps needed to exhaust each file
            for pass_idx in range(self.steps_per_file):
                batch = []

                # Global offset at this pass for each file; disjoint across ranks
                base_offset_global = pass_idx * self.per_file_global + self.rank * self.per_file_local
                for fid in selected_files:
                    start, step = per_file_rng[fid]
                    N = self.events_per_file

                    # Indices for this rank from this file for this step
                    for j in range(self.per_file_local):
                        pos = base_offset_global + j
                        ev = (start + pos * step) % N
                        batch.append(SampleKey(file_idx=fid, event_idx=ev))

                # Optional light within-batch shuffle (deterministic)
                g_batch = torch.Generator()
                g_batch.manual_seed(
                    self.seed * 65537 + self.epoch * 131071 + \
                    group_idx * 524287 + pass_idx * 8191 + self.rank
                )
                perm = torch.randperm(len(batch), generator=g_batch).tolist()

                yield [batch[i] for i in perm]