"""Deterministic image batches with one aspect-ratio bucket across all ranks."""

import hashlib
import math
import random


def nearest_bucket(height, width, buckets):
    # Log distance treats reciprocal portrait/landscape ratios symmetrically.
    return min(range(len(buckets)),
               key=lambda i: abs(math.log((width / height) / (buckets[i][1] / buckets[i][0]))))


class AspectRatioBatchSampler:
    """Sample buckets in proportion to source counts, cycling shuffled sources.

    Every rank constructs the same global batches and takes its local slice.
    Integer sample IDs are globally unique within an epoch and seed augmentation;
    regenerating and skipping batches therefore preserves exact resume.
    Small buckets are repeated to fill a batch rather than silently discarded.
    """

    def __init__(self, dataset, batch_size, world_size=1, rank=0):
        if batch_size < 1 or world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("invalid batch size, world size or rank")
        self.dataset, self.batch_size = dataset, batch_size
        self.world_size, self.rank = world_size, rank
        self.groups = {}
        for index, record in enumerate(dataset.records):
            bucket = nearest_bucket(record["height"], record["width"], dataset.stage.crop_buckets)
            self.groups.setdefault(bucket, []).append(index)

    def __len__(self):
        return len(self.dataset) // (self.batch_size * self.world_size)

    def __iter__(self):
        key = f"{self.dataset.seed}:{self.dataset.epoch}:{self.dataset.stage.name}:buckets"
        rng = random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest(), "big"))
        buckets = sorted(self.groups)
        weights = [len(self.groups[bucket]) for bucket in buckets]
        pools = {bucket: [] for bucket in buckets}
        global_size = self.batch_size * self.world_size
        for batch in range(len(self)):
            bucket = rng.choices(buckets, weights=weights, k=1)[0]
            indices = []
            for offset in range(global_size):
                if not pools[bucket]:
                    pools[bucket] = list(self.groups[bucket])
                    rng.shuffle(pools[bucket])
                indices.append((batch * global_size + offset, pools[bucket].pop(), bucket))
            start = self.rank * self.batch_size
            yield indices[start:start + self.batch_size]
