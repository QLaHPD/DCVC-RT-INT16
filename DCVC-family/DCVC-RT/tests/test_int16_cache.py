from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from src.utils.common import load_int16_prep_state, save_int16_prep_state


class Int16PreparedCacheTests(unittest.TestCase):
    def test_concurrent_cache_publication_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth.tar"
            prepared = {
                "version": 3,
                "values": torch.arange(128, dtype=torch.int16),
            }

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(save_int16_prep_state, checkpoint, prepared)
                    for _ in range(5)
                ]
                for future in futures:
                    future.result()

            loaded = load_int16_prep_state(checkpoint)
            self.assertEqual(loaded["version"], prepared["version"])
            self.assertTrue(torch.equal(loaded["values"], prepared["values"]))
            self.assertEqual(list(Path(directory).glob(".*.tmp.*")), [])


if __name__ == "__main__":
    unittest.main()
