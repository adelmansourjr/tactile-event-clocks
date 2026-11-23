import itertools
import unittest

import numpy as np
from study import encode, histogram, transformed


class EventClock(unittest.TestCase):
    def test_rank_matches_independent_slot_enumeration(self):
        # In fine rank units, spread each tied event uniformly over its group's slots.
        t = np.array([0, .2, .2, .2, .3, .3, .9])
        ids = np.array([0, 1, 1, 2, 0, 2, 1])
        bins = 4
        want = np.zeros((bins, 3))
        rank = 0
        for _, group in itertools.groupby(t):
            count = len(list(group))
            for slot in range(rank * bins, (rank + count) * bins):
                for k in range(rank, rank + count):
                    want[slot // len(t), ids[k]] += 1 / (bins * count)
            rank += count
        np.testing.assert_allclose(histogram(t, ids, bins, 3, "rank"), want, atol=1e-12)

    def test_preserves_mass_and_invariance(self):
        rng = np.random.default_rng(63)
        for i in range(100):
            t = rng.integers(0, 20, size=121) / 12
            taxel = rng.integers(0, 256, size=len(t))
            p = rng.random(len(t)) < .5
            first = encode(t, taxel, p, "rank")
            warped = encode(*transformed(i, t, taxel, p, "warp"), "rank")
            perm = rng.permutation(len(t))
            shuffled = encode(t[perm], taxel[perm], p[perm], "rank")
            np.testing.assert_allclose(first, warped, atol=1e-12)
            np.testing.assert_allclose(first, shuffled, atol=1e-12)
            self.assertAlmostEqual(float(first @ first), 1)

    def test_tie_merge_and_deletion_can_fail(self):
        t = np.array([0, .2, .3, .5])
        a = np.array([0, 32, 64, 96])
        p = np.zeros(4, dtype=bool)
        clean = encode(t, a, p, "rank")
        merged = encode(np.array([0, .2, .2, .5]), a, p, "rank")
        removed = encode(t[:-1], a[:-1], p[:-1], "rank")
        self.assertFalse(np.allclose(clean, merged))
        self.assertFalse(np.allclose(clean, removed))

    def test_compact_dimensions_and_blank_inputs(self):
        for name in ("fixed", "range", "spatial", "rank", "hybrid", "full"):
            empty = encode(np.empty(0), np.empty(0, dtype=int), np.empty(0, dtype=bool), name)
            self.assertEqual(len(empty), 2048 if name == "full" else 512)
            self.assertEqual(int(empty.sum()), 0)


if __name__ == "__main__":
    unittest.main()
