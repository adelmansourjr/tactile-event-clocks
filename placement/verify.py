"""Independently replay the archived writer endpoints and check voxel invariance."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import unittest

import study

import numpy as np

ROOT = Path(__file__).resolve().parent


class RepresentativeTests(unittest.TestCase):
    def test_in_frame_translation(self):
        rng = np.random.default_rng(596)
        for _ in range(256):
            n = rng.integers(1, 122)
            t = rng.integers(0, 24, n)
            x = rng.integers(4, 12, n)
            y = rng.integers(4, 12, n)
            on = rng.random(n) > .5
            shift_x, shift_y = rng.integers(-4, 5, 2)
            original = study.features(t, x, y, on)
            modified = study.features(t, x+shift_x, y+shift_y, on)
            np.testing.assert_array_equal(original[-2:], modified[-2:])
            np.testing.assert_allclose(np.square(original).sum(axis=1), 1., atol=1e-6)
        for _ in range(10):
            permutation = rng.permutation(len(t))
            permuted = study.features(t[permutation], x[permutation], y[permutation], on[permutation])
            np.testing.assert_allclose(original, permuted, atol=1e-7)

    def test_border_or_missing(self):
        t, x, y = np.array([0, 1, 2]), np.array([3, 6, 15]), np.array([5, 7, 7])
        on = np.array([True, False, True])
        before = study.features(t, x, y, on)
        moved = x + 1
        keep = moved < 16
        after = study.features(t[keep], moved[keep], y[keep], on[keep])
        self.assertFalse(np.array_equal(before[-1], after[-1]))
        self.assertTrue(np.all(study.features(t[:0], x[:0], y[:0], on[:0]) == 0))


def ledger(directory):
    summary = json.loads((directory/"summary.json").read_text())
    lock = json.loads((ROOT/"validation-lock.json").read_text())
    predictions = directory/"predictions.jsonl"
    if summary["predictions_sha256"] != study.digest(predictions):
        raise ValueError("Ledger hash changed")
    if summary["protocol"] != study.fingerprints() or lock["files"] != study.fingerprints():
        raise ValueError("Frozen source changed")
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert len(rows) == 1925
    by_writer = defaultdict(list)
    for row in rows:
        assert set(row["predictions"]) == set(lock["validation"])
        assert len(row["counts"]) == len(study.TEST_CONDITIONS)
        assert all(0 <= n <= study.CAP for n in row["counts"])
        assert 0 <= row["label"] < study.N_CLASSES
        assert all(len(pred)==len(row["counts"]) and all(0 <= y < study.N_CLASSES for y in pred)
                   for pred in row["predictions"].values())
        by_writer[row["writer"]].append(row)
    assert sorted(by_writer) == summary["writer_order"]
    assert len(by_writer) == 10
    for k, condition in enumerate(study.TEST_CONDITIONS):
        assert np.median([r["counts"][k] for r in rows]) == summary["event_count_medians"][condition]
    values = {}
    for name in lock["validation"]:
        values[name] = np.array([
            [sum(r["label"]==r["predictions"][name][k] for r in group)/len(group)
             for k in range(len(study.TEST_CONDITIONS))]
            for _, group in sorted(by_writer.items())
        ])
        np.testing.assert_allclose(list(summary["accuracy"][name].values()), values[name].mean(axis=0))
    baseline = values[lock["baseline"]]
    candidate = values[lock["candidate"]]
    assert len(lock["validation"]) == 16
    use = [3, 5, 6]
    draws = np.random.default_rng(519308).integers(0,10,size=(12000,10))

    def minimum_gap(a,b):
        return a[...,use].mean(axis=-2).min(axis=-1)-b[...,use].mean(axis=-2).min(axis=-1)

    difference = minimum_gap(candidate,baseline)
    limits = np.quantile(minimum_gap(candidate[draws],baseline[draws]),[.025,.975])
    np.testing.assert_allclose([difference,*limits], summary["paired_worst"])
    clean = candidate[:,0]-baseline[:,0]
    clean_limits = np.quantile(clean[draws].mean(axis=1),[.025,.975])
    np.testing.assert_allclose([clean.mean(),*clean_limits], summary["clean"])
    assert summary["primary_pass"] == (difference >= .03 and limits[0] > 0)
    assert summary["clean_preserved"] == (clean_limits[0] > -.02)
    same_clock = values["centroid_rank_mlp64"][:,3] - values["extent_rank_mlp64"][:,3]
    descriptive = [same_clock.mean(), *np.quantile(same_clock[draws].mean(axis=1), [.025,.975])]
    np.testing.assert_allclose(descriptive, [.06514285714285714,.03828571428571427,.09714285714285717])
    assert all(same_clock > 0)
    print("Verified 1,925 common event counts, 16 sets of predictions, and both paired writer gates.")
    print("Placement-only, same-clock gain and interval (points): " +
          ", ".join(f"{result*100:.2f}" for result in descriptive))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path)
    args=parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RepresentativeTests)
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.results is not None:
        ledger(args.results)
