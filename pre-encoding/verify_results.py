"""Verify result integrity and recompute paired metrics from frozen predictions."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "level-crossing"))
import study
import study_crossing
import numpy as np


def read(path):
    return json.loads(path.read_text())


def verify_inputs(path):
    for line in (path / "PRE_VALIDATION_SHA256SUMS").read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert sha256((ROOT / filename).read_bytes()).hexdigest() == expected
    lock = read(path / "validation-lock.json")
    for name, expected in lock["files"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == expected
    return lock


def independently_score(predictions, targets, writers):
    unique_writers = sorted(set(writers))
    subjects = [[idx for idx, subject in enumerate(writers) if subject == w] for w in unique_writers]
    return np.array([
        [
            [sum(row[idx] == targets[idx] for idx in inds) / len(inds) for inds in subjects]
            for row in method
        ]
        for method in predictions
    ])


def verify_case(path, mechanism):
    lock = verify_inputs(path)
    report = read(path / "results/summary.json")
    raw_path = path / "results/predictions.json"
    raw = read(raw_path)
    predicted = np.array(raw["predictions"])
    actual = raw["truth" if mechanism else "actual"]
    writers = raw["writers"]
    cases = dict(Counter(writers))
    assert len(actual) == 1925 and len(cases) == 10
    assert cases == {int(k): v for k, v in report["test_recordings_per_writer"].items()}
    expected_digest = "raw_predictions_sha256" if mechanism else "predictions_sha256"
    assert sha256(raw_path.read_bytes()).hexdigest() == report[expected_digest]
    writer_means = np.asarray(report["writer_means" if mechanism else "writer_accuracies"])
    np.testing.assert_array_equal(independently_score(raw["predictions"], actual, writers), writer_means)
    if mechanism:
        analysis = study_crossing.analyze(predicted, np.asarray(actual), np.asarray(writers), lock)
        counts = np.asarray(raw["counts"])
        np.testing.assert_array_equal(counts[:, :, 3], np.minimum(121, counts[:, :, 1:3].min(axis=2)))
        expected_medians = list(report["event_count_medians_by_condition"].values())
    else:
        analysis = study.summaries(predicted, np.asarray(actual), np.asarray(writers), lock)
        counts = np.asarray(raw["event_counts"])
        expected_medians = list(report["event_median"].values())
    np.testing.assert_array_equal(np.median(counts, axis=1), expected_medians)
    for key, value in analysis.items():
        assert value == report[key], key
    return writer_means


def verify_level_backend():
    td, kernel = study_crossing.compile_backend()
    with td:
        pressure = np.zeros((3, 256))
        pressure[:, :3] = [[0, 0, 0], [16, 4, -16], [0, 8, 0]]
        finer = np.stack((pressure[0], (pressure[0] + pressure[1]) / 2, pressure[1],
                          (pressure[1] + pressure[2]) / 2, pressure[2]))
        raw = study_crossing.level_events(pressure, np.diff(pressure, axis=0), kernel)
        refined = study_crossing.level_events(finer, np.diff(finer, axis=0), kernel)
        base = sorted(zip(raw[0], raw[1]))
        expected = [(0.5, 2), (0.5, 256), (1, 2), (1, 256), (1.5, 0), (1.5, 258),
                    (2, 0), (2, 257), (2, 258)]
        assert base == expected
        assert sorted(zip(refined[0] / 2, refined[1])) == base
        rng = np.random.default_rng(462591)
        for _ in range(100):
            pressure = rng.integers(0, 32, size=(8, 256)).astype(float)
            diffs = np.diff(pressure, axis=0)
            t, a = study_crossing.level_events(pressure, diffs, kernel)
            variation = np.abs(diffs).sum(axis=0)
            by_taxel = np.bincount(a % 256, minlength=256)
            assert np.all(by_taxel * 8 <= variation)
            assert np.all((t > 0) & (t <= len(pressure) - 1))
        features, counts = study_crossing.triplet(pressure, .9, 31, kernel)
        assert counts[3] == min(121, *counts[1:3])
        assert all(v.shape == (512,) and np.isclose(np.linalg.norm(v), 1) for v in features)


if __name__ == "__main__":
    verify_case(ROOT, mechanism=False)
    verify_case(ROOT / "level-crossing", mechanism=True)
    verify_level_backend()
    print("Source locks, prediction digests, independent writer scores, matched budgets, and send-on-delta kernel verified.")
