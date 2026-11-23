"""Writer-held-out pre-encoding stress benchmark. Raw pressure arrays never leave memory."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import re
import resource
import time
from zipfile import ZipFile

for setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[setting] = "1"

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
ARCHIVE_MD5 = "6ca4638b2f95bf34f59873ab62399bd8"
PATTERN = re.compile(r"([A-Z]+)_([1-9A-Z])_(\d+)\.h5")
BASELINES = ("fixed", "range", "rank", "concatenated", "spatial", "direct_pressure")
CANDIDATES = ("blend_25", "blend_50", "blend_75", "pressure_progress")
METHODS = BASELINES + CANDIDATES
LAMBDAS = (.03, .3, 3., 30.)
DEVELOPMENT = ("clean", "mild_rate", "mild_threshold", "mild_dead")
FINAL = ("clean", "rate", "coarse", "threshold", "dropout", "composite")
CLOCK = np.linspace(0., 1., 240)
LOW_GRID = ((np.arange(256) // 16) // 2) * 8 + (np.arange(256) % 16) // 2
NOMINAL = (4200, 1575, 1925)


def digest(path, kind="sha256"):
    with Path(path).open("rb") as inp:
        return hashlib.file_digest(inp, kind).hexdigest()


def manifest(archive):
    records = []
    for member in archive.namelist():
        if "/RawCharacters/" not in member:
            continue
        match = PATTERN.fullmatch(member.rsplit("/", 1)[-1])
        if match is None:
            continue
        writer, label, repetition = match.groups()
        records.append((writer, label, int(repetition), member))
    assert len(records) == 7700 and len(set(r[:3] for r in records)) == 7700
    codes = sorted({r[0] for r in records})
    labels = sorted({r[1] for r in records})
    assert len(codes) == 34 and len(labels) == 35
    split = np.random.default_rng(2026112311).permutation(codes)
    anonymous = {w: i for i, w in enumerate(split)}
    groups = [[], [], []]
    for idx, (writer, label, _, member) in enumerate(sorted(records)):
        w = anonymous[writer]
        s = 0 if w < 18 else 1 if w < 24 else 2
        groups[s].append((idx, w, labels.index(label), member))
    assert tuple(map(len, groups)) == NOMINAL
    return groups


def resample(pressure, locations):
    pixels = locations * (len(pressure) - 1)
    lo = pixels.astype(np.int64)
    hi = np.minimum(lo + 1, len(pressure) - 1)
    fraction = (pixels - lo)[:, None]
    return pressure[lo] * (1 - fraction) + pressure[hi] * fraction


def corruption(pressure, condition, seed):
    rng = np.random.default_rng(seed)
    quantile = .90
    if condition == "mild_rate":
        amp = rng.uniform(-.35, .35)
        pressure = resample(pressure, CLOCK + amp * np.sin(2 * np.pi * CLOCK) / (2 * np.pi))
    if condition in ("rate", "composite"):
        x = rng.uniform(.42, .58)
        y = rng.uniform(.15, .25) if rng.integers(2) else rng.uniform(.75, .85)
        locations = np.interp(CLOCK, [0, x, 1], [0, y, 1])
        pressure = resample(pressure, locations)
    if condition in ("coarse", "composite"):
        keep = np.r_[np.arange(0, 239, 3), 239]
        positions = np.interp(np.arange(240), keep, np.linspace(0, 1, len(keep)))
        pressure = np.clip(8 * np.round(resample(pressure[keep], positions) / 8), 0, 255)
    if condition == "mild_threshold":
        quantile = rng.uniform(.88, .94)
    if condition in ("threshold", "composite"):
        quantile = rng.uniform(.95, .97)
    if condition in ("mild_dead", "dropout", "composite"):
        pressure = pressure.copy()
        if condition == "mild_dead":
            dead = rng.random(256) < .05
            duration = int(rng.integers(15, 26))
        else:
            start_column = int(rng.integers(0, 13))
            dead = (np.arange(256) % 16 >= start_column) & (np.arange(256) % 16 < start_column + 4)
            duration = 36
        pressure[:, dead] = pressure[0, dead]
        start = int(rng.integers(45, 170))
        pressure[start:start + duration] = pressure[start - 1]
    return pressure, quantile


def observations(pressure, quantile):
    differences = np.diff(pressure, axis=0)
    absolute = np.abs(differences)
    nonzero = absolute[absolute > 0]
    threshold = max(4., np.quantile(nonzero, quantile)) if len(nonzero) else 4.
    frames, taxels = np.nonzero(absolute > threshold)
    return absolute, frames, taxels, differences[frames, taxels] > 0


def tent(clock, ids, width, bins, event_weights=None):
    x = np.clip(clock, 0, 1) * (bins - 1)
    lo = x.astype(np.int64)
    hi = np.minimum(lo + 1, bins - 1)
    frac = x - lo
    if event_weights is None:
        event_weights = np.ones_like(frac)
    return (
        np.bincount(lo * width + ids, weights=event_weights * (1 - frac), minlength=bins * width)
        + np.bincount(hi * width + ids, weights=event_weights * frac, minlength=bins * width)
    )


def encoding(pressure, quantile):
    absolute, frames, taxels, on = observations(pressure, quantile)
    mass = absolute.sum(axis=1)
    event_count = len(frames)
    if event_count:
        fixed = CLOCK[frames + 1]
        span = np.ptp(fixed)
        event_range = (fixed - fixed.min()) / span if span else np.zeros_like(fixed)
        per_frame = np.bincount(frames, minlength=239)
        midpoint_cdf = (np.cumsum(per_frame) - .5 * per_frame) / event_count
        rank = midpoint_cdf[frames]
        progress = (mass.cumsum() - .5 * mass) / mass.sum()
        progress = progress[frames]
    else:
        fixed = rank = event_range = progress = np.empty(0)
    coarse = LOW_GRID[taxels] + 64 * on
    fine = taxels + 256 * on
    timed = {
        "fixed": fixed,
        "range": event_range,
        "rank": rank,
        "blend_25": .75 * fixed + .25 * rank,
        "blend_50": .50 * fixed + .50 * rank,
        "blend_75": .25 * fixed + .75 * rank,
        "pressure_progress": progress,
    }
    values = {k: tent(v, coarse, 128, 4) for k, v in timed.items()}
    values["concatenated"] = np.r_[tent(event_range, coarse, 128, 2), tent(rank, coarse, 128, 2)] / 2
    values["spatial"] = np.bincount(fine, minlength=512)
    # Combine raw absolute differences on 2x2 spatial blocks. Weight time bins by variation.
    pooled = absolute.reshape(239, 8, 2, 8, 2).sum(axis=(2, 4)).reshape(239, 64)
    direct = tent(np.repeat(CLOCK[1:], 64), np.tile(np.arange(64), 239), 64, 8, pooled.ravel())
    values["direct_pressure"] = direct
    encoded = {k: np.sqrt(v / max(v.sum(), 1)) for k, v in values.items()}
    return encoded, event_count


def score_groups(correct, writers):
    return np.array([np.mean(correct[writers == w]) for w in np.unique(writers)])


def collect(archive, records, conditions, seed):
    arrays = {m: np.empty((len(conditions), len(records), 512), dtype=np.float32) for m in METHODS}
    writers = np.array([r[1] for r in records])
    truth = np.array([r[2] for r in records])
    counts = np.empty((len(conditions), len(records)), dtype=np.int32)
    start = time.perf_counter()
    for i, (index, _, _, member) in enumerate(records):
        with h5py.File(io.BytesIO(archive.read(member)), "r") as stream:
            p = stream["pressure_data"][:].reshape(240, 256).astype(np.float32)
        for j, condition in enumerate(conditions):
            altered, quantile = corruption(p, condition, seed + (index * 16) + j)
            values, count = encoding(altered, quantile)
            counts[j, i] = count
            for method, v in values.items():
                arrays[method][j, i] = v
    return arrays, labels_to_array(truth), writers, counts, time.perf_counter() - start


def labels_to_array(labels):
    return np.eye(35, dtype=np.float32)[labels]


def fit(x, y):
    x = x.reshape(-1, x.shape[-1])
    rows = np.tile(y, (len(x) // len(y), 1))
    xm = x.mean(axis=0, dtype=np.float64)
    ym = rows.mean(axis=0, dtype=np.float64)
    xc = x - xm
    gram = xc.T @ xc
    rhs = xc.T @ (rows - ym)
    eye = np.eye(x.shape[1])
    for lam in LAMBDAS:
        fitted = np.linalg.solve(gram + lam * eye, rhs)
        yield (lam, fitted, ym - xm @ fitted)


def prehashes():
    return {n: digest(Path(ROOT, n)) for n in ("PROTOCOL.md", "study.py", "PROOF.md", "test_study.py")}


def validate(archive_path, output):
    if digest(archive_path, "md5") != ARCHIVE_MD5:
        raise ValueError("Unrecognised dataset archive")
    with ZipFile(archive_path) as archive:
        records = manifest(archive)
        x, y, _, _, elapsed_train = collect(archive, records[0], DEVELOPMENT, 70000000)
        v, vy, vw, _, elapsed_valid = collect(archive, records[1], DEVELOPMENT, 81000000)
    scores = {}
    parameters = {}
    actual = vy.argmax(axis=1)
    for method in METHODS:
        candidates = []
        for lam, weights, bias in fit(x[method], y):
            dev = (v[method] @ weights + bias).argmax(axis=2)
            results = [float(score_groups(p == actual, vw).mean()) for p in dev]
            candidates.append((min(results), float(np.mean(results)), lam, results, weights, bias))
        worst, avg, lam, acc, weights, bias = max(candidates, key=lambda r: r[:2])
        scores[method] = {"worst": worst, "overall": avg, "lambda": lam, "accuracies": acc}
        parameters[method] = weights
        parameters[method + "_bias"] = bias
    select = lambda methods: max(methods, key=lambda m: (scores[m]["worst"], scores[m]["overall"]))
    model = output / "weights.npz"
    np.savez_compressed(model, **parameters)
    result = {
        "files": prehashes(), "weights_sha256": digest(model), "archive_sha256": digest(archive_path),
        "validation": scores, "baseline": select(BASELINES), "candidate": select(CANDIDATES),
        "train_encoding_seconds": elapsed_train, "validation_encoding_seconds": elapsed_valid,
        "training_samples_per_method": len(records[0]) * len(DEVELOPMENT),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    (output / "lock.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"baseline": result["baseline"], "candidate": result["candidate"], "validation": scores}, indent=2))


def grouped(predicted, actual, writers):
    return np.stack([
        np.stack([score_groups(row == actual, writers) for row in method]) for method in predicted
    ])


def summaries(predicted, actual, writers, lock):
    matrix = grouped(predicted, actual, writers)
    results = {
        method: {condition: float(matrix[m, c].mean()) for c, condition in enumerate(FINAL)}
        for m, method in enumerate(METHODS)
    }
    b = METHODS.index(lock["baseline"])
    p = METHODS.index(lock["candidate"])
    rng = np.random.default_rng(2026112402)
    idx = rng.integers(0, len(np.unique(writers)), size=(10000, len(np.unique(writers))))
    cand = matrix[p, 1:, :][:, idx].mean(axis=2).min(axis=0)
    base = matrix[b, 1:, :][:, idx].mean(axis=2).min(axis=0)
    clean = (matrix[p, 0] - matrix[b, 0])[idx].mean(axis=1)
    ci = np.quantile(cand - base, [.025, .975])
    ci_clean = np.quantile(clean, [.025, .975])
    worst_p = min(results[lock["candidate"]][c] for c in FINAL[1:])
    worst_b = min(results[lock["baseline"]][c] for c in FINAL[1:])
    return {
        "accuracies": results, "writer_accuracies": matrix.tolist(),
        "baseline": lock["baseline"], "candidate": lock["candidate"],
        "worst_candidate": worst_p, "worst_baseline": worst_b, "worst_gain": float(worst_p - worst_b),
        "paired_cluster_interval": ci.tolist(), "clean_interval": ci_clean.tolist(),
        "success": bool((worst_p - worst_b >= .02) and (ci[0] > 0)),
        "clean_gate": bool(ci_clean[0] > -.02),
    }


def test(archive_path, prepared, output):
    lock = json.loads((prepared / "lock.json").read_text())
    if prehashes() != lock["files"]:
        raise ValueError("Code or preregistration changed after validation")
    if digest(prepared / "weights.npz") != lock["weights_sha256"]:
        raise ValueError("Model weights changed after validation")
    if digest(archive_path) != lock["archive_sha256"]:
        raise ValueError("Dataset changed after validation")
    with ZipFile(archive_path) as archive:
        records = manifest(archive)
        features, y, w, counts, elapsed_test = collect(archive, records[2], FINAL, 92000000)
    elapsed = {}
    predictions = []
    with np.load(prepared / "weights.npz") as weights:
        for method in METHODS:
            start = time.perf_counter()
            pred = (features[method] @ weights[method] + weights[method + "_bias"]).argmax(axis=2)
            elapsed[method] = time.perf_counter() - start
            predictions.append(pred)
    predictions = np.stack(predictions)
    actual = y.argmax(axis=1)
    score = summaries(predictions, actual, w, lock)
    result = {
        **score,
        "input_archive_sha256": lock["archive_sha256"],
        "test_recordings_per_writer": dict(Counter(w.tolist())),
        "event_median": {k: float(np.median(counts[i])) for i, k in enumerate(FINAL)},
        "event_ratio_to_clean_median": {k: float(np.median(counts[i] / np.maximum(counts[0], 1))) for i, k in enumerate(FINAL)},
        "fraction_changed_event_counts": {k: float(np.mean(counts[i] != counts[0])) for i, k in enumerate(FINAL)},
        "joint_encoding_seconds": elapsed_test, "method_readout_seconds": elapsed,
        "readout_trainable_coefficients": 512 * 35 + 35,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    path = output / "predictions.json"
    path.write_text(json.dumps({
        "methods": METHODS, "conditions": FINAL, "actual": actual.tolist(),
        "writers": w.tolist(), "predictions": predictions.tolist(), "event_counts": counts.tolist(),
    }, separators=(",", ":")) + "\n")
    result["predictions_sha256"] = digest(path)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "writer_accuracies"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen-fit", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    if args.frozen_fit is None:
        validate(args.archive, args.output)
    else:
        test(args.archive, args.frozen_fit, args.output)


if __name__ == "__main__":
    main()
