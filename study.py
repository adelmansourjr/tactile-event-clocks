"""Writer-held-out evaluation of fixed-size tactile event representations."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import re
import sys
import time
from zipfile import ZipFile

for thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_setting] = "1"

import h5py
import numpy as np

ARCHIVE_MD5 = "6ca4638b2f95bf34f59873ab62399bd8"
METHODS = ("fixed", "range", "spatial", "rank", "hybrid", "full")
CONDITIONS = ("clean", "warp", "loss", "merge")
LAMBDAS = (0.03, 0.3, 3, 30)
ROOT = Path(__file__).resolve().parent
LOCKED = ("study.py", "PROTOCOL.md", "test_study.py")


def digest(path, kind="sha256"):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, kind).hexdigest()


def read_archive(path):
    if digest(path, "md5") != ARCHIVE_MD5:
        raise ValueError("Archive checksum differs from the published Zenodo file")
    pattern = re.compile(r"([A-Z]+)_([1-9A-Z])_(\d+)_spikes\.h5")
    records = []
    with ZipFile(path) as archive:
        for member in sorted(archive.namelist()):
            match = pattern.fullmatch(member.rsplit("/", 1)[-1])
            if "/ProcessedSpikes/" not in member or match is None:
                continue
            writer, label, repetition = match.groups()
            with h5py.File(io.BytesIO(archive.read(member)), "r") as stream:
                events = stream["spikes"][:]
            timestamp = events["timestamp"].astype(np.float64)
            taxel = events["taxel_id"].astype(np.int16) - 1
            polarity = events["polarity"]
            if not (np.isfinite(timestamp).all() and ((0 <= timestamp) & (timestamp < 2)).all()
                    and ((0 <= taxel) & (taxel < 256)).all() and np.isin(polarity, [-1, 1]).all()):
                raise ValueError("Unexpected event encoding")
            records.append((writer, label, int(repetition), timestamp, taxel, polarity > 0))
    if len(records) != 7700 or len(set(r[:3] for r in records)) != len(records):
        raise ValueError("Unexpected archive size or duplicate records")
    return records


def split(records):
    codes = sorted({r[0] for r in records})
    if len(codes) != 34:
        raise ValueError("Unexpected number of writers")
    order = np.random.default_rng(2026112311).permutation(codes)
    anonymous = {writer: i for i, writer in enumerate(order)}
    labels = {letter: i for i, letter in enumerate(sorted({r[1] for r in records}))}
    assert len(labels) == 35
    groups = [[], [], []]
    for index, r in enumerate(records):
        w = anonymous[r[0]]
        part = 0 if w < 18 else 1 if w < 24 else 2
        groups[part].append((index, w, labels[r[1]], *r[3:]))
    return groups


def transformed(index, t, taxel, polarity, condition):
    if condition == "clean" or not len(t):
        return t, taxel, polarity
    rng = np.random.default_rng(800100 + index)
    if condition in ("warp", "merge"):
        pivot = rng.uniform(.30, 1.70)
        warped = np.where(t <= 1, t * pivot, pivot + (t - 1) * (2 - pivot))
        if condition == "merge":
            warped = np.floor(warped * 5) / 5
        return warped, taxel, polarity
    if condition == "loss":
        start = rng.uniform(.10, .70) * 2
        keep = ((t < start) | (t >= start + .40)) & (rng.random(len(t)) >= .10)
        return t[keep], taxel[keep], polarity[keep]
    raise ValueError(condition)


def spatial_ids(taxel, polarity, side):
    reduce = 16 // side
    return (((taxel // 16) // reduce) * side + ((taxel % 16) // reduce)
            + polarity * side * side).astype(np.int64)


def histogram(t, ids, bins, width, clock):
    if len(t) == 0:
        return np.zeros((bins, width))
    if clock == "rank":
        _, inverse, lengths = np.unique(t, return_inverse=True, return_counts=True)
        ends = lengths.cumsum()
        starts = ends - lengths
        # Multiply rank coordinates by bins; overlaps are exact integers until the final division.
        layers = []
        for i in range(bins):
            overlap = np.maximum(0, np.minimum(ends * bins, (i + 1) * len(t))
                                 - np.maximum(starts * bins, i * len(t)))
            mass = overlap[inverse] / (bins * lengths[inverse])
            layers.append(np.bincount(ids, weights=mass, minlength=width))
        return np.array(layers)
    if clock == "range":
        scale = float(np.ptp(t))
        ratio = (t - t.min()) / scale if scale else np.zeros_like(t)
    elif clock == "fixed":
        ratio = t / 2
    else:
        raise ValueError(clock)
    index = np.minimum((ratio * bins).astype(int), bins - 1)
    return np.bincount(ids + width * index, minlength=width * bins).reshape(bins, width)


def encode(t, taxel, p, method):
    side = 16 if method in ("spatial", "full") else 8
    width = 2 * side ** 2
    ids = spatial_ids(taxel, p, side)
    if method == "spatial":
        values = np.bincount(ids, minlength=width)
    elif method == "hybrid":
        values = np.concatenate((histogram(t, ids, 2, width, "range"),
                                 histogram(t, ids, 2, width, "rank"))).ravel() / 2
    else:
        values = histogram(t, ids, 4, width, "rank" if method == "full" else method).ravel()
    return np.sqrt(values / max(len(t), 1))


def features(records, method, condition):
    return np.stack([encode(*transformed(i, t, a, p, condition), method)
                     for i, _, _, t, a, p in records])


def eq_writer(ok, w):
    return np.array([ok[w == writer].mean() for writer in sorted(set(w))])


def train(train_records, validation, method):
    train_x = features(train_records, method, "clean")
    y = np.eye(35)[[r[2] for r in train_records]]
    mean_x = train_x.mean(axis=0)
    mean_y = y.mean(axis=0)
    train_x -= mean_x
    gram = train_x.T @ train_x
    rhs = train_x.T @ (y - mean_y)
    validation_x = [features(validation, method, condition) - mean_x for condition in CONDITIONS]
    w = np.array([r[1] for r in validation])
    target = np.array([r[2] for r in validation])
    candidates = []
    for lam in LAMBDAS:
        weights = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
        accuracies = [float(eq_writer(np.argmax(x @ weights + mean_y, axis=1) == target, w).mean())
                      for x in validation_x]
        candidates.append((float(np.mean(accuracies)), lam, accuracies, weights))
    _, lam, accuracies, weights = max(candidates, key=lambda v: (v[0], -v[1]))
    return mean_x, mean_y, weights, lam, accuracies


def score(records, fits):
    conditions = {}
    for condition in CONDITIONS:
        conditions[condition] = {}
        for method in METHODS:
            x0 = time.perf_counter()
            encoded = features(records, method, condition)
            x1 = time.perf_counter()
            mean_x, mean_y, weights, _, _ = fits[method]
            predicted = np.argmax((encoded - mean_x) @ weights + mean_y, axis=1)
            conditions[condition][method] = {
                "predicted": predicted.tolist(), "encode_seconds": x1 - x0,
                "readout_seconds": time.perf_counter() - x1,
                "features": weights.shape[0], "readout_bytes_fp32": int((weights.size + mean_y.size) * 4),
            }
    return conditions


def summarize(rows, predictions, comparator):
    w = np.array([r[0] for r in rows])
    y = np.array([r[1] for r in rows])
    n = len(set(w))
    bootstrap = np.random.default_rng(832146).integers(0, n, size=(10000, n))
    summary = {}
    for condition, methods in predictions.items():
        acc = {}
        for method, outputs in methods.items():
            ok = np.array(outputs["predicted"]) == y
            acc[method] = {"equal_writer": float(eq_writer(ok, w).mean()),
                           "pooled": float(ok.mean()), "writer_accuracy": eq_writer(ok, w).tolist()}
        delta = np.array(acc["rank"]["writer_accuracy"]) - acc[comparator]["writer_accuracy"]
        summary[condition] = {
            "accuracy": acc, "paired_delta": float(delta.mean()),
            "writer_bootstrap95": np.quantile(delta[bootstrap].mean(axis=1), [.025, .975]).tolist(),
            "writer_deltas": delta.tolist(),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    exist = args.output.exists()
    if exist:
        raise SystemExit("Use a fresh output path")
    lock = {name: digest(ROOT / name) for name in LOCKED}
    start = time.perf_counter()
    train_records, validation, test_records = split(read_archive(args.archive))
    print("Group sizes", *(len(part) for part in (train_records, validation, test_records)), flush=True)
    fits = {}
    for method in METHODS:
        fits[method] = train(train_records, validation, method)
        print(method, fits[method][3], fits[method][4], flush=True)
    comparator = max(METHODS[:3], key=lambda name: fits[name][4][1])
    print("Primary comparator", comparator, flush=True)
    nominal = validation if args.development else test_records
    rows = [(r[1], r[2]) for r in nominal]
    predictions = score(nominal, fits)
    result = summarize(rows, predictions, comparator)
    args.output.mkdir(parents=True)
    raw = {"comparator": comparator, "label_and_writer_index": rows, "predictions": predictions}
    (args.output / "predictions.json").write_text(json.dumps(raw, separators=(",", ":")) + "\n")
    warp = result["warp"]
    clean = result["clean"]
    primary = warp["paired_delta"] >= .02 and warp["writer_bootstrap95"][0] > 0
    deployment = primary and clean["paired_delta"] > -.02 and clean["writer_bootstrap95"][0] > -.02
    info = {"validation_only": args.development, "sizes": [len(x) for x in (train_records, validation, test_records)],
            "test_samples_by_writer": dict(Counter(str(r[0]) for r in rows)),
            "archive_sha256": digest(args.archive), "frozen_hashes": lock, "comparator": comparator,
            "tuning": {k: {"lambda": v[3], "validation_accuracies": v[4]} for k, v in fits.items()},
            "claims": {"primary_pass": primary, "deployment_gate_pass": deployment},
            "results": result, "raw_sha256": digest(args.output / "predictions.json"),
            "elapsed_seconds": time.perf_counter() - start, "python": sys.version, "numpy": np.__version__,
            "h5py": h5py.__version__, "platform": sys.platform,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
            (1 if sys.platform == "darwin" else 1024)}
    (args.output / "summary.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Completed", args.output, info["elapsed_seconds"], info["peak_rss_bytes"], info["claims"])


if __name__ == "__main__":
    main()
