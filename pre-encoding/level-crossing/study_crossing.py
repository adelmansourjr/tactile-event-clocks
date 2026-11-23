"""Compare a differential encoder and send-on-delta with exactly matched event counts."""
import argparse
from collections import Counter
import ctypes
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import study as common
import numpy as np
import h5py

METHODS = ("nominal_range", "nominal_rank", "derivative_range", "derivative_rank", "level_range", "level_rank")
GROUPS = ("nominal", "derivative", "level")


def compile_backend():
    td = tempfile.TemporaryDirectory()
    library = Path(td.name, "crossing.so")
    subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(ROOT / "events.c"), "-o", str(library)], check=True)
    kernel = ctypes.CDLL(str(library)).send_delta
    kernel.restype = ctypes.c_size_t
    kernel.argtypes = [
        np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_double, np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.int32, flags="C_CONTIGUOUS"), ctypes.c_size_t,
    ]
    return td, kernel


def level_events(pressure, diffs, kernel, step=8):
    pressure = np.ascontiguousarray(pressure, dtype=np.float64)
    capacity = int(np.abs(diffs).sum() // step) + 256
    t = np.empty(capacity, dtype=np.float64)
    ids = np.empty(capacity, dtype=np.int32)
    count = kernel(pressure, len(pressure), pressure.shape[1], step, t, ids, capacity)
    if count > capacity:
        raise RuntimeError("Crossing count exceeded the total-variation allowance")
    return t[:count], ids[:count]


def readout_features(t, labels, clock):
    if not len(t):
        return np.zeros(512)
    if clock == "range":
        spread = np.ptp(t)
        u = (t-t.min()) / spread if spread else np.zeros_like(t)
    elif clock == "rank":
        _, inv, sizes = np.unique(t, return_inverse=True, return_counts=True)
        u = (sizes.cumsum() - .5 * sizes)[inv] / len(t)
    else:
        raise ValueError(clock)
    xyz = common.LOW_GRID[labels % 256] + (labels >= 256) * 64
    return np.sqrt(common.tent(u, xyz, 128, 4) / len(t))


def triplet(pressure, q, seed, kernel):
    diffs = np.diff(pressure.astype(np.float64), axis=0)
    absolute = np.abs(diffs)
    nx = absolute[absolute > 0]
    threshold = max(4., np.quantile(nx, q)) if len(nx) else 4.
    t, a = np.nonzero(absolute > threshold)
    nominal = (t, a + 256 * (diffs[t, a] > 0))
    tx, ax = np.nonzero(absolute >= 4.)
    lx, la = level_events(pressure, diffs, kernel)
    cap = min(121, len(tx), len(lx))
    rng = np.random.default_rng(seed)
    dx = np.lexsort((rng.random(len(tx)), -absolute[tx, ax]))[:cap]
    ordering = np.argsort(lx, kind="stable")
    grid = ((np.arange(cap) + .5) * len(lx) / max(1, cap)).astype(int)
    select = ordering[grid]
    streams = (nominal,
               (tx[dx], ax[dx] + 256 * (diffs[tx[dx], ax[dx]] > 0)),
               (lx[select], la[select]))
    assert len(streams[1][0]) == len(streams[2][0]) == cap
    out = [readout_features(t, a, clock) for t, a in streams for clock in ("range", "rank")]
    return out, (len(nominal[0]), len(tx), len(lx), cap)


def collect(archive, records, conditions, seed, kernel):
    n = len(records)
    xs = np.empty((len(METHODS), len(conditions), n, 512), dtype=np.float32)
    ys, writers = np.array([r[2] for r in records]), np.array([r[1] for r in records])
    counts = np.empty((len(conditions), n, 4), np.int32)
    for i, (index, _, _, member) in enumerate(records):
        with h5py.File(io.BytesIO(archive.read(member)), "r") as stream:
            values = stream["pressure_data"][:].reshape(240, 256).astype(np.float32)
        for j, name in enumerate(conditions):
            sample_seed = seed + 16*index + j
            pressure, q = common.corruption(values, name, sample_seed)
            xs[:, j, i], counts[j, i] = triplet(pressure, q, sample_seed + 70000000, kernel)
    return xs, ys, writers, counts


def hashes():
    return {str(p.relative_to(ROOT.parent)): common.digest(p) for p in (
        ROOT / "PROTOCOL.md", ROOT / "events.c", ROOT / "study_crossing.py", ROOT.parent / "study.py")}


def fit(archive_path, out, kernel):
    if common.digest(archive_path, "md5") != common.ARCHIVE_MD5:
        raise ValueError("Unrecognised dataset archive")
    with ZipFile(archive_path) as archive:
        groups = common.manifest(archive)
        x, y, _, _ = collect(archive, groups[0], common.DEVELOPMENT, 149000000, kernel)
        v, vy, vw, _ = collect(archive, groups[1], common.DEVELOPMENT, 189000000, kernel)
    encoded = {}
    scores = {}
    for m, method in enumerate(METHODS):
        choices = []
        for lam, weights, bias in common.fit(x[m], common.labels_to_array(y)):
            pred = (v[m] @ weights + bias).argmax(axis=2)
            acc = [float(common.score_groups(p == vy, vw).mean()) for p in pred]
            choices.append((min(acc), float(np.mean(acc)), lam, weights, bias, acc))
        worst, avg, lam, weights, bias, acc = max(choices, key=lambda c: c[:2])
        encoded[method], encoded[method + "_bias"] = weights, bias
        scores[method] = {"worst": worst, "overall": avg, "lambda": lam, "conditions": acc}
    selected = {
        g: max(METHODS[2*i:2*i+2], key=lambda m: (scores[m]["worst"], scores[m]["overall"]))
        for i, g in enumerate(GROUPS)
    }
    np.savez_compressed(out / "weights.npz", **encoded)
    data = {"files": hashes(), "dataset": common.digest(archive_path), "weights": common.digest(out / "weights.npz"),
            "scores": scores, "chosen": selected,
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    (out / "lock.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(selected))


def analyze(pred, y, writers, lock):
    matrix = common.grouped(pred, y, writers)
    acc = {m: dict(zip(common.FINAL, list(map(float, matrix[i].mean(axis=-1))))) for i, m in enumerate(METHODS)}
    ids = [METHODS.index(lock["chosen"][m]) for m in ("level", "derivative")]
    a, b = matrix[ids]
    shifts = np.arange(1, len(common.FINAL))
    sensor = shifts[shifts != common.FINAL.index("threshold")]
    rng = np.random.default_rng(190012129)
    idx = rng.integers(0, len(np.unique(writers)), size=(10000, len(np.unique(writers))))
    worst_diff = lambda pa, pb: pa.min(axis=0) - pb.min(axis=0)
    gap = worst_diff(a[shifts].mean(axis=1), b[shifts].mean(axis=1))
    gap_sensor = worst_diff(a[sensor].mean(axis=1), b[sensor].mean(axis=1))
    bootstrap = worst_diff(a[shifts][:,idx].mean(axis=2), b[shifts][:,idx].mean(axis=2))
    ci = np.quantile(bootstrap, [.025, .975])
    clean = (a[0] - b[0])[idx].mean(axis=1)
    clean_ci = np.quantile(clean, [.025, .975])
    return {
        "accuracies": acc, "winner_picked_on_development": lock["chosen"],
        "contrast": float(gap), "contrast_excluding_threshold": float(gap_sensor),
        "paired_writer_interval": ci.tolist(), "paired_clean_interval": clean_ci.tolist(),
        "primary_pass": bool(gap >= .02 and gap_sensor >= 0 and ci[0] > 0),
        "clean_preservation": bool(clean_ci[0] > -.02),
        "writer_means": matrix.tolist(),
    }


def test(archive_path, model, out, kernel):
    lock = json.loads((model / "lock.json").read_text())
    assert lock["files"] == hashes()
    assert lock["dataset"] == common.digest(archive_path)
    assert lock["weights"] == common.digest(model / "weights.npz")
    started = time.perf_counter()
    with ZipFile(archive_path) as archive:
        x, y, writers, counts = collect(archive, common.manifest(archive)[2], common.FINAL, 203000000, kernel)
    with np.load(model / "weights.npz") as params:
        pred = np.stack([(x[i] @ params[m] + params[m + "_bias"]).argmax(axis=2) for i, m in enumerate(METHODS)])
    data = {"writers": writers.tolist(), "truth": y.tolist(), "predictions": pred.tolist(), "counts": counts.tolist()}
    path = out / "predictions.json"
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    result = {
        **analyze(pred, y, writers, lock),
        "event_count_medians_by_condition": dict(zip(common.FINAL, np.median(counts, axis=1).tolist())),
        "event_count_columns": ["percentile", "available_derivative", "raw_level", "exact_matched_count"],
        "test_recordings_per_writer": dict(Counter(writers.tolist())), "source_sha256": lock["dataset"],
        "wall_seconds_test": time.perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "raw_predictions_sha256": common.digest(path),
    }
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "writer_means"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frozen-fit", type=Path)
    args = parser.parse_args()
    args.output.mkdir()
    tmp, kernel = compile_backend()
    with tmp:
        if args.frozen_fit:
            test(args.archive, args.frozen_fit, args.output, kernel)
        else:
            fit(args.archive, args.output, kernel)


if __name__ == "__main__":
    main()
