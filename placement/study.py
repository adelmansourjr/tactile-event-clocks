"""Writer-held-out tactile replay at exactly shared event inputs."""

import argparse
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
from zipfile import ZipFile

for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
source_spec = importlib.util.spec_from_file_location("pressure_replay", ROOT.parent / "pre-encoding/study.py")
source = importlib.util.module_from_spec(source_spec)
source_spec.loader.exec_module(source)

METHODS = ("range", "rank", "smooth_range", "smooth_rank", "extent_range", "extent_rank",
           "centroid_range", "centroid_rank")
BASES = METHODS[:6]
CANDIDATES = METHODS[6:]
FIT_CONDITIONS = ("clean", "mild_motion", "mild_place", "mild_both", "mild_sensor")
TEST_CONDITIONS = ("clean", "rate", "coarse", "placement", "masked",
                   "coarse_masked", "combined")
SEEDS = (47407319, 79276811, 68171639)
CAP = 121
N_CLASSES = 35
HIDDEN = 64
EPOCHS = (14, 28)
RATE = (.001, .003)
RIDGE = (.03, .3, 3., 30.)
PROTOCOL_FILES = (ROOT / "PROTOCOL.md", Path(__file__), ROOT.parent / "pre-encoding/study.py")


def digest(path):
    with Path(path).open("rb") as inp:
        return hashlib.file_digest(inp, "sha256").hexdigest()


def fingerprints():
    return {str(p.relative_to(ROOT.parent)): digest(p) for p in PROTOCOL_FILES}


def transfer(values, condition, rng):
    # Perturb pressure before selecting the common event stream.
    motion = condition in ("rate", "combined", "mild_motion", "mild_both")
    coarse = condition in ("coarse", "coarse_masked", "combined", "mild_sensor", "mild_both")
    masked = condition in ("masked", "coarse_masked", "combined", "mild_sensor", "mild_both")
    placed = condition in ("placement", "combined", "mild_place", "mild_both")
    mild = condition.startswith("mild")
    if motion:
        midpoint = rng.uniform(.43, .57)
        amplitude = rng.uniform(.035, .13) if mild else rng.uniform(.19, .29)
        side = rng.choice((-1, 1))
        grid = np.interp(source.CLOCK, [0, midpoint, 1], [0, midpoint+side*amplitude, 1])
        values = source.resample(values, grid)
    if coarse:
        step = 2 if mild else int(rng.integers(3, 6))
        keep = np.r_[np.arange(rng.integers(0, step), 239, step), 239]
        fraction = np.interp(np.arange(240), keep, np.linspace(0, 1, len(keep)))
        level = 4 if mild else int(rng.choice([8, 12, 16]))
        values = np.clip(level*np.round(source.resample(values[keep], fraction)/level), 0, 255)
    delta = np.diff(values, axis=0)
    t, k = np.nonzero(np.abs(delta) >= 4)
    x, y = k % 16, k // 16
    on = delta[t, k] > 0
    mag = np.abs(delta[t, k])
    if placed:
        axis = rng.integers(0, 2)
        signed = rng.choice((-1, 1))
        offset = rng.integers(1, 3) if mild else rng.integers(2, 6)
        if axis == 0:
            x = x + signed * offset
        else:
            y = y + signed * offset
    valid = (x >= 0) & (x < 16) & (y >= 0) & (y < 16)
    if masked:
        # Two orthogonal failure modes, independent of labels and coordinates of the stroke.
        width = 2 if mild else int(rng.integers(2, 5))
        start = rng.integers(0, 17-width)
        axis = x if rng.integers(0, 2) else y
        valid &= (axis < start) | (axis >= start+width)
        duration = 20 if mild else int(rng.integers(25, 55))
        burst = rng.integers(28, 211-duration)
        valid &= (t < burst) | (t >= burst+duration)
    picks = np.flatnonzero(valid)
    tie = rng.random(len(picks))
    order = picks[np.lexsort((tie, -mag[picks]))[:CAP]]
    return t[order], x[order], y[order], on[order]


def voxel(clock, x, y, on):
    # Bilinear spatial interpolation followed by two-neighbor temporal splatting.
    xx, yy = np.clip(x, 0, 7), np.clip(y, 0, 7)
    lx, ly = xx.astype(int), yy.astype(int)
    hx, hy = np.minimum(lx+1, 7), np.minimum(ly+1, 7)
    dx, dy = xx-lx, yy-ly
    off = on.astype(int)*64
    mass = sum(source.tent(clock, off+yc*8+xc, 128, 4, wx*wy)
               for xc, wx in ((lx, 1-dx), (hx, dx))
               for yc, wy in ((ly, 1-dy), (hy, dy)))
    return np.sqrt(mass/max(1, len(x)))


def centroid(v):
    # Form the integer centered numerator first so translations yield exactly identical bits.
    return (v*len(v)-v.sum())*(7/(18*len(v)))+3.5


def extent(v):
    # Coordinate-wise bounding box normalization with symmetric slack for tiny extents.
    center = (int(v.min())+int(v.max()))/2
    return (v-center) * (7 / max(10, np.ptp(v))) + 3.5


def features(t, x, y, on):
    if not len(t):
        return np.zeros((len(METHODS), 512), dtype=np.float32)
    elapsed = (t-t.min()) / max(1, np.ptp(t))
    _, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    rank = (counts.cumsum()-.5*counts)[inverse] / len(t)
    ids = x//2 + (y//2)*8 + on*64
    fixed = [np.sqrt(source.tent(kind, ids, 128, 4)/len(t)) for kind in (elapsed, rank)]
    smooth = [voxel(kind, x*7/15, y*7/15, on) for kind in (elapsed, rank)]
    bounding = [voxel(kind, extent(x), extent(y), on) for kind in (elapsed, rank)]
    centered = [voxel(kind, centroid(x), centroid(y), on) for kind in (elapsed, rank)]
    return np.array([*fixed, *smooth, *bounding, *centered], dtype=np.float32)


def collect(path, split, conditions):
    with ZipFile(path) as archive:
        records = source.manifest(archive)[split]
        n = len(records)
        arrays = np.empty((len(METHODS), len(conditions), n, 512), dtype=np.float32)
        counts = np.empty((len(conditions), n), np.uint8)
        for i, (record, _, _, member) in enumerate(records):
            with h5py.File(io.BytesIO(archive.read(member)), "r") as stream:
                pressure = stream["pressure_data"][:].reshape(240, 256).astype(np.float32)
            for j, condition in enumerate(conditions):
                rng = np.random.default_rng(SEEDS[split] + record*31 + j)
                events = transfer(pressure, condition, rng)
                arrays[:, j, i] = features(*events)
                counts[j, i] = len(events[0])
            if i and i % 700 == 0:
                print(f"split {split}: encoded {i}/{n}", flush=True)
    y = np.array([r[2] for r in records])
    writers = np.array([r[1] for r in records])
    return arrays, counts, y, writers


def score(params, arrays):
    if len(params) == 2:
        return (arrays @ params[0] + params[1]).argmax(axis=-1)
    first, first_bias, last, last_bias = params
    return (np.maximum(arrays @ first + first_bias, 0) @ last + last_bias).argmax(axis=-1)


def ridge(x, y):
    centered = x-x.mean(axis=0)
    target = y-y.mean(axis=0)
    gram = centered.T @ centered
    rhs = centered.T @ target
    for lam in RIDGE:
        w = np.linalg.solve(gram + np.eye(x.shape[-1], dtype=np.float32)*lam, rhs)
        yield {"kind":"ridge", "lambda":lam}, (w, y.mean(axis=0)-x.mean(axis=0) @ w)


def mlp(x, y, rate, seed=17931):
    rng = np.random.default_rng(seed)
    params = [
        (rng.standard_normal((512,HIDDEN))*np.sqrt(2/512)).astype(np.float32),
        np.zeros(HIDDEN, np.float32),
        (rng.standard_normal((HIDDEN,N_CLASSES))*np.sqrt(2/HIDDEN)).astype(np.float32),
        np.zeros(N_CLASSES, np.float32),
    ]
    mu = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    steps = 0
    n = len(x)
    for epoch in range(1, max(EPOCHS)+1):
        order = rng.permutation(n)
        for ix in np.array_split(order,(n+255)//256):
            a, labels = x[ix], y[ix]
            h = np.maximum(a @ params[0] + params[1],0)
            error = h @ params[2] + params[3]
            error = np.exp(error-error.max(axis=1,keepdims=True))
            error /= error.sum(axis=1,keepdims=True)
            error = (error-labels)/len(ix)
            d_h = (error @ params[2].T) * (h > 0)
            gradients = (a.T @ d_h, d_h.sum(axis=0), h.T @ error, error.sum(axis=0))
            steps += 1
            for i,(p,g) in enumerate(zip(params,gradients)):
                mu[i] = .9*mu[i]+.1*g
                v[i] = .999*v[i]+.001*g*g
                step = mu[i]/(1-.9**steps) / (np.sqrt(v[i]/(1-.999**steps))+1e-8)
                p -= rate*(step + (1e-3*p if p.ndim==2 else 0))
        if epoch in EPOCHS:
            yield {"kind":"mlp64", "rate":rate, "epochs":epoch}, tuple(p.copy() for p in params)


def grouped(pred, actual, writers):
    matrix = []
    for w in np.unique(writers):
        idx = writers == w
        matrix.append((pred[...,idx] == actual[idx]).mean(axis=-1))
    return np.stack(matrix, axis=-1)


def fit(archive, out):
    clock = time.perf_counter()
    x, _, y, _ = collect(archive, 0, FIT_CONDITIONS)
    dev, _, truth, writers = collect(archive, 1, FIT_CONDITIONS)
    labels = source.labels_to_array(np.tile(y, len(FIT_CONDITIONS)))
    records, models = {}, {}
    for i, method in enumerate(METHODS):
        flat = x[i].reshape(-1,512)
        choices = {}

        def consider(grid):
            for config, params in grid:
                acc = grouped(score(params, dev[i]), truth, writers).mean(axis=-1)
                key = (float(acc[1:].min()), float(acc.mean()))
                name = config["kind"]
                if name not in choices or key > choices[name][0]:
                    choices[name] = (key, dict(config, validation=acc.tolist()), params)

        consider(ridge(flat,labels))
        for lr in RATE:
            consider(mlp(flat,labels,lr))
        for kind, (_, config, params) in choices.items():
            name = method + "_" + kind
            records[name] = config
            models[name] = np.array(len(params))
            for k, param in enumerate(params):
                models[f"{name}_{k}"] = param
        print(f"fit {method}: {[round(min(c[1]['validation'][1:]), 3) for c in choices.values()]}", flush=True)

    def best(group):
        return max(group, key=lambda n:(min(records[n]["validation"][1:]),
                                        np.mean(records[n]["validation"])))

    names = list(records)
    baseline = best([n for n in names if n.rsplit("_",1)[0] in BASES])
    candidate = best([n for n in names if n.rsplit("_",1)[0] in CANDIDATES
                      and records[n]["kind"]==records[baseline]["kind"]])
    weights = out / "weights.npz"
    np.savez_compressed(weights, **models)
    lock = dict(archive=digest(archive), files=fingerprints(), weights=digest(weights),
                validation=records, baseline=baseline, candidate=candidate,
                fit_elapsed=time.perf_counter()-clock, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/"lock.json").write_text(json.dumps(lock,indent=2)+"\n")
    print(f"locked {baseline} versus {candidate}", flush=True)


def analysis(predictions, truth, writers, lock):
    matrix = {name:grouped(values, truth, writers)
              for name, values in predictions.items()}
    results = {name:dict(zip(TEST_CONDITIONS, val.mean(axis=-1).tolist()))
               for name,val in matrix.items()}
    a,b = (matrix[lock[name]] for name in ("candidate","baseline"))
    n = len(np.unique(writers))
    draws = np.random.default_rng(519308).integers(0,n,size=(12000,n))
    shift_ids = [TEST_CONDITIONS.index(k) for k in ("placement", "coarse_masked", "combined")]

    def estimate(x,y):
        d = x[shift_ids].mean(axis=1).min()-y[shift_ids].mean(axis=1).min()
        boot = x[shift_ids][:,draws].mean(axis=2).min(axis=0)-y[shift_ids][:,draws].mean(axis=2).min(axis=0)
        return [float(d), *np.quantile(boot,[.025,.975]).tolist()]

    d,lo,hi = estimate(a,b)
    clean = a[0]-b[0]
    clean_ci = [float(clean.mean()), *np.quantile(clean[draws].mean(axis=1),[.025,.975]).tolist()]
    increments = {name:[float(diff.mean()), *np.quantile(diff[draws].mean(axis=1),[.025,.975]).tolist()]
                  for name,diff in zip(TEST_CONDITIONS,a-b)}
    return {"accuracy":results, "paired_worst":[d,lo,hi],
            "paired_by_condition":increments,"clean":clean_ci,"primary_pass":bool(d>=.03 and lo>0),
            "clean_preserved":bool(clean_ci[1]>-.02),
            "best_validation_baseline":lock["baseline"],"best_capacity_matched_candidate":lock["candidate"],
            "writer_order":np.unique(writers).tolist()}


def evaluate(archive, model, out):
    start = time.perf_counter()
    lock = json.loads((model/"lock.json").read_text())
    if (lock["archive"],lock["files"],lock["weights"]) != (
            digest(archive), fingerprints(), digest(model/"weights.npz")):
        raise ValueError("Data, code or fitted weights differ from the lock")
    x, counts, y, writers = collect(archive, 2, TEST_CONDITIONS)
    predictions = {}
    with np.load(model/"weights.npz") as fitted:
        for i, method in enumerate(METHODS):
            for kind in ("ridge","mlp64"):
                name = method+"_"+kind
                params = tuple(fitted[f"{name}_{k}"] for k in range(int(fitted[name])))
                predictions[name] = score(params,x[i])
    report = analysis(predictions,y,writers,lock)
    report.update(event_count_medians=dict(zip(TEST_CONDITIONS,np.median(counts,axis=1).tolist())),
                  elapsed=time.perf_counter()-start, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  numpy_version=np.__version__, protocol=fingerprints())
    # Keep the audit ledger compact and exclude raw inputs and writer identifiers.
    with (out/"predictions.jsonl").open("w") as file:
        for i in range(len(y)):
            row = {"writer":int(writers[i]),"label":int(y[i]),"counts":counts[:,i].tolist(),
                   "predictions":{name:values[:,i].tolist() for name,values in predictions.items()}}
            print(json.dumps(row,separators=(",",":")),file=file)
    report["predictions_sha256"] = digest(out/"predictions.jsonl")
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print({k:report[k] for k in ("paired_worst","clean","primary_pass","clean_preserved")},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--locked-fit", type=Path)
    args = parser.parse_args()
    if source.digest(args.archive,"md5") != source.ARCHIVE_MD5:
        raise ValueError("Unknown source archive")
    args.out.mkdir(parents=True,exist_ok=False)
    if args.locked_fit is None:
        fit(args.archive,args.out)
    else:
        evaluate(args.archive,args.locked_fit,args.out)


if __name__ == "__main__":
    main()
