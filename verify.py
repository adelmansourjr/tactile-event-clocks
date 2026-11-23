"""Recalculate reported outcomes and exact paired sensitivity from saved predictions."""
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

from study import digest

root = Path(__file__).resolve().parent
raw_path = root / "results/predictions.json"
raw = json.loads(raw_path.read_text())
report = json.loads((root / "results/summary.json").read_text())
assert report["raw_sha256"] == digest(raw_path)
for name, sha in report["frozen_hashes"].items():
    assert digest(root / name) == sha
rows = raw["label_and_writer_index"]
assert len(rows) == 1925 and set(r[0] for r in rows) == set(range(24, 34))
assert raw["predictions"]["clean"]["rank"]["predicted"] == raw["predictions"]["warp"]["rank"]["predicted"]
comparison = {}
for condition, methods in raw["predictions"].items():
    scores = {}
    for name, method in methods.items():
        assert len(method["predicted"]) == len(rows)
        counts = defaultdict(list)
        for (writer, actual), predicted in zip(rows, method["predicted"]):
            counts[writer].append(actual == predicted)
        acc = [sum(c) / len(c) for _, c in sorted(counts.items())]
        assert acc == report["results"][condition]["accuracy"][name]["writer_accuracy"]
        scores[name] = acc
    delta = [a-b for a, b in zip(scores["rank"], scores[raw["comparator"]])]
    assert delta == report["results"][condition]["writer_deltas"]
    total = sum(delta)
    extreme = sum(abs(sum(sign * d for sign, d in zip(signs, delta))) >= abs(total) - 1e-14
                  for signs in product((-1, 1), repeat=len(delta)))
    comparison[condition] = {
        "writers_improved": sum(x > 0 for x in delta),
        "writers_tied": sum(x == 0 for x in delta),
        "sign_flip_two_sided": extreme / 2 ** len(delta),
    }
print(json.dumps({"rows": len(rows), "prediction_invariance": True, "paired_sensitivity": comparison}, indent=2))
