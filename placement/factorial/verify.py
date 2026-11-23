"""Read-only verification of matched exposure and writer-level results."""
from collections import Counter
import json
from pathlib import Path

import run

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    expected = Counter({0:2,1:2,2:2,3:2})
    for schedule in run.BANKS.values():
        assert len(schedule)==5 and schedule.count(())==1
        assert Counter(factor for view in schedule for factor in view)==expected
    lock = json.loads((ROOT/"validation-lock.json").read_text())
    results=ROOT/"results"
    summary=json.loads((results/"summary.json").read_text())
    ledger=results/"predictions.jsonl"
    assert summary["ledger_sha256"]==run.study.digest(ledger)
    assert summary["hashes"]==lock["hashes"]==run.hashes()
    rows=[json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows)==1925
    assert all(len(r["counts"])==len(run.ASSESS) for r in rows)
    assert all(0 <= count <= run.study.CAP for r in rows for count in r["counts"])
    expected_methods={f"{bank}_{feat}" for bank in run.BANKS for feat in run.FEATURES}
    assert all(set(row["pred"])==expected_methods for row in rows)
    writers=sorted(set(row["writer"] for row in rows))
    assert len(writers)==10
    readout={}
    for method in expected_methods:
        rates = np.empty((len(writers),len(run.ASSESS)))
        for i,w in enumerate(writers):
            group = [row for row in rows if row["writer"]==w]
            for j in range(len(run.ASSESS)):
                assert all(0<=row["pred"][method][j]<run.study.N_CLASSES for row in group)
                rates[i,j] = sum(row["pred"][method][j]==row["truth"] for row in group)/len(group)
        np.testing.assert_allclose(rates.mean(axis=0),list(summary["accuracy"][method].values()))
        readout[method]=rates
    selected=lock["selected"]
    assert selected==summary["selected"]
    assert selected==max(run.FEATURES,key=lambda f:(min(lock["development"]["paired_"+f]["accuracy"][-2:]),
                                                      np.mean(lock["development"]["paired_"+f]["accuracy"])))
    paired=readout[f"paired_{selected}"]
    joint=readout[f"joint_{selected}"]
    draw=np.random.default_rng(773827).integers(0,10,size=(12000,10))
    point=joint.mean(axis=0)[-2:].min()-paired.mean(axis=0)[-2:].min()
    gaps=joint[draw].mean(axis=1)[:,-2:].min(axis=1)-paired[draw].mean(axis=1)[:,-2:].min(axis=1)
    lower, upper = np.quantile(gaps,[.025,.975])
    np.testing.assert_allclose(summary["primary"],[point,lower,upper])
    assert summary["efficacy"]==(point>=.05 and lower>0)
    clean=joint[:,0]-paired[:,0]
    clean_ci=np.quantile(clean[draw].mean(axis=1),[.025,.975])
    np.testing.assert_allclose(summary["clean"],[clean.mean(),*clean_ci])
    assert summary["retention"]==(clean_ci[0]>-.02)
    counts=np.median([r["counts"] for r in rows],axis=0)
    np.testing.assert_equal(counts,list(summary["count_median"].values()))
    print("Verified matched perturbation doses, frozen selection, all 1,925 records and paired gates.")


if __name__=="__main__":
    main()
