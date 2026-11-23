"""A matched perturbation dose intervention for tactile augmentation."""
from collections import Counter
import argparse
import io
import json
from pathlib import Path
import resource
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import study

import h5py
import numpy as np

BANKS = {
    "paired": ((), (0,1), (2,3), (0,2), (1,3)),
    "joint": ((), (0,1,2,3), (2,3), (0,), (1,)),
}
FEATURES = ("smooth_rank", "extent_range", "centroid_rank")
ASSESS = ((), (2,), (1,3), (0,1,2,3), (0,1,2,3))
ASSESS_NAMES = ("clean", "placement", "coarse_mask", "joint_stripe", "joint_patch")
SEED = (77452109, 94173821, 126970391)
FILES = tuple(Path(p) for p in (__file__,ROOT/"PROTOCOL.md",ROOT.parent/"study.py",ROOT.parent.parent/"pre-encoding/study.py"))


def hashes():
    return {str(p.relative_to(ROOT.parent.parent)):study.digest(p) for p in FILES}


def replay(pressure, factors, seed, row, occurrence, slot, patch):
    def stream(factor):
        return np.random.default_rng(np.random.SeedSequence([seed,row,factor,occurrence[factor]]))
    if 0 in factors:
        r=stream(0)
        x=r.uniform(.36,.64)
        shift=r.uniform(.10,.26)*r.choice([-1,1])
        pressure=study.source.resample(pressure,np.interp(study.source.CLOCK,[0,x,1],[0,x+shift,1]))
    if 1 in factors:
        r=stream(1)
        step=int(r.integers(2,6))
        keep=np.r_[np.arange(r.integers(0,step),239,step),239]
        grid=np.interp(np.arange(240),keep,np.linspace(0,1,len(keep)))
        q=r.choice([4,8,12,16])
        pressure=np.clip(q*np.round(study.source.resample(pressure[keep],grid)/q),0,255)
    diff=np.diff(pressure,axis=0)
    t,k=np.nonzero(np.abs(diff)>=4)
    signed=diff[t,k]
    x,y=k%16,k//16
    if 2 in factors:
        r=stream(2)
        displace=int(r.integers(1,6))*r.choice([-1,1])
        if r.integers(2):
            x=x+displace
        else:
            y=y+displace
    keep=(0<=x)&(x<16)&(0<=y)&(y<16)
    if 3 in factors:
        r=stream(3)
        axis=x if r.integers(2) else y
        width=r.integers(2,6)
        start=r.integers(0,17-width)
        spatial=(start<=axis)&(axis<start+width)
        if patch:
            alt=y if axis is x else x
            first=r.integers(0,8)
            spatial=spatial & (first<=alt)&(alt<first+9)
            second=(start+8)%16
            spatial|=(axis==second)&(alt>=max(0,first-3))
        duration=r.integers(22,58)
        a=r.integers(30,205-duration)
        keep&=~(spatial|((a<=t)&(t<a+duration)))
    choices=np.flatnonzero(keep)
    tie=np.random.default_rng(np.random.SeedSequence([seed,row,85,slot])).random(len(choices))
    choose=choices[np.lexsort((tie,-np.abs(signed[choices])))[:study.CAP]]
    return t[choose],x[choose],y[choose],signed[choose]>0


def collect(archive_path, split, schedule):
    used=tuple(study.METHODS.index(name) for name in FEATURES)
    with ZipFile(archive_path) as archive:
        records=study.source.manifest(archive)[split]
        matrix=np.empty((len(used),len(schedule),len(records),512),np.float32)
        counts=np.empty((len(schedule),len(records)),np.uint8)
        for i,(record,writer,label,member) in enumerate(records):
            with h5py.File(io.BytesIO(archive.read(member)),"r") as handle:
                pressure=handle["pressure_data"][:].reshape(240,256).astype(np.float32)
            occurrence=Counter()
            for j,factors in enumerate(schedule):
                if split==0:
                    r=occurrence
                else:
                    r=Counter({f:j for f in factors})
                args=replay(pressure,factors,SEED[split],record,r,j,split!=0 and j==4)
                counts[j,i]=len(args[0])
                matrix[:,j,i]=study.features(*args)[list(used)]
                occurrence.update(factors)
            if i and i%1000==0:
                print(f"encoded split {split}: {i}/{len(records)}",flush=True)
    return matrix,counts,np.array([r[2] for r in records]),np.array([r[1] for r in records])


def fit(archive,out):
    assert all(Counter(x for v in s for x in v)==Counter({i:2 for i in range(4)})
               and len(s)==5 and s.count(())==1 for s in BANKS.values())
    start=time.perf_counter()
    dev,_,dy,dw=collect(archive,1,ASSESS)
    models={}
    scores={}
    for bank,schedule in BANKS.items():
        train,_,y,_=collect(archive,0,schedule)
        y=study.source.labels_to_array(np.tile(y,len(schedule)))
        for i,method in enumerate(FEATURES):
            opt=None
            x=train[i].reshape(-1,512)
            for lr in study.RATE:
                for config,params in study.mlp(x,y,lr):
                    acc=study.grouped(study.score(params,dev[i]),dy,dw).mean(axis=1)
                    key=(min(acc[-2:]),np.mean(acc))
                    if opt is None or key>opt[0]:
                        opt=(key,params,config,acc.tolist())
            _,params,config,acc=opt
            name=bank+"_"+method
            scores[name]=dict(config,accuracy=acc)
            for k,v in enumerate(params):
                models[f"{name}_{k}"]=v
            print(name,acc,flush=True)
        del x,train
    model=out/"weights.npz"
    np.savez_compressed(model,**models)
    choice=max(FEATURES,key=lambda f:(min(scores["paired_"+f]["accuracy"][-2:]),
                                    np.mean(scores["paired_"+f]["accuracy"])))
    print("locked",choice,flush=True)
    (out/"lock.json").write_text(json.dumps(dict(hashes=hashes(),archive=study.digest(archive),
        weight_sha256=study.digest(model),selected=choice,development=scores,
        elapsed=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        indent=2)+"\n")


def summarize(predictions, y, writers, lock):
    matrix={k:study.grouped(v,y,writers) for k,v in predictions.items()}
    a,b=(matrix[x+"_"+lock["selected"]] for x in ("joint","paired"))
    rng=np.random.default_rng(773827)
    draw=rng.integers(0,10,size=(12000,10))
    primary=a[3:].mean(axis=1).min()-b[3:].mean(axis=1).min()
    pair=a[3:][:,draw].mean(axis=2).min(axis=0)-b[3:][:,draw].mean(axis=2).min(axis=0)
    interval=np.quantile(pair,[.025,.975]).tolist()
    clean=a[0]-b[0]
    clean_interval=np.quantile(clean[draw].mean(axis=1),[.025,.975]).tolist()
    per_condition={name:[float(diff.mean()),*np.quantile(diff[draw].mean(axis=1),[.025,.975]).tolist()]
                   for name,diff in zip(ASSESS_NAMES,a-b)}
    return dict(accuracy={k:dict(zip(ASSESS_NAMES,v.mean(axis=1).tolist())) for k,v in matrix.items()},
                primary=[float(primary),*interval],clean=[float(clean.mean()),*clean_interval],
                paired_by_condition=per_condition,efficacy=bool(primary>=.05 and interval[0]>0),
                retention=bool(clean_interval[0]>-.02))


def evaluate(archive, out, model):
    start=time.perf_counter()
    lock=json.loads((model/"lock.json").read_text())
    assert lock["hashes"]==hashes() and lock["archive"]==study.digest(archive)
    assert lock["weight_sha256"]==study.digest(model/"weights.npz")
    x,counts,truth,writers=collect(archive,2,ASSESS)
    predictions={}
    with np.load(model/"weights.npz") as weights:
        for bank in BANKS:
            for i,method in enumerate(FEATURES):
                name=bank+"_"+method
                params=[weights[f"{name}_{k}"] for k in range(4)]
                predictions[name]=study.score(params,x[i])
    ledger=out/"predictions.jsonl"
    with ledger.open("w") as inp:
        for i,(y,w) in enumerate(zip(truth,writers)):
            print(json.dumps(dict(truth=int(y),writer=int(w),counts=counts[:,i].tolist(),
                                  pred={key:p[:,i].tolist() for key,p in predictions.items()}),
                             separators=(",",":")),file=inp)
    result=summarize(predictions,truth,writers,lock)
    result.update(hashes=hashes(),selected=lock["selected"],ledger_sha256=study.digest(ledger),
                  count_median=dict(zip(ASSESS_NAMES,np.median(counts,axis=1).tolist())),
                  elapsed=time.perf_counter()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result["selected"],result["primary"],result["clean"],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--locked-fit",type=Path)
    a=parser.parse_args()
    assert study.source.digest(a.archive,"md5")==study.source.ARCHIVE_MD5
    a.out.mkdir(parents=True,exist_ok=False)
    if a.locked_fit is None:
        fit(a.archive,a.out)
    else:
        evaluate(a.archive,a.out,a.locked_fit)


if __name__=="__main__":
    main()
