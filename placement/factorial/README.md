# Joint augmentation at matched perturbation doses

Packing all four acquisition perturbations into a single training view did not improve worst-shift accuracy over an equally sized schedule that exposed the perturbations in pairs. Across two fully combined held-out replay conditions, the paired-training model reached **42.20%** on its harder condition and the joint-training model reached **41.43%**. The paired difference was **−0.77 points [−2.17,+0.86]**. The upper endpoint falls below the intended five-point gain. Both models used the same selected centroid-rank descriptor and 64-hidden-unit readout.

## Matched exposure

The paired schedule applied `clean, AB, CD, AC, BD`; the joint schedule applied `clean, ABCD, CD, A, B`. A is a motion warp, B is coarse sampling and pressure quantization, C is an offset in taxel placement, and D is masking a set of taxels and an interval. Each training schedule uses five views of every labelled trace, includes one clean view, and exposes each factor twice with identical factor parameter draws at both occurrences. This controls the marginal perturbation doses. The joint co-occurrence and resulting training event distribution differ.

The comparison uses new replay seeds and the same ten held-out STEMNIST writers as the earlier study. Development-only selection fixes the descriptor, learning rate and training length. Each held-out comparison receives exactly the same selected events. The localized patch includes an additional damaged line and was absent from the training generator. This isolates a controlled shift in the coverage pattern. Other perturbation ranges overlap training. [Frozen protocol](PROTOCOL.md) · [Development policy and source lock](validation-lock.json)

| Equal-writer accuracy | Clean | Placement | Coarse and masked | Joint stripe | Joint patch |
|---|---:|---:|---:|---:|---:|
| Paired training | 90.74% | 84.97% | 44.29% | 42.20% | 48.00% |
| Joint training | 90.89% | 83.89% | 44.37% | 41.43% | 46.26% |
| Shared median events | 121 | 121 | 82 | 69 | 74 |

The clean contrast was **+0.14 points [−0.40,+0.71]**, meeting the clean-preservation gate. The efficacy gate failed. The patch-specific contrast was **−1.74 points [−3.74,+0.29]**. These are conditional writer intervals from 12,000 paired resamples. The data do not identify how well any other grouping or physical sensor behaves. The results support using matched-input challenge tests to distinguish coordinate alignment from loss tolerance. Training peaked at 267 MB process RSS, including fitting and development evaluation.

## Reproduce and audit

Run from the repository root with the dependencies and original STEMNIST archive specified in the [placement report](../README.md).

```sh
python placement/factorial/verify.py
python placement/factorial/run.py /path/to/stemnist.zip /tmp/factorial-fit
python placement/factorial/run.py /path/to/stemnist.zip /tmp/factorial-test --locked-fit /tmp/factorial-fit
```

The audit independently checks the schedules' matched marginal counts, source and prediction hashes, the rule for selecting the development control, shared event counts, and the two writer-level decision gates. [Predictions](results/predictions.jsonl) · [All six classifier results](results/summary.json)
