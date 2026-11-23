# When a tactile clock cannot repair event starvation

Replaying the raw pressure before event generation changed the number of emitted spikes on **98.4%** of rate-perturbed recordings. Under a coarse acquisition stress, a percentile derivative encoder retained a median of eight spikes per record and scored **11.86%**. A budgeted magnitude baseline recovered **67.23%** using a median of 121 spikes, the count of a typical clean recording. At exactly the same count as that baseline on each recording, send-on-delta reached **66.34%** under coarse acquisition and scored **3.57 points lower** in the worst shifted condition.

These two frozen studies use 7,700 real STEMNIST pressure measurements from 34 writers. The 18 train, 6 validation and 10 test writers are disjoint within each run. Both studies reuse this writer split, including its 1,925 test recordings. The measured pressures are real and the acquisition stresses are simulated. Each experiment uses separate corruption seeds.

## 1. Move the stress upstream of the clock

The [original benchmark](../README.md) preserves event identities under a monotone clock transformation. This experiment starts from the 240×16×16 pressure traces and generates new pulses after stress is applied. Every classifier receives equally augmented training data and uses 512 features. Select the event-range baseline and a fixed-time/rank blend with 75% rank weight on validation performance only.

| Writer-macro accuracy | Clean | Rate | Coarse | Threshold | Dropout | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Event range | 71.23% | 67.94% | 12.00% | 59.89% | 37.89% | 4.11% |
| Event rank | 71.11% | 71.57% | 9.23% | 56.83% | 31.43% | 4.57% |
| Fixed/rank blend | 72.54% | 67.20% | 11.49% | 59.91% | 36.31% | 5.46% |
| Pressure-progress clock | 70.17% | 70.86% | 17.17% | 60.66% | 32.94% | 6.34% |
| Direct pressure | 51.46% | 25.80% | 51.49% | 51.46% | 30.51% | 17.77% |

The blend missed its preset two-point worst-shift improvement gate. Its gain over the selected control was **+1.34 points** with paired 95% writer-bootstrap interval **+0.51 to +2.40**. It met the clean preservation margin of −2 points (interval **−0.80 to +4.00**). The direct pressure comparator used the same observed pressure and feature dimension. It bypassed the event bottleneck and used more input information than an event-only classifier. Its coarse-condition result showed that substantial class information remained when the percentile detector retained few events.

The [technical derivation](PROOF.md) bounds midpoint rank shift by the deleted event fraction, with a sharp coefficient. It also gives an upper bound on the change in the joint soft histogram, conditional on preserving order, ties, taxel labels, polarity and a common time scale. New events, merged timestamps, and pre-encoding changes fall outside that guarantee.

[Protocol](PROTOCOL.md) · [Validation fit lock](validation-lock.json) · [Full report](results/summary.json) · [Predictions and counts](results/predictions.json)

## 2. Test the acquisition bottleneck at matched event counts

The first experiment motivated a separate test of two low-cost encoders. In the second study we froze the protocol and fresh replay seeds before scoring. For each observed trace, the budgeted derivative encoder keeps the strongest K one-step changes, and the established send-on-delta encoder keeps K evenly spaced threshold crossings. K is the least of 121, the number of absolute one-step changes at least 4, and the number of threshold crossings. The protocol exactly matches retained inputs to the classifier for those two methods. It uses offline selection and access to both streams to set K. Each method had the same four training views per recording, the same selection grid for its clock and ridge coefficient, and a 512-feature linear readout.

| Writer-macro accuracy | Clean | Rate | Coarse | Threshold | Dropout | Composite |
|---|---:|---:|---:|---:|---:|---:|
| Percentile derivative | 71.14% | 68.26% | 11.86% | 60.74% | 38.71% | 4.69% |
| Budgeted strongest derivative | 67.03% | 67.60% | 67.23% | 66.60% | 37.29% | 35.23% |
| Send-on-delta | 72.17% | 62.83% | 66.34% | 72.17% | 37.74% | 31.66% |
| Median percentile event count | 121 | 121 | 8 | 51 | 65 | 1 |
| Median matched count | 121 | 121 | 121 | 121 | 86 | 74 |

The **+55.37-point** coarse contrast between strongest derivative and the percentile reference had a descriptive paired 95% writer-bootstrap interval of **+49.83 to +60.51**. All ten writers improved. This contrast includes the increase from 8 to 121 median retained events. With K set to the original percentile count on a trace, top-K derivative uses the same event locations as that percentile rule at a common clock. This control shows how much classification evidence is available at a clean-sized event budget after coarse acquisition. It gives no evidence that magnitude selection retains more class information than the original rule at its own count.

At the exact matched count, send-on-delta changed coarse accuracy by **−0.89 points** with a descriptive interval of **−2.54 to +0.57**. The predeclared primary contrast was its worst shifted accuracy minus the strongest derivative encoder's worst shifted accuracy, **−3.57 points [−4.91, −2.03]**. Composite was the worst condition for both. The contrast stayed the same when the encoder-specific threshold condition was left out. Clean accuracy improved by **+5.14 points [ +1.77, +8.29 ]**, a trade-off against the poorer composite score. Intervals resample paired writers and recompute the minima in each bootstrap draw. The ten observed subjects and fixed families limit the interpretation. See individual and rank-clock results in the full table.

[Protocol](level-crossing/PROTOCOL.md) · [Validation fit lock](level-crossing/validation-lock.json) · [Full report](level-crossing/results/summary.json) · [Predictions and counts](level-crossing/results/predictions.json)

## Link to the literature

- Brayshaw, Ward-Cherrier and Pearson vary sensor velocity on recorded textures and compare spatiotemporal coding in neuroTac. Our replay can diagnose where to examine sensor loss in a repeat of that design. Acquiring the same trajectory at separately commanded speeds would distinguish changes in contact mechanics from changes in sampling. [Neuromorphic tactile texture classification](https://doi.org/10.3390/electronics13112159)
- Xu, Saal and Ward-Cherrier use tactile signals to construct biomimetic spike trains. Our send-on-delta control tests an established reset-relative coding mechanism on a distinct task and dataset and limits what can be inferred from time-order invariance alone. [TouchReal](https://doi.org/10.1088/2634-4386/ae98b0) · [Reset-relative event sensing](https://tilde.ini.uzh.ch/~tobi/wiki/lib/exe/fetch.php?media=lichtsteiner_dvs_jssc08.pdf)

## Reproduce

Download the original archive from [STEMNIST](https://zenodo.org/records/19469535). The methodology for the archive uses its raw pressures and approximates the event rule described in the [STEMNIST paper](https://arxiv.org/html/2601.01658v1). Install the parent project's requirements and have a C compiler available for the second study. Run from this directory with a fresh temporary output directory. Test predictions and weights can vary on different BLAS builds near rounding ties.

```sh
python -m pip install -r ../requirements.txt
python test_study.py
python verify_results.py

data=/path/to/stemnist.zip
out=$(mktemp -d)
python study.py "$data" "$out/clock-validation"
python study.py "$data" "$out/clock-test" --frozen-fit "$out/clock-validation"
python level-crossing/study_crossing.py "$data" "$out/encoder-validation"
python level-crossing/study_crossing.py "$data" "$out/encoder-test" --frozen-fit "$out/encoder-validation"
```

The runs use one BLAS thread. On the development machine the clock study peaked at 664 MB RSS for fitting and 370 MB for evaluation. The matched-budget study peaked at 474 MB for fitting and 275 MB for evaluation. The test runs took 10.1 seconds for joint clock feature construction and 12.3 seconds for complete matched-budget evaluation. The reports include source hashes and counts. These are offline CPU measurements.
