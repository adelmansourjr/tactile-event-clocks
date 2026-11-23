# Tactile event clocks

Can cumulative event order preserve tactile recognition when a movement is replayed at a different rate? This study compares fixed-size event representations for recognizing characters drawn on a tactile pressure array. The test holds out whole writers.

The rank clock preserved every prediction under a strictly monotone remapping of time. It improved accuracy by **15.0 percentage points** over the strongest validation-selected timing baseline under that shift, but lost **9.69 points** when events were missing. It also missed the predeclared clean-data noninferiority gate. This suggests retaining an alternative representation when the sensor can drop bursts of events.

**Raw-pressure follow-up.** Resampling before generating spikes exposes an acquisition failure hidden by post-spike retiming. At coarse acquisition, a compact magnitude-based baseline recovered 67.23% accuracy from 121 median events versus 11.86% from eight median events under the percentile rule. At exactly matched counts, send-on-delta missed its worst-shift improvement gate. [Two frozen experiments and reproducible results](pre-encoding/README.md)

**Matched-event placement follow-up.** At identical event identities, feature width and classifier capacity, centroid alignment raised placement accuracy from 77.06% to 83.57% against the strongest tested bounding-extent model using the same rank clock. The stronger combined perturbation remained difficult, including after matched-dose joint augmentation. [Placement results, exact analysis and training-schedule test](placement/README.md)

## Real recordings and fixed comparisons

The [STEMNIST dataset](https://doi.org/10.5281/zenodo.19469535) contains 7,700 recorded tactile traces of 35 characters by 34 participants. Train, validation and test partitions used 18, 6 and 10 disjoint writers, with 4,200, 1,575 and 1,925 traces. A fixed linear ridge classifier was independently trained for each representation, with the same regularization grid and validation procedure. The published spike stream already reflects the authors' offline adaptive threshold. The project starts from that stream, so the experiments measure representation and readout.

Elapsed-time, range-aligned, rank-clock, hybrid, and spatial-only representations all used 512 features. The spatial-only control used all 16×16 taxels. The time-aware 512-feature methods used an 8×8 spatial grid with polarity. A 2,048-feature rank model is a size reference. Accuracy is averaged across writers, with each writer contributing equally.

| Condition | Fixed clock | Range-aligned clock | Spatial only | Rank clock | Hybrid | Rank clock, 2,048 features |
|---|---:|---:|---:|---:|---:|---:|
| Original data | 53.51% | 70.09% | 50.51% | 69.54% | 62.11% | 66.69% |
| Timing warp | 40.83% | 54.54% | 50.51% | **69.54%** | 57.06% | 66.69% |
| Burst and random event loss | 39.34% | **42.34%** | 33.63% | 32.66% | 37.29% | 31.77% |
| Warp plus timestamp merging | 41.06% | 48.57% | 50.51% | **63.40%** | 55.37% | 61.63% |

The validation-selected range clock was the primary comparator. The timing-warp difference was **+15.00 points** with paired writer-bootstrap interval **+12.00 to +18.51**. All 10 writers improved in this perturbation. All 1,925 rank-clock predictions matched the corresponding original trace predictions. The clean difference was **−0.54 points** with interval **−2.23 to +1.31**, missing the predeclared −2-point clean margin. Under event loss, all 10 writers did worse with the rank clock. The inferred range of effects applies to this test and its constructed shifts. Real changes in writing speed can change the pressure pattern and delete or add events.

For the 512-feature rank clock, feature construction took 0.062–0.083 seconds per 1,925-trace test condition on one CPU. See the separate timings in the raw artifact. The 512-feature readout takes 71,820 bytes as float32 weights plus bias. That does not count source-buffer memory, wall-clock sensor latency, telemetry, or hardware energy. The complete sequential run peaked at about 542 MB process RSS. Fine-grained timings are hardware-dependent.

## Why the invariance holds

Sort events by timestamp, allowing equal timestamps to form one group. Place the group in cumulative rank space with length equal to its number of events. For each rank slice, allocate the same fractional overlap to every event in the group. Sum these masses by taxel and polarity. Any strictly increasing mapping of timestamps preserves the order and ties, so it preserves each overlap and the feature vector. This yields exactly the same prediction for a fixed readout. It also removes arbitrariness from the ordering of events measured at the same timestamp.

The proof is conditional on preservation of the observed ordered tie groups. Merging timestamps can change which slice a taxel contributes to. Removing an early event can change ranks of later events. Unit tests include explicit examples and an independent enumeration of the fractional weights.

## Paper connection

[TouchReal](https://doi.org/10.1088/2634-4386/ae98b0), from Xu, Saal and Ward-Cherrier, constructs biomimetic spike trains from tactile sensors. [Brayshaw, Ward-Cherrier and Pearson](https://research-information.bris.ac.uk/files/335151637/Temporal_and_Spatio-temporal_domains_for_0ANeuromorphic_Tactile_Texture_Classification_IEEE.pdf) compare temporal and spatiotemporal codes for neuromorphic touch. [Stone, Lepora and Barton](https://arxiv.org/abs/1909.05808) use online generative learning for tactile contour following. This benchmark asks which representation remains useful under specified changes to timing and event transmission. It uses handwriting and does not reproduce those sensors or control experiments. The adjacent research question is whether a confidence signal can tell a controller when loss of events makes an otherwise stable clock unreliable.

## Reproduce

Download the original 35 MB archive from [Zenodo](https://zenodo.org/records/19469535) and supply its local path. The archive and source identifiers are excluded from Git. The archive MD5 matches the publisher's value, and the SHA-256 of the downloaded copy is in the results. A fresh run writes a new directory and refuses to overwrite the frozen output.

```sh
python -m pip install -r requirements.txt
python -m unittest -v
python verify.py
python study.py /path/to/stemnist.zip reproduction
```

[Protocol](PROTOCOL.md) · [Pre-run hashes](PROTOCOL_SHA256SUMS) · [Full numerical report](results/summary.json) · [Raw predictions](results/predictions.json) · [Independent count and paired checks](results/independent_counts.json).
