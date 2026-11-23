# Tactile recognition at the same event budget

**Moving a complete tactile event stream and losing part of it are distinct problems.** Centering the observed taxel positions raised held-out placement accuracy from **77.06% to 83.57%** against the strongest tested placement control, at identical event identities, rank clock, feature width, and classifier capacity. All ten writers gained. The paired writer interval for this descriptive **+6.51-point** difference was **+3.83 to +9.71**. The prespecified worst-shift comparison, which combines placement and missing observations, did not pass.

## Matched design

The study reused the 18 training, six development, and ten final writers from the [earlier STEMNIST split](../pre-encoding/README.md). Every method received the same derivative event stream for a given record and perturbation. The cap was 121 highest-magnitude eligible events. Each method discarded magnitudes after selection. The same 512-feature budget and four temporal bins were available to all descriptors. Ridge and a 64-hidden-unit MLP were equally available to every spatial representation. Development selected an MLP for both main arms and the same 28-epoch, 0.003-rate configuration. Each of these trained heads stores 35,107 float32 parameters (137.1 KiB). The controls included fixed taxel bins, bilinear interpolation and a bounding-extent-aligned representation that also has in-frame translation invariance. Each could choose elapsed or rank timing. All 16 fitted feature/readout combinations are archived.

Pressure resampling and quantization act before differencing. The synthetic placement shifts derivative events across taxels and clips events that leave the array. The masking controls omit a random stripe and burst interval, with identical retained inputs across methods. The raw contacts are real. Movement, quantization and sensor loss are controlled perturbations of those recordings. The data do not contain new pressure acquisitions or movement at different physical locations. [Frozen protocol and revision history](PROTOCOL.md) · [Selected development policies](validation-lock.json)

| Equal-writer accuracy | Clean | Rate | Coarse | Placement | Masked | Coarse and masked | Combined |
|---|---:|---:|---:|---:|---:|---:|---:|
| Bilinear fixed-frame rank MLP | 87.91% | 88.71% | 85.60% | 34.71% | 45.66% | 45.69% | 17.11% |
| Bounding extent, elapsed MLP (selected control) | 84.97% | 90.46% | 84.14% | 75.29% | 40.40% | 41.26% | **39.51%** |
| Bounding extent, rank MLP | 87.14% | 90.94% | 86.80% | 77.06% | 40.63% | 39.94% | 39.26% |
| Centroid, rank MLP (selected candidate) | **92.97%** | 93.17% | **90.26%** | **83.57%** | 41.89% | 39.97% | 37.94% |
| Median events shared by all arms | 121 | 121 | 121 | 121 | 95 | 70 | 55 |

The paired comparison between the two development-selected methods gave **−1.57 points** for the worst shifted condition (95% interval **−3.37 to +0.37**) and **+8.00 points** on clean data (95% interval **+3.26 to +13.72**). Against the control selected before test for worst-shift performance, the secondary pure-placement contrast was **+8.29 points**, paired interval **+4.11 to +13.43**. Extent-aligned rank was the best tested control on placement alone. Comparing to it holds the rank clock fixed and gives the descriptive contrast in the opening paragraph, with an unadjusted interval conditional on these writers. The identical-clock clean comparison is **+5.83 points** [**+2.46 to +10.11**]. Both controls already align their geometry. The methods use the same 512 input features by count and the same number of model parameters. A larger contrast against the fixed-frame model is visible in the table. The comparison does not establish a global robustness advantage for centroid alignment.

[Read the exact translation identity and a deletion bound](analysis.md). When a translation preserves the set of observed events, the integer numerator of each centered taxel position is unchanged. If clipping alters the set, the pooling frame can move and the resulting input can change. The conditional bound depends on how many events were deleted and the strength of spatial or rank shifts. Training and testing used a single CPU thread for numerical kernels. The reported fit peaked at 604 MB RSS, with no raw data or weights committed.

## Can the training schedule compensate for joint loss?

A [second locked experiment](factorial/README.md) compared paired and all-factor augmentation with identical marginal perturbation draws. All-factor augmentation gave 41.43% versus 42.20% for the paired control in the harder fully shifted condition. Clean accuracy held. This limits the case for changing co-occurrence alone within this replay protocol.

## Reproduce

From the repository root, use Python 3.12, NumPy, h5py and the original downloaded [STEMNIST archive](https://zenodo.org/records/19469535).

```sh
python -m pip install -r requirements.txt
python placement/verify.py --results placement/results
python placement/study.py /path/to/stemnist.zip /tmp/placement-fit
python placement/study.py /path/to/stemnist.zip /tmp/placement-test --locked-fit /tmp/placement-fit
```

The verifier checks the protocol and prediction hashes, exact in-frame synthetic invariance and its boundary, all 1,925 event counts, and the paired writer-level intervals. [Held-out predictions](results/predictions.jsonl) · [Numerical summary](results/summary.json)

[STEMNIST](https://arxiv.org/abs/2601.01658) supplies the tactile recordings. [Gehrig et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.html) formalize grid-based event tensor representations. [Ward-Cherrier and colleagues](https://doi.org/10.1002/aisy.202401078) study invariance to contact force and scan speed for tactile texture classification. The present test asks whether the spatial frame of a small event voxel array absorbs a different perturbation and where that mechanism breaks.
