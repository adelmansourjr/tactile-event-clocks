# Comparing clocks for compact tactile recognition

## Question

At the same fixed feature dimension, does splitting a tactile event stream by cumulative event count improve writer-held-out recognition when the speed of a recording changes? A secondary question tests the failure mode when events are lost or when distinct timestamps become tied. This is an offline decoding study of a public pressure-sensor dataset. It does not measure hardware power, physical changes in movement speed, prosthesis performance, or neural activity.

TouchReal (Xu, Saal and Ward-Cherrier, 2026, doi:10.1088/2634-4386/ae98b0) maps tactile measurements to afferent-like spikes. Prior neuroTac work compares temporal and spatiotemporal codes. STEMNIST (Tripathi and Basu, 2026, arXiv:2601.01658) supplies tactile events from human handwriting. The contribution here is a matched-dimension, unseen-writer stress audit and an exact test of what representation survives a time warp. Rank-based time normalization is an established idea. We claim no new learning algorithm.

## Data and split

The Zenodo v1 archive has MD5 6ca4638b2f95bf34f59873ab62399bd8 and contains 7,700 spike recordings from 34 writer codes across 35 classes. Read only the published event fields and labels, converting one-based taxel IDs to zero-based indices. Ignore pressure files, timestamps in metadata, threshold metadata, writer identifiers as input features, and paths recorded in HDF5. Sort unique writer codes, then permute with NumPy PCG64 seed 2026112311. Split first 18 writers for training, next 6 for validation, last 10 for final. All recordings by one writer remain in the same split. Fit every readout on nominal training inputs. Shifted validation selects settings; no final tuning.

## Compared representations

All methods output 512 features and use the same normalized counts (square root after dividing by total events), centered multiclass ridge train objective, and same validation grid {0.03, 0.3, 3, 30}. Fixed, range-aligned, event-rank and hybrid clocks use 8x8 spatial bins and separate change polarities. Each uses 4 time slices, except the hybrid which concatenates 2 range-clock and 2 rank-clock slices and halves their masses. The high-spatial-resolution comparator uses all 16x16 taxels and two polarities, pooled across time. A full representation with 4 event slices and all 16x16 taxels is an unmatched 2,048-feature reference.

- Fixed time uses the nominal 2-second acquisition window.
- Range alignment divides the observed time span into equal slices.
- Event rank divides cumulative event mass into equal slices. A timestamp tie block is allocated fractionally in proportion to its overlap with each slice. It cannot be broken by within-timestamp input order.
- Hybrid retains both range and rank information.
- Pooled spatial control trades time for space at the same feature dimension.

Each lambda is selected by mean equal-writer validation accuracy across clean, warp, loss and timestamp-merge conditions, breaking ties by lower lambda. The primary comparator is the best of fixed time, range alignment and high-resolution pooled spatial on warped validation (ties in listed order). Hybrid is a diagnostic challenger reported in all conditions.

## Locked conditions and inference

Clean uses the published event stream. Warp maps timestamps using a monotone piecewise-linear function fixing 0 and 2 seconds but compressing or expanding the first half, with breakpoint chosen uniformly in [0.30, 1.70] s by a sample-specific seeded random generator. This changes recorded timing only, never inventing or removing events. Event loss drops a contiguous 0.40 s interior interval of the two-second window and independently drops another 10% of remaining events using fixed per-record seeds. Merge applies the warp then assigns all timestamps to 10 equal clock bins. Randomness is identical across methods per record and independent of labels. The benchmark does not claim these simulated shifts describe real unseen writing speeds.

Primary endpoint is paired equal-writer accuracy difference on the warp condition, rank minus development-selected strongest clock comparator. Pass requires a point gain of at least 2 percentage points and positive lower endpoint of a percentile 95% writer-cluster bootstrap (10,000 draws, fixed seed). Deployment gate additionally requires clean rank accuracy no worse than 2 points in point estimate, with the bootstrap lower bound above -2 points vs that same comparator. Report all conditions, 10 writer-level paired differences, and whole-dataset accuracy as a descriptive sensitivity. Bootstrap precision does not establish population-level robustness from 10 writers. A positive comparison on a shifted recording does not imply competitive accuracy on general recognition. Keep observed failures and the locked policy. Algorithmic cost includes event counting and feature construction. Count feature dimensions and ridge parameter storage separately. CPU timing is machine specific. No inference about energy.

## Exact property and counterexample

Assign each observed timestamp group a cumulative-rank interval of length equal to the number of events in that group. The feature is the sum of polarity and taxel event masses weighted by fractional overlaps of these intervals with equal-rank slices. A strictly increasing remapping preserves the ordered tie partition, so every term and the output agree. With a fixed readout, its prediction is therefore unchanged. Merging previously distinct timestamps can change the partition and the encoding. Loss can also change the ranks of surviving events. Tests must independently show strict-warp equality, equality after within-tie shuffling, and counterexamples for merge and deletion. The no-transform control pooled over time is also invariant to strict retiming, so the empirical comparison tests whether allocating some features to order helps recognition relative to allocating them to spatial detail.

## Replay

Use Python 3.12, NumPy 2.3.5 and h5py 3.15.1 with one CPU thread. Keep the archive outside the public Git repository. Saved results include anonymous writer index, split record counts, class and method predictions, pipeline timings and results hashes. Original HDF5 filenames and writer codes will not be included.
