# Translation and clipping bounds for centered voxels

Let a trace contain signed events with integer taxel positions `(x_i, y_i)`. Let `N` be its event count and let `s = 7/18`. In each spatial coordinate, the pooling location is `3.5 + s (x_i - mean(x))`, clipped to the interval from 0 to 7. The temporal coordinate is either elapsed event range or midpoint rank. Each event contributes a product of three one-dimensional two-neighbor interpolation weights in its own polarity channel. Add the contributions, divide by the number of events, and take square roots entry by entry.

## In-frame translation

Translate every taxel by the same integer offset while preserving the event identities, polarity and temporal coordinate. The centered numerator `N x_i - sum_j x_j` is identical before and after translation. This identity holds in the implementation's integer arithmetic, so every spatial weight and every feature is unchanged. The result does not depend on classifier training. Clipping at the *sensor boundary* can discard events before feature construction and breaks the premise. Clipping of unchanged coordinates during feature construction preserves the identity.

## Stability under deletion, with rank timing

Suppose boundary clipping or loss deletes a fraction `d < 1` of the events without adding others. Let `D_x, D_y` denote the original coordinate ranges. For every surviving event, the centroid shift in each axis is at most the deleted fraction times that axis's range. For example, the original mean is the convex combination of the survivor and deleted means, with deleted weight `d`. The midpoint rank numerator satisfies the same mixture identity: the full trace's midpoint rank is the survivor midpoint rank times `1-d` plus the deleted events' fractional rank relative to the same timestamp times `d`. This deleted fractional rank lies between zero and one. Thus the surviving event's time-grid coordinate shifts by at most `3d`.

For an individual axis, the two interpolation weights have L1 difference at most twice the grid-coordinate displacement. This follows by splitting a displacement at cell boundaries and summing the difference within each cell. The bound extends additively to the product of the three normalized weight vectors by inserting and subtracting one intermediate product for each differing factor. Clipping from real to grid coordinates is one-Lipschitz. Averaging each surviving event's bound and accounting for deletion-induced reweighting gives

```text
L1(hit_histogram, surviving_histogram) <= min(2, 2d [1 + 3 + s(D_x + D_y)]).
```

The square-root feature distance squared is no greater than this L1 bound, since `(sqrt(a)-sqrt(b))^2 <= abs(a-b)` for nonnegative `a,b`. A fixed linear decoder retains its winner if its logged margin over every rival exceeds that feature-distance bound times the Euclidean norm of the corresponding difference in decoder weights. Centroid displacement alone can reach its bound by removing events at one end of a segment while retaining events at the other. The displayed stability bound can be conservative. It assumes no newly generated observable events. In particular, a change in the top-magnitude selection after clipping may promote events that were originally below the cap, so the expression does not certify the composite replay measurements. Elapsed-range rescaling can jump when a single earliest or latest event is deleted and lacks this rank bound without stronger assumptions.
