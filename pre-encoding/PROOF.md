# What can survive deletion and retiming?

The guarantees below concern a fixed event multiset and only the specified comparison transformations. The empirical study allows events to be created or changed by thresholding re-sampled pressure signals and consequently cannot inherit these guarantees.

## Midpoint rank has a sharp deletion bound

For an event at time t, let F(t) be the fraction of events strictly before t plus half the fraction tied at t. This midpoint empirical CDF is the normalized rank used by the encoder. If fraction d of events is deleted, write S and D for the surviving and deleted event sets and use F_S and F_D for their midpoint CDFs. If any events survive,

`F(t) = (1 − d) F_S(t) + d F_D(t).`

Thus `|F_S(t) − F(t)| = d |F_S(t) − F_D(t)| ≤ d`. This holds at each surviving event, including ties. A common strictly increasing remapping of surviving timestamps preserves their midpoint ranks. The coefficient one in this uniform bound cannot be improved: place all deleted events before every survivor, and evaluate at the first survivor of N distinct surviving events. F_D = 1 and F_S = 1/(2N), giving d(1 − 1/(2N)), arbitrarily close to d as N grows.

## Bound on the joint histogram after deletion and retiming

Time is normalized relative to the same fixed acquisition endpoints in both streams. If surviving times move by at most e, all surviving labels (joint spatial bin and polarity) are retained, and their ordering and equality partition are unchanged, then the blended clock `c = (1 − a)t + aF(t)` moves by at most `(1 − a)e + ad` at each survivor. This condition can fail if the timestamp quantizer merges two previously distinct instants.

A linear tent map splitting unit mass between adjacent knots of B equispaced temporal bins is 2(B−1)-Lipschitz in L1. Within one interval the two changing coordinates move at speed B−1 in opposite directions. Across intervals the triangle inequality along the intervening knots gives the same global constant. Appending a fixed spatial bin and polarity retains that constant. Average the distances over the survivor coupling, then decompose the original empirical histogram into the survivor fraction and the deleted fraction. The L1 perturbation is at most

`min(2, 2d + (1 − d) × 2(B−1) × ((1 − a)e + ad))`.

The normalized event histogram sums to one if there is at least one event. Square-root features obey `||sqrt(h) − sqrt(h')||_2² ≤ ||h − h'||_1`, by the elementary coordinate inequality for two nonnegative numbers. If a linear classifier has clean winner i, each score difference versus rival j larger than the bound on square-root distance times `||W_i − W_j||_2` certifies unchanged class under those perturbations. It does not certify correctness. This requires an actual bound on the removed mass and on surviving timestamps.

The blend trades dependence on the two disturbance budgets. The bound is affine in a before clipping, so its best policy is an endpoint when both bounds are fixed. An interior blend has no claimed theoretical minimax advantage. Also, at four bins and d=0.5 the L1 bound already saturates for the rank clock even if timestamps are untouched. For a particular image near the decision boundary, even a small feature motion cannot certify the label. The sharp deletion statement alone does not establish the empirical value of blending.

## Arc length and why upstream thresholding is different

In an L1 pressure space, a discrete linear interpolation segment from x to y has length `||y − x||_1`. Subdividing that segment at ordered fractions `q_0 = 0 ≤ ... ≤ q_k = 1` yields differences `(q_i − q_{i−1})(y − x)`, whose L1 lengths sum exactly to the original length. Thus the cumulative length at a common path location is independent of where one samples along any single straight segment, assuming no skipped corner. Evaluating the progress clock at a nonzero measured change uses half the current segment's length and ranks that segment midpoint. The guarantee refers to that same geometric path point, not arbitrary segment midpoints produced by different sampling.

Now take one taxel changing from 0 to 8, and threshold strictly above 4. Sampling only endpoints creates an event. Sampling at 0, 4, 8 produces no events under the exact same fixed threshold. Even if the pressure distance of the path is unchanged, the measured events are not. Pre-encoding retiming can also skip turns, change adaptation quantiles, or shift pressure between two channels. This is the failure mode motivating the raw-pressure benchmark.
