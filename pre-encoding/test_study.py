import numpy as np

import study


def midrank(t):
    return np.array([((t < x).sum() + .5 * (t == x).sum()) / len(t) for x in t])


def test_polyline_and_threshold():
    p = np.array([[0, 0], [8, 0], [8, -4]], dtype=float)
    q = np.array([[0, 0], [4, 0], [8, 0], [8, -2], [8, -4]], dtype=float)
    assert np.abs(np.diff(p, axis=0)).sum() == np.abs(np.diff(q, axis=0)).sum() == 12
    assert np.count_nonzero(np.abs(np.diff(p, axis=0)) > 4) == 1
    assert not np.any(np.abs(np.diff(q, axis=0)) > 4)


def test_bounded_feature_shift():
    rng = np.random.default_rng(313133)
    for _ in range(400):
        n = int(rng.integers(3, 160))
        t = np.sort(rng.integers(101, size=n)) / 100.
        labels = rng.integers(9, size=n)
        surviving = (rng.random(n) > .35)
        surviving[0] = True
        warp = np.sqrt(t[surviving])
        d = 1. - surviving.mean()
        e = np.abs(warp - t[surviving]).max()
        old_rank, new_rank = midrank(t), midrank(warp)
        assert np.abs(old_rank[surviving] - new_rank).max() <= d + 1.e-12
        for a in (0., .25, .5, .75, 1.):
            old = study.tent((1-a)*t + a*old_rank, labels, 9, 4) / n
            new = study.tent((1-a)*warp + a*new_rank, labels[surviving], 9, 4) / surviving.sum()
            bound = min(2., 2*d + (1-d) * 6 * ((1-a)*e + a*d))
            assert np.abs(old - new).sum() <= bound + 1.e-12
            assert np.square(np.sqrt(old) - np.sqrt(new)).sum() <= bound + 1.e-12
    separated = np.arange(110)
    assert np.isclose(
        np.abs(midrank(separated)[10:] - midrank(separated[10:])).max(),
        10/110 * (1 - 1/200),
    )


def test_resample_and_feature_budget():
    rng = np.random.default_rng(123)
    recording = rng.integers(0, 255, size=(240, 256)).astype(np.float32)
    assert np.allclose(recording, study.resample(recording, study.CLOCK))
    for condition in (*study.DEVELOPMENT, *study.FINAL):
        altered, quantile = study.corruption(recording, condition, 987)
        filtered, count = study.encoding(altered, quantile)
        assert count > 0
        assert set(filtered) == set(study.METHODS)
        for vector in filtered.values():
            assert vector.shape == (512,)
            assert np.isclose(np.linalg.norm(vector), 1)


if __name__ == "__main__":
    test_polyline_and_threshold()
    test_bounded_feature_shift()
    test_resample_and_feature_budget()
    print("400 tied-rank deletion trials, 2,000 soft-clock bounds, pressure path counterexample, and dimensions: passed.")
