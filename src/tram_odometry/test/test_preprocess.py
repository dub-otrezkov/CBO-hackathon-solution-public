import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tram_odometry.preprocess import KMH_TO_MS, WheelPreprocessor, WheelSample

KMH = 3.6


def feed(pre, bogie, t0, n, v_ms, dt=0.1):
    out = []
    for i in range(n):
        out.append(pre.push(bogie, t0 + i * dt, v_ms * KMH))
    return out


def test_kmh_to_ms_exactly_once():
    pre = WheelPreprocessor()
    s = pre.push("front", 100.0, 36.0)
    assert isinstance(s, WheelSample)
    assert s.v == pytest.approx(10.0, rel=1e-12)
    assert s.raw == 36.0
    assert KMH_TO_MS == pytest.approx(1 / 3.6)
    assert s.bogie == "front" and s.stamp == 100.0
    assert s.flags == frozenset({"partner_stale"})


def test_nonfinite_and_bad_types_rejected_without_state_change():
    pre = WheelPreprocessor()
    assert pre.push("rear", 1.0, 10.0) is not None
    for bad in (math.nan, math.inf, -math.inf, None, "abc", object(), True):
        assert pre.push("rear", 1.1, bad) is None
        assert pre.last_reject == "nonfinite"
    for bad_t in (math.nan, math.inf, None, "x"):
        assert pre.push("rear", bad_t, 10.0) is None
        assert pre.last_reject == "nonfinite"
    s = pre.push("rear", 1.1, 10.0)
    assert s is not None and "gap" not in s.flags and s.dt == pytest.approx(0.1)
    assert pre.counts["nonfinite"] == 11


def test_unknown_bogie_and_range():
    pre = WheelPreprocessor()
    assert pre.push("middle", 1.0, 10.0) is None and pre.last_reject == "bogie"
    assert pre.push(None, 1.0, 10.0) is None and pre.last_reject == "bogie"
    assert pre.push("front", 1.0, 200.0) is None and pre.last_reject == "range"
    s = pre.push("front", 1.0, -0.3)
    assert s is not None and s.v == pytest.approx(-0.3 / 3.6)


def test_backstep_and_duplicate_dropped():
    pre = WheelPreprocessor()
    assert pre.push("front", 1.0, 10.0) is not None
    assert pre.push("front", 1.1, 10.0) is not None
    assert pre.push("front", 1.1, 10.0) is None and pre.last_reject == "duplicate"
    assert pre.push("front", 1.1, 11.0) is None and pre.last_reject == "duplicate"
    assert pre.push("front", 0.5, 10.0) is None and pre.last_reject == "backstep"
    s = pre.push("front", 1.2, 10.0)
    assert s is not None and s.dt == pytest.approx(0.1)
    assert pre.latest("front") is s
    assert pre.push("rear", 1.2, 10.0) is not None


def test_interleaved_late_packets_are_dropped_fresh_kept():
    pre = WheelPreprocessor()
    feed(pre, "front", 0.0, 10, 5.0)
    accepted, dropped = [], 0
    for i in range(10):
        fresh = 1.0 + 0.1 * i
        late = fresh - 1.05
        for t in (fresh, late):
            s = pre.push("front", t, 5.0 * KMH)
            if s is None:
                dropped += 1
                assert pre.last_reject == "backstep"
            else:
                accepted.append(t)
    assert accepted == [pytest.approx(1.0 + 0.1 * i) for i in range(10)]
    assert dropped == 10
    assert pre.counts["epoch"] == 0


def test_persistent_backstep_resyncs_epoch_for_both_bogies():
    pre = WheelPreprocessor(resync_n=5)
    feed(pre, "front", 1000.0, 3, 5.0)
    feed(pre, "rear", 1000.0, 3, 5.0)
    res = feed(pre, "front", 10.0, 5, 5.0)
    assert res[:4] == [None] * 4
    assert res[4] is not None and "epoch" in res[4].flags
    s = pre.push("rear", 10.45, 5.0 * KMH)
    assert s is not None and "stale" not in s.flags
    assert "epoch" in s.flags
    s = pre.push("rear", 10.55, 5.0 * KMH)
    assert s is not None and "epoch" not in s.flags


def test_dropout_of_one_bogie_flags_partner_stale_and_gap():
    pre = WheelPreprocessor()
    for i in range(50):
        t = 0.1 * i
        pre.push("front", t, 8.0 * KMH)
        pre.push("rear", t, 8.0 * KMH)
    last_front = pre.latest("front").stamp
    flags = []
    for i in range(1, 199):
        s = pre.push("rear", last_front + 0.1 * i, 8.0 * KMH)
        flags.append(s.flags)
    assert all(not f for f in flags[:4])
    assert all(f == frozenset({"partner_stale"}) for f in flags[6:])
    assert pre.age("front", last_front + 19.8) == pytest.approx(19.8)
    assert pre.latest("front").stamp == last_front
    back = pre.push("front", last_front + 19.8, 8.0 * KMH)
    assert back is not None and "gap" in back.flags and back.usable
    assert back.dt == pytest.approx(19.8)


def test_lagging_bogie_is_stale():
    pre = WheelPreprocessor()
    pre.push("front", 11.2, 0.0)
    s = pre.push("rear", 10.0, 0.0)
    assert s is not None and "stale" in s.flags and not s.usable
    s = pre.push("rear", 11.1, 0.0)
    assert s is not None and "stale" not in s.flags and s.usable


def test_outlier_spike_and_level_change():
    pre = WheelPreprocessor()
    feed(pre, "front", 0.0, 10, 10.0)
    spike = pre.push("front", 1.0, 0.0)
    assert spike is not None and "outlier" in spike.flags and not spike.usable and spike.v == 0.0
    nxt = pre.push("front", 1.1, 10.0 * KMH)
    assert "outlier" not in nxt.flags
    res = feed(pre, "front", 1.2, 8, 3.0)
    n_flagged = sum("outlier" in s.flags for s in res)
    assert 1 <= n_flagged <= 3
    assert all("outlier" not in s.flags for s in res[3:])


def test_no_outlier_on_emergency_braking_or_normal_noise():
    pre = WheelPreprocessor()
    v = 12.0
    for i in range(40):
        s = pre.push("rear", 0.1 * i, max(0.0, v) * KMH)
        v -= 0.5
        assert "outlier" not in s.flags
    pre = WheelPreprocessor()
    for i in range(200):
        s = pre.push("rear", 10.0 + 0.1 * i, (10.0 + 0.03 * math.sin(1.7 * i)) * KMH)
        assert "outlier" not in s.flags


def test_bounded_memory_and_never_raises():
    pre = WheelPreprocessor()
    for i in range(20000):
        pre.push("front" if i % 2 else "rear", 0.05 * i, (i % 97) * 0.5)
    st = pre._state["front"]
    assert len(st.values) <= pre.median_n and len(st.stamps) <= pre.median_n
    for junk in ([], {}, b"1", complex(1, 1)):
        assert pre.push("front", 1e9, junk) is None


def test_params():
    assert WheelPreprocessor({"gap_s": 1.0}).gap_s == 1.0
    assert WheelPreprocessor(gap_s=2.0).gap_s == 2.0
    with pytest.raises(ValueError):
        WheelPreprocessor({"no_such_param": 1})


def test_single_far_future_stamp_is_held_and_does_not_blank_the_other_bogie():
    pre = WheelPreprocessor()
    for i in range(20):
        pre.push("front", 0.1 * i, 36.0)
        pre.push("rear", 0.1 * i, 36.0)
    assert pre.push("front", 1e6, 36.0) is None and pre.last_reject == "future"
    for i in range(20, 40):
        f = pre.push("front", 0.1 * i, 36.0)
        r = pre.push("rear", 0.1 * i, 36.0)
        assert f is not None and r is not None
        assert not f.flags and not r.flags and f.usable and r.usable
    assert pre.counts["backstep"] == 0 and pre.counts["stale"] == 0


def test_forward_jump_after_joint_silence_is_confirmed_by_the_other_bogie():
    pre = WheelPreprocessor()
    for i in range(10):
        pre.push("front", 0.1 * i, 36.0)
        pre.push("rear", 0.1 * i, 36.0)
    t = 30.0
    assert pre.push("front", t, 36.0) is None and pre.last_reject == "future"
    r = pre.push("rear", t, 36.0)
    assert r is not None and "gap" in r.flags and r.usable
    f = pre.push("front", t + 0.1, 36.0)
    assert f is not None and "gap" in f.flags and f.usable


def test_lone_bogie_forward_jump_is_accepted_after_confirmation():
    pre = WheelPreprocessor()
    for i in range(10):
        pre.push("rear", 0.1 * i, 36.0)
    res = [pre.push("rear", 20.0 + 0.1 * k, 36.0) for k in range(5)]
    n = pre.params["future_confirm_n"]
    assert res[:n] == [None] * n and pre.counts["future"] == n
    assert res[n] is not None and "gap" in res[n].flags
    assert all(x is not None for x in res[n:])


def test_seeded_fuzz_never_raises_and_keeps_invariants():
    rng = random.Random(20260926)
    junk = [math.nan, math.inf, -math.inf, None, "1", b"1", True, [], object(), 1e308, -1e308, 10 ** 400]
    pre = WheelPreprocessor()
    t, last = 100.0, {}
    for _ in range(20000):
        b = rng.choice(["front", "rear", "front", "rear", "x", None])
        k = rng.random()
        if k < 0.05:
            st = rng.choice(junk)
        elif k < 0.10:
            st = t - rng.uniform(0.0, 3.0)
        elif k < 0.12:
            st = t + rng.uniform(3.0, 1e6)
        elif k < 0.13:
            st = rng.uniform(-1e9, 1e9)
        else:
            t += rng.uniform(0.0, 0.1)
            st = t
        raw = rng.choice(junk) if rng.random() < 0.05 else rng.uniform(-5.0, 200.0)
        s = pre.push(b, st, raw)
        if s is None:
            assert pre.last_reject in pre.counts
            continue
        assert math.isfinite(s.v) and math.isfinite(s.stamp) and abs(s.v) <= pre.v_max
        if s.bogie in last and "epoch" not in s.flags:
            assert s.stamp > last[s.bogie]
        last[s.bogie] = s.stamp
    assert pre.counts["error"] == 0


def test_bag_start_backlog_first_fresh_sample_is_not_held():
    pre = WheelPreprocessor()
    assert pre.push("rear", 97.5, 0.0) is not None
    f = pre.push("front", 100.0, 0.0)
    assert f is not None and f.usable and pre.counts["future"] == 0
    r = pre.push("rear", 97.6, 0.0)
    assert r is not None and "stale" in r.flags and not r.usable
