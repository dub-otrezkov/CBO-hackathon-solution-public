import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tram_odometry.preprocess import WheelPreprocessor, WheelSample
from tram_odometry.slip import DEFAULTS, SlipDetector, WheelTrust

KMH = 3.6
DT = 0.1
NAN = math.nan


def jitter(i, amp=0.01):
    return amp * math.sin(1.37 * i + 0.3) * math.cos(0.71 * i)


def run(frames, pre=None, det=None, order=("front", "rear")):
    pre = pre or WheelPreprocessor()
    det = det or SlipDetector()
    out = []
    for f in frames:
        row = {}
        for b in order:
            v = f.get(b)
            if v is None:
                continue
            t = f["t"] + (f.get("rear_dt", 0.0) if b == "rear" else 0.0)
            s = pre.push(b, t, v * KMH)
            if s is None:
                continue
            row[b] = (s, det.update(s, f.get("v_pred", NAN), f.get("var_pred", NAN), f.get("a_model", NAN)))
        out.append(row)
    return out


def weights(res, bogie, lo=0, hi=None):
    return [r[bogie][1].weight for r in res[lo:hi] if bogie in r]


def slips(res, bogie, lo=0, hi=None):
    return [r[bogie][1].slip for r in res[lo:hi] if bogie in r]


def bump(i, start, rise, hold, fall, height):
    if i < start:
        return 0.0
    if i < start + rise:
        return height * (i - start + 1) / rise
    if i < start + rise + hold:
        return height
    if i < start + rise + hold + fall:
        return height * (1 - (i - start - rise - hold + 1) / fall)
    return 0.0


def test_clean_constant_speed_full_trust():
    frames = [dict(t=i * DT, front=10 + jitter(i), rear=10 + jitter(i + 50), v_pred=10.0, var_pred=0.01, a_model=0.0)
              for i in range(200)]
    res = run(frames)
    for b in ("front", "rear"):
        assert min(weights(res, b)) >= 0.99
        assert not any(slips(res, b))
    tr = res[-1]["front"][1]
    assert isinstance(tr, WheelTrust) and abs(tr.residual) < 0.02 and abs(tr.z) < 1


def test_turn_like_difference_is_not_slip():
    for v, d in ((5.0, 0.04), (10.0, 0.15)):
        frames = []
        for i in range(300):
            sgn = 1.0 if (i // 20) % 2 == 0 else -1.0
            frames.append(dict(t=i * DT, front=v + sgn * d / 2, rear=v - sgn * d / 2, v_pred=v, var_pred=0.0025, a_model=0.0))
        res = run(frames)
        for b in ("front", "rear"):
            assert min(weights(res, b)) >= 0.95, (v, d, b)
            assert not any(slips(res, b))


def test_bogie_time_offset_during_acceleration_is_tolerated():
    frames = []
    for i in range(100):
        t = i * DT
        v = 3.0 + 1.2 * t
        frames.append(dict(t=t, front=v, rear=v + 1.2 * 0.05, rear_dt=0.05, v_pred=v + 0.03, var_pred=0.0025, a_model=1.2))
    res = run(frames)
    for b in ("front", "rear"):
        assert min(weights(res, b, 5)) >= 0.99
        assert not any(slips(res, b))


def spin_frames():
    frames = []
    for i in range(120):
        t = i * DT
        v = 5.0 + 0.5 * t
        ex = bump(i, 30, 3, 17, 3, 2.0)
        frames.append(dict(t=t, front=v + ex + jitter(i), rear=v + jitter(i + 7), v_pred=v, var_pred=0.0025, a_model=0.5))
    return frames


def test_single_wheel_spin_downweights_only_that_wheel_and_recovers():
    res = run(spin_frames())
    assert any(slips(res, "front", 31, 53)) and min(weights(res, "front", 31, 53)) <= DEFAULTS["slip_weight_cap"]
    assert not any(slips(res, "rear")) and min(weights(res, "rear")) >= 0.95
    end = 30 + 3 + 17 + 3
    assert not any(slips(res, "front", end + 7))
    assert min(weights(res, "front", end + 7 + 11)) >= 0.99
    w = weights(res, "front", end, None)
    steps = [b - a for a, b in zip(w, w[1:])]
    assert max(steps) <= (1 - DEFAULTS["slip_weight_cap"]) * DT / DEFAULTS["recover_s"] + 1e-6
    assert min(steps) >= -1e-9


def test_single_wheel_skid_without_filter_uses_braking_physics():
    frames = []
    for i in range(100):
        t = i * DT
        v = 10.0 - 1.0 * t
        drop = bump(i, 20, 2, 15, 2, 2.0)
        frames.append(dict(t=t, front=v, rear=v - drop, v_pred=NAN, var_pred=NAN, a_model=-1.0))
    res = run(frames)
    assert any(slips(res, "rear", 20, 40))
    assert min(weights(res, "rear", 21, 38)) <= 0.05
    assert not any(slips(res, "front")) and min(weights(res, "front")) >= 0.95
    assert math.isnan(res[30]["rear"][1].residual)


def test_common_skid_of_both_wheels_is_flagged_on_both():
    frames = []
    for i in range(120):
        t = i * DT
        v = 10.0 - 0.8 * t
        drop = bump(i, 30, 4, 15, 4, 4.0)
        frames.append(dict(t=t, front=v - drop, rear=v - drop + 0.01, v_pred=v, var_pred=0.01, a_model=-0.8))
    res = run(frames)
    during = range(34, 49)
    for b in ("front", "rear"):
        assert all(res[i][b][1].slip for i in during), b
        assert max(res[i][b][1].weight for i in during) <= DEFAULTS["slip_weight_cap"]
    assert any(res[i]["rear"][1].common for i in during)
    for b in ("front", "rear"):
        assert min(weights(res, b, 53 + 5 + 11)) >= 0.99


def test_common_spin_of_both_wheels_is_flagged():
    frames = []
    for i in range(100):
        t = i * DT
        v = 4.0 + 0.8 * t
        ex = bump(i, 30, 3, 12, 3, 3.0)
        frames.append(dict(t=t, front=v + ex, rear=v + ex, v_pred=v, var_pred=0.01, a_model=0.8))
    res = run(frames)
    for b in ("front", "rear"):
        assert all(res[i][b][1].slip for i in range(33, 45)), b
        assert res[40][b][1].weight <= DEFAULTS["slip_weight_cap"]


def test_emergency_braking_unknown_to_model_is_not_rejected():
    pre, det = WheelPreprocessor(), SlipDetector()
    x, P, tk = 10.0, 0.01, 0.0
    q = 0.5
    worst_err, ws = 0.0, []
    for i in range(80):
        t = i * DT
        v = max(1.0, 10.0 - 4.5 * max(0.0, t - 1.0))
        for b in ("front", "rear"):
            s = pre.push(b, t, (v + jitter(i + (b == "rear"))) * KMH)
            Pp = P + q * max(0.0, t - tk)
            tr = det.update(s, x, Pp, 0.0)
            ws.append(tr.weight)
            assert not tr.slip
            P, tk = Pp, t
            if tr.weight > 0:
                R = (0.03 + 0.01 * abs(s.v)) ** 2 / tr.weight
                K = P / (P + R)
                x += K * (s.v - x)
                P *= 1 - K
        worst_err = max(worst_err, abs(x - v))
    assert min(ws) >= 1 - DEFAULTS["soft_gain"] - 1e-9
    assert worst_err < 0.5


def test_filter_drift_does_not_lock_out_consistent_wheels():
    frames = [dict(t=i * DT, front=8.0 + jitter(i), rear=8.0 + jitter(i + 3), v_pred=9.0, var_pred=0.0025, a_model=0.0)
              for i in range(80)]
    res = run(frames)
    relax = DEFAULTS["relax_s"] + DEFAULTS["relax_ramp_s"]
    for b in ("front", "rear"):
        assert not any(slips(res, b))
        assert max(weights(res, b, 5, 25)) <= 1 - DEFAULTS["soft_gain"] + 0.01
        assert min(weights(res, b, int(relax / DT) + 5)) >= 0.99


def test_soft_evidence_still_acts_after_a_long_clean_run():
    n0 = 600
    frames = []
    for i in range(n0 + 80):
        dev = 0.0 if i < n0 else min(1.5, 0.05 * (i - n0))
        frames.append(dict(t=i * DT, front=8.0 + dev, rear=8.0 + dev, v_pred=8.0, var_pred=0.0025, a_model=0.0))
    res = run(frames)
    for b in ("front", "rear"):
        assert min(weights(res, b, 0, n0)) >= 0.99
        assert not any(slips(res, b))
        w = weights(res, b, n0 + 15, n0 + 35)
        assert max(w) <= 1 - DEFAULTS["soft_gain"] * 0.3 and min(w) >= 1 - DEFAULTS["soft_gain"] - 1e-9
        first = next(i for i in range(n0, len(res)) if res[i][b][1].soft > 0)
        assert first <= n0 + 10
        assert min(weights(res, b, first + int((DEFAULTS["relax_s"] + DEFAULTS["relax_ramp_s"]) / DT) + 1)) >= 0.99


def test_flickering_soft_evidence_does_not_defeat_anti_lockout():
    frames = []
    for i in range(120):
        off = 1.0 if (i // 2) % 2 == 0 else 0.0
        frames.append(dict(t=i * DT, front=8.0 + jitter(i), rear=8.0 + jitter(i + 3), v_pred=8.0 + off,
                           var_pred=0.0025, a_model=0.0))
    res = run(frames)
    relax = DEFAULTS["relax_s"] + DEFAULTS["relax_ramp_s"]
    for b in ("front", "rear"):
        assert min(weights(res, b, int(relax / DT) + 3)) >= 0.99


def test_weak_intermittent_cross_evidence_does_not_lock_out_a_wheel():
    frames = []
    for i in range(300):
        frames.append(dict(t=i * DT, front=8.0 + (0.45 if i % 20 == 10 else 0.0), rear=8.0, v_pred=9.0,
                           var_pred=0.0025, a_model=0.0))
    res = run(frames)
    for b in ("front", "rear"):
        assert not any(slips(res, b, 100))
        assert min(weights(res, b, 100)) >= 0.8, b


def test_dropout_of_one_bogie():
    frames = []
    for i in range(300):
        t = i * DT
        front = None if 50 <= i < 248 else 8.0
        frames.append(dict(t=t, front=front, rear=8.0 + jitter(i), v_pred=8.0, var_pred=0.01, a_model=0.0))
    res = run(frames)
    assert min(weights(res, "rear")) >= 0.99 and not any(slips(res, "rear"))
    assert "partner_stale" in res[100]["rear"][0].flags
    back = res[248]["front"]
    assert "gap" in back[0].flags and back[1].weight >= 0.99 and not back[1].slip
    assert not math.isnan(res[250]["front"][1].cross)


def test_outlier_and_stale_samples_get_zero_weight():
    pre, det = WheelPreprocessor(), SlipDetector()
    for i in range(20):
        det.update(pre.push("front", i * DT, 36.0), 10.0, 0.01, 0.0)
    s = pre.push("front", 2.0, 0.0)
    assert "outlier" in s.flags
    tr = det.update(s, 10.0, 0.01, 0.0)
    assert tr.weight == 0.0 and tr.reason == "outlier"
    tr = det.update(pre.push("front", 2.1, 36.0), 10.0, 0.01, 0.0)
    assert tr.weight >= 0.99 and not tr.slip
    s = pre.push("rear", 0.5, 36.0)
    assert "stale" in s.flags
    tr = det.update(s, 10.0, 0.01, 0.0)
    assert tr.weight == 0.0 and tr.reason == "stale"


def test_missing_or_invalid_filter_inputs_are_skipped():
    for vp, vv, am in ((NAN, NAN, NAN), (None, None, None), (math.inf, 0.01, 0.0), (10.0, -1.0, 0.0),
                       (10.0, math.nan, math.inf), ("10", "0.01", "0")):
        res = run([dict(t=i * DT, front=10.0, rear=10.0, v_pred=vp, var_pred=vv, a_model=am) for i in range(30)])
        for b in ("front", "rear"):
            assert min(weights(res, b)) >= 0.99, (vp, vv, am)
            assert math.isnan(res[-1][b][1].z)


def test_bad_samples_never_raise():
    det = SlipDetector()
    for bad in (None, object(), 3.0, WheelSample("front", NAN, 1.0), WheelSample("front", 1.0, math.inf),
                WheelSample("left", 1.0, 1.0)):
        tr = det.update(bad, 10.0, 0.01, 0.0)
        assert tr.weight == 0.0 and tr.reason == "invalid" and not tr.slip
    tr = det.update(WheelSample("rear", 1.0, 5.0, flags=None), 5.0, 0.01, 0.0)
    assert tr.weight >= 0.99
    det = SlipDetector()
    for i in range(30):
        for b, v in (("front", 1e154 * (-1) ** i), ("rear", -1e154)):
            tr = det.update(WheelSample(b, i * DT, v), 1e154, 1e300, 1e308)
            assert tr.weight == 0.0 and tr.reason == "invalid"
    for vp, vv, am in ((1e308, 1e308, 1e308), (-1e308, 0.0, -1e308), (10.0, 1e308, 0.0)):
        tr = det.update(WheelSample("front", 10.0, 5.0), vp, vv, am)
        assert math.isfinite(tr.weight) and 0.0 <= tr.weight <= 1.0


def test_seeded_fuzz_weight_always_in_unit_interval():
    rng = random.Random(4242)
    junk = [math.nan, math.inf, None, "1", True, 1e308, -1e308, 10 ** 400, -0.0]
    pre, det = WheelPreprocessor(), SlipDetector()
    t = 0.0
    for _ in range(20000):
        t += rng.uniform(-0.05, 0.15)
        s = pre.push(rng.choice(["front", "rear"]), t if rng.random() > 0.02 else rng.choice(junk),
                     rng.uniform(-2.0, 60.0) if rng.random() > 0.05 else rng.choice(junk))
        if s is None:
            continue
        args = [rng.uniform(-5, 20) if rng.random() > 0.1 else rng.choice(junk) for _ in range(3)]
        tr = det.update(s, *args)
        assert isinstance(tr, WheelTrust) and math.isfinite(tr.weight) and 0.0 <= tr.weight <= 1.0
        assert 0.0 <= tr.hard <= 1.0 and 0.0 <= tr.soft <= 1.0 and tr.reason != "error"


def test_deterministic_and_bounded():
    a = [(r[b][1].weight, r[b][1].slip, r[b][1].reason) for r in run(spin_frames()) for b in sorted(r)]
    b_ = [(r[b][1].weight, r[b][1].slip, r[b][1].reason) for r in run(spin_frames()) for b in sorted(r)]
    assert a == b_
    det = SlipDetector()
    pre = WheelPreprocessor()
    for i in range(5000):
        s = pre.push("front", i * DT, 36.0 + (i % 7))
        if s is not None:
            det.update(s, 10.0, 0.01, 0.0)
    assert len(det._st["front"].hist) <= DEFAULTS["history_n"]


def test_params_and_reset():
    assert SlipDetector({"z_lo": 4.0}).p["z_lo"] == 4.0
    assert SlipDetector(z_lo=5.0).p["z_lo"] == 5.0
    with pytest.raises(ValueError):
        SlipDetector({"bogus": 1})
    det = SlipDetector()
    run(spin_frames()[:40], det=det)
    assert det.any_slip()
    det.reset()
    assert not det.any_slip() and det.state("front")["latest"] is None
