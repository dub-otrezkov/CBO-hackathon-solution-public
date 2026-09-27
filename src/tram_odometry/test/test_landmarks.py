import json
import math
import os
import random
import sys
import time

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from tram_odometry import landmarks as LM
from tram_odometry.route import RouteMap, _Path

MAPS = os.path.join(PKG, "maps")
BIN = 0.5
MAP_RMS = 1.5e-3
NOISE = 2.5e-3


def _route(length=4000.0, name="route_test.json"):
    pts = [(1000.0 + i, 2000.0, 100.0) for i in range(int(length) + 1)]
    route = RouteMap([_Path(pts, closed=False)])
    route.names = [name]
    return route


def _pattern(n, seed, period_bins=None):
    rng = random.Random(seed)
    if period_bins:
        base = _pattern(period_bins, seed)
        return [base[i % period_bins] for i in range(n)]
    out, prev = [], 0.0
    for _ in range(n):
        prev = 0.6 * prev + rng.gauss(0.0, 1.0)
        out.append(prev)
    scale = MAP_RMS / math.sqrt(sum(x * x for x in out) / n)
    return [x * scale for x in out]


def _write_fp(directory, route, values, name="test", length=None):
    doc = {"version": 1, "route_file": f"route_{name}.json",
           "route_length_m": route.length(0) if length is None else length,
           "bin_m": BIN, "s_first": BIN / 2, "r": values, "n": [10] * len(values)}
    with open(os.path.join(directory, f"fingerprint_{name}.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)


def _value(values, s):
    q = (s - BIN / 2) / BIN
    i = math.floor(q)
    if i < 0 or i + 1 >= len(values):
        return 0.0
    f = q - i
    return values[i] + f * (values[i + 1] - values[i])


def _drive(matcher, truth, s0_true, offset, var0, dist, v=10.0, seed=1, noise=NOISE, scale=1.0, t0=100.0,
           pid=0, s_shift=0.0, times=None):
    rng = random.Random(seed)
    dt = 0.1
    s_true, s_est = s0_true, s0_true + offset
    var = var0
    matcher.reset(pid, s_est + s_shift)
    fixes, worst = [], 0.0
    for k in range(int(dist / (v * dt))):
        s_true += v * dt
        s_est += scale * v * dt
        var = (math.sqrt(var) + 0.005 * v * dt) ** 2
        r = (_value(truth, s_true) if truth is not None else 0.0) + rng.gauss(0.0, noise)
        vf, vr = v * (1 + r / 2), v * (1 - r / 2)
        c0 = time.perf_counter()
        fix = matcher.update(t0 + k * dt, vf, vr, v, s_est + s_shift, var)
        spent = time.perf_counter() - c0
        worst = max(worst, spent)
        if times is not None:
            times.append(spent)
        if fix is not None:
            fixes.append((fix, s_true + s_shift))
            gain = var / (var + fix.var)
            s_est += gain * (fix.s_route - s_shift - s_est)
            var = var * fix.var / (var + fix.var)
    return fixes, s_true - s_est, worst


@pytest.fixture()
def synthetic(tmp_path):
    route = _route()
    values = _pattern(int(route.length(0) / BIN), seed=7)
    _write_fp(str(tmp_path), route, values)
    return route, values, str(tmp_path)


def test_recovers_known_offset(synthetic):
    route, values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    fixes, _, _ = _drive(m, values, 300.0, offset=1.5, var0=1.0, dist=400.0)
    assert fixes, m.stats
    first, s_true = fixes[0]
    assert first.source == "fingerprint" and first.var == pytest.approx(0.16)
    assert abs(first.s_route - s_true) <= 0.5
    assert first.delta == pytest.approx(-1.5, abs=0.5)
    assert not first.reacq


def test_wide_gate_and_reacquisition(synthetic):
    route, values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    fixes, err, _ = _drive(m, values, 300.0, offset=25.0, var0=15.0 ** 2, dist=600.0)
    assert fixes and abs(fixes[0][0].s_route - fixes[0][1]) <= 0.5
    assert abs(err) <= 0.5
    m = LM.RouteMatcher(route, maps_dir=d)
    fixes, err, _ = _drive(m, values, 300.0, offset=25.0, var0=0.25, dist=1500.0)
    assert fixes, m.stats
    assert fixes[0][0].reacq and fixes[0][0].d_since >= 200.0
    assert abs(fixes[0][0].s_route - fixes[0][1]) <= 0.5
    assert abs(err) <= 0.5


def test_noise_only_gives_no_fix(synthetic):
    route, _values, d = synthetic
    for seed in (1, 2, 3):
        m = LM.RouteMatcher(route, maps_dir=d)
        fixes, _, _ = _drive(m, None, 200.0, offset=0.0, var0=4.0, dist=3500.0, seed=seed)
        assert not fixes, [(f.delta, f.rho, f.pr, f.reacq) for f, _ in fixes]
        assert m.stats["jobs"] > 100


@pytest.mark.parametrize("period_m, sigma", [(8.0, 0.5), (12.5, 20.0)])
def test_periodic_map_is_ambiguous(tmp_path, period_m, sigma):
    route = _route()
    values = _pattern(int(route.length(0) / BIN), seed=3, period_bins=int(period_m / BIN))
    _write_fp(str(tmp_path), route, values)
    m = LM.RouteMatcher(route, maps_dir=str(tmp_path))
    fixes, _, _ = _drive(m, values, 300.0, offset=0.0, var0=sigma ** 2, dist=1500.0, noise=1e-3)
    assert not fixes, [(f.delta, f.rho, f.pr) for f, _ in fixes]
    assert m.stats["reject"] > 10


def test_worst_case_update_time(synthetic):
    route, _values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    times = []
    _drive(m, None, 100.0, offset=0.0, var0=0.25, dist=3700.0, v=12.0, times=times)
    assert m.stats["jobs"] > 50
    _drive(m, None, 100.0, offset=0.0, var0=0.25, dist=400.0, v=2.5, times=times)
    times.sort()
    assert times[int(0.99 * (len(times) - 1))] <= 0.015
    assert times[-1] <= 0.100
    widest = 2 * int((150.0 + 10.0) / 0.25) + 1
    assert widest * m.p["max_cells"] > 10 * m.p["work_per_call"]


def test_garbage_inputs_never_raise(synthetic):
    route, values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    m.reset(0, 300.0)
    bad = [float("nan"), float("inf"), -float("inf"), None, "x", 1e308, -1e308, -5.0, 0.0]
    t = 10.0
    for a in bad:
        for b in bad:
            t += 0.05
            assert m.update(t, a, b, a, 300.0, b) is None
            assert m.update(a, 5.0, 5.0, 5.0, a, 1.0) is None
    m.update(t - 100.0, 5.0, 5.0, 5.0, 300.0, 1.0)
    m.update(t + 1e6, 5.0, 5.0, 5.0, 1e7, 1.0)
    m.update(t, 5.0, 5.0, 5.0, 300.0, -1.0, path_id=99)
    assert m.stats["errors"] == 0
    fixes, _, _ = _drive(m, values, 300.0, offset=1.0, var0=1.0, dist=400.0, t0=2e6)
    assert fixes and abs(fixes[0][0].s_route - fixes[0][1]) <= 0.5


def test_stops_and_slow_travel_give_no_samples(synthetic):
    route, _values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    m.reset(0, 300.0)
    for k in range(600):
        m.update(10.0 + 0.1 * k, 1.5, 1.5, 1.5, 300.0 + 0.15 * k, 1.0)
    for k in range(600):
        m.update(80.0 + 0.1 * k, 0.0, 0.0, 0.0, 390.0, 1.0)
    assert m.stats["samples"] == 0 and m.stats["fixes"] == 0


def test_reversing_clears_the_window(synthetic):
    route, values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    _drive(m, values, 300.0, offset=0.0, var0=1.0, dist=150.0)
    before = m.stats["window_resets"]
    s = 450.0
    for k in range(50):
        s -= 0.1
        m.update(1000.0 + 0.1 * k, -1.0, -1.0, 1.0, s, 1.0)
    assert m.stats["window_resets"] > before
    assert m._c_count == 0


def test_route_without_fingerprint_is_inactive(tmp_path):
    route = _route(name="shchukinskaya-tallinskaya.json")
    m = LM.RouteMatcher(route, maps_dir=str(tmp_path))
    m.reset(0, 100.0)
    assert not m.active and not m.maps
    assert all(m.update(1.0 + 0.1 * k, 10.0, 10.0, 10.0, 100.0 + k, 1.0) is None for k in range(3000))
    assert m.stats["inactive"] == 3000
    route = _route()
    _write_fp(str(tmp_path), route, [0.0] * 100, length=route.length(0) + 50.0)
    m = LM.RouteMatcher(route, maps_dir=str(tmp_path))
    m.reset(0, 100.0)
    assert not m.active and "route_test.json" in m.load_errors


def test_terminal_connector_maps_back(synthetic):
    route, values, d = synthetic
    p = route.paths[0]
    pid = route._dynamic([(p.x[0] - 20.0, p.y[0], p.z[0])] + list(zip(p.x, p.y, p.z)))
    shift = route.paths[pid].s[1]
    assert pid == 1 and shift == pytest.approx(20.0, abs=0.01)
    m = LM.RouteMatcher(route, maps_dir=d)
    m.reset(pid, 320.0)
    assert m.active
    fixes, _, _ = _drive(m, values, 300.0, offset=-1.2, var0=1.0, dist=400.0, pid=pid, s_shift=shift)
    assert fixes and abs(fixes[0][0].s_route - fixes[0][1]) <= 0.5
    pid = route._dynamic([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
    m.reset(pid, 0.0)
    assert not m.active


def test_deterministic(synthetic):
    route, values, d = synthetic
    runs = []
    for _ in range(2):
        m = LM.RouteMatcher(route, maps_dir=d)
        fixes, err, _ = _drive(m, values, 300.0, offset=3.0, var0=4.0, dist=800.0, scale=1.01)
        runs.append(([(f.stamp, f.s_route, f.rho, f.pr) for f, _ in fixes], err, dict(m.stats)))
    assert runs[0] == runs[1]


def test_scale_rule():
    fix = LM.RouteFix(stamp=0.0, s_route=0.0, var=0.16, delta=1.0, d_since=100.0)
    assert LM.update_scale(1.0, fix, 0.5) == pytest.approx(1.0 + 0.3 * 0.5 * 0.01)
    assert LM.update_scale(1.0, LM.RouteFix(0.0, 0.0, 0.16, delta=1.0, d_since=10.0), 0.5) == 1.0
    assert LM.update_scale(1.0, LM.RouteFix(0.0, 0.0, 0.16, delta=500.0, d_since=60.0), 1.0) == 1.03


def test_package_fingerprints_match_the_trip_routes():
    route = RouteMap.load(MAPS)
    m = LM.RouteMatcher(route, maps_dir=MAPS)
    assert not m.load_errors
    assert sorted(m.maps) == [i for i, n in enumerate(route.names) if n.startswith("route_")]
    for pid in m.maps:
        m.reset(pid, 1000.0)
        assert m.active


def _drive_on(matcher, truth, s_true, s_est, var, dist, t0, v=10.0, seed=3):
    rng = random.Random(seed)
    dt = 0.1
    fixes = []
    for k in range(int(dist / (v * dt))):
        s_true += v * dt
        s_est += v * dt
        var = (math.sqrt(var) + 0.005 * v * dt) ** 2
        r = _value(truth, s_true) + rng.gauss(0.0, NOISE)
        fix = matcher.update(t0 + k * dt, v * (1 + r / 2), v * (1 - r / 2), v, s_est, var)
        if fix is not None:
            fixes.append(fix)
            gain = var / (var + fix.var)
            s_est += gain * (fix.s_route - s_est)
            var = var * fix.var / (var + fix.var)
    return fixes, s_true - s_est


@pytest.mark.parametrize("calls", [
    [(50.0, 5.0, 5.0, 1e308, 1e300, 1.0), (1e6, 5.0, 5.0, 1e308, -1e300, 1.0)],
    [(50.0, 5.0, 5.0, 5.0, 1e308, 1.0), (50.1, 5.0, 5.0, 5.0, -1e308, 1.0)],
    [(50.0, 5.0, 5.0, 5.0, 10 ** 400, 1.0), (50.1, 5.0, 5.0, 5.0, float("inf"), 1.0)],
    [(1e308, 5.0, 5.0, 5.0, 300.0, 1.0)],
])
def test_extreme_inputs_do_not_poison_the_state(synthetic, calls):
    route, values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    m.reset(0, 300.0)
    for a in calls:
        assert m.update(*a) is None
    fixes, err = _drive_on(m, values, 298.5, 300.0, 1.0, 800.0, t0=2e6)
    assert m.stats["errors"] == 0
    assert fixes and abs(err) <= 0.5 and math.isfinite(m._u) and abs(m._u) < 1e4


def test_fuzz_never_raises_and_recovers_without_reset(synthetic):
    route, values, d = synthetic
    garbage = [float("nan"), float("inf"), -float("inf"), None, "x", b"1", [], 1e308, -1e308, 1e12, -5.0, 0.0,
               10 ** 400, True, complex(1, 1)]
    for seed in range(4):
        rng = random.Random(seed)
        m = LM.RouteMatcher(route, maps_dir=d)
        m.reset(0, 300.0)
        t, s = 50.0, 300.0
        for _ in range(1500):
            x = rng.random()
            if x < 0.02:
                t -= rng.choice([0.05, 1.0, 3.0, 1e4])
            elif x < 0.03:
                t += rng.choice([5.0, 60.0, 3600.0])
                s += rng.choice([0.0, 50.0, 600.0])
            elif x < 0.04:
                for _ in range(rng.randint(10, 300)):
                    t += 0.1
                    m.update(t, 0.0, 0.0, 0.0, s + rng.gauss(0.0, 0.01), 1.0)
            else:
                t += 0.1
                s += 1.0
            g = [rng.choice(garbage) if rng.random() < 0.3 else good for good in (t, 10.0, 10.0, 10.0, s, 1.0)]
            pid = rng.choice([None, 0, 1, -1, 99, "a"]) if rng.random() < 0.02 else None
            if rng.random() < 0.01:
                m.applied(rng.choice(garbage))
            m.update(*g, path_id=pid)
        assert m.stats["errors"] == 0, m.stats
        m.update(t + 0.1, 10.0, 10.0, 10.0, 1000.0, 1.0, path_id=0)
        fixes, err = _drive_on(m, values, 998.5, 1000.0, 1.0, 800.0, t0=t + 0.2)
        assert fixes and abs(err) <= 0.6, (seed, len(fixes), err, m.stats)


def test_pair_dropout_drops_samples_next_to_it(synthetic):
    route, _values, d = synthetic
    m = LM.RouteMatcher(route, maps_dir=d)
    m.reset(0, 300.0)
    t, s = 10.0, 300.0
    for k in range(200):
        t += 5.0 if k == 100 else 0.1
        s += 50.0 if k == 100 else 1.0
        m.update(t, 10.0, 10.0, 10.0, s, 1.0)
    assert m.stats["samples"] == 200 - 5 - 5


def test_reacquisition_confirm_gap(synthetic):
    route, values, d = synthetic
    base = LM.RouteMatcher(route, maps_dir=d)
    fx0, _err0, _ = _drive(base, values, 300.0, offset=25.0, var0=0.25, dist=1500.0)
    gapped = LM.RouteMatcher(route, {"reacq_confirm_gap_m": 60.0}, maps_dir=d)
    fx1, err1, _ = _drive(gapped, values, 300.0, offset=25.0, var0=0.25, dist=1500.0)
    assert fx0 and fx1 and fx0[0][0].reacq and fx1[0][0].reacq
    assert abs(fx1[0][0].s_route - fx1[0][1]) <= 0.5 and abs(err1) <= 0.5
    assert fx1[0][0].d_since >= fx0[0][0].d_since + 60.0 - 10.0


def test_maps_dir_resolution(tmp_path, monkeypatch):
    route = RouteMap.load(MAPS)
    m = LM.RouteMatcher(route)
    assert m.maps_dir in LM._package_maps_dirs() and sorted(m.maps) == [0, 1]
    monkeypatch.setattr(LM, "_package_maps_dirs", lambda: [str(tmp_path / "missing"), MAPS])
    assert LM.RouteMatcher(route).maps_dir == MAPS
    share = tmp_path / "share"
    share.mkdir()
    for n in os.listdir(MAPS):
        if n.startswith("fingerprint_"):
            with open(os.path.join(MAPS, n), encoding="utf-8") as src:
                (share / n).write_text(src.read(), encoding="utf-8")
    monkeypatch.setattr(LM, "_package_maps_dirs", lambda: [str(share), MAPS])
    m = LM.RouteMatcher(route)
    assert m.maps_dir == str(share) and sorted(m.maps) == [0, 1]
    route.maps_dir = MAPS
    assert LM.RouteMatcher(route).maps_dir == MAPS
    empty = tmp_path / "empty"
    empty.mkdir()
    m = LM.RouteMatcher(route, maps_dir=str(empty))
    assert m.maps_dir == str(empty) and not m.maps and not m.active
