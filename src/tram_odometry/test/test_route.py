import json
import math
import os
import sys
import time

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG)

from tram_odometry import route
from tram_odometry.route import RouteMap, grid_scale, to_grid

MAPS = os.path.join(PKG, "maps")
FIXES = [(55.80661953166667, 37.45784378333333), (55.80521452666667, 37.41377159), (55.800290565, 37.39131927)]


@pytest.fixture(scope="module")
def official(tmp_path_factory):
    d = tmp_path_factory.mktemp("official")
    for name in route.PATH_FILES:
        with open(os.path.join(MAPS, name), encoding="utf-8") as src:
            (d / name).write_text(src.read(), encoding="utf-8")
    return RouteMap.load(str(d))


@pytest.fixture(scope="module")
def trips():
    return RouteMap.load(MAPS)


def _write_route(directory, name, xyz, **meta):
    points = [{"x": x, "y": y, "z": z} for x, y, z in xyz]
    path = {"ext_id": None, "point_indices": list(range(len(points)))}
    path.update(meta)
    with open(os.path.join(directory, name), "w", encoding="utf-8") as f:
        json.dump({"points": points, "paths": [path]}, f)


def test_to_grid_central_meridian_and_symmetry():
    x, y = to_grid(0.0, 39.0)
    assert x == pytest.approx(500000.0 - route.X0, abs=1e-6)
    assert y == pytest.approx(-route.Y0, abs=1e-6)
    east_plus = to_grid(55.8, 39.0 + 1.5)[0] + route.X0 - route.E0
    east_minus = to_grid(55.8, 39.0 - 1.5)[0] + route.X0 - route.E0
    assert east_plus == pytest.approx(-east_minus, abs=1e-6)


def test_grid_scale_on_the_line():
    k = grid_scale(101000.0)
    assert 0.99970 < k < 0.99975


def test_real_fixes_lie_on_the_official_paths(official):
    for lat, lon in FIXES:
        x, y = to_grid(lat, lon)
        path_id, s, lateral = official.localize(x, y)
        px, py, _, _ = official.pose(path_id, s)
        assert math.hypot(px - x, py - y) < 1.0
        assert abs(lateral) < 1.0


def test_official_paths_loaded_with_ground_length(official):
    assert official.names == list(route.PATH_FILES)
    for p in official.paths:
        grid = sum(math.hypot(p.x[i + 1] - p.x[i], p.y[i + 1] - p.y[i]) for i in range(len(p.x) - 1))
        assert 4700.0 < grid < 4715.0
        assert p.length - grid == pytest.approx(grid * (1.0 / 0.99972 - 1.0), abs=0.05)


def test_pose_localize_round_trip(official):
    for path_id in range(len(official.paths)):
        for s in (0.0, 12.3, 1000.0, 2500.5, official.length(path_id) - 1.0):
            x, y, z, yaw = official.pose(path_id, s)
            got_id, got_s, lateral = official.localize(x, y, yaw)
            assert got_id == path_id
            assert got_s == pytest.approx(s, abs=1e-6)
            assert abs(lateral) < 1e-6


def test_heading_selects_the_direction_track(official):
    x, y, _, yaw = official.pose(0, 2000.0)
    assert official.localize(x, y, yaw)[0] == 0
    assert official.localize(x, y, yaw + math.pi)[0] == 1


def test_z_follows_the_antenna_height(official, tmp_path):
    p = official.paths[0]
    assert official.pose(0, 0.0)[2] == pytest.approx(p.z[0] + route.Z_OFFSET_M)
    for name in route.PATH_FILES:
        with open(os.path.join(MAPS, name), encoding="utf-8") as src:
            (tmp_path / name).write_text(src.read(), encoding="utf-8")
    (tmp_path / route.PARAMS_FILE).write_text(json.dumps({"z_offset_m": 0.0}), encoding="utf-8")
    assert RouteMap.load(str(tmp_path)).pose(0, 0.0)[2] == pytest.approx(p.z[0])


def test_open_end_extrapolates_straight(official):
    length = official.length(0)
    x1, y1, _, yaw = official.pose(0, length)
    x2, y2, _, _ = official.pose(0, length + 10.0)
    k = grid_scale(x1)
    assert math.hypot(x2 - x1, y2 - y1) == pytest.approx(10.0 * k, abs=1e-3)
    assert math.atan2(y2 - y1, x2 - x1) == pytest.approx(yaw, abs=1e-6)


def test_closed_loop_wraps(tmp_path):
    square = [(0.0, 0.0, 1.0), (100.0, 0.0, 2.0), (100.0, 100.0, 3.0), (0.0, 100.0, 4.0)]
    _write_route(str(tmp_path), route.LOOP_FILE, square)
    loop = RouteMap.load(str(tmp_path))
    assert loop.names == [route.LOOP_FILE]
    length = loop.length(0)
    assert length == pytest.approx(400.0 / grid_scale(50.0), rel=1e-6)
    for s in (0.0, 37.0, 250.0):
        a, b = loop.pose(0, s), loop.pose(0, s + length)
        assert a == pytest.approx(b, abs=1e-6)


def test_trip_routes_preferred_and_departure_snap(tmp_path):
    _write_route(str(tmp_path), "route_a.json", [(0.0, 0.0, 0.0), (500.0, 0.0, 0.0)], departure_m=50.0, arrival_m=100.0)
    _write_route(str(tmp_path), "route_b.json", [(495.0, 3.0, 0.0), (495.0, 600.0, 0.0)], departure_m=100.0, arrival_m=50.0)
    _write_route(str(tmp_path), route.PATH_FILES[0], [(0.0, 50.0, 0.0), (10.0, 50.0, 0.0)])
    trips = RouteMap.load(str(tmp_path))
    assert trips.names == ["route_a.json", "route_b.json"]
    pid, s, lateral = trips.localize(499.0, 10.5)
    assert pid == 1 and s == pytest.approx((10.5 - 3.0) / grid_scale(495.0), abs=1e-6) and abs(lateral) == pytest.approx(4.0)
    assert trips.localize(499.0, 0.5, 0.0)[0] == 0
    assert trips.localize(250.0, 1.0)[0] == 0


def test_far_point_and_speed(official):
    path_id, s, lateral = official.localize(0.0, 0.0)
    assert path_id == 2 and s == 0.0 and lateral == 0.0
    x, y, _, _ = official.pose(path_id, 0.0)
    assert (x, y) == pytest.approx((0.0, 0.0))
    x, y = to_grid(*FIXES[1])
    start = time.perf_counter()
    for _ in range(200):
        official.localize(x + 1.0, y - 1.0)
        official.pose(0, 1234.5)
    assert (time.perf_counter() - start) / 200 < 0.005


def test_missing_maps_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        RouteMap.load(str(tmp_path))


def test_package_trip_routes(trips):
    assert trips.names == ["route_east_west.json", "route_west_east.json"]
    for pid in (0, 1):
        assert trips.paths[pid].departure_s > 150.0 and trips.paths[pid].arrival_s > 300.0
        x, y, z, yaw = trips.pose(pid, 3000.0)
        got_id, got_s, _ = trips.localize(x, y, yaw)
        assert got_id == pid and got_s == pytest.approx(3000.0, abs=1e-6)


def test_waiting_in_the_west_loop_departs_west_east(trips):
    x, y = 98999.0, 84907.3
    pid, s0, _ = trips.localize(x, y)
    assert pid == 2 and s0 == 0.0
    assert trips.pose(pid, 0.0)[:2] == pytest.approx((x, y))
    start = trips.paths[1]
    gap = math.hypot(start.x[0] - x, start.y[0] - y)
    px, py, _, _ = trips.pose(pid, gap / grid_scale(x) + 100.0)
    qx, qy, _, _ = trips.pose(1, 100.0)
    assert math.hypot(px - qx, py - qy) < 0.05


def test_waiting_at_the_east_terminal_departs_east_west(trips):
    x, y, _, _ = trips.pose(0, 0.0)
    pid, s0, _ = trips.localize(x + 0.5, y + 0.5)
    base, shift = trips.base_of(pid)
    assert base == 0 and s0 + shift < 4.0


def test_connector_maps_back_to_the_route_it_joins():
    import math
    rm = RouteMap.load(MAPS)
    for pid in range(len(rm.paths)):
        assert rm.base_of(pid) == (pid, 0.0)
    assert rm.base_of(None) == (None, 0.0)
    p = rm.paths[0]
    ux, uy = p.x[1] - p.x[0], p.y[1] - p.y[0]
    n = math.hypot(ux, uy)
    x, y = p.x[0] - 30.0 * ux / n, p.y[0] - 30.0 * uy / n
    pid, s0, _ = rm.localize(x, y, heading=math.atan2(uy, ux))
    if pid < rm._n_map:
        return
    base, shift = rm.base_of(pid)
    assert base is not None and shift < 0
    for s in (40.0, 500.0, 2000.0):
        a, b = rm.pose(pid, s), rm.pose(base, s + shift)
        assert math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-6
    line = rm._dynamic([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], None)
    assert rm.base_of(line) == (None, 0.0)


def _line_map(tmp_path, departures=None):
    pts = [{"x": 100000.0 + i, "y": 85000.0, "z": 150.0} for i in range(1001)]
    with open(tmp_path / "route_line.json", "w", encoding="utf-8") as f:
        json.dump({"points": pts, "paths": [{"point_indices": list(range(1001))}]}, f)
    if departures is not None:
        with open(tmp_path / route.DEPARTURES_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "departures": departures}, f)
    return RouteMap.load(str(tmp_path))


def test_a_tram_standing_behind_the_route_start_goes_on_its_extension(tmp_path):
    rm = _line_map(tmp_path)
    pid, s0, lateral = rm.localize(100000.0 - 3.0, 85000.4)
    assert pid == 0 and abs(s0 - (-3.0 / grid_scale(100000.0))) < 1e-6 and abs(lateral - 0.4) < 1e-9
    x, y, _, _ = rm.pose(0, s0)
    assert abs(x - (100000.0 - 3.0)) < 1e-6 and abs(y - 85000.0) < 1e-9
    assert rm.localize(100000.0 + 5.0, 85000.0)[1] > 0.0
    far = rm.localize(100000.0 - route.CONNECT_MIN_M - 2.0, 85000.0)
    assert far[0] == 0 and far[1] == 0.0


def test_the_route_departure_track_replaces_the_route_line_until_it_joins(tmp_path):
    k = grid_scale(100000.0)
    pts, s_route = [], []
    for i in range(60):
        lat = 0.5 if i < 50 else 0.5 * (60 - i) / 10.0
        pts.append([100000.0 + i, 85000.0 + lat, 150.0])
        s_route.append(i / k)
    pts.append([100060.0, 85000.0, 150.0])
    s_route.append(60 / k)
    rm = _line_map(tmp_path, [{"route": "route_line.json", "join_index": 60, "points": pts, "s_route": s_route}])
    assert len(rm.departures) == 1
    pid, s0, lateral = rm.localize(100010.0, 85000.45)
    assert pid == 1 and abs(lateral + 0.05) < 1e-9
    base, shift = rm.base_of(pid, s0)
    assert base == 0 and abs((s0 + shift) - 10.0 / k) < 1e-9
    track_len = rm.departures[0][2].length
    b_join, sh_join = rm.base_of(pid, track_len)
    assert abs((track_len + sh_join) - 60.0 / k) < 1e-9 and rm.base_of(pid) == (0, sh_join)
    assert abs(rm.base_of(pid, track_len + 300.0)[1] - sh_join) < 1e-12
    x, y, _, _ = rm.pose(pid, track_len + 100.0)
    assert abs(x - (100060.0 + 100.0 * k)) < 1e-3 and abs(y - 85000.0) < 1e-9
    assert rm.localize(100200.0, 85000.0)[0] == 0


def test_departures_that_do_not_end_on_their_route_point_are_ignored(tmp_path):
    bad = {"route": "route_line.json", "join_index": 60, "points": [[100000.0, 85000.5, 150.0], [100060.0, 85000.2, 150.0]],
           "s_route": [0.0, 60.0]}
    assert _line_map(tmp_path, [bad]).departures == []


def test_real_departure_tracks_are_on_their_routes_and_join_them(trips):
    assert {rm_name for rm_name in (trips.names[d[0]] for d in trips.departures)} == set(trips.names)
    for pid, j, dp, s_route in trips.departures:
        base = trips.paths[pid]
        assert (dp.x[-1], dp.y[-1]) == (base.x[j], base.y[j]) and abs(s_route[-1] - base.s[j]) < 1e-3
        assert len(s_route) == len(dp.s) and all(b > a for a, b in zip(s_route, s_route[1:]))
        assert abs(dp.length - (s_route[-1] - s_route[0])) < 3.0


FORK_I = 100
BRANCH_PTS = 121
NB = 31


def _loc(j):
    return 2.7 + 0.3 * ((j % 3) - 1)


def _fork_model(**over):
    m = {"kind": "route_envelope", "bin_m": 5.0, "route_loc": [_loc(j) for j in range(NB)], "route_scale": [0.1] * NB,
         "nu": 3.0, "scale_floor": 0.35, "branch_v_max": 9.0, "evidence_m": [25.0, 100.0], "corr_m": 10.0,
         "prior_branch": 0.1, "stop_revert_m": [40.0, 112.0], "stop_v": 0.3, "dead_end_m": 185.0}
    m.update(over)
    return m


def _fork_map(tmp_path, model="default"):
    pts = [{"x": 100000.0 + i, "y": 85000.0, "z": 150.0} for i in range(401)]
    with open(tmp_path / "route_line.json", "w", encoding="utf-8") as f:
        json.dump({"points": pts, "paths": [{"point_indices": list(range(401))}]}, f)
    if model is not None:
        a = 0.3
        bpts = [[100000.0 + FORK_I + i * math.cos(a), 85000.0 + i * math.sin(a), 150.0] for i in range(BRANCH_PTS)]
        entry = {"name": "test_branch", "route": "route_line.json", "fork_index": FORK_I, "points": bpts}
        if model != "none-block":
            entry["model"] = _fork_model() if model == "default" else model
        with open(tmp_path / route.BRANCHES_FILE, "w", encoding="utf-8") as f:
            json.dump({"version": 3, "branches": [entry]}, f)
    return RouteMap.load(str(tmp_path))


def _feed(sel, fork_s, speed, d0=-10.0, d1=200.0, step=0.25):
    out, i = [], 0
    while True:
        d = d0 + i * step
        if d > d1:
            return out
        out.append((d, sel.update(fork_s + d, speed(d))))
        i += 1


def _first(res):
    return next((d for d, c in res if c), None)


def _run(rm, speed, until_d=250.0, trace=False):
    import tram_odometry.estimator as core

    core_to_grid = core.to_grid
    core.to_grid = lambda lat, lon: (lon, lat)
    try:
        est = core.Estimator(route=rm)
        est.on_controller(0.0, 0)
        est.on_wheel("front", 0.0, 0.0)
        est.on_gnss_fix(0.0, 85000.0, 100030.0, 153.0, "master")
    finally:
        core.to_grid = core_to_grid
    fork_s = rm.paths[0].s[FORK_I]
    t, outs, s_pub, ws = 0.0, [], [], []
    if trace:
        est.trace_w = ws
    while t < 300.0:
        est.on_controller(t, 0)
        if abs(t * 10 - round(t * 10)) < 1e-6:
            for b in ("front", "rear"):
                est.on_wheel(b, t, speed(t) * 3.6)
        e = est.output(t)
        if e is not None:
            outs.append(e)
            s_pub.append(est.route_s_ref + e.s - est.s_ref + est.p["output_along_offset_m"])
            if trace:
                ws.append(est._branch[2].w if est._branch is not None else 0.0)
            if s_pub[-1] > fork_s + until_d:
                break
        t = round(t + 0.05, 6)
    return est, outs, s_pub


def _same(a, b):
    return len(a) == len(b) and all((p.stamp, p.x, p.y, p.z, p.yaw, p.v, p.s) == (q.stamp, q.x, q.y, q.z, q.yaw, q.v, q.s)
                                    for p, q in zip(a, b))


def test_branches_load_with_their_model_and_share_the_route_up_to_the_fork(tmp_path):
    rm = _fork_map(tmp_path)
    assert len(rm.branches) == 1
    b = rm.branches[0]
    assert b.model and len(b.terms) == NB and b.fork_s == rm.paths[0].s[FORK_I]
    assert all(rm.paths[0].s[i] == b.path.s[i] for i in range(FORK_I + 1))
    assert rm.branch_pose(0, b.fork_s - 1.0) is None
    x, y, _, yaw = rm.branch_pose(0, b.fork_s + 50.0)
    assert y > 85000.0 + 10.0 and abs(yaw - 0.3) < 1e-9
    assert rm.branch_of(0) == (0, 0.0)
    assert b.prior_log_odds == math.log(0.1 / 0.9)


def test_branch_pose_holds_the_end_pose_past_the_branch_end(tmp_path):
    rm = _fork_map(tmp_path)
    b = rm.branches[0]
    end = rm.branch_pose(0, b.path.length)
    assert abs(end[0] - (100000.0 + FORK_I + 120 * math.cos(0.3))) < 1e-6
    assert abs(end[1] - (85000.0 + 120 * math.sin(0.3))) < 1e-6
    for extra in (0.1, 10.0, 60.0, 1e4):
        assert rm.branch_pose(0, b.path.length + extra) == end
    assert rm.branch_pose(0, b.path.length - 1.0)[0] < end[0]


def test_invalid_model_blocks_never_select(tmp_path):
    legacy = {"diverge_s": 100.0, "centres_m": [7.5], "route": {"nu": 4.0, "loc": [2.7], "scale": [0.3]},
            "branch": {"nu": 4.0, "loc": [6.0], "scale": [1.0]}, "prior_branch": 0.15, "posterior_threshold": 0.99}
    bad = [legacy, {"kind": "other"}, {"kind": None}, {"prior_branch": 1.0}, {"prior_branch": 0.0}, {"nu": 0.0},
           {"route_scale": [-0.1] * NB}, {"route_loc": [2.7] * (NB - 1)}, {"route_loc": []},
           {"evidence_m": [100.0, 25.0]}, {"evidence_m": [-5.0, 25.0]}, {"corr_m": 0.0}, {"bin_m": 0.0},
           {"scale_floor": 0.0}, {"branch_v_max": -1.0}, {"stop_revert_m": [112.0, 40.0]}, {"dead_end_m": math.nan},
           {"route_loc": [2.7] * (NB - 1) + [math.nan]}, {"stop_v": "fast"}]
    for i, over in enumerate(bad):
        m = over if i == 0 else _fork_model(**over)
        d = tmp_path / str(i)
        d.mkdir()
        rm = _fork_map(d, m)
        b = rm.branches[0]
        assert len(rm.branches) == 1 and not b.model, over
        sel = route.BranchSelector(b)
        assert _first(_feed(sel, b.fork_s, lambda d: 7.0)) is None
    missing = _fork_model()
    del missing["corr_m"]
    (tmp_path / "missing").mkdir()
    assert not _fork_map(tmp_path / "missing", missing).branches[0].model
    (tmp_path / "none").mkdir()
    assert not _fork_map(tmp_path / "none", "none-block").branches[0].model


def test_selector_route_like_speeds_never_choose(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    for dv in (-0.5, -0.3, 0.0, 0.3, 0.5):
        sel = route.BranchSelector(b)
        res = _feed(sel, b.fork_s, lambda d, dv=dv: _loc(min(int(max(d, 0.0) / 5.0), NB - 1)) + dv)
        assert _first(res) is None and sel.log_odds < b.prior_log_odds, dv
    sel = route.BranchSelector(b)
    assert _first(_feed(sel, b.fork_s, lambda d: 2.7 if d < 60.0 else 0.0, d1=60.0 + 1e-9)) is None
    assert sel.dead == route.BranchSelector.DEAD_STOP


def test_selector_fast_speeds_choose_inside_the_window(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    for v in (5.0, 6.5, 8.0, 12.0):
        sel = route.BranchSelector(b)
        res = _feed(sel, b.fork_s, lambda d, v=v: v)
        first = _first(res)
        assert first is not None and 25.0 < first < 40.0, (v, first)
        assert all(c for d, c in res if first <= d <= 185.0) and not any(c for d, c in res if d > 185.0)
        assert sel.dead == route.BranchSelector.DEAD_END and math.isfinite(sel.log_odds)


def test_selector_stop_in_the_platform_zone_reverts_for_good(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    sel = route.BranchSelector(b)
    res = _feed(sel, b.fork_s, lambda d: 6.5, d1=50.0)
    assert res[-1][1]
    assert not sel.update(b.fork_s + 50.0, 0.1)
    assert sel.dead == route.BranchSelector.DEAD_STOP
    lo = sel.log_odds
    assert _first(_feed(sel, b.fork_s, lambda d: 6.5, d0=50.25)) is None and sel.log_odds == lo
    for d_stop in (30.0, 115.0):
        sel = route.BranchSelector(b)
        _feed(sel, b.fork_s, lambda d: 6.5, d1=d_stop)
        assert sel.update(b.fork_s + d_stop, 0.0) and not sel.dead


def test_selector_past_the_dead_end_reverts_for_good(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    sel = route.BranchSelector(b)
    assert _feed(sel, b.fork_s, lambda d: 6.5, d1=185.0)[-1][1]
    assert not sel.update(b.fork_s + 185.5, 6.5) and sel.dead == route.BranchSelector.DEAD_END
    assert not sel.update(b.fork_s + 60.0, 6.5) and not sel.update(b.fork_s + 61.0, 6.5)


def test_selector_is_not_latched(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    sel = route.BranchSelector(b)
    res = _feed(sel, b.fork_s, lambda d: 6.5 if d <= 35.0 else _loc(min(int(d / 5.0), NB - 1)), d1=100.0)
    first = _first(res)
    assert first is not None and first <= 35.0
    back = next(d for d, c in res if d > first and not c)
    assert back < 100.0 and not res[-1][1] and not sel.dead and sel.log_odds < 0.0


def test_selector_weight_is_the_clipped_posterior(tmp_path, monkeypatch):
    b = _fork_map(tmp_path).branches[0]
    sel = route.BranchSelector(b)
    assert sel.w == 0.0 and sel.weight() == 0.0
    sel.update(b.fork_s - 5.0, 2.7)
    assert abs(sel.w - 0.1) < 1e-12 and sel.weight() == sel.w
    for lo in (-800.0, -3.0, 0.0, 2.0, 800.0):
        sel.log_odds, sel._d_prev = lo, b.fork_s + 200.0
        sel.update(b.fork_s - 5.0, 2.7)
        assert abs(sel.w - 1.0 / (1.0 + math.exp(-max(min(lo, 700.0), -700.0)))) < 1e-12
    monkeypatch.setattr(route, "BRANCH_W_LO", 0.2)
    monkeypatch.setattr(route, "BRANCH_W_HI", 0.8)
    for w, want in ((0.0, 0.0), (0.19, 0.0), (0.2, 0.2), (0.5, 0.5), (0.8, 0.8), (0.81, 1.0), (1.0, 1.0)):
        sel.w = w
        assert sel.weight() == want, w
    sel = route.BranchSelector(b)
    _feed(sel, b.fork_s, lambda d: 6.5, d1=100.0)
    assert sel.w > 0.999 and sel.weight() == 1.0
    sel.update(b.fork_s + 186.0, 6.5)
    assert sel.dead and sel.w == 0.0 and sel.weight() == 0.0


def test_estimator_survives_a_broken_branch_selector(tmp_path):
    rm = _fork_map(tmp_path)
    est, outs, _ = _run(rm, lambda t: 6.5, until_d=10.0)

    class Boom:
        def update(self, s, v):
            raise RuntimeError("broken branch map")

    est._branch = (0, 0.0, Boom())
    e = est.output(est.t)
    assert e is not None and est._branch is None
    x, y, _, _ = rm.pose(est.path_id, est.route_s_ref + e.s - est.s_ref + est.p["output_along_offset_m"])
    assert abs(e.x - x) < 1e-9 and abs(e.y - y) < 1e-9

def test_selector_reset_clears_the_decision(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    sel = route.BranchSelector(b)
    _feed(sel, b.fork_s, lambda d: 6.5, d1=190.0)
    assert sel.dead
    sel.reset()
    assert sel.log_odds == b.prior_log_odds and not sel.dead and not sel.chosen
    assert not sel.update(b.fork_s + 50.0, 6.5) and sel.log_odds == b.prior_log_odds
    assert sel.update(b.fork_s + 60.0, 6.5)


def test_selector_evidence_matches_the_specified_increment_inside_a_bin(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    for d0, d1, v in ((26.0, 27.0, 3.3), (31.0, 34.5, 6.0), (95.5, 99.0, 1.0), (50.0, 50.2, 11.0)):
        sel = route.BranchSelector(b)
        sel.update(b.fork_s + d0, v)
        sel.update(b.fork_s + d1, v)
        j = int(d1 / 5.0)
        sc = 0.35
        z = (min(v, 9.0) - _loc(j)) / sc
        c3 = math.gamma(2.0) / (math.gamma(1.5) * math.sqrt(3.0 * math.pi))
        llr = math.log(1.0 / 9.0) - (math.log(c3) - 2.0 * math.log1p(z * z / 3.0) - math.log(sc))
        assert abs(sel.log_odds - (b.prior_log_odds + llr * (d1 - d0) / 10.0)) < 1e-12


def test_selector_evidence_does_not_depend_on_the_output_rate(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    step = 0.37

    def v_step(k):
        return 3.5 + 2.5 * math.sin(0.1 * k)

    sel_c, sel_f = route.BranchSelector(b), route.BranchSelector(b)
    trace_c, trace_f = [], []
    for k in range(0, 311):
        sel_c.update(b.fork_s - 5.0 + step * k, v_step(k))
        trace_c.append(sel_c.log_odds)
    for i in range(0, 3101):
        sel_f.update(b.fork_s - 5.0 + step / 10.0 * i, v_step((i + 9) // 10))
        if i % 10 == 0:
            trace_f.append(sel_f.log_odds)
    assert sel_c.log_odds != b.prior_log_odds
    assert max(abs(a - f) for a, f in zip(trace_c, trace_f)) < 1e-9


def test_selector_evidence_counts_only_new_ground_when_d_steps_back(tmp_path):
    b = _fork_map(tmp_path).branches[0]
    mono, jit = route.BranchSelector(b), route.BranchSelector(b)
    ds = [b.fork_s + 20.0 + 0.5 * k for k in range(0, 161)]
    for s in ds:
        mono.update(s, 6.0)
    for k, s in enumerate(ds):
        if k % 3 == 1:
            jit.update(s - 0.8, 6.0)
        jit.update(s, 6.0)
    assert mono.log_odds != b.prior_log_odds
    assert abs(jit.log_odds - mono.log_odds) < 1e-12

def test_estimator_on_the_route_is_bit_identical_with_and_without_branches(tmp_path, monkeypatch):
    monkeypatch.setattr(route, "BRANCH_W_LO", 0.2)
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir(), (tmp_path / "c").mkdir()
    with_b, outs_b, s_b = _run(_fork_map(tmp_path / "a"), lambda t: 2.7)
    _, outs_n, _ = _run(_fork_map(tmp_path / "b", None), lambda t: 2.7)
    _, outs_m, _ = _run(_fork_map(tmp_path / "c", "none-block"), lambda t: 2.7)
    assert with_b._branch is not None and not with_b._branch[2].chosen
    assert s_b[-1] > with_b._branch[2].branch.fork_s + 200.0
    assert _same(outs_b, outs_n) and _same(outs_m, outs_n)
    assert all(e.y == 85000.0 for e in outs_b)


def test_estimator_on_the_route_blends_by_the_branch_weight_and_returns_to_the_route(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    rm = _fork_map(tmp_path / "a")
    est, outs, s_b = _run(rm, lambda t: 2.7, trace=True)
    _, outs_n, _ = _run(_fork_map(tmp_path / "b", None), lambda t: 2.7)
    assert len(outs) == len(outs_n)
    ws = est.trace_w
    off = [i for i, e in enumerate(outs) if e.y != 85000.0]
    assert off and all(route.BRANCH_W_LO <= ws[i] <= route.BRANCH_W_HI for i in off)
    for i in off:
        bx, by, _, _ = rm.branch_pose(0, s_b[i])
        n = outs_n[i]
        assert abs(outs[i].x - (n.x + ws[i] * (bx - n.x))) < 1e-9 and abs(outs[i].y - (n.y + ws[i] * (by - n.y))) < 1e-9
    assert s_b[off[-1]] - rm.branches[0].fork_s < 100.0
    assert _same(outs[off[-1] + 1:], outs_n[off[-1] + 1:])


def test_estimator_follows_the_branch_holds_its_end_and_returns_past_the_dead_end(tmp_path):
    (tmp_path / "a").mkdir(), (tmp_path / "b").mkdir()
    rm = _fork_map(tmp_path / "a")
    b = rm.branches[0]
    est, outs, s_pub = _run(rm, lambda t: 6.5, until_d=190.0, trace=True)
    _, outs_main, _ = _run(_fork_map(tmp_path / "b", None), lambda t: 6.5, until_d=190.0)
    assert all(e.y == 85000.0 for e in outs_main)
    ws = est.trace_w
    on = [i for i, e in enumerate(outs) if e.y != 85000.0]
    assert on and on == list(range(on[0], on[-1] + 1))
    full = [i for i in on if ws[i] > route.BRANCH_W_HI]
    assert full and full == list(range(full[0], on[-1] + 1))
    d_full, d_last = s_pub[full[0]] - b.fork_s, s_pub[on[-1]] - b.fork_s
    assert 25.0 < d_full < 60.0 and 184.0 < d_last <= 185.0
    assert s_pub[on[0]] - b.fork_s > 0.0
    assert _same(outs[:on[0]], outs_main[:on[0]]) and _same(outs[on[-1] + 1:], outs_main[on[-1] + 1:])
    end = rm.branch_pose(0, b.path.length)
    for i in on:
        x, y, _, _ = rm.branch_pose(0, s_pub[i])
        w = 1.0 if ws[i] > route.BRANCH_W_HI else ws[i]
        n = outs_main[i]
        assert abs(outs[i].x - (n.x + w * (x - n.x))) < 1e-9 and abs(outs[i].y - (n.y + w * (y - n.y))) < 1e-9
        if i in full:
            assert abs(outs[i].x - x) < 1e-9 and abs(outs[i].y - y) < 1e-9
            if s_pub[i] > b.path.length:
                assert (outs[i].x, outs[i].y) == (end[0], end[1])
    assert any(s_pub[i] > b.path.length + 30.0 for i in on)
    assert est._branch[2].dead == route.BranchSelector.DEAD_END


def test_relocalisation_resets_the_branch_decision(tmp_path):
    import tram_odometry.estimator as core

    rm = _fork_map(tmp_path)
    est, _, _ = _run(rm, lambda t: 6.5, until_d=60.0)
    sel = est._branch[2]
    assert sel.chosen
    core_to_grid = core.to_grid
    core.to_grid = lambda lat, lon: (lon, lat)
    try:
        est.on_gnss_fix(est.t, 85000.0, 100050.0, 153.0, "master")
    finally:
        core.to_grid = core_to_grid
    assert est._branch[2] is sel and not sel.chosen and sel.log_odds == sel.branch.prior_log_odds


def test_package_branch_model_is_valid_and_separates_route_from_fast_speeds(trips):
    assert len(trips.branches) == 1
    b = trips.branches[0]
    assert b.model and trips.names[b.pid] == "route_east_west.json" and abs(b.fork_s - 5379.525) < 0.01
    with open(os.path.join(MAPS, route.BRANCHES_FILE), encoding="utf-8") as f:
        m = json.load(f)["branches"][0]["model"]
    assert m["kind"] == route.MODEL_KIND and len(m["route_loc"]) == NB and "train" in m["fitted_on"]
    assert all(n >= 4 for n in m["route_n"][5:20])
    loc = m["route_loc"]
    sel = route.BranchSelector(b)
    assert _first(_feed(sel, b.fork_s, lambda d: loc[min(int(max(d, 0.0) / 5.0), NB - 1)], d1=150.0)) is None
    first = _first(_feed(route.BranchSelector(b), b.fork_s, lambda d: 6.5))
    assert first is not None and 25.0 < first < 40.0
