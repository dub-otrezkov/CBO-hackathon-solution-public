import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tram_odometry.estimator import Estimator
from tram_odometry.model import table_accel

BL_AHEAD = 9.873
BASELINE = 12.436
ANTENNA_H = 3.10


def drive(est, profile, duration, dt=0.05, wheel=lambda t, v, b: v, notch=lambda t: 0, wheel_period=0.1):
    t, next_wheel, outs = 0.0, 0.0, []
    while t < duration:
        est.on_controller(t, notch(t))
        if t >= next_wheel - 1e-9:
            for b in ("front", "rear"):
                raw = wheel(t, profile(t), b)
                if raw is not None:
                    est.on_wheel(b, t, raw * 3.6)
            next_wheel += wheel_period
        e = est.output(t)
        if e:
            outs.append((t, e, profile(t)))
        t = round(t + dt, 6)
    return outs


def ramp(t):
    return min(10.0, 0.8 * t)


def test_clean_wheels_track_speed_and_distance():
    outs = drive(Estimator(), ramp, 30.0, notch=lambda t: 10 if t < 12.5 else 0)
    err = [abs(e.v - v) for t, e, v in outs if t > 1]
    assert max(err) < 0.3
    truth = 0.5 * 0.8 * 12.5**2 + 10.0 * (30.0 - 12.5)
    assert abs(outs[-1][1].s - truth) / truth < 0.01


def test_one_wheel_dropout_keeps_publishing():
    outs = drive(Estimator(), ramp, 20.0, notch=lambda t: 10, wheel=lambda t, v, b: None if (b == "front" and 5 < t < 15) else v)
    assert all(abs(e.v - v) < 0.3 for t, e, v in outs if t > 1)


def test_single_wheel_spin_is_rejected():
    spin = lambda t, v, b: v + (3.0 if (b == "front" and 8 < t < 12) else 0.0)
    outs = drive(Estimator(), lambda t: 6.0, 20.0, notch=lambda t: 0, wheel=spin)
    assert max(abs(e.v - v) for t, e, v in outs if t > 1) < 0.5
    assert any(e.slip for t, e, v in outs if 8 < t < 12)


def test_nonfinite_and_out_of_range_are_ignored():
    bad = lambda t, v, b: float("nan") if b == "rear" and int(t * 10) % 7 == 0 else (500.0 if b == "front" and int(t * 10) % 11 == 0 else v)
    outs = drive(Estimator(), lambda t: 5.0, 10.0, wheel=bad)
    assert all(math.isfinite(e.v) and abs(e.v - 5.0) < 0.3 for t, e, v in outs if t > 1)


def test_agreeing_wheels_win_over_a_wrong_model():
    outs = drive(Estimator(), lambda t: 0.3 * t, 15.0, notch=lambda t: -12)
    assert abs(outs[-1][1].v - outs[-1][2]) < 0.3


def test_backward_stamp_does_not_move_time_back():
    est = Estimator()
    drive(est, lambda t: 5.0, 5.0)
    s = est.kf.s
    est.on_wheel("front", 4.0, 5.0 * 3.6)
    assert est.t >= 4.95 and est.kf.s >= s - 1e-6


def test_table_is_monotone_in_traction_at_low_speed():
    assert table_accel(1, 3.0) < table_accel(5, 3.0) < table_accel(9, 3.0)
    assert table_accel(-7, 5.0) < table_accel(-1, 5.0) < table_accel(0, 5.0) + 0.05


def test_late_stamp_gets_the_state_of_its_time():
    est = Estimator()
    drive(est, lambda t: 5.0, 10.0)
    now = est.output(9.95)
    late = est.output(9.0)
    assert abs((now.s - late.s) - 5.0 * 0.95) < 0.2
    assert est.t >= 9.95 - 1e-9


def test_common_spin_of_both_bogies_is_not_followed():
    spin = lambda t, v, b: v * (1.25 if 10 < t < 14 else 1.0)
    outs = drive(Estimator(), lambda t: 8.0, 20.0, notch=lambda t: 0, wheel=spin)
    assert max(abs(e.v - v) for t, e, v in outs if 10 < t < 16) < 0.8
    assert any(e.slip for t, e, v in outs if 10 < t < 14)


def test_braking_to_a_stop_follows_agreeing_wheels():
    stop = lambda t: max(0.0, 3.0 - 1.7 * max(0.0, t - 5.0))
    outs = drive(Estimator(), stop, 10.0, notch=lambda t: -14 if t > 5 else 0)
    assert abs(outs[-1][1].v) < 0.2


class _LineRoute:

    def localize(self, x, y, heading=None):
        return 0, x, y

    def pose(self, path_id, s):
        return s, 0.0, 0.0, 0.0


def test_route_fixes_estimate_the_wheel_scale():
    est = Estimator(route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    true_v, scale = 10.0, 0.985
    t, next_fix = 0.0, 250.0
    while t < 200.0:
        est.on_controller(t, 0)
        if abs(t * 10 - round(t * 10)) < 1e-6:
            for b in ("front", "rear"):
                est.on_wheel(b, t, true_v * scale * 3.6)
        if true_v * t >= next_fix:
            est.on_route_fix(t, true_v * t, 9.0)
            next_fix += 250.0
        t = round(t + 0.05, 6)
    e = est.output(t)
    assert abs(est.kf.scale - scale) < 0.006
    assert abs(e.v - true_v) < 0.05
    assert abs(e.s - true_v * t) < 5.0


def test_without_route_fixes_the_scale_stays_put():
    est = Estimator()
    drive(est, lambda t: 8.0, 60.0)
    assert est.kf.scale == 1.0


def _stop_at_500(t):
    return 10.0 if t < 45.0 else max(0.0, 10.0 - (t - 45.0))


def test_a_stop_at_a_known_platform_corrects_the_path(tmp_path):
    import json

    stops = tmp_path / "stops.json"
    stops.write_text(json.dumps({"routes": {"line.json": [{"s": 500.0, "std": 1.0}]}}))
    route = _LineRoute()
    route.names = ["line.json"]
    est = Estimator(params={"stops_file": str(stops)}, route=route)
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    drive(est, _stop_at_500, 70.0, wheel=lambda t, v, b: v * 1.02, notch=lambda t: 0 if t < 45 else -9)
    assert abs(est.output(70.0).s - 500.0) < 3.0


def test_a_stop_far_from_every_platform_gives_no_fix(tmp_path):
    import json

    stops = tmp_path / "stops.json"
    stops.write_text(json.dumps({"routes": {"line.json": [{"s": 900.0, "std": 1.0}]}}))
    route = _LineRoute()
    route.names = ["line.json"]
    est = Estimator(params={"stops_file": str(stops)}, route=route)
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    drive(est, _stop_at_500, 70.0, notch=lambda t: 0 if t < 45 else -9)
    assert abs(est.output(70.0).s - 500.0) < 3.0


def test_route_matcher_fixes_reach_the_filter(monkeypatch):
    import tram_odometry.estimator as core

    calls = []

    class Fix:
        def __init__(self, stamp, s_route, var):
            self.stamp, self.s_route, self.var = stamp, s_route, var

    class FakeMatcher:
        def __init__(self, route):
            pass

        def reset(self, path_id, s0):
            calls.append(("reset", path_id, s0))

        def update(self, stamp, vf, vr, v_est, s_est, var_s, path_id=None):
            calls.append(("update", stamp))
            return Fix(stamp, 10.0 * stamp, 1.0) if abs(stamp - 30.0) < 1e-6 else None

        def applied(self, shift):
            calls.append(("applied", shift))

    monkeypatch.setattr(core, "RouteMatcher", FakeMatcher)
    monkeypatch.setattr(core, "to_grid", lambda lat, lon: (lon, lat))
    route = _LineRoute()
    est = Estimator(route=route)
    est.on_controller(0.0, 0)
    est.on_wheel("front", 0.0, 0.0)
    est.on_gnss_fix(0.0, 5.0, 0.0, 0.0, "master")
    assert calls and calls[0][0] == "reset"
    drive(est, lambda t: 10.0, 40.0, wheel=lambda t, v, b: v * 1.03)
    assert any(c[0] == "update" for c in calls)
    assert any(c[0] == "applied" and abs(c[1]) > 5.0 for c in calls)
    assert abs(est.output(40.0).s - 403.0) < 3.0


def test_a_broken_matcher_is_dropped_not_fatal(monkeypatch):
    import tram_odometry.estimator as core

    class Broken:
        def __init__(self, route):
            pass

        def reset(self, path_id, s0):
            pass

        def update(self, *a):
            raise RuntimeError("boom")

    monkeypatch.setattr(core, "RouteMatcher", Broken)
    monkeypatch.setattr(core, "to_grid", lambda lat, lon: (lon, lat))
    est = Estimator(route=_LineRoute())
    est.on_controller(0.0, 0)
    est.on_wheel("front", 0.0, 0.0)
    est.on_gnss_fix(0.0, 5.0, 0.0, 0.0, "master")
    drive(est, lambda t: 5.0, 10.0)
    assert est.matcher is None and abs(est.output(10.0).v - 5.0) < 0.3


def test_a_stamp_far_in_the_future_does_not_hang():
    import time

    est = Estimator()
    drive(est, lambda t: 5.0, 5.0)
    t0 = time.perf_counter()
    est.on_controller(1e9, 0)
    est.on_wheel("front", 1e9 + 0.1, 18.0)
    assert time.perf_counter() - t0 < 0.5
    assert est.output(1e9 + 0.1) is None
    est.on_controller(5.05, 0)
    assert est.output(5.05) is not None


def _aligned(params=None, fixes=(), route=None):
    import tram_odometry.estimator as core

    core_to_grid = core.to_grid
    core.to_grid = lambda lat, lon: (lon, lat)
    try:
        est = Estimator(params=params, route=route or _LineRoute())
        est.on_controller(0.0, 0)
        est.on_wheel("front", 0.0, 0.0)
        for stamp, lat, lon, alt, antenna in fixes:
            est.on_gnss_fix(stamp, lat, lon, alt, antenna)
    finally:
        core.to_grid = core_to_grid
    return est


def test_start_offset_is_published_at_standstill_and_fades_with_path():
    est = _aligned(fixes=[(0.0, 2.0, 100.0, 0.0, "master")])
    e = est.output(0.0)
    assert abs(e.x - (100.0 + BL_AHEAD)) < 1e-6 and abs(e.y - 2.0) < 1e-6
    drive(est, lambda t: 10.0, 20.0)
    assert abs(est.output(20.0).y) < 1e-6


def test_rover_aligns_only_without_master_and_is_shifted_back():
    est = _aligned(fixes=[(0.0, 1.0, 100.0, 0.0, "rover")])
    assert abs(est.route_s_ref - (100.0 - BASELINE)) < 1e-6
    est2 = _aligned(fixes=[(0.0, 1.0, 100.0, 0.0, "master"), (0.1, 1.0, 150.0, 0.0, "rover")])
    assert abs(est2.route_s_ref - 100.0) < 1e-6


def test_one_corrupt_forward_stamp_adds_no_path():
    est = Estimator()
    drive(est, lambda t: 10.0, 60.0)
    s_before = est.kf.s
    est.on_controller(60.0 + 30.0, 0)
    assert est.output(90.0) is None
    t = 60.0
    while t < 62.0:
        est.on_controller(t, 0)
        for b in ("front", "rear"):
            est.on_wheel(b, t, 36.0)
        t = round(t + 0.05, 6)
    assert abs(est.kf.s - (s_before + 20.0)) < 3.0


def test_a_real_pause_is_confirmed_by_the_other_sources():
    est = Estimator()
    drive(est, lambda t: 10.0, 30.0)
    s_before = est.kf.s
    t = 35.0
    while t < 36.0:
        est.on_controller(t, 0)
        for b in ("front", "rear"):
            est.on_wheel(b, t, 36.0)
        t = round(t + 0.05, 6)
    assert abs(est.kf.s - (s_before + 60.0)) < 5.0


def test_output_can_be_shifted_along_the_track():
    est = _aligned(params={"output_along_offset_m": 12.44}, fixes=[(0.0, 1.0, 100.0, 0.0, "master")])
    assert abs(est.output(0.0).x - 112.44) < 1e-6


def _ahead(x, y, yaw, dist):
    return x + dist * math.cos(yaw), y + dist * math.sin(yaw)


def _flat_route_map(tmp_path, z_map=150.0, z_offset=0.0):
    import json

    from tram_odometry.route import RouteMap

    pts = [{"x": 10.0 * i, "y": 10.0, "z": z_map} for i in range(101)]
    doc = {"points": pts, "paths": [{"point_indices": list(range(101))}]}
    (tmp_path / "route_test.json").write_text(json.dumps(doc))
    (tmp_path / "map_params.json").write_text(json.dumps({"z_offset_m": z_offset}))
    return RouteMap.load(str(tmp_path))


def test_published_point_is_base_link_ahead_of_the_master_antenna():
    est = _aligned(fixes=[(0.0, 1.0, 100.0, 0.0, "master"), (0.0, 1.0, 100.0 + BASELINE, 0.0, "rover")])
    e = est.output(0.0)
    assert abs(e.x - (100.0 + BL_AHEAD)) < 1e-6 and abs(e.y - 1.0) < 1e-6
    assert abs(e.s - est.s_ref) < 1e-9
    outs = drive(est, lambda t: 10.0, 30.0)
    for _t, e, _v in outs:
        master_s = est.route_s_ref + e.s - est.s_ref
        assert abs(e.x - (master_s + BL_AHEAD)) < 1e-6
    assert outs[-1][1].x > 100.0 + BL_AHEAD + 250.0


def test_start_z_is_base_link_height_not_the_antenna_height(tmp_path):
    route = _flat_route_map(tmp_path)
    alt = 150.0 + 3.10
    params = {"use_matcher": False}
    est = _aligned(params, fixes=[(0.0, 10.0, 300.0, alt, "master")], route=route)
    assert abs(est.output(0.0).z - (alt - ANTENNA_H)) < 0.05
    as_main = _aligned(dict(params, antenna_height_m=0.0), fixes=[(0.0, 10.0, 300.0, alt, "master")], route=route)
    assert abs(as_main.output(0.0).z - alt) < 0.05
    drive(est, lambda t: 10.0, 50.0)
    assert abs(est.output(50.0).z - 150.0) < 1e-6


def test_start_z_plausibility_guard_includes_the_antenna_height():
    assert abs(_aligned(fixes=[(0.0, 1.0, 100.0, 21.0, "master")]).output(0.0).z - (21.0 - ANTENNA_H)) < 1e-9
    assert _aligned(fixes=[(0.0, 1.0, 100.0, -18.0, "master")]).output(0.0).z == 0.0


def test_start_offset_takes_the_antenna_baseline_heading_at_standstill():
    yaw = math.radians(20.0)
    mx, my = 100.0, 2.0
    rx, ry = _ahead(mx, my, yaw, BASELINE)
    bx, by = _ahead(mx, my, yaw, BL_AHEAD)
    master, rover = (0.0, my, mx, 0.0, "master"), (0.0, ry, rx, 0.0, "rover")
    for fixes in ([master, rover], [rover, master]):
        est = _aligned(fixes=fixes)
        e = est.output(0.0)
        assert abs(e.x - bx) < 1e-6 and abs(e.y - by) < 1e-6
    rover_first = [(0.0, my, mx, 0.0, "master"), (0.2, ry, rx, 0.0, "rover"), (0.1, my, mx, 0.0, "master"),
                   (0.3, ry, rx, 0.0, "rover"), (0.2, my, mx, 0.0, "master")]
    est = _aligned(fixes=rover_first)
    assert abs(est.t - 0.2) < 1e-9
    e = est.output(0.2)
    assert abs(e.x - bx) < 1e-6 and abs(e.y - by) < 1e-6
    e = _aligned(params={"baseline_buffer": 1}, fixes=rover_first).output(0.2)
    assert abs(e.x - (mx + BL_AHEAD)) < 1e-6 and abs(e.y - my) < 1e-6
    est = _aligned(fixes=[master, rover, (0.3, my, mx, 0.0, "master")])
    e = est.output(0.3)
    assert abs(e.x - bx) < 1e-6 and abs(e.y - by) < 1e-6
    along = (mx + BL_AHEAD, my)
    stale = _aligned(fixes=[master, rover, (0.7, my, mx, 0.0, "master")])
    wx, wy = _ahead(mx, my, yaw, BASELINE - 0.5)
    wrong = _aligned(fixes=[master, (0.0, wy, wx, 0.0, "rover")])
    alone = _aligned(fixes=[master])
    for est, t in ((stale, 0.7), (wrong, 0.0), (alone, 0.0)):
        e = est.output(t)
        assert abs(e.x - along[0]) < 1e-6 and abs(e.y - along[1]) < 1e-6
    assert math.hypot(bx - along[0], by - along[1]) > 3.0


def test_zero_lever_arm_reproduces_the_master_antenna_output():
    main = {"output_along_offset_m": 0.0, "antenna_height_m": 0.0, "fallback_offset_m": 12.44}
    yaw = math.radians(20.0)
    rx, ry = _ahead(100.0, 2.0, yaw, BASELINE)
    fixes = [(0.0, 2.0, 100.0, 5.0, "master"), (0.0, ry, rx, 5.0, "rover")]
    with_rover, without = _aligned(main, fixes), _aligned(main, fixes[:1])
    assert with_rover._start_offset == without._start_offset == (0.0, 2.0, 5.0)
    e = with_rover.output(0.0)
    assert (e.x, e.y, e.z) == (100.0, 2.0, 5.0)
    a = drive(with_rover, lambda t: 10.0, 20.0)
    b = drive(without, lambda t: 10.0, 20.0)
    assert [(e.x, e.y, e.z) for _t, e, _v in a] == [(e.x, e.y, e.z) for _t, e, _v in b]
    assert abs(a[-1][1].x - (100.0 + a[-1][1].s - with_rover.s_ref)) < 1e-6
    rover_only = _aligned(main, [(0.0, 1.0, 100.0, 0.0, "rover")])
    assert abs(rover_only.route_s_ref - (100.0 - 12.44)) < 1e-9
    assert abs(rover_only.output(0.0).x - (100.0 - 12.44)) < 1e-9


def test_rover_only_alignment_publishes_base_link():
    est = _aligned(fixes=[(0.0, 1.0, 112.436, 4.0, "rover")])
    assert est._fallback_active and abs(est.route_s_ref - 100.0) < 1e-6
    e = est.output(0.0)
    assert abs(e.x - (112.436 - 2.563)) < 1e-6 and e.y == 0.0 and e.z == 0.0
    outs = drive(est, lambda t: 10.0, 10.0)
    assert all(abs(e.x - (est.route_s_ref + e.s - est.s_ref + BL_AHEAD)) < 1e-6 for _t, e, _v in outs)
    later = _aligned(fixes=[(0.0, 1.0, 112.436, 4.0, "rover"), (0.1, 1.0, 100.5, 4.0, "master"),
                            (0.1, 1.0, 100.5 + BASELINE, 4.0, "rover")])
    assert not later._fallback_active and abs(later.route_s_ref - 100.5) < 1e-6
    e = later.output(0.1)
    assert abs(e.x - (100.5 + BL_AHEAD)) < 1e-6 and abs(e.y - 1.0) < 1e-6 and abs(e.z - (4.0 - ANTENNA_H)) < 1e-6


def test_t063_defaults():
    from tram_odometry.estimator import DEFAULTS

    assert DEFAULTS["scale_epoch_min_m"] == 50.0
    assert DEFAULTS["output_stamp_comp"] is True
    est = Estimator()
    assert est.p["scale_epoch_min_m"] == 50.0 and est.p["output_stamp_comp"] is True


def test_scale_epoch_admits_a_fix_only_far_enough_from_the_last_entry():
    est = Estimator(route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    est.on_controller(0.0, 0)
    for t, s in ((0.0, 0.0), (1.0, 49.9), (2.0, 50.0), (3.0, 60.0), (4.0, 100.0), (5.0, 50.0)):
        est.on_route_fix(t, s, 9.0)
    assert [e[1] for e in est._scale_fixes] == [0.0, 50.0, 100.0, 50.0]
    off = Estimator(params={"scale_epoch_min_m": 0.0}, route=_LineRoute())
    off.path_id, off.route_s_ref, off.s_ref = 0, 0.0, 0.0
    off.on_controller(0.0, 0)
    for t, s in ((0.0, 0.0), (1.0, 49.9), (2.0, 50.0)):
        off.on_route_fix(t, s, 9.0)
    assert [e[1] for e in off._scale_fixes] == [0.0, 49.9, 50.0]


def _dense_fixes(params, every_m=20.0):
    est = Estimator(params=params, route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    true_v, scale = 10.0, 0.985
    t, next_fix, log = 0.0, every_m, []
    while t < 200.0:
        est.on_controller(t, 0)
        if abs(t * 10 - round(t * 10)) < 1e-6:
            for b in ("front", "rear"):
                est.on_wheel(b, t, true_v * scale * 3.6)
        if true_v * t >= next_fix - 1e-9:
            before = (est.kf.scale, est._scale_var, len(est._scale_fixes))
            est.on_route_fix(t, true_v * t, 9.0)
            log.append((before, (est.kf.scale, est._scale_var, len(est._scale_fixes))))
            next_fix += every_m
        t = round(t + 0.05, 6)
    return est, log


def test_scale_epochs_widen_the_scale_base_and_skipped_fixes_keep_w():
    est, log = _dense_fixes({})
    s_buf = [e[1] for e in est._scale_fixes]
    assert all(b - a >= 50.0 for a, b in zip(s_buf, s_buf[1:]))
    skipped = [(b, a) for b, a in log if a[2] == b[2] and len(s_buf) < 40]
    assert skipped and all(a[:2] == b[:2] for b, a in skipped)
    assert abs(est.kf.scale - 0.985) < 0.006
    off, _ = _dense_fixes({"scale_epoch_min_m": 0.0})
    s_off = [e[1] for e in off._scale_fixes]
    assert len(s_off) == 40 and all(abs(b - a - 20.0) < 1e-6 for a, b in zip(s_off, s_off[1:]))
    assert s_buf[-1] - s_buf[0] > 2.0 * (s_off[-1] - s_off[0])


def test_output_stamp_comp_moves_a_fresh_late_stamp_back_to_its_time():
    on, off = Estimator(), Estimator(params={"output_stamp_comp": False})
    for est in (on, off):
        drive(est, lambda t: 5.0, 10.0)
    assert on.kf.s == off.kf.s and on.t == off.t
    t = on.t
    a, b = on.output(t - 0.1), off.output(t - 0.1)
    assert b.s == off.kf.s
    assert abs(a.s - (on.kf.s - on.kf.v * 0.1)) < 1e-9 and abs((b.s - a.s) - 5.0 * 0.1) < 0.02
    assert a.x == a.s
    assert abs(a.v - (b.v - on.a_model * 0.1)) < 1e-9
    for stamp in (t, t - 0.5):
        assert on.output(stamp).s == off.output(stamp).s


def test_diagnostics_expose_slip_and_adaptive_terms():
    est = Estimator()
    spin = lambda t, v, b: v * (1.3 if (b == "front" and 8 < t < 12) else 1.0)
    drive(est, lambda t: 6.0, 10.0, wheel=spin)
    d = est.diagnostics()
    assert set(d) >= {"mode", "slip", "weight_front", "weight_rear", "wheel_scale", "traction_gain", "accel_bias"}
    assert d["slip"] is True and d["weight_front"] < 0.5 and d["mode"].endswith("/slip")
    assert d["slip_front"] is True and d["slip_rear"] is False and abs(d["v_rear"] - 6.0) < 0.1
    assert {"path_id", "s_route", "last_fix_age_s", "v"} <= set(d)


def test_a_stop_on_a_connector_uses_the_platforms_of_the_joined_route(tmp_path):
    import json

    stops = tmp_path / "stops.json"
    stops.write_text(json.dumps({"routes": {"line.json": [{"s": 470.0, "std": 1.0}]}}))

    class ConnectorRoute(_LineRoute):
        names = ["line.json", None]

        def base_of(self, path_id, s=None):
            return (0, 0.0) if path_id == 0 else (0, -30.0)

    est = Estimator(params={"stops_file": str(stops)}, route=ConnectorRoute())
    est.path_id, est.route_s_ref, est.s_ref = 1, 0.0, 0.0
    drive(est, _stop_at_500, 70.0, wheel=lambda t, v, b: v * 1.02, notch=lambda t: 0 if t < 45 else -9)
    assert est.fix_counts["stop"] == 1
    assert abs(est.output(70.0).s - 500.0) < 3.0


def test_scale_pairs_across_a_wheel_outage_are_not_used():
    est = Estimator(route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    true_v, scale = 10.0, 0.985
    t, next_fix = 0.0, 250.0
    while t < 300.0:
        est.on_controller(t, 0)
        outage = 120.0 <= t < 140.0
        if not outage and abs(t * 10 - round(t * 10)) < 1e-6:
            for b in ("front", "rear"):
                est.on_wheel(b, t, true_v * scale * 3.6)
        if true_v * t >= next_fix:
            est.on_route_fix(t, true_v * t, 9.0)
            next_fix += 250.0
        t = round(t + 0.05, 6)
    assert est._scale_epoch >= 1
    assert abs(est.kf.scale - scale) < 0.006


def test_first_speed_keeps_the_covariance_consistent():
    est = Estimator()
    est.on_controller(0.0, 5)
    est.on_gnss_vel(2.0, 3.0, 0.0, est.p["gnss_antenna"])
    t = 0.1
    while t < 10.0:
        for bogie in ("front", "rear"):
            est.on_wheel(bogie, t, 3.0 * 3.6)
        est.on_controller(t, 5)
        t += 0.1
    P = est.kf.P
    for i in range(len(P)):
        assert P[i][i] >= 0.0
        for j in range(len(P)):
            assert abs(P[i][j]) <= math.sqrt(P[i][i] * P[j][j]) + 1e-9


def test_route_fix_without_scale_refit_moves_position_but_not_the_scale():
    est = Estimator(route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    true_v, scale = 10.0, 0.985
    t, next_fix = 0.0, 250.0
    while t < 200.0:
        est.on_controller(t, 0)
        if abs(t * 10 - round(t * 10)) < 1e-6:
            for b in ("front", "rear"):
                est.on_wheel(b, t, true_v * scale * 3.6)
        if true_v * t >= next_fix:
            est.on_route_fix(t, true_v * t, 9.0, refit_scale=False)
            next_fix += 250.0
        t = round(t + 0.05, 6)
    e = est.output(t)
    assert est.kf.scale == 1.0 and not est._scale_fixes
    assert abs(e.s - true_v * t) < 20.0
