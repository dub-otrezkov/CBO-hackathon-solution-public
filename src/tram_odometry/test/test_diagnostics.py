import math
import os
import struct
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tram_odometry.estimator import Estimator
from tram_odometry.node import (DIAG_ERROR, DIAG_MAX_HZ, DIAG_OK, DIAG_STALE, DIAG_WARN, Diagnostics,
                                OdometryCore, _text, diagnostics_period_s)

NS = 1_000_000_000
T0 = 1_787_000_000 * NS
NODE_KEYS = {"gnss_window", "published", "duplicate_stamps_skipped", "core_errors", "input_age_wall_s",
             "wheel_msg_age_front_s", "wheel_msg_age_rear_s"}
SLIP_KEYS = {"longitudinal_slip_front", "longitudinal_slip_rear"}


class _LineRoute:

    names = ["line.json"]

    def localize(self, x, y, heading=None):
        return 0, x, y

    def pose(self, path_id, s):
        return s, 0.0, 0.0, 0.0


def _on_route(**params):
    est = Estimator(params=params or None, route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 100.0, 0.0
    return est


def _feed(core, t_from, t_to, v=lambda t: 8.0, wheel=lambda t, v, b: v, notch=lambda t: 0, sink=None, every=None):
    for k in range(round(t_from * 20), round(t_to * 20)):
        t = k / 20
        ns = T0 + k * 50_000_000
        outs = [(ns, "cmd", core.controller(ns, notch(t)))]
        if k % 2 == 0:
            for b, lag in (("front", 20_000_000), ("rear", 25_000_000)):
                raw = wheel(t, v(t), b)
                if raw is not None:
                    outs.append((ns + lag, b, core.wheel(b, ns + lag, raw * 3.6)))
        for out in outs:
            if sink is not None:
                sink(*out)
        if every is not None:
            every(t)


def test_normal_state_is_ok_and_passes_the_core_dict_through():
    core = OdometryCore(_on_route(), gnss_init_s=0.0, diagnostics=True)
    _feed(core, 0.0, 20.0)
    level, message, values, stamp = Diagnostics().build(core, now=100.0)
    kv, d = dict(values), core.est.diagnostics()
    assert level == DIAG_OK, message
    assert set(kv) == (set(d) - {"last_fix_age_s"}) | NODE_KEYS | SLIP_KEYS
    assert all(kv[k] == _text(v, Diagnostics.DIGITS.get(k, 4)) for k, v in d.items() if k in kv)
    assert kv["mode"] == "route/wheels" and kv["slip_front"] == kv["slip_rear"] == kv["slip"] == "false"
    assert kv["path_id"] == "0" and kv["localized"] == "true" and abs(float(kv["s_route"]) - 260.0) < 2.0
    assert abs(float(kv["v"]) - 8.0) < 0.1 and float(kv["weight_front"]) > 0.9
    assert abs(float(kv["longitudinal_slip_front"])) < 0.02 and float(kv["wheel_msg_age_rear_s"]) < 0.1
    assert kv["gnss_window"] == "disabled" and kv["core_errors"] == "0" and kv["route_fixes_stop"] == "0"
    assert stamp == core.last_input_ns and stamp > T0


def test_single_bogie_spin_is_a_warn_slip_episode():
    spin = lambda t, v, b: v + (3.0 if (b == "front" and 8.0 < t < 12.0) else 0.0)
    core = OdometryCore(_on_route(), gnss_init_s=0.0, diagnostics=True)
    diag, seen = Diagnostics(), []

    def tick(t):
        if 9.0 < t < 11.0:
            seen.append(diag.build(core, now=t))

    _feed(core, 0.0, 11.0, v=lambda t: 6.0, wheel=spin, every=tick)
    warn = [(msg, dict(vals)) for lvl, msg, vals, _ in seen if lvl == DIAG_WARN]
    assert warn, [m for _, m, _, _ in seen]
    msg, kv = warn[-1]
    assert "slip episode: front" in msg and kv["slip_front"] == "true" and kv["slip"] == "true"
    assert float(kv["weight_front"]) <= 0.05
    assert float(kv["longitudinal_slip_front"]) > 0.3
    assert abs(float(kv["longitudinal_slip_rear"])) < 0.05


def test_silent_bogie_then_no_wheel_speed_is_warn():
    core, diag = OdometryCore(_on_route(), gnss_init_s=0.0, diagnostics=True), Diagnostics()
    _feed(core, 0.0, 10.0)
    _feed(core, 10.0, 12.0, wheel=lambda t, v, b: None if b == "rear" else v)
    level, message, values, _ = diag.build(core, now=12.0)
    kv = dict(values)
    assert level == DIAG_WARN and message == "no fresh message: rear" and kv["mode"] == "route/wheels"
    assert abs(float(kv["wheel_msg_age_rear_s"]) - 2.0) < 0.1 and "longitudinal_slip_rear" not in kv
    _feed(core, 12.0, 14.0, wheel=lambda t, v, b: None)
    level, message, values, _ = diag.build(core, now=14.0)
    kv = dict(values)
    assert level == DIAG_WARN and "model only" in message and kv["mode"] == "route/model"
    assert not SLIP_KEYS & set(kv)


def test_diagnostics_rate_is_capped_and_bad_values_disable_it():
    assert diagnostics_period_s(2.0) == 0.5 and diagnostics_period_s(2) == 0.5
    assert diagnostics_period_s(1000.0) == diagnostics_period_s(math.inf) == 1.0 / DIAG_MAX_HZ
    for bad in (0, 0.0, -1.0, math.nan, "x", None):
        assert diagnostics_period_s(bad) is None


def test_relative_position_after_the_window_is_warn():
    core = OdometryCore(Estimator(), gnss_init_s=0.0, diagnostics=True)
    _feed(core, 0.0, 3.0)
    level, message, values, _ = Diagnostics().build(core, now=3.0)
    kv = dict(values)
    assert level == DIAG_WARN and "not on the route" in message
    assert kv["mode"] == "relative/wheels" and kv["localized"] == "false" and kv["path_id"] == "-1"
    assert "s_route" not in kv


def test_stale_before_the_first_estimate_and_after_inputs_stop():
    core = OdometryCore(_on_route(), gnss_init_s=5.0, diagnostics=True)
    diag = Diagnostics()
    level, message, values, stamp = diag.build(core, now=0.0)
    assert level == DIAG_STALE and stamp == 0 and dict(values)["gnss_window"] == "waiting"
    _feed(core, 0.0, 3.0)
    assert diag.build(core, now=10.0)[0] == DIAG_OK
    level, message, values, _ = diag.build(core, now=12.5)
    assert level == DIAG_STALE and "no input" in message and dict(values)["input_age_wall_s"] == "2.5"


def test_core_errors_are_error_then_clear():
    class Failing(Estimator):
        fail = False

        def on_controller(self, stamp, position):
            if self.fail:
                raise RuntimeError("boom")
            super().on_controller(stamp, position)

    est = Failing(route=_LineRoute())
    est.path_id, est.route_s_ref, est.s_ref = 0, 0.0, 0.0
    core, diag = OdometryCore(est, gnss_init_s=0.0, diagnostics=True), Diagnostics()
    _feed(core, 0.0, 2.0)
    est.fail = True
    _feed(core, 2.0, 2.5)
    level, message, values, _ = diag.build(core, now=10.0)
    assert level == DIAG_ERROR and dict(values)["core_errors"] == str(core.errors) and core.errors > 0
    est.fail = False
    _feed(core, 2.5, 3.0)
    assert diag.build(core, now=16.0)[0] == DIAG_OK


class _Stub:

    def on_wheel(self, bogie, stamp, raw):
        pass

    def on_controller(self, stamp, position):
        pass

    def output(self, stamp):
        return SimpleNamespace(v=1.0, x=0.0, y=0.0, z=0.0, mode="route/wheels")


def test_missing_or_broken_core_diagnostics_never_raise():
    class Raising(_Stub):
        def diagnostics(self):
            raise RuntimeError("boom")

    class Garbage(_Stub):
        def diagnostics(self):
            return {"mode": 5, "v": "x", "v_front": None, "wheel_scale": 0.0, "slip_front": "yes", "localized": None}

    for est in (_Stub(), Raising(), Garbage(), object()):
        core = OdometryCore(est, gnss_init_s=5.0, diagnostics=True)
        _feed(core, 0.0, 1.0)
        level, message, values, stamp = Diagnostics().build(core, now=1.0)
        keys = {k for k, _ in values}
        assert level in (DIAG_OK, DIAG_WARN, DIAG_STALE, DIAG_ERROR) and stamp == core.last_input_ns > T0
        assert NODE_KEYS <= keys and not SLIP_KEYS & keys
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in values)
        if isinstance(est, Garbage):
            assert level == DIAG_OK and dict(values)["mode"] == "5"
        elif isinstance(est, _Stub):
            assert level == DIAG_OK and keys == NODE_KEYS


def _bits(e):
    return tuple(struct.pack("<d", v).hex() if isinstance(v, float) else v for v in vars(e).values())


def _run(diagnostics):
    core = OdometryCore(_on_route(), gnss_init_s=5.0, diagnostics=diagnostics)
    diag, out, levels = Diagnostics(), [], set()
    v = lambda t: min(10.0, 0.8 * t) if t < 45 else max(0.0, 10.0 - (t - 45.0))
    wheel = lambda t, v, b: (None if b == "rear" and 30 < t < 33 else
                             v + (2.5 if b == "front" and 20 < t < 23 else 0.0))

    def sink(ns, src, e):
        ready = None if e is None else core.position_ready(ns, e)
        out.append((ns, src, None if e is None else _bits(e), ready))

    def tick(t):
        if diagnostics and round(t * 20) % 10 == 0:
            levels.add(diag.build(core, now=t)[0])
        if round(t * 20) == 800:
            core.est.on_route_fix(core.est.t, core.est._route_s() + 3.0, 4.0)

    _feed(core, 0.0, 60.0, v=v, wheel=wheel, notch=lambda t: 8 if t < 12 else (0 if t < 45 else -9), sink=sink,
          every=tick)
    return out, core, levels


def test_result_topics_are_bit_exact_with_diagnostics_on_or_off():
    off, core_off, _ = _run(False)
    on, core_on, levels = _run(True)
    assert len(on) == len(off) > 2300 and sum(e is not None for _, _, e, _ in on) > 2300
    assert on == off
    assert core_on.published == core_off.published and {DIAG_OK, DIAG_WARN} <= levels
