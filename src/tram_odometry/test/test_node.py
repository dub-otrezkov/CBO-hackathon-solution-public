import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tram_odometry.node import GnssWindow, OdometryCore, StampDedup

NS = 1_000_000_000


@dataclass
class Est:
    stamp: float
    v: float = 1.0
    a: float = 0.0
    s: float = 0.0
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    var_v: float = 0.01
    var_s: float = 0.1
    slip: bool = False
    mode: str = "stub"


class StubEstimator:
    def __init__(self, fail_on=None, nan=False):
        self.calls = []
        self.fail_on = fail_on
        self.nan = nan

    def _log(self, name, *args):
        self.calls.append((name,) + args)
        if name == self.fail_on:
            raise RuntimeError("boom")

    def on_wheel(self, bogie, stamp, raw):
        self._log("on_wheel", bogie, stamp, raw)

    def on_controller(self, stamp, position):
        self._log("on_controller", stamp, position)

    def on_gnss_fix(self, stamp, lat, lon, alt, antenna):
        self._log("on_gnss_fix", stamp, antenna)

    def on_gnss_vel(self, stamp, vx, vy, antenna):
        self._log("on_gnss_vel", stamp, antenna)

    def output(self, stamp):
        self._log("output", stamp)
        return Est(stamp=stamp, v=math.nan if self.nan else 1.0)


def test_gnss_window_from_first_message_then_closed_for_good():
    w = GnssWindow(5.0)
    assert w.accept(100.0) and w.accept(104.9) and w.accept(105.0)
    assert not w.accept(105.1) and w.closed
    assert not w.accept(101.0)


def test_gnss_window_zero_disables_gnss_and_ignores_zero_stamps():
    assert not GnssWindow(0.0).accept(100.0)
    w = GnssWindow(5.0)
    assert not w.accept(0.0) and w.start is None


def test_gnss_reaches_estimator_only_inside_window():
    est = StubEstimator()
    core = OdometryCore(est, gnss_init_s=5.0)
    assert core.gnss_fix(100 * NS, 55.0, 37.0, 150.0, "master")
    assert core.gnss_vel(103 * NS, 1.0, 0.0, "rover")
    assert not core.gnss_fix(106 * NS, 55.0, 37.0, 150.0, "master")
    assert [c[0] for c in est.calls] == ["on_gnss_fix", "on_gnss_vel"]
    assert est.calls[1][2] == "rover"


def test_one_output_per_input_stamp():
    est = StubEstimator()
    core = OdometryCore(est, gnss_init_s=5.0)
    t = 1_787_000_000 * NS + 123_456_789
    assert core.wheel("front", t, 36.0) is not None
    assert core.wheel("rear", t, 36.0) is None
    assert core.controller(t + 50_000_000, 3) is not None
    assert core.published == 2 and core.skipped_duplicates == 1
    inputs = [c for c in est.calls if c[0] != "output"]
    assert inputs == [("on_wheel", "front", t * 1e-9, 36.0), ("on_wheel", "rear", t * 1e-9, 36.0),
                      ("on_controller", (t + 50_000_000) * 1e-9, 3)]


def test_zero_stamp_is_never_published():
    core = OdometryCore(StubEstimator(), gnss_init_s=5.0)
    assert core.wheel("front", 0, 10.0) is None and core.published == 0


def test_estimator_exceptions_are_contained():
    errors = []
    core = OdometryCore(StubEstimator(fail_on="on_wheel"), gnss_init_s=5.0, on_error=lambda w, e: errors.append(w))
    assert core.wheel("front", 5 * NS, 10.0) is not None
    core2 = OdometryCore(StubEstimator(fail_on="output"), gnss_init_s=5.0, on_error=lambda w, e: errors.append(w))
    assert core2.controller(6 * NS, 1) is None
    assert errors == ["on_wheel", "output"] and core.errors == 1 and core2.errors == 1


def test_non_finite_estimate_is_not_published():
    core = OdometryCore(StubEstimator(nan=True), gnss_init_s=5.0)
    assert core.wheel("front", 7 * NS, 10.0) is None and core.published == 0


def test_dedup_forgets_old_stamps_beyond_capacity():
    d = StampDedup(size=3)
    assert all(d.first(i) for i in range(5))
    assert d.first(0)
    assert not d.first(4)


def test_position_waits_for_route_or_the_end_of_the_gnss_window():
    core = OdometryCore(StubEstimator(), gnss_init_s=5.0)
    t0 = 1_000 * NS
    relative, on_route = Est(stamp=0, mode="relative/wheels"), Est(stamp=0, mode="route/wheels")
    core.wheel("front", t0, 1.0)
    assert not core.position_ready(t0 + NS, relative)
    assert core.position_ready(t0 + NS, on_route)
    assert core.position_ready(t0 + 6 * NS, relative)
    no_gnss = OdometryCore(StubEstimator(), gnss_init_s=0.0)
    assert no_gnss.position_ready(t0, relative)


def test_late_gnss_keeps_waiting_for_the_route():
    core = OdometryCore(StubEstimator(), gnss_init_s=5.0)
    t0 = 1_000 * NS
    relative = Est(stamp=0, mode="relative/wheels")
    core.wheel("front", t0, 1.0)
    core.gnss_fix(t0 + 4 * NS, 55.0, 37.0, 150.0, "master")
    assert not core.position_ready(t0 + 7 * NS, relative)
    core.gnss_fix(t0 + 10 * NS, 55.0, 37.0, 150.0, "master")
    assert core.position_ready(t0 + 10 * NS, relative)


def _feed_all(core, events):
    out = []
    for kind, stamp, value in events:
        out += core.feed(kind, stamp, value)
    return out


def test_feed_order_within_a_stamp_does_not_matter():
    a, b = StubEstimator(), StubEstimator()
    base = [("cmd", 1 * NS, 3), ("front", 2 * NS, 10.0), ("rear", 2 * NS, 11.0), ("cmd", 3 * NS, 4)]
    swapped = [base[0], base[2], base[1], base[3]]
    out_a, out_b = _feed_all(OdometryCore(a, 0.0), base), _feed_all(OdometryCore(b, 0.0), swapped)
    assert a.calls == b.calls and [s for s, _ in out_a] == [s for s, _ in out_b] == [1 * NS, 2 * NS, 3 * NS]
    wheels = [c[1] for c in a.calls if c[0] == "on_wheel"]
    assert wheels == ["front", "rear"] and sum(c[0] == "output" for c in a.calls) == 3


def test_feed_lone_bogie_waits_for_the_next_wheel_stamp_and_later_inputs_wait_behind_it():
    est = StubEstimator()
    core = OdometryCore(est, 0.0)
    t = NS // 10
    assert [s for s, _ in _feed_all(core, [("front", 10 * t, 1.0), ("rear", 10 * t, 1.0)])] == [10 * t]
    assert core.feed("front", 11 * t, 10.0) == []
    assert core.feed("cmd", 11 * t + t // 2, 3) == []
    out = core.feed("front", 12 * t, 10.0)
    assert [s for s, _ in out] == [11 * t, 11 * t + t // 2]
    assert [c[0] for c in est.calls if c[0] != "output"][-2:] == ["on_wheel", "on_controller"]


def test_feed_releases_everything_held_on_a_clock_reset():
    core = OdometryCore(StubEstimator(), 0.0)
    core.feed("front", 100 * NS, 10.0)
    out = core.feed("rear", 5 * NS, 10.0)
    assert [s for s, _ in out] == [100 * NS]


def test_feed_stops_waiting_during_a_dropout_of_one_bogie():
    core = OdometryCore(StubEstimator(), 0.0)
    out = []
    for k in range(5):
        out.append([s for s, _ in core.feed("rear", (10 + k) * NS // 10, 10.0)])
    assert out == [[], [1 * NS], [11 * NS // 10], [12 * NS // 10], [13 * NS // 10]]


def test_release_held_sends_the_lone_bogie_alone_and_later_complete_stamps_follow():
    core = OdometryCore(StubEstimator(), 0.0)
    t = NS // 10
    assert core.feed("front", 10 * t, 1.0) == [] and core.held_stamp() == 10 * t
    assert core.feed("cmd", 10 * t + t // 2, 3) == []
    out = core.release_held()
    assert [s for s, _ in out] == [10 * t, 10 * t + t // 2] and core.held_stamp() is None
    assert core.feed("front", 11 * t, 1.0) == []
    assert [s for s, _ in core.feed("rear", 11 * t, 1.0)] == [11 * t]


def test_default_order_is_front_then_rear():
    est = StubEstimator()
    _feed_all(OdometryCore(est, 0.0), [("rear", 2 * NS, 11.0), ("front", 2 * NS, 10.0)])
    assert [c[1] for c in est.calls if c[0] == "on_wheel"] == ["front", "rear"]


class LateEstimator(StubEstimator):
    def on_gnss_late(self, stamp, lat, lon, alt, antenna, status):
        self._log("on_gnss_late", stamp, antenna, status)


def test_late_fix_goes_to_the_estimators_rule_only_after_the_window():
    est = LateEstimator()
    core = OdometryCore(est, gnss_init_s=5.0)
    assert core.gnss_fix(100 * NS, 55.0, 37.0, 150.0, "master", 2)
    assert not core.gnss_fix(106 * NS, 55.0, 37.0, 150.0, "master", 2)
    assert not core.gnss_fix(300 * NS, 55.0, 37.0, 150.0, "rover", 1)
    assert [c[0] for c in est.calls] == ["on_gnss_fix", "on_gnss_late", "on_gnss_late"]
    assert est.calls[-1][1:] == (300.0, "rover", 1)
    plain = StubEstimator()
    core = OdometryCore(plain, gnss_init_s=5.0)
    core.gnss_fix(100 * NS, 55.0, 37.0, 150.0, "master", 2)
    assert not core.gnss_fix(300 * NS, 55.0, 37.0, 150.0, "master", 2)
    assert [c[0] for c in plain.calls] == ["on_gnss_fix"]
