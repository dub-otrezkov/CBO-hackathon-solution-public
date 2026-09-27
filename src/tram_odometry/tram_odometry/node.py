from __future__ import annotations

import dataclasses
import math
import os
import signal
import time
from collections import deque
from typing import Any, Callable

FRONT = "/vehicle/front_bogie_velocity"
REAR = "/vehicle/rear_bogie_velocity"
CONTROLLER = "/vehicle/driver_position_cmd"
OUT_VELOCITY = "/result/velocity"
OUT_POSITION = "/result/position"
OUT_DIAGNOSTICS = "/diagnostics"
DIAG_NAME = "tram_odometry/estimator"
DIAG_OK, DIAG_WARN, DIAG_ERROR, DIAG_STALE = 0, 1, 2, 3
DIAG_MAX_HZ = 10.0
BOGIES = ("front", "rear")
ANTENNAS = ("master", "rover")
ESTIMATOR_PREFIX = "estimator."


class GnssWindow:

    def __init__(self, seconds: float):
        self.seconds = float(seconds)
        self.start: float | None = None
        self.closed = self.seconds <= 0.0

    def accept(self, stamp: float) -> bool:
        if self.closed or not math.isfinite(stamp) or stamp <= 0.0:
            return False
        if self.start is None:
            self.start = stamp
        if stamp - self.start > self.seconds:
            self.closed = True
            return False
        return stamp >= self.start


class StampDedup:

    def __init__(self, size: int = 1024):
        self._order: deque[int] = deque()
        self._seen: set[int] = set()
        self.size = size

    def first(self, stamp_ns: int) -> bool:
        if stamp_ns in self._seen:
            return False
        self._seen.add(stamp_ns)
        self._order.append(stamp_ns)
        if len(self._order) > self.size:
            self._seen.discard(self._order.popleft())
        return True


WHEEL_ORDER = {"front_first": ("front", "rear"), "rear_first": ("rear", "front")}
HOLD_BEHIND = 3
DROPOUT_NS = 1_000_000_000
BACKSTEP_NS = 1_000_000_000
HOLD_WALL_S = 0.05
HOLD_CHECK_S = 0.01
FILL_PERIOD_S = 0.05
FILL_ACCEL_S = 1.0


class OdometryCore:
    def __init__(self, estimator: Any, gnss_init_s: float, on_error: Callable[[str, Exception], None] | None = None,
                 diagnostics: bool = False, wheel_order: str = "front_first"):
        self.est = estimator
        self._rank = {k: i for i, k in enumerate(WHEEL_ORDER[wheel_order] + ("cmd",))}
        self._held: list[tuple[int, int, str, Any, bool]] = []
        self._newest_wheel_ns = 0
        self._last_wheel_stamp: dict[str, int | None] = {b: None for b in BOGIES}
        self.window = GnssWindow(gnss_init_s)
        self.dedup = StampDedup()
        self.on_error = on_error or (lambda where, exc: None)
        self.errors = 0
        self.first_input_ns: int | None = None
        self.published = 0
        self.skipped_duplicates = 0
        self.diagnostics = diagnostics
        self.last_input_ns: int | None = None
        self.last_wheel_ns: dict[str, int | None] = {b: None for b in BOGIES}

    def _call(self, where: str, fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as exc:
            self.errors += 1
            self.on_error(where, exc)
            return None

    def position_ready(self, stamp_ns: int, est: Any) -> bool:
        waited = (self.window.start is None and self.first_input_ns is not None
                  and (stamp_ns - self.first_input_ns) * 1e-9 > self.window.seconds)
        return str(getattr(est, "mode", "")).startswith("route") or self.window.closed or waited

    def _estimate(self, stamp_ns: int) -> Any:
        if stamp_ns <= 0:
            return None
        if self.first_input_ns is None or stamp_ns < self.first_input_ns:
            self.first_input_ns = stamp_ns
        if not self.dedup.first(stamp_ns):
            self.skipped_duplicates += 1
            return None
        est = self._call("output", lambda: self.est.output(stamp_ns * 1e-9))
        if est is None or not all(math.isfinite(getattr(est, k)) for k in ("v", "x", "y", "z")):
            return None
        self.published += 1
        return est

    def _apply(self, kind: str, stamp_ns: int, value: Any) -> None:
        if kind == "cmd":
            self._call("on_controller", lambda: self.est.on_controller(stamp_ns * 1e-9, int(value)))
        else:
            self._call("on_wheel", lambda: self.est.on_wheel(kind, stamp_ns * 1e-9, value))
        if self.diagnostics and stamp_ns > 0:
            self.last_input_ns = stamp_ns
            if kind in self.last_wheel_ns:
                self.last_wheel_ns[kind] = stamp_ns

    def wheel(self, bogie: str, stamp_ns: int, raw: float) -> Any:
        self._apply(bogie, stamp_ns, raw)
        return self._estimate(stamp_ns)

    def controller(self, stamp_ns: int, position: int) -> Any:
        self._apply("cmd", stamp_ns, position)
        return self._estimate(stamp_ns)

    def feed(self, kind: str, stamp_ns: int, value: Any) -> list[tuple[int, Any]]:
        out = []
        if kind in BOGIES and self._held and stamp_ns < self._newest_wheel_ns - BACKSTEP_NS:
            out = self._release(force=True)
            self._newest_wheel_ns = 0
            self._last_wheel_stamp = {b: None for b in BOGIES}
        dropout = False
        if kind in BOGIES:
            self._newest_wheel_ns = max(self._newest_wheel_ns, stamp_ns)
            other = self._last_wheel_stamp[BOGIES[1] if kind == BOGIES[0] else BOGIES[0]]
            prev, self._last_wheel_stamp[kind] = self._last_wheel_stamp[kind], stamp_ns
            dropout = prev is not None and (other is None or prev - other > DROPOUT_NS)
        self._held.append((stamp_ns, self._rank[kind], kind, value, dropout))
        self._held.sort(key=lambda h: (h[0], h[1]))
        return out + self._release()

    def held_stamp(self) -> int | None:
        return self._held[0][0] if self._held else None

    def release_held(self) -> list[tuple[int, Any]]:
        return self._release(force_first=True)

    def _release(self, force: bool = False, force_first: bool = False) -> list[tuple[int, Any]]:
        out = []
        while self._held:
            stamp = self._held[0][0]
            n = next((i for i, h in enumerate(self._held) if h[0] != stamp), len(self._held))
            kinds = {h[2] for h in self._held[:n]}
            lone = kinds & set(BOGIES) and not set(BOGIES) <= kinds
            dropout = any(h[4] for h in self._held[:n])
            behind = len(self._held) - n
            if not (force or force_first) and lone and (behind < 1 if dropout else
                                       self._newest_wheel_ns <= stamp and behind < HOLD_BEHIND):
                break
            force_first = False
            group, self._held = self._held[:n], self._held[n:]
            for _, _, kind, value, _ in group:
                self._apply(kind, stamp, value)
            est = self._estimate(stamp)
            if est is not None:
                out.append((stamp, est))
        return out

    def gnss_fix(self, stamp_ns: int, lat: float, lon: float, alt: float, antenna: str, status: int = 0) -> bool:
        if not self.window.accept(stamp_ns * 1e-9):
            late = getattr(self.est, "on_gnss_late", None) if self.window.closed else None
            if late is not None:
                self._call("on_gnss_late", lambda: late(stamp_ns * 1e-9, lat, lon, alt, antenna, status))
            return False
        self._call("on_gnss_fix", lambda: self.est.on_gnss_fix(stamp_ns * 1e-9, lat, lon, alt, antenna))
        return True

    def gnss_vel(self, stamp_ns: int, vx: float, vy: float, antenna: str) -> bool:
        if not self.window.accept(stamp_ns * 1e-9):
            return False
        self._call("on_gnss_vel", lambda: self.est.on_gnss_vel(stamp_ns * 1e-9, vx, vy, antenna))
        return True


def _read(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except Exception:
        return None


def _text(value: Any, digits: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}" if math.isfinite(value) else None
    return str(value)


def diagnostics_period_s(hz: Any) -> float | None:
    try:
        hz = float(hz)
    except (TypeError, ValueError):
        return None
    if math.isnan(hz) or hz <= 0.0:
        return None
    return 1.0 / min(hz, DIAG_MAX_HZ)


def window_state(core: OdometryCore) -> str:
    w = core.window
    if w.seconds <= 0.0:
        return "disabled"
    t_in = core.last_input_ns * 1e-9 if core.last_input_ns else None
    if w.closed or (w.start is not None and t_in is not None and t_in - w.start > w.seconds):
        return "closed"
    if w.start is not None:
        return "open"
    first = core.first_input_ns * 1e-9 if core.first_input_ns else None
    return "none" if t_in is not None and first is not None and t_in - first > w.seconds else "waiting"


class Diagnostics:

    STALE_S = 2.0
    ERROR_HOLD_S = 5.0
    FRESH_S = 0.5
    V_MIN_MPS = 0.5
    DIGITS = {"v": 3, "v_front": 3, "v_rear": 3, "s_route": 2, "last_fix_age_s": 2, "weight_front": 3,
              "weight_rear": 3, "wheel_scale": 5, "var_v": 6, "scale_var": 8}

    def __init__(self) -> None:
        self._sig: Any = None
        self._sig_wall = 0.0
        self._errors = 0
        self._error_wall: float | None = None

    def build(self, core: OdometryCore, now: float) -> tuple[int, str, list[tuple[str, str]], int]:
        values: list[tuple[str, str]] = []

        def put(key: str, value: Any, digits: int = 4) -> None:
            text = _text(value, digits)
            if text is not None:
                values.append((key, text))

        d = _read(lambda: dict(core.est.diagnostics())) or {}
        for key, value in d.items():
            put(str(key), value, self.DIGITS.get(key, 4))
        mode = d.get("mode") if isinstance(d.get("mode"), str) else ""

        stale = []
        for b in BOGIES:
            age = self._age_s(core, b)
            put(f"wheel_msg_age_{b}_s", age, 2)
            if age is None:
                continue
            if age >= self.FRESH_S:
                stale.append(b)
            elif not mode.endswith("/model"):
                put(f"longitudinal_slip_{b}", _read(lambda: self._slip_ratio(d, b)))
        put("gnss_window", window_state(core))
        put("published", core.published)
        put("duplicate_stamps_skipped", core.skipped_duplicates)
        put("core_errors", core.errors)

        sig = (core.last_input_ns, core.published, core.skipped_duplicates)
        if sig != self._sig:
            self._sig, self._sig_wall = sig, now
        silence = max(0.0, now - self._sig_wall)
        put("input_age_wall_s", silence, 1)

        if core.errors > self._errors:
            self._errors, self._error_wall = core.errors, now
        if self._error_wall is not None and now - self._error_wall <= self.ERROR_HOLD_S:
            level, message = DIAG_ERROR, f"core errors: {core.errors}"
        elif core.published == 0:
            level, message = DIAG_STALE, "no estimate yet"
        elif silence > self.STALE_S:
            level, message = DIAG_STALE, f"no input for {silence:.1f} s"
        else:
            warn = []
            slipping = [b for b in BOGIES if d.get(f"slip_{b}") is True]
            if slipping:
                warn.append("slip episode: " + ", ".join(slipping))
            elif d.get("slip") is True:
                warn.append("wheel trust reduced")
            if mode.endswith("/model"):
                warn.append("model only: no fresh wheel speed")
            elif stale:
                warn.append("no fresh message: " + ", ".join(stale))
            if d.get("localized") is False and core.position_ready(core.last_input_ns or 0, None):
                warn.append("position relative to the start: not on the route")
            level = DIAG_WARN if warn else DIAG_OK
            message = "; ".join(warn) if warn else f"ok ({mode or 'no core diagnostics'})"
        return level, message, values, core.last_input_ns or 0

    @staticmethod
    def _age_s(core: OdometryCore, bogie: str) -> float | None:
        t = core.last_wheel_ns[bogie]
        if t is None or core.last_input_ns is None or core.last_input_ns < t:
            return None
        return (core.last_input_ns - t) * 1e-9

    def _slip_ratio(self, d: dict, bogie: str) -> float:
        v = float(d["v"])
        return (float(d[f"v_{bogie}"]) / float(d["wheel_scale"]) - v) / max(v, self.V_MIN_MPS)


def stamp_ns_of(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_ns(msg: Any) -> int:
    return stamp_ns_of(msg.header.stamp)


def default_maps_dir() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("tram_odometry"), "maps")
    except Exception:
        return ""
    return path if os.path.isdir(path) and any(n.endswith(".json") for n in os.listdir(path)) else ""


def load_route(maps_dir: str, log: Callable[[str], None]) -> Any:
    if not maps_dir:
        log("no route maps found: position is published as relative distance along x")
        return None
    try:
        from .route import RouteMap
    except ImportError:
        log("tram_odometry.route is not available yet: position is published as relative distance along x")
        return None
    return RouteMap.load(maps_dir)


def main(args: list[str] | None = None) -> None:
    import rclpy
    from geometry_msgs.msg import TwistStamped
    from rclpy.clock import Clock, ClockType
    from nav_msgs.msg import Odometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import NavSatFix
    from tram_vehicle_msgs.msg import VelocitySensor
    try:
        from tram_vehicle_msgs.msg import DriverControllerCommand
    except ImportError:
        DriverControllerCommand = None

    try:
        from .estimator import Estimator
    except ImportError as exc:
        raise SystemExit(f"tram_odometry: the core estimator module tram_odometry/estimator.py is missing ({exc})") from exc

    class OdometryNode(Node):
        def __init__(self) -> None:
            super().__init__("tram_odometry", automatically_declare_parameters_from_overrides=True)
            p = self._param
            self.frame_id = p("frame_id", "map")
            self.child_frame_id = p("child_frame_id", "base_link")
            self.velocity_frame_id = p("velocity_frame_id", "base_link")
            gnss_init_s = float(p("gnss_init_s", 5.0))
            maps_dir = p("maps_dir", "") or default_maps_dir()
            depth = int(p("qos_depth", 100))

            params = {name: prm.value for name, prm in self.get_parameters_by_prefix(ESTIMATOR_PREFIX.rstrip(".")).items()}
            route = load_route(maps_dir, self.get_logger().warning)
            self.core = OdometryCore(Estimator(params, route), gnss_init_s, self._on_error)
            self.get_logger().info(f"estimator params {params or 'defaults'}; gnss_init_s={gnss_init_s}; "
                                   f"maps={'loaded from ' + maps_dir if route is not None else 'none'}")
            self._setup_diagnostics(p("diagnostics_hz", 2.0))

            sub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=depth, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub_v = self.create_publisher(VelocitySensor, OUT_VELOCITY, 10)
            self.pub_p = self.create_publisher(Odometry, OUT_POSITION, 10)
            self.create_subscription(VelocitySensor, FRONT, lambda m: self._wheel("front", m), sub_qos)
            self.create_subscription(VelocitySensor, REAR, lambda m: self._wheel("rear", m), sub_qos)
            if DriverControllerCommand is not None:
                self.create_subscription(DriverControllerCommand, CONTROLLER, self._controller, sub_qos)
            else:
                self.get_logger().warning(f"tram_vehicle_msgs has no DriverControllerCommand: {CONTROLLER} not used, "
                                          "wheels only (the traction model gets no controller position); "
                                          f"predictions every {FILL_PERIOD_S} s between wheel samples")
                self.create_timer(FILL_PERIOD_S, self._fill, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.gnss_subs, self.fix_subs = [], []
            if gnss_init_s > 0:
                for ant in ANTENNAS:
                    self.fix_subs.append(self.create_subscription(
                        NavSatFix, f"/sensing/gnss/{ant}/fix", lambda m, a=ant: self._fix(a, m), sub_qos))
                    self.gnss_subs.append(self.create_subscription(
                        TwistStamped, f"/sensing/gnss/{ant}/vel", lambda m, a=ant: self._vel(a, m), sub_qos))
            self._velocity_msg, self._odometry_msg = VelocitySensor, Odometry
            self._last_out = None
            self._hold = None
            self.create_timer(HOLD_CHECK_S, self._release_held, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.create_timer(30.0, self._report)

        def _setup_diagnostics(self, hz: Any) -> None:
            self.diag = None
            period = diagnostics_period_s(hz)
            if period is None:
                return
            try:
                from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
            except ImportError as exc:
                self.get_logger().warning(f"diagnostic_msgs is not available ({exc}): no {OUT_DIAGNOSTICS}")
                return
            self._diag_types = (DiagnosticArray, DiagnosticStatus, KeyValue)
            self.diag = Diagnostics()
            self.core.diagnostics = True
            self.pub_d = self.create_publisher(DiagnosticArray, OUT_DIAGNOSTICS, 10)
            self.create_timer(period, self._diagnostics, clock=Clock(clock_type=ClockType.STEADY_TIME))
            self.get_logger().info(f"{OUT_DIAGNOSTICS} at {1.0 / period:g} Hz (diagnostics_hz={hz!r})")

        def _diagnostics(self) -> None:
            try:
                level, message, values, stamp = self.diag.build(self.core, time.monotonic())
                array_t, status_t, kv_t = self._diag_types
                msg = array_t()
                msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(int(stamp), 1_000_000_000)
                st = status_t()
                st.level = bytes((level,))
                st.name = DIAG_NAME
                st.message = message
                st.hardware_id = self.get_name()
                st.values = [kv_t(key=k, value=v) for k, v in values]
                msg.status = [st]
                self.pub_d.publish(msg)
            except Exception as exc:
                self.get_logger().error(f"diagnostics failed: {exc!r}", throttle_duration_sec=5.0)

        def _param(self, name: str, default: Any) -> Any:
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value

        def _on_error(self, where: str, exc: Exception) -> None:
            self.get_logger().error(f"estimator.{where} failed: {exc!r}", throttle_duration_sec=5.0)

        def _report(self) -> None:
            c = self.core
            self.get_logger().info(f"published {c.published}, duplicate stamps skipped {c.skipped_duplicates}, "
                                   f"estimator errors {c.errors}, gnss window {'closed' if c.window.closed else 'open'}")

        def _wheel(self, bogie: str, msg: Any) -> None:
            self._feed(msg.header.stamp, bogie, float(msg.velocity))

        def _controller(self, msg: Any) -> None:
            self._feed(msg.header.stamp, "cmd", int(msg.position))

        def _feed(self, stamp_msg: Any, kind: str, value: Any) -> None:
            self._emit(type(stamp_msg), self.core.feed(kind, stamp_ns_of(stamp_msg), value))

        def _emit(self, stamp_type: Any, out: list) -> None:
            for stamp, est in out:
                self._publish(stamp_type(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000), est)
            held = self.core.held_stamp()
            if held is None:
                self._hold = None
            elif self._hold is None or self._hold[0] != held:
                self._hold = (held, time.monotonic(), stamp_type)

        def _release_held(self) -> None:
            if self._hold is not None and time.monotonic() - self._hold[1] >= HOLD_WALL_S:
                self._emit(self._hold[2], self.core.release_held())

        def _fix(self, antenna: str, msg: Any) -> None:
            self.core.gnss_fix(stamp_ns(msg), msg.latitude, msg.longitude, msg.altitude, antenna, int(msg.status.status))
            self._close_gnss_if_done()

        def _vel(self, antenna: str, msg: Any) -> None:
            lin = msg.twist.linear
            self.core.gnss_vel(stamp_ns(msg), lin.x, lin.y, antenna)
            self._close_gnss_if_done()

        def _close_gnss_if_done(self) -> None:
            if self.core.window.closed and self.gnss_subs:
                for sub in self.gnss_subs:
                    self.destroy_subscription(sub)
                self.gnss_subs = []
                self.get_logger().info("GNSS alignment window closed: unsubscribed from GNSS speed; fixes go to the "
                                       "estimator's late-fix rule" if hasattr(self.core.est, "on_gnss_late") else
                                       "GNSS alignment window closed: GNSS speed unsubscribed, later fixes ignored")

        def _fill(self) -> None:
            last = self._last_out
            age = time.monotonic() - last[1] if last is not None else 0.0
            if last is None or age < FILL_PERIOD_S:
                return
            stamp, e = last[0] + int(age * 1e9), last[3]
            v0, a, ta = max(0.0, e.v), e.a, min(age, FILL_ACCEL_S)
            if a < 0.0 and v0 + a * ta <= 0.0:
                d, v = v0 * v0 / (-2.0 * a), 0.0
            else:
                v = v0 + a * ta
                d = v0 * ta + 0.5 * a * ta * ta + v * (age - ta)
            yaw = e.yaw if math.isfinite(e.yaw) else 0.0
            d = max(0.0, d)
            est = dataclasses.replace(e, stamp=stamp * 1e-9, s=e.s + d, v=max(0.0, v),
                                      x=e.x + d * math.cos(yaw), y=e.y + d * math.sin(yaw))
            self._publish(last[2](sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000), est, fill=True)

        def _publish(self, stamp: Any, est: Any, fill: bool = False) -> None:
            if est is None:
                return
            if not fill:
                self._last_out = (int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec), time.monotonic(), type(stamp), est)
            v = self._velocity_msg()
            v.header.stamp = stamp
            v.header.frame_id = self.velocity_frame_id
            v.velocity = float(est.v)
            self.pub_v.publish(v)
            if not self.core.position_ready(int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec), est):
                return

            o = self._odometry_msg()
            o.header.stamp = stamp
            o.header.frame_id = self.frame_id
            o.child_frame_id = self.child_frame_id
            o.pose.pose.position.x = float(est.x)
            o.pose.pose.position.y = float(est.y)
            o.pose.pose.position.z = float(est.z)
            yaw = float(est.yaw) if math.isfinite(est.yaw) else 0.0
            o.pose.pose.orientation.z = math.sin(yaw / 2.0)
            o.pose.pose.orientation.w = math.cos(yaw / 2.0)
            var_s = float(est.var_s) if math.isfinite(est.var_s) and est.var_s >= 0 else 0.0
            var_v = float(est.var_v) if math.isfinite(est.var_v) and est.var_v >= 0 else 0.0
            cov = [0.0] * 36
            cov[0] = cov[7] = cov[14] = var_s
            o.pose.covariance = cov
            o.twist.twist.linear.x = float(est.v)
            tcov = [0.0] * 36
            tcov[0] = var_v
            o.twist.covariance = tcov
            self.pub_p.publish(o)

    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for step in (node.destroy_node, rclpy.try_shutdown, lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)):
            try:
                step()
            except (KeyboardInterrupt, ExternalShutdownException):
                pass


if __name__ == "__main__":
    main()
