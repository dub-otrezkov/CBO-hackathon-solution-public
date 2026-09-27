import json
import math
import os
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass

from .filter import LongitudinalFilter
from .model import TractionModel

try:
    from .preprocess import WheelPreprocessor
except ImportError:
    WheelPreprocessor = None
try:
    from .slip import SlipDetector
except ImportError:
    SlipDetector = None
try:
    from .landmarks import RouteMatcher
except ImportError:
    RouteMatcher = None
try:
    from .route import to_grid
except ImportError:
    to_grid = None

KMH_TO_MPS = 1.0 / 3.6

DEFAULTS = {
    "wheel_scale": KMH_TO_MPS,
    "wheel_var": 0.01,
    "wheel_max_age_s": 0.3,
    "wheel_min_mps": -3.0,
    "wheel_max_mps": 30.0,
    "slip_gate_sigma": 4.0,
    "slip_gate_floor_mps": 0.6,
    "slip_release_s": 1.0,
    "slip_weight": 0.0001,
    "bogie_disagree_mps": 0.8,
    "bogie_agree_mps": 0.4,
    "lockout_s": 1.5,
    "abrupt_rate": 1.5,
    "max_slip_s": 8.0,
    "low_speed_trust_mps": 3.0,
    "wheel_max_accel": 2.5,
    "use_t013": True,
    "slip_episode_skip": True,
    "slip_partner_skip_s": 4.0,
    "jump_accel": 8.0,
    "jump_release_mps": 0.5,
    "jump_max_s": 8.0,
    "gnss_vel_var": 0.01,
    "gnss_antenna": "master",
    "fallback_antenna": "rover",
    "fallback_offset_m": 12.436,
    "output_along_offset_m": 9.873,
    "antenna_height_m": 3.10,
    "baseline_tol_m": 0.3,
    "baseline_max_age_s": 0.5,
    "baseline_buffer": 8,
    "start_offset_xy_m": 160.0,
    "start_offset_z_m": 400.0,
    "stop_speed_mps": 0.1,
    "epoch_reset_s": 2.0,
    "epoch_forward_s": 60.0,
    "confirm_forward_s": 2.0,
    "confirm_window_s": 2.0,
    "confirm_count": 20,
    "substep_s": 0.1,
    "history_s": 3.0,
    "m8_hysteresis": True,
    "wheel_time_align": True,
    "wheel_align_accel_std": 0.3,
    "late_stamp_s": 0.25,
    "output_stamp_comp": True,
    "grade_half_window_m": 10.0,
    "q_accel": 0.05,
    "q_bias": 0.0,
    "q_gain": 0.0,
    "b0_var": 0.04,
    "c0_var": 0.01,
    "w0_var": 0.0,
    "q_scale": 0.0,
    "scale_prior_var": 0.0001,
    "scale_min_span_m": 200.0,
    "scale_epoch_min_m": 50.0,
    "scale_q_per_km": 1e-6,
    "scale_outage_s": 5.0,
    "scale_outage_mode": "epoch",
    "scale_refit_min_entries": 3,
    "use_matcher": True,
    "stops_with_matcher": True,
    "matcher_pair_s": 0.05,
    "stops_file": "",
    "stop_min_s": 5.0,
    "stop_gate_m": 20.0,
    "stop_gate_per_m": 0.015,
    "stop_extra_std_m": 3.0,
    "frame_id": "map",
}


@dataclass
class Estimate:
    stamp: float
    v: float
    a: float
    s: float
    x: float
    y: float
    z: float
    yaw: float
    var_v: float
    var_s: float
    slip: bool
    mode: str


class _SimpleWheels:

    def __init__(self, p):
        self.p = p

    def push(self, bogie, stamp, raw):
        if raw is None or not math.isfinite(raw):
            return None
        v = raw * self.p["wheel_scale"]
        if not self.p["wheel_min_mps"] <= v <= self.p["wheel_max_mps"]:
            return None
        return v


class Estimator:
    def __init__(self, params: dict | None = None, route=None):
        self.p = dict(DEFAULTS)
        self.p.update(params or {})
        self.route = route
        self.model = TractionModel(m8_hysteresis=self.p["m8_hysteresis"])
        self.kf = LongitudinalFilter(q_accel=self.p["q_accel"], q_bias=self.p["q_bias"], q_gain=self.p["q_gain"],
                                     q_scale=self.p["q_scale"], b0_var=self.p["b0_var"], c0_var=self.p["c0_var"],
                                     w0_var=self.p["w0_var"])
        self.detector = None
        if self.p["use_t013"] and WheelPreprocessor is not None and SlipDetector is not None:
            self.wheels = WheelPreprocessor()
            self.detector = SlipDetector()
        else:
            self.wheels = _SimpleWheels(self.p)
        self.episode_start = {}
        self.jump_since = {}
        self.prev_sample = {}
        self.t = None
        self.started = False
        self.latest = {}
        self.slip_since = {}
        self.reject_start = {}
        self.abrupt = {}
        self.prev_innov = {}
        self.prev_wheel = {}
        self.slip = False
        self.a_model = 0.0
        self.path_id = None
        self.route_s_ref = 0.0
        self.s_ref = 0.0
        self._fixes = []
        self._scale_var = self.p["scale_prior_var"]
        self._stops = self._load_stops(self.p["stops_file"])
        self._still_since = None
        self._stop_used = False
        self.matcher = None
        self._matcher_failed = False
        self._paired = (None, None)
        self._last_route_fix = None
        self._u = 0.0
        self._scale_fixes = deque(maxlen=40)
        self._model_only_s = 0.0
        self._scale_reset = False
        self._scale_epoch = 0
        self._start_offset = (0.0, 0.0, 0.0)
        self._align_fix = None
        n_buf = max(1, int(self.p["baseline_buffer"]))
        self._recent_main = deque(maxlen=n_buf)
        self._recent_rover = deque(maxlen=n_buf)
        self._axis = None
        self.wheel_weight = {"front": 1.0, "rear": 1.0}
        self.fix_counts = {"stop": 0, "matcher": 0}
        self._last_fix_stamp = None
        self._last_mode = "init"
        self._main_fix_seen = False
        self._fallback_active = False
        self._history = deque()
        self._fwd_pending = None
        self._branch = None

    @staticmethod
    def _load_stops(path: str) -> dict:
        if not path:
            try:
                from ament_index_python.packages import get_package_share_directory
                path = os.path.join(get_package_share_directory("tram_odometry"), "maps", "stops.json")
            except Exception:
                path = os.path.join(os.path.dirname(__file__), "..", "maps", "stops.json")
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            return {name: [(float(m["s"]), float(m["std"])) for m in marks] for name, marks in doc["routes"].items()}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def _reset_matcher(self) -> None:
        if not self.p["use_matcher"] or RouteMatcher is None or self._matcher_failed or self.path_id is None:
            return
        try:
            if self.matcher is None:
                self.matcher = RouteMatcher(self.route)
            self.matcher.reset(self.path_id, self._route_s())
        except Exception:
            self.matcher, self._matcher_failed = None, True

    def _feed_matcher(self, stamp: float) -> None:
        if self.matcher is None or self.path_id is None:
            return
        f, r = self.latest.get("front"), self.latest.get("rear")
        if f is None or r is None or abs(f[0] - r[0]) > self.p["matcher_pair_s"]:
            return
        if f[0] == self._paired[0] or r[0] == self._paired[1]:
            return
        self._paired = (f[0], r[0])
        since = self.kf.s - (self._last_route_fix[0] if self._last_route_fix else self.s_ref)
        var_s = self.kf.P[0][0] + since * since * self._scale_var
        try:
            fix = self.matcher.update(stamp, f[1], r[1], self.kf.v, self._route_s(), var_s, path_id=self.path_id)
            if fix is not None:
                before = self._route_s()
                self.on_route_fix(fix.stamp, fix.s_route, fix.var)
                self.fix_counts["matcher"] += 1
                self.matcher.applied(self._route_s() - before)
        except Exception:
            self.matcher, self._matcher_failed = None, True

    def _check_stop(self, stamp: float) -> None:
        if self.matcher is not None and not self.p["stops_with_matcher"]:
            return
        standing = self.kf.v < 0.05 and self._wheels_still(stamp)
        if not standing:
            self._still_since, self._stop_used = None, False
            return
        if self._still_since is None:
            self._still_since = stamp
        if self._stop_used or stamp - self._still_since < self.p["stop_min_s"] or self.path_id is None:
            return
        base, shift = self.path_id, 0.0
        base_of = getattr(self.route, "base_of", None)
        if base_of is not None:
            try:
                base, shift = base_of(self.path_id, self._route_s())
            except Exception:
                base, shift = None, 0.0
        names = getattr(self.route, "names", None)
        marks = self._stops.get(names[base]) if base is not None and names and 0 <= base < len(names) else None
        if not marks:
            return
        s_pred = self._route_s() + shift
        since = self.kf.s - (self._last_route_fix[0] if self._last_route_fix else self.s_ref)
        gate = self.p["stop_gate_m"] + self.p["stop_gate_per_m"] * abs(since)
        near = [(abs(m - s_pred), m, sd) for m, sd in marks if abs(m - s_pred) <= gate]
        self._stop_used = True
        if len(near) != 1:
            return
        _, m, sd = near[0]
        self.on_route_fix(stamp, m - shift, sd * sd + self.p["stop_extra_std_m"] ** 2)
        self.fix_counts["stop"] += 1

    def _advance(self, stamp: float, source: str | None = None) -> None:
        if self.t is None:
            self.t = stamp
            return
        dt = stamp - self.t
        if dt > self.p["confirm_forward_s"]:
            if source is None:
                return
            pend = self._fwd_pending
            if pend is not None and pend[0] != source and abs(stamp - pend[1]) <= self.p["confirm_window_s"]:
                self._fwd_pending = None
            elif pend is not None and pend[0] == source and abs(stamp - pend[1]) <= self.p["confirm_window_s"] + 5.0:
                if pend[2] + 1 < self.p["confirm_count"]:
                    self._fwd_pending = (source, stamp, pend[2] + 1)
                    return
                self._fwd_pending = None
            else:
                self._fwd_pending = (source, stamp, 1)
                return
        elif source is not None:
            self._fwd_pending = None
        if dt < 0.0:
            if -dt > self.p["epoch_reset_s"]:
                self.t = stamp
                self._history.clear()
            return
        if dt > self.p["epoch_forward_s"]:
            self.t = stamp
            self._history.clear()
            return
        step = self.p["substep_s"]
        while dt > 1e-9:
            h = min(step, dt)
            v = self.kf.v
            a_table = self.model.table(self.t + h, v)
            a_grade = self.model.grade_accel(self._grade())
            self.a_model = self.kf.accel(a_table, a_grade)
            hold = (v < self.p["stop_speed_mps"] and self.a_model <= 0.0 and self.model.notch_at(self.t + h) <= 0
                    and self._wheels_still(self.t + h))
            v_before = self.kf.v
            if self.p["scale_outage_s"] > 0.0:
                lf, lr = self.latest.get("front"), self.latest.get("rear")
                tn = self.t + h
                if not ((lf is not None and tn - lf[0] < 0.5) or (lr is not None and tn - lr[0] < 0.5)):
                    self._model_only_s += h
            self.kf.predict(h, a_table, a_grade, hold=hold)
            self._u += self.kf.scale * 0.5 * (v_before + self.kf.v) * h
            self.t += h
            dt -= h
            self._history.append((self.t, self.kf.s, self.kf.v))
        while self._history and self._history[0][0] < self.t - self.p["history_s"]:
            self._history.popleft()

    def _route_s(self) -> float:
        return self.route_s_ref + self.kf.s - self.s_ref

    def _grade(self) -> float:
        if self.path_id is None:
            return 0.0
        s = self._route_s()
        half = self.p["grade_half_window_m"]
        z_ahead = self.route.pose(self.path_id, s + half)[2]
        z_behind = self.route.pose(self.path_id, s - half)[2]
        return (z_ahead - z_behind) / (2.0 * half)

    def on_controller(self, stamp: float, position: int) -> None:
        self.model.on_controller(stamp, position)
        self._advance(stamp, "cmd")
        if self.started:
            self._check_stop(stamp)

    def on_wheel(self, bogie: str, stamp: float, raw: float) -> None:
        if self.detector is not None:
            self._on_wheel_trust(bogie, stamp, raw)
            return
        v = self.wheels.push(bogie, stamp, raw)
        self._advance(stamp, bogie)
        if v is None or stamp < self.t - self.p["wheel_max_age_s"]:
            return
        self.latest[bogie] = (stamp, v)
        if not self.started:
            self._init_speed(max(0.0, v / self.kf.scale), self.p["wheel_var"])
            self.started = True
            return
        wgt = self._weight(bogie, stamp, v)
        self.wheel_weight[bogie] = wgt
        r = self.p["wheel_var"] / wgt
        z, r = self._align_wheel(stamp, v, r)
        self.kf.update_wheel(z, r)
        self._feed_matcher(stamp)

    def _on_wheel_trust(self, bogie: str, stamp: float, raw: float) -> None:
        sample = self.wheels.push(bogie, stamp, raw)
        if sample is None:
            return
        self._advance(stamp, bogie)
        det = self.detector
        late = stamp < self.t - self.p["wheel_max_age_s"]
        if late or not self.started:
            det.update(sample, None, None, None)
            self.slip = det.any_slip()
            if late or self.started or not sample.usable:
                return
            self.latest[bogie] = (stamp, sample.v)
            self._init_speed(max(0.0, sample.v / self.kf.scale), self.p["wheel_var"])
            self.started = True
            return
        w_scale = self.kf.scale
        lag = self.t - stamp if self.p["wheel_time_align"] and stamp < self.t else 0.0
        trust = det.update(sample, w_scale * (self.kf.v - self.a_model * lag), w_scale * w_scale * self.kf.P[1][1],
                           self.a_model)
        self.slip = det.any_slip()
        if sample.usable:
            self.latest[bogie] = (stamp, sample.v)
        w = trust.weight
        if sample.usable:
            prev = self.prev_sample.get(bogie)
            self.prev_sample[bogie] = (stamp, sample.v)
            if prev is not None and 0.02 < stamp - prev[0] < 1.0 and abs(sample.v - prev[1]) / (stamp - prev[0]) > self.p["jump_accel"]:
                self.jump_since.setdefault(bogie, stamp)
            t_jump = self.jump_since.get(bogie)
            if t_jump is not None:
                back = abs(sample.v - w_scale * self.kf.v) < self.p["jump_release_mps"]
                if back or stamp - t_jump > self.p["jump_max_s"]:
                    self.jump_since.pop(bogie, None)
                else:
                    w = 0.0
                    self.slip = True
        if trust.slip:
            self.episode_start.setdefault(bogie, stamp)
            if self.p["slip_episode_skip"]:
                w = 0.0
        else:
            self.episode_start.pop(bogie, None)
        t_ep = self.episode_start.get("rear" if bogie == "front" else "front")
        if t_ep is not None and self.p["slip_partner_skip_s"] > 0.0 and stamp - t_ep < self.p["slip_partner_skip_s"]:
            w = 0.0
        self.wheel_weight[bogie] = w
        if w <= 0.0:
            return
        sigma = det.p["meas_sigma0"] + det.p["meas_sigma_rel"] * abs(sample.v)
        z, r = self._align_wheel(stamp, sample.v, sigma * sigma / w)
        self.kf.update_wheel(z, r)
        self._feed_matcher(stamp)

    def diagnostics(self) -> dict:
        kf = self.kf
        det = self.detector
        slip_bogie = {}
        for b in ("front", "rear"):
            if det is not None:
                slip_bogie[b] = b in self.episode_start or b in self.jump_since
            else:
                slip_bogie[b] = b in self.slip_since and self.t is not None and self.t - self.slip_since[b] < self.p["slip_release_s"]
        return {
            "mode": self._last_mode, "slip": bool(self.slip),
            "slip_front": bool(slip_bogie["front"]), "slip_rear": bool(slip_bogie["rear"]),
            "v_front": float(self.latest["front"][1]) if "front" in self.latest else float("nan"),
            "v_rear": float(self.latest["rear"][1]) if "rear" in self.latest else float("nan"),
            "v": max(0.0, kf.v),
            "path_id": -1 if self.path_id is None else int(self.path_id),
            "s_route": float(self._route_s()) if self.path_id is not None else float("nan"),
            "last_fix_age_s": float(self.t - self._last_fix_stamp) if self._last_fix_stamp is not None and self.t is not None else float("nan"),
            "weight_front": float(self.wheel_weight.get("front", 1.0)), "weight_rear": float(self.wheel_weight.get("rear", 1.0)),
            "wheel_scale": kf.scale, "traction_gain": kf.gain, "accel_bias": kf.bias,
            "var_v": kf.P[1][1], "var_s": kf.P[0][0], "scale_var": self._scale_var,
            "route_fixes_stop": self.fix_counts["stop"], "route_fixes_matcher": self.fix_counts["matcher"],
            "matcher_active": self.matcher is not None, "localized": self.path_id is not None,
        }

    def _align_wheel(self, stamp: float, z: float, r: float) -> tuple[float, float]:
        if not self.p["wheel_time_align"] or stamp >= self.t:
            return z, r
        dt = self.t - stamp
        w = self.kf.scale
        return z + w * self.a_model * dt, r + (w * self.p["wheel_align_accel_std"] * dt) ** 2

    def _wheels_still(self, stamp: float) -> bool:
        return all(abs(v) < 2 * self.p["stop_speed_mps"] for t, v in self.latest.values() if stamp - t < 0.5)

    def _plausible(self, bogie: str, stamp: float, v: float) -> bool:
        prev = self.prev_wheel.get(bogie)
        self.prev_wheel[bogie] = (stamp, v)
        if prev is None or not 0.02 < stamp - prev[0] < 1.0:
            return True
        return abs(v - prev[1]) / (stamp - prev[0]) <= self.p["wheel_max_accel"]

    def _weight(self, bogie: str, stamp: float, v: float) -> float:
        p = self.p
        plausible = self._plausible(bogie, stamp, v)
        y, pvv = self.kf.innovation(v)
        prev = self.prev_innov.get(bogie)
        self.prev_innov[bogie] = (stamp, y)
        jump = not plausible or (prev is not None and 0.02 < stamp - prev[0] < 1.0
                                 and abs(y - prev[1]) / (stamp - prev[0]) > p["abrupt_rate"])
        gate = max(p["slip_gate_floor_mps"], p["slip_gate_sigma"] * math.sqrt(pvv + p["wheel_var"]))
        outside = abs(y) > gate
        other = self.latest.get("rear" if bogie == "front" else "front")
        if other and abs(other[0] - stamp) < 0.3 and abs(other[1] - v) > p["bogie_disagree_mps"]:
            outside = outside or abs(y) > abs(other[1] - self.kf.v)
        outside = outside or not plausible
        if outside:
            if bogie not in self.reject_start:
                self.reject_start[bogie] = stamp
                self.abrupt[bogie] = jump
            elif jump and stamp - self.reject_start[bogie] < 1.0:
                self.abrupt[bogie] = True
            start = self.reject_start[bogie]
            agree = other and abs(other[0] - stamp) < 0.3 and abs(other[1] - v) < p["bogie_agree_mps"]
            wait = p["max_slip_s"] if self.abrupt.get(bogie) else p["lockout_s"]
            if max(v, self.kf.v) < p["low_speed_trust_mps"]:
                wait = 0.0
            if plausible and agree and stamp - start > wait:
                self.kf.P[1][1] += y * y
                self.kf.P[2][2] += 0.1
                self.reject_start.pop(bogie, None)
                self.slip_since.pop(bogie, None)
                self.slip = bool(self.slip_since)
                return 1.0
            self.slip_since[bogie] = stamp
        else:
            self.reject_start.pop(bogie, None)
        suspect = bogie in self.slip_since and stamp - self.slip_since[bogie] < p["slip_release_s"]
        self.slip = any(stamp - t0 < p["slip_release_s"] for t0 in self.slip_since.values())
        return p["slip_weight"] if suspect else 1.0

    def on_gnss_fix(self, stamp: float, lat: float, lon: float, alt: float, antenna: str) -> None:
        if to_grid is None or self.route is None:
            return
        if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) < 1.0:
            return
        main = antenna == self.p["gnss_antenna"]
        if not main:
            if antenna != self.p["fallback_antenna"]:
                return
            if self._main_fix_seen:
                rx, ry = to_grid(lat, lon)
                self._pair_baseline(stamp, rx, ry, False)
                if (self._align_fix is not None and self._align_fix[0] == stamp and self.path_id is not None
                        and not self._fallback_active):
                    self._start_offset = self._start_offset_xy() + (self._start_offset[2],)
                return
        else:
            self._main_fix_seen = True
            if self._fallback_active:
                self._fixes, self._fallback_active = [], False
        self._advance(stamp)
        x, y = to_grid(lat, lon)
        self._pair_baseline(stamp, x, y, main)
        self._fixes.append((stamp, x, y))
        heading = None
        first = self._fixes[0]
        if math.hypot(x - first[1], y - first[2]) > 3.0:
            heading = math.atan2(y - first[2], x - first[1])
        loc = self.route.localize(x, y, heading)
        if loc is None:
            return
        path_id, s0, _lateral = loc
        self.path_id, self.s_ref = path_id, self.kf.s
        if main:
            self.route_s_ref = s0
            pz = self.route.pose(path_id, s0)[2]
            h = self.p["antenna_height_m"]
            oz = alt - pz - h if math.isfinite(alt) and abs(alt - pz - h) < 20.0 else 0.0
            self._align_fix = (stamp, x, y)
            self._start_offset = self._start_offset_xy() + (oz,)
        else:
            self.route_s_ref = s0 - self.p["fallback_offset_m"]
            self._align_fix = None
            self._start_offset = (0.0, 0.0, 0.0)
            self._fallback_active = True
        self._scale_fixes.clear()
        self._reset_matcher()
        self._reset_branch()

    def _reset_branch(self) -> None:
        try:
            find = getattr(self.route, "branch_of", None)
            found = find(self.path_id) if find is not None and self.path_id is not None else None
            if found is None:
                self._branch = None
                return
            k, shift = found
            if self._branch is not None and self._branch[0] == k:
                sel = self._branch[2]
                sel.reset()
            else:
                sel = self.route.branch_selector(k)
        except Exception:
            self._branch = None
            return
        self._branch = (k, shift, sel)

    def _pair_baseline(self, stamp: float, x: float, y: float, main: bool) -> None:
        mine, other = (self._recent_main, self._recent_rover) if main else (self._recent_rover, self._recent_main)
        mine.append((stamp, x, y))
        for t, ox, oy in reversed(other):
            if t != stamp:
                continue
            bx, by = (ox - x, oy - y) if main else (x - ox, y - oy)
            length = math.hypot(bx, by)
            if length > 0.0 and abs(length - self.p["fallback_offset_m"]) < self.p["baseline_tol_m"]:
                self._axis = (stamp, bx / length, by / length)
            return

    def _start_offset_xy(self) -> tuple[float, float]:
        stamp, x, y = self._align_fix
        d = self.p["output_along_offset_m"]
        ax = self._axis
        if ax is not None and abs(stamp - ax[0]) <= self.p["baseline_max_age_s"]:
            qx, qy = self.route.pose(self.path_id, self.route_s_ref + d)[:2]
            return x + d * ax[1] - qx, y + d * ax[2] - qy
        px, py = self.route.pose(self.path_id, self.route_s_ref)[:2]
        return x - px, y - py

    def on_route_fix(self, stamp: float, s_route: float, var: float, refit_scale: bool = True) -> None:
        if self.path_id is None or not (math.isfinite(s_route) and math.isfinite(var)) or var <= 0.0:
            return
        self._advance(stamp)
        s_filter = s_route - self.route_s_ref + self.s_ref
        if stamp < self.t:
            s_filter += self.kf.v * (self.t - stamp)
        last = self._last_route_fix
        if refit_scale:
            u_fix = self._u - (self.kf.scale * self.kf.v * (self.t - stamp) if stamp < self.t else 0.0)
            dmin = self.p["scale_epoch_min_m"]
            if self.p["scale_outage_s"] > 0.0 and self._model_only_s >= self.p["scale_outage_s"]:
                if self.p["scale_outage_mode"] == "epoch":
                    self._scale_epoch += 1
                    self._model_only_s = 0.0
                else:
                    self._scale_fixes.clear()
                    self._scale_reset = True
            fresh = dmin <= 0.0 or not self._scale_fixes or abs(s_route - self._scale_fixes[-1][1]) >= dmin
            if fresh:
                self._model_only_s = 0.0
                self._scale_fixes.append((u_fix, s_route, self._scale_epoch))
            slopes = []
            refill = self._scale_reset and len(self._scale_fixes) < self.p["scale_refit_min_entries"]
            if self._scale_reset and not refill:
                self._scale_reset = False
            fixes = list(self._scale_fixes) if fresh and not refill else []
            for i in range(len(fixes)):
                for j in range(i + 1, len(fixes)):
                    ds = fixes[j][1] - fixes[i][1]
                    if fixes[i][2] == fixes[j][2] and abs(ds) >= self.p["scale_min_span_m"]:
                        slopes.append((fixes[j][0] - fixes[i][0]) / ds)
            if slopes:
                slopes.sort()
                n = len(slopes)
                w_ts = slopes[n // 2] if n % 2 else 0.5 * (slopes[n // 2 - 1] + slopes[n // 2])
                lo, hi = self.kf.scale_limits
                w_new = max(lo, min(hi, w_ts))
                self._scale_var = self.p["scale_prior_var"] / (1.0 + n)
                if w_new != self.kf.scale:
                    self.kf.x[1] *= self.kf.scale / w_new
                    self.kf.x[4] = w_new
        since = self.kf.s - (last[0] if last is not None else self.s_ref)
        self.kf.P[0][0] += since * since * self._scale_var
        self.kf.update_position(s_filter, var)
        self._last_route_fix = (self.kf.s, s_route, var)
        self._last_fix_stamp = stamp

    def _init_speed(self, v: float, var: float) -> None:
        self.kf.x[1] = v
        for j in range(len(self.kf.P)):
            self.kf.P[1][j] = self.kf.P[j][1] = 0.0
        self.kf.P[1][1] = var

    def on_gnss_vel(self, stamp: float, vx: float, vy: float, antenna: str) -> None:
        if antenna != self.p["gnss_antenna"] or not (math.isfinite(vx) and math.isfinite(vy)):
            return
        self._advance(stamp)
        v = math.hypot(vx, vy)
        if not self.started:
            self._init_speed(v, self.p["gnss_vel_var"])
            self.started = True
            return
        self.kf.update_speed(v, self.p["gnss_vel_var"])

    def _state_at(self, stamp: float) -> tuple[float, float]:
        h = self._history
        if stamp >= self.t - self.p["late_stamp_s"] or not h or stamp < h[0][0]:
            return self.kf.s, self.kf.v
        i = bisect_left([e[0] for e in h], stamp)
        if i == 0:
            return h[0][1], h[0][2]
        (t0, s0, v0), (t1, s1, v1) = h[i - 1], h[i]
        f = (stamp - t0) / (t1 - t0) if t1 > t0 else 1.0
        return s0 + f * (s1 - s0), v0 + f * (v1 - v0)

    def output(self, stamp: float) -> Estimate | None:
        if not self.started:
            return None
        self._advance(stamp)
        if stamp > self.t + self.p["confirm_forward_s"]:
            return None
        kf = self.kf
        s_out, v_out = self._state_at(stamp)
        if self.p["output_stamp_comp"] and self.t - self.p["late_stamp_s"] <= stamp < self.t:
            s_out -= kf.v * (self.t - stamp)
            if self.p["wheel_time_align"]:
                v_out -= self.a_model * (self.t - stamp)
        if self.path_id is not None:
            s_pub = self.route_s_ref + s_out - self.s_ref + self.p["output_along_offset_m"]
            x, y, z, yaw = self.route.pose(self.path_id, s_pub)
            if self._branch is not None:
                k, shift, sel = self._branch
                try:
                    sel.update(s_pub + shift, max(0.0, v_out))
                    w = sel.weight()
                    if w > 0.0:
                        bp = self.route.branch_pose(k, s_pub + shift)
                        if bp is not None:
                            if w >= 1.0:
                                x, y, z, yaw = bp
                            else:
                                bx, by, bz, byaw = bp
                                x, y, z = x + w * (bx - x), y + w * (by - y), z + w * (bz - z)
                                yaw += w * math.atan2(math.sin(byaw - yaw), math.cos(byaw - yaw))
                except Exception:
                    self._branch = None
            d = abs(s_out - self.s_ref)
            fxy = max(0.0, 1.0 - d / self.p["start_offset_xy_m"])
            fz = max(0.0, 1.0 - d / self.p["start_offset_z_m"])
            ox, oy, oz = self._start_offset
            x, y, z = x + ox * fxy, y + oy * fxy, z + oz * fz
            mode = "route"
        else:
            x, y, z, yaw = s_out, 0.0, 0.0, 0.0
            mode = "relative"
        fresh = [b for b, (t, _) in self.latest.items() if stamp - t < 0.5]
        mode += "/model" if not fresh else "/slip" if self.slip else "/wheels"
        self._last_mode = mode
        return Estimate(stamp=stamp, v=max(0.0, v_out), a=self.a_model, s=s_out, x=x, y=y, z=z, yaw=yaw,
                        var_v=kf.P[1][1], var_s=kf.P[0][0], slip=self.slip, mode=mode)
