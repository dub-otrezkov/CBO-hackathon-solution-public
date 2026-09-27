import json
import math
import os
from collections import deque
from dataclasses import dataclass
from operator import mul

DEFAULTS = {
    "v_min_mps": 2.0,
    "dv_max_mps": 0.5,
    "a_max_mps2": 2.5,
    "acc_half_s": 0.5,
    "acc_min_span_s": 0.3,
    "acc_max_span_s": 2.0,
    "cell_m": 0.5,
    "window_m": 200.0,
    "check_m": 10.0,
    "min_samples": 100,
    "min_cells": 50,
    "max_cells": 200,
    "min_coverage": 0.9,
    "step_m": 0.25,
    "scan_pad_m": 10.0,
    "excl_m": 5.0,
    "gate_sigma": 3.0,
    "gate_min_m": 2.0,
    "gate_max_m": 60.0,
    "rho_min": 0.25,
    "pr_min": 1.5,
    "reacq_m": 200.0,
    "reacq_gate_max_m": 150.0,
    "reacq_frac": 0.05,
    "reacq_rho_min": 0.30,
    "reacq_pr_min": 1.5,
    "reacq_confirm_m": 1.0,
    "reacq_confirm_gap_m": 0.0,
    "init_gate_m": 0.0,
    "fix_sigma_m": 0.4,
    "sigma0_m": 0.5,
    "sigma_rate": 0.005,
    "jump_tol_m": 3.0,
    "jump_time_s": 0.5,
    "jump_dt_max_s": 120.0,
    "v_max_mps": 40.0,
    "s_abs_max_m": 1.0e7,
    "reverse_reset_m": 1.0,
    "epoch_reset_s": 2.0,
    "min_bags": 3,
    "max_length_mismatch_m": 1.0,
    "work_per_call": 16000,
}

_HIST_N = 64


@dataclass
class RouteFix:
    stamp: float
    s_route: float
    var: float
    source: str = "fingerprint"
    rho: float = 0.0
    pr: float = 0.0
    delta: float = 0.0
    d_since: float = 0.0
    reacq: bool = False
    path_id: int = -1


def update_scale(k: float, fix: RouteFix, gain: float, k_gain: float = 0.3, min_d_m: float = 50.0,
                 k_min: float = 0.97, k_max: float = 1.03) -> float:
    if fix is None or not (fix.d_since >= min_d_m) or not math.isfinite(fix.delta) or not math.isfinite(k):
        return k
    return min(k_max, max(k_min, k * (1.0 + k_gain * gain * fix.delta / fix.d_since)))


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(x)
    except (TypeError, ValueError, OverflowError):
        return False


def _package_maps_dirs():
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps")
    try:
        from ament_index_python.packages import get_package_share_directory
        return [os.path.join(get_package_share_directory("tram_odometry"), "maps"), source]
    except Exception:
        return [source]


def _find_maps_dir(maps_dir, route, files) -> str:
    if maps_dir is not None:
        return str(maps_dir)
    files = list(files)
    cands = [getattr(route, "maps_dir", None)] + _package_maps_dirs()
    for d in cands:
        if d and any(os.path.isfile(os.path.join(d, f)) for f in files):
            return str(d)
    return ""


class _Fingerprint:

    __slots__ = ("h", "known", "length", "m", "m2", "s0", "size")

    def __init__(self, doc: dict, step_m: float, pad_m: float, min_bags: int):
        if int(doc.get("version", 0)) != 1:
            raise ValueError("unsupported fingerprint version")
        bin_m = float(doc["bin_m"])
        r = [float(x) for x in doc["r"]]
        n = [int(x) for x in doc["n"]]
        if bin_m <= 0.0 or len(r) != len(n) or len(r) < 2 or not all(math.isfinite(x) for x in r):
            raise ValueError("malformed fingerprint")
        up = max(1, round(bin_m / step_m))
        h = bin_m / up
        pad = math.ceil(pad_m / h) + 2
        m = [0.0] * pad
        known = [False] * pad
        for i in range(len(r) - 1):
            a, b = r[i], r[i + 1]
            ka, kb = n[i] >= min_bags, n[i + 1] >= min_bags
            for j in range(up):
                f = j / up
                m.append(a + f * (b - a))
                known.append(ka if j == 0 else (ka and kb))
        m.append(r[-1])
        known.append(n[-1] >= min_bags)
        m.extend([0.0] * pad)
        known.extend([False] * pad)
        self.m = m
        self.m2 = [x * x for x in m]
        self.known = known
        self.h = h
        self.s0 = float(doc["s_first"]) - pad * h
        self.size = len(m)
        self.length = float(doc.get("route_length_m", float("nan")))


class _Job:
    __slots__ = ("base", "gate_i", "i", "i_hi", "i_lo", "n", "pr_min", "reacq", "rho", "rho_min", "syy", "yc")


class RouteMatcher:
    def __init__(self, route, params: dict | None = None, maps_dir: str | None = None):
        self.p = dict(DEFAULTS)
        self.p.update(params or {})
        self.route = route
        self.stats = {"calls": 0, "samples": 0, "jobs": 0, "fixes": 0, "reacq_fixes": 0, "errors": 0,
                      "bad_input": 0, "inactive": 0, "window_resets": 0, "skip_samples": 0, "skip_coverage": 0,
                      "reject": 0, "reacq_unconfirmed": 0}
        self.maps = {}
        self.load_errors = {}
        names = list(getattr(route, "names", []) or [])
        self._n_map = len(names)
        files = {pid: f"fingerprint_{n.removesuffix('.json').removeprefix('route_')}.json"
                 for pid, n in enumerate(names)}
        self.maps_dir = _find_maps_dir(maps_dir, route, files.values())
        pad = self.p["reacq_gate_max_m"] + self.p["gate_max_m"] + self.p["scan_pad_m"]
        for pid, name in enumerate(names):
            fn = os.path.join(self.maps_dir, files[pid]) if self.maps_dir else ""
            if not os.path.isfile(fn):
                continue
            try:
                with open(fn, encoding="utf-8") as f:
                    fp = _Fingerprint(json.load(f), self.p["step_m"], pad, int(self.p["min_bags"]))
                path = route.paths[pid]
                if getattr(path, "closed", False):
                    raise ValueError("closed paths are not supported")
                if not abs(fp.length - route.length(pid)) <= self.p["max_length_mismatch_m"]:
                    raise ValueError(f"route length {route.length(pid):.3f} m != fingerprint {fp.length}")
                self.maps[pid] = fp
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.load_errors[name] = str(exc)
        self._cells_cap = math.ceil(self.p["window_m"] / self.p["cell_m"]) + 8
        self._pid_raw = None
        self._path_obj = None
        self._fp = None
        self._offset = 0.0
        self.reset(None, 0.0)

    @property
    def active(self) -> bool:
        return self._fp is not None

    def _current_path(self):
        paths = getattr(self.route, "paths", None)
        pid = self._pid_raw
        return paths[pid] if paths is not None and pid is not None and 0 <= pid < len(paths) else None

    def reset(self, path_id, s0) -> None:
        try:
            path_id = None if path_id is None else int(path_id)
        except (TypeError, ValueError, OverflowError):
            path_id = -1
        self._pid_raw = path_id
        self._resolve(path_id)
        if self._local and _finite(s0):
            try:
                self._offset = -float(self.route.base_of(path_id, float(s0))[1])
            except (AttributeError, TypeError, ValueError, IndexError):
                pass
        self._s_prev = float(s0) - self._offset if _finite(s0) else None
        self._t = None
        self._u = 0.0
        self._u_max = 0.0
        self._pending = 0.0
        self._clear_window()
        self._fl_var0 = self.p["sigma0_m"] ** 2
        self._fl_u0 = 0.0
        self._last_fix_u = 0.0
        self._acquired = False

    def applied(self, shift_m: float) -> None:
        if _finite(shift_m):
            self._pending = float(shift_m)

    def _clear_window(self) -> None:
        cap = self._cells_cap
        self._c_r = [0.0] * cap
        self._c_u = [0.0] * cap
        self._c_n = [0] * cap
        self._c_head = 0
        self._c_count = 0
        self._open = None
        self._hist = deque()
        self._hist_i = 0
        self._job = None
        self._reacq_cand = None
        self._last_check_u = self._u if hasattr(self, "_u") else 0.0

    def _resolve(self, path_id) -> None:
        self._fp, self._offset, self._path_obj, self._local = None, 0.0, None, False
        if path_id is None or self.route is None:
            return
        try:
            pid = int(path_id)
            paths = self.route.paths
            if pid < 0 or pid >= len(paths):
                return
            self._path_obj = paths[pid]
            if pid < self._n_map:
                self._fp = self.maps.get(pid)
                return
            dyn = paths[pid]
            for mid, fp in self.maps.items():
                mp = paths[mid]
                k = len(dyn.x) - len(mp.x)
                if k not in (0, 1) or dyn.x[-1] != mp.x[-1] or dyn.y[-1] != mp.y[-1]:
                    continue
                if dyn.x[k] == mp.x[0] and dyn.y[k] == mp.y[0] and abs((dyn.length - dyn.s[k]) - mp.length) < 1e-6:
                    self._fp, self._offset = fp, dyn.s[k]
                    return
            base_of = getattr(self.route, "base_of", None)
            base, shift = base_of(pid) if base_of is not None else (None, 0.0)
            if base is not None and base in self.maps:
                self._fp, self._offset, self._local = self.maps[base], -float(shift), True
        except (AttributeError, TypeError, ValueError, IndexError):
            self._fp, self._offset = None, 0.0

    def update(self, stamp, v_front, v_rear, v_est, s_route_est, var_s, path_id=None):
        try:
            return self._update(stamp, v_front, v_rear, v_est, s_route_est, var_s, path_id)
        except Exception:
            self.stats["errors"] += 1
            self._job = None
            return None

    def _update(self, stamp, v_front, v_rear, v_est, s_route_est, var_s, path_id):
        st = self.stats
        st["calls"] += 1
        if path_id is not None and path_id != self._pid_raw:
            self.reset(path_id, s_route_est)
        elif self._path_obj is not None and self._current_path() is not self._path_obj:
            self.reset(self._pid_raw, s_route_est)
        if self._fp is None:
            st["inactive"] += 1
            return None
        p = self.p
        if not (_finite(stamp) and _finite(s_route_est) and abs(s_route_est) <= p["s_abs_max_m"]):
            st["bad_input"] += 1
            return None
        t = float(stamp)
        if self._local:
            try:
                self._offset = -float(self.route.base_of(self._pid_raw, float(s_route_est))[1])
            except (AttributeError, TypeError, ValueError, IndexError):
                pass
        s_map = float(s_route_est) - self._offset
        ve = min(abs(float(v_est)), p["v_max_mps"]) if _finite(v_est) else 0.0
        dt = 0.0
        if self._t is not None:
            dt = t - self._t
            if dt < -p["epoch_reset_s"]:
                self._clear_window()
                self._t = None
                st["window_resets"] += 1
        if self._s_prev is not None:
            ds = s_map - self._s_prev - self._pending
            tol = p["jump_tol_m"] + ve * (p["jump_time_s"] + min(max(dt, 0.0), p["jump_dt_max_s"]))
            if not abs(ds) <= tol:
                self._clear_window()
                st["window_resets"] += 1
            else:
                self._u += ds
                if self._u > self._u_max:
                    self._u_max = self._u
                elif self._u < self._u_max - p["reverse_reset_m"]:
                    self._clear_window()
                    self._u_max = self._u
                    st["window_resets"] += 1
        self._pending = 0.0
        self._s_prev = s_map
        if self._t is None or t > self._t:
            self._t = t
            self._sample(t, v_front, v_rear)
        job = self._job
        if job is None:
            if self._u - self._last_check_u >= p["check_m"]:
                self._last_check_u = self._u
                self._start(s_map, var_s)
            return None
        return self._work(job, t, s_map, var_s)

    def _sample(self, t, vf, vr) -> None:
        p = self.p
        r = None
        v = float("nan")
        if _finite(vf) and _finite(vr):
            vf, vr = float(vf), float(vr)
            v = 0.5 * (vf + vr)
            if v > p["v_min_mps"] and abs(vf - vr) <= p["dv_max_mps"]:
                r = (vf - vr) / v
        h = self._hist
        if len(h) >= _HIST_N:
            h.popleft()
            self._hist_i = max(0, self._hist_i - 1)
        h.append((t, v, r, self._u))
        half = p["acc_half_s"]
        while self._hist_i < len(h):
            tc, _vc, rc, uc = h[self._hist_i]
            if t < tc + half:
                break
            if rc is not None:
                im = self._hist_i
                while im > 0 and h[im - 1][0] >= tc - half:
                    im -= 1
                ip = self._hist_i
                while h[ip][0] < tc + half:
                    ip += 1
                t0, v0 = h[im][0], h[im][1]
                t1, v1 = h[ip][0], h[ip][1]
                span = t1 - t0
                if p["acc_min_span_s"] < span <= p["acc_max_span_s"] and math.isfinite(v0) and math.isfinite(v1) \
                        and abs(v1 - v0) <= p["a_max_mps2"] * span:
                    self._add_cell(rc, uc)
            self._hist_i += 1
            while self._hist_i > 0 and h[0][0] < h[self._hist_i if self._hist_i < len(h) else -1][0] - half - 1e-9:
                h.popleft()
                self._hist_i -= 1

    def _add_cell(self, r, u) -> None:
        self.stats["samples"] += 1
        c = math.floor(u / self.p["cell_m"])
        o = self._open
        if o is not None and o[0] == c:
            o[1] += r
            o[2] += u
            o[3] += 1
            return
        if o is not None:
            self._push_cell(o)
        self._open = [c, r, u, 1]

    def _push_cell(self, o) -> None:
        i = self._c_head
        self._c_r[i] = o[1] / o[3]
        self._c_u[i] = o[2] / o[3]
        self._c_n[i] = o[3]
        self._c_head = (i + 1) % self._cells_cap
        self._c_count = min(self._c_count + 1, self._cells_cap)

    def _sigma(self, var_s) -> float:
        p = self.p
        floor = math.sqrt(self._fl_var0) + p["sigma_rate"] * max(0.0, self._u - self._fl_u0)
        if _finite(var_s) and var_s > 0.0:
            return max(math.sqrt(var_s), floor)
        return floor

    def _start(self, s_map, var_s) -> None:
        p, fp, st = self.p, self._fp, self.stats
        if self._open is not None:
            self._push_cell(self._open)
            self._open = None
        cap, u_now = self._cells_cap, self._u
        u_lo = u_now - p["window_m"]
        idx = []
        n_samples = 0
        i = self._c_head
        for _ in range(self._c_count):
            i = (i - 1) % cap
            if self._c_u[i] < u_lo:
                break
            idx.append(i)
            n_samples += self._c_n[i]
        if n_samples < p["min_samples"] or len(idx) < p["min_cells"]:
            st["skip_samples"] += 1
            return
        stride = -(-len(idx) // int(p["max_cells"]))
        h, s0, size = fp.h, fp.s0, fp.size
        d_since = u_now - self._last_fix_u
        g = min(p["gate_max_m"], max(p["gate_min_m"], p["gate_sigma"] * self._sigma(var_s)))
        reacq = d_since >= p["reacq_m"]
        init = not self._acquired and p["init_gate_m"] > 0.0
        if reacq or init:
            g = min(p["reacq_gate_max_m"], max(g, p["reacq_frac"] * d_since, p["init_gate_m"] if init else 0.0))
            reacq = True
        gate_i = math.floor(g / h + 1e-9)
        span_i = gate_i + math.ceil(p["scan_pad_m"] / h)
        base, ys = [], []
        known = 0
        lo_ok, hi_ok = span_i, size - 1 - span_i
        for k in idx:
            q = round((s_map - (u_now - self._c_u[k]) - s0) / h)
            if lo_ok <= q <= hi_ok and fp.known[q]:
                known += 1
        if known < p["min_coverage"] * len(idx):
            st["skip_coverage"] += 1
            return
        for k in idx[::stride]:
            q = round((s_map - (u_now - self._c_u[k]) - s0) / h)
            base.append(min(hi_ok, max(lo_ok, q)))
            ys.append(self._c_r[k])
        n = len(base)
        ym = sum(ys) / n
        yc = [y - ym for y in ys]
        syy = sum(map(mul, yc, yc))
        if not syy > 0.0:
            st["skip_samples"] += 1
            return
        job = _Job()
        job.base, job.yc, job.n, job.syy = base, yc, n, syy
        job.i_lo, job.i_hi, job.i = -span_i, span_i, -span_i
        job.rho = []
        job.gate_i = gate_i
        job.reacq = reacq
        job.rho_min = p["reacq_rho_min"] if reacq else p["rho_min"]
        job.pr_min = p["reacq_pr_min"] if reacq else p["pr_min"]
        self._job = job
        st["jobs"] += 1

    def _work(self, job, t, s_map, var_s):
        m, m2 = self._fp.m, self._fp.m2
        base, yc, n = job.base, job.yc, job.n
        steps = max(1, int(self.p["work_per_call"]) // n)
        i_end = min(job.i_hi + 1, job.i + steps)
        rho = job.rho
        syy = job.syy
        for i in range(job.i, i_end):
            vals = [m[b + i] for b in base]
            sx = sum(vals)
            vxx = sum([m2[b + i] for b in base]) - sx * sx / n
            if vxx > 1e-30:
                rho.append(sum(map(mul, yc, vals)) / math.sqrt(syy * vxx))
            else:
                rho.append(0.0)
        job.i = i_end
        if i_end <= job.i_hi:
            return None
        self._job = None
        return self._decide(job, t, s_map, var_s)

    def _decide(self, job, t, s_map, var_s):
        p, st = self.p, self.stats
        rho, i_lo = job.rho, job.i_lo
        best_k, best = -1, -2.0
        for k in range(job.gate_i * -1 - i_lo, job.gate_i - i_lo + 1):
            if rho[k] > best:
                best_k, best = k, rho[k]
        excl = math.floor(p["excl_m"] / self._fp.h + 1e-9)
        sec = -2.0
        for k in range(len(rho)):
            if abs(k - best_k) > excl and rho[k] > sec:
                sec = rho[k]
        pr = best / sec if sec > 0.0 else float("inf")
        gap = p["reacq_confirm_gap_m"]
        if not (best >= job.rho_min and pr >= job.pr_min):
            st["reject"] += 1
            if job.reacq and not gap > 0.0:
                self._reacq_cand = None
            return None
        delta = (best_k + i_lo) * self._fp.h
        tol = p["reacq_confirm_m"]
        if job.reacq and tol > 0.0:
            cand, prev = s_map + delta, self._reacq_cand
            if prev is None or abs(cand - (prev[0] + self._u - prev[1])) > tol:
                self._reacq_cand = (cand, self._u)
                st["reacq_unconfirmed"] += 1
                return None
            if self._u - prev[1] < gap:
                st["reacq_unconfirmed"] += 1
                return None
            self._reacq_cand = None
        var_fix = p["fix_sigma_m"] ** 2
        sigma = self._sigma(var_s)
        var_prior = float(var_s) if _finite(var_s) and var_s > 0.0 else sigma * sigma
        d_since = self._u - self._last_fix_u
        fix = RouteFix(stamp=t, s_route=s_map + delta + self._offset, var=var_fix, rho=best, pr=pr, delta=delta,
                       d_since=d_since, reacq=job.reacq, path_id=-1 if self._pid_raw is None else int(self._pid_raw))
        self._pending = var_prior / (var_prior + var_fix) * delta
        fl = sigma * sigma
        self._fl_var0 = fl * var_fix / (fl + var_fix)
        self._fl_u0 = self._u
        self._last_fix_u = self._u
        self._acquired = True
        st["fixes"] += 1
        if job.reacq:
            st["reacq_fixes"] += 1
        return fix
