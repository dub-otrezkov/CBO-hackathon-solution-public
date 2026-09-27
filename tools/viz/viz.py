import argparse
import glob
import json
import math
import os
import sqlite3
import sys
import time
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PKG = os.path.join(REPO, "src", "tram_odometry")
sys.path.insert(0, PKG)
from tram_odometry import estimator as core_est
from tram_odometry import route as R
from tram_odometry.estimator import Estimator
from tram_odometry.filter import LongitudinalFilter
from tram_odometry.model import TractionModel
from tram_odometry.node import OdometryCore

assert os.path.realpath(core_est.__file__).startswith(os.path.realpath(PKG)), core_est.__file__

MAPS = os.path.join(PKG, "maps")
DATA = "."
REF = ""
TEMPLATE = os.path.join(HERE, "page.html")
KMH_TO_MPS = 1.0 / 3.6
FRONT, REAR, CMD = "/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity", "/vehicle/driver_position_cmd"
GNSS = {"/sensing/gnss/master/fix": ("fix", "master"), "/sensing/gnss/rover/fix": ("fix", "rover"),
        "/sensing/gnss/master/vel": ("vel", "master"), "/sensing/gnss/rover/vel": ("vel", "rover")}
NS = 1_000_000_000
NAN = float("nan")


def _conv(v):
    v = v.strip().strip('"').strip("'")
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return v


def load_config():
    params, gnss_init = {}, 5.0
    fn = os.path.join(PKG, "config", "params.yaml")
    in_est, ind = False, 0
    with open(fn, encoding="utf-8") as fh:
        for line in fh:
            s = line.split("#", 1)[0].rstrip()
            if not s.strip():
                continue
            lead = len(s) - len(s.lstrip())
            key, _, val = s.strip().partition(":")
            if in_est and lead <= ind:
                in_est = False
            if in_est:
                params[key] = _conv(val)
            elif key == "gnss_init_s":
                gnss_init = float(val)
            elif key == "estimator" and not val.strip():
                in_est, ind = True, lead
    return params, gnss_init


def _store():
    try:
        from rosbags.typesys import Stores, get_types_from_msg, get_typestore
    except ImportError:
        return _RclpyStore()
    s = get_typestore(Stores.ROS2_HUMBLE)
    types = {}
    types.update(get_types_from_msg("std_msgs/Header header\nfloat64 velocity\n", "tram_vehicle_msgs/msg/VelocitySensor"))
    types.update(get_types_from_msg("std_msgs/Header header\nint8 position\n",
                                    "tram_vehicle_msgs/msg/DriverControllerCommand"))
    s.register(types)
    return s


class _RclpyStore:
    def __init__(self):
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        self._deserialize = deserialize_message
        self._get_message = get_message
        self._types = {}

    def deserialize_cdr(self, raw, typ):
        if typ not in self._types:
            self._types[typ] = self._get_message(typ)
        return self._deserialize(bytes(raw), self._types[typ])


def read_bag(bag, data_dir):
    store = _store()
    files = sorted(glob.glob(os.path.join(data_dir, bag, "*.db3")))
    if not files:
        raise SystemExit(f"no .db3 in {os.path.join(data_dir, bag)}")
    rows = []
    for fi, db in enumerate(files):
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
            topics = {tid: (name, typ) for tid, name, typ in con.execute("SELECT id, name, type FROM topics")}
            want = {tid: v for tid, v in topics.items() if v[0] in (FRONT, REAR, CMD) or v[0] in GNSS}
            q = f"SELECT topic_id, timestamp, id, data FROM messages WHERE topic_id IN ({','.join(map(str, want))})"
            for tid, rec, mid, raw in con.execute(q):
                rows.append((rec, fi, mid, want[tid], raw))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    ev = []
    for _rec, _fi, _mid, (name, typ), raw in rows:
        m = store.deserialize_cdr(raw, typ)
        h = int(m.header.stamp.sec) * NS + int(m.header.stamp.nanosec)
        if name == FRONT or name == REAR:
            ev.append(("front" if name == FRONT else "rear", h, float(m.velocity)))
        elif name == CMD:
            ev.append(("cmd", h, int(m.position)))
        else:
            kind, ant = GNSS[name]
            if kind == "fix":
                ev.append(("fix", h, (m.latitude, m.longitude, m.altitude, ant, int(m.status.status))))
            else:
                ev.append(("vel", h, (m.twist.linear.x, m.twist.linear.y, ant)))
    return ev


class Log:
    def __init__(self):
        self.fixes = []
        self.fp_detail = None
        self.trust = []
        self.cells = []
        self.jobs = {"decided": 0, "accepted": 0}
        self.gnss = []
        self.align = []
        self.matcher_map = None


def instrument(est, L):
    src = {"v": 2}
    pos = {}
    o_fix, o_feed, o_stop, o_reset = est.on_route_fix, est._feed_matcher, est._check_stop, est._reset_matcher
    o_pos = est.kf.update_position

    def update_position(z, r):
        pos["k"] = (est.kf.P[0][0], r)
        o_pos(z, r)

    def on_route_fix(stamp, s_route, var):
        ok = est.path_id is not None
        before = est._route_s() if ok else NAN
        p0, w0 = est.kf.P[0][0], est.kf.scale
        pos.clear()
        o_fix(stamp, s_route, var)
        after = est._route_s() if est.path_id is not None else NAN
        pk = pos.get("k")
        d = {"t": stamp, "src": src["v"], "s_fix": s_route, "s_before": before, "s_after": after, "var": var,
             "p0": p0, "p1": est.kf.P[0][0], "w0": w0, "w1": est.kf.scale, "pid": est.path_id,
             "K": pk[0] / (pk[0] + pk[1]) if pk is not None and pk[0] + pk[1] > 0 else NAN,
             "p_prior": pk[0] if pk is not None else NAN}
        if src["v"] == 0 and L.fp_detail is not None:
            d["job"] = L.fp_detail
        L.fp_detail = None
        L.fixes.append(d)

    def feed(stamp):
        src["v"] = 0
        try:
            o_feed(stamp)
        finally:
            src["v"] = 2

    def stop(stamp):
        src["v"] = 1
        try:
            o_stop(stamp)
        finally:
            src["v"] = 2

    def reset_matcher():
        o_reset()
        m = est.matcher
        L.align.append((est.t, est.path_id, est.route_s_ref, est.kf.v))
        if m is None:
            return
        mid = next((k for k, fp in m.maps.items() if fp is m._fp), None)
        L.matcher_map = (mid, m._offset)
        if getattr(m, "_viz", False):
            return
        m._viz = True
        o_add, o_decide = m._add_cell, m._decide

        def add_cell(r, u):
            h, i = m._hist, m._hist_i
            tc = h[i][0] if 0 <= i < len(h) else NAN
            s_est = m._s_prev - (m._u - u) + m._offset if m._s_prev is not None else NAN
            L.cells.append((tc, r, s_est, -1 if m._pid_raw is None else m._pid_raw))
            o_add(r, u)

        def decide(job, t, s_map, var_s):
            fix = o_decide(job, t, s_map, var_s)
            L.jobs["decided"] += 1
            if fix is not None:
                L.jobs["accepted"] += 1
                fp, h = m._fp, m._fp.h
                L.fp_detail = {
                    "pos": [fp.s0 + b * h for b in job.base],
                    "y": list(job.yc),
                    "rho": list(job.rho), "i_lo": job.i_lo, "h": h, "gate": job.gate_i * h,
                    "delta": fix.delta, "rho_best": fix.rho, "pr": fix.pr, "reacq": bool(fix.reacq),
                    "d_since": fix.d_since, "offset": m._offset}
            return fix

        m._add_cell, m._decide = add_cell, decide

    est.on_route_fix, est._feed_matcher, est._check_stop, est._reset_matcher = on_route_fix, feed, stop, reset_matcher
    est.kf.update_position = update_position

    det = est.detector
    if det is not None:
        cur = {}
        o_det, o_upd, o_trust = det.update, est.kf.update_wheel, est._on_wheel_trust

        def det_update(sample, v_pred, var_pred, a_model):
            tr = o_det(sample, v_pred, var_pred, a_model)
            cur["tr"] = tr
            return tr

        def update_wheel(z, r):
            hph = est.kf.innovation(z)[1]
            cur["upd"] = (z, r, hph / (hph + r) if hph + r > 0 else NAN)
            o_upd(z, r)

        def on_wheel_trust(bogie, stamp, raw):
            cur.clear()
            o_trust(bogie, stamp, raw)
            tr = cur.get("tr")
            if tr is None:
                return
            if "upd" in cur:
                z, r, k = cur["upd"]
                sig = det.p["meas_sigma0"] + det.p["meas_sigma_rel"] * abs(z)
                w_app = sig * sig / r if r > 0 else NAN
            else:
                w_app, k = 0.0, 0.0
            L.trust.append((stamp, 0 if bogie == "front" else 1, tr.weight, w_app, bool(tr.slip), tr.reason, k))

        det.update, est._on_wheel_trust = det_update, on_wheel_trust
        est.kf.update_wheel = update_wheel


def make(params):
    route = R.RouteMap.load(MAPS)
    route.maps_dir = MAPS
    return Estimator(dict(params), route), route


def replay(events, params, gnss_init, log=None):
    est, route = make(params)
    if log is not None:
        instrument(est, log)
    nc = OdometryCore(est, gnss_init)
    rows, extra = [], []
    for kind, hdr, val in events:
        if kind == "fix":
            acc = nc.gnss_fix(hdr, val[0], val[1], val[2], val[3])
            if log is not None and acc and math.isfinite(val[0]) and math.isfinite(val[1]) and abs(val[0]) > 1.0:
                x, y = R.to_grid(val[0], val[1])
                log.gnss.append((hdr * 1e-9, x, y, val[3], val[4]))
            continue
        if kind == "vel":
            nc.gnss_vel(hdr, val[0], val[1], val[2])
            continue
        feed = getattr(nc, "feed", None)
        if feed is not None:
            out = feed(kind, hdr, val)
        else:
            e = nc.controller(hdr, val) if kind == "cmd" else nc.wheel(kind, hdr, val)
            out = [] if e is None else [(hdr, e)]
        for stamp, e in out:
            pub = nc.position_ready(stamp, e)
            rows.append((stamp, e.x, e.y, e.z, e.v, pub))
            if log is not None:
                on = est.path_id is not None
                extra.append((est.route_s_ref + e.s - est.s_ref if on else NAN, -1 if not on else est.path_id,
                              est.kf.P[0][0], est.kf.scale, e.yaw, 1 if e.slip else 0,
                              0 if e.mode.startswith("route") else 1, e.var_v))
    return rows, extra, est, route, nc


def check_bitexact(events, params, gnss_init, rows_instr):
    rows_plain, _, _, _, _ = replay(events, params, gnss_init, None)
    same = len(rows_plain) == len(rows_instr) and all(a == b or (a != a and b != b)
                                                       for ra, rb in zip(rows_plain, rows_instr)
                                                       for a, b in zip(ra, rb))
    return same, len(rows_plain), len(rows_instr)


def integrate_from(t, v, t_start):
    t, v = np.asarray(t, float), np.asarray(v, float)
    if len(t) == 0:
        return np.array([t_start]), np.array([0.0])
    m = t > t_start
    tt = np.r_[t_start, t[m]]
    vv = np.r_[np.interp(t_start, t, v), v[m]]
    return tt, np.r_[0.0, np.cumsum(0.5 * (vv[1:] + vv[:-1]) * np.diff(tt))]


def model_only(events, t_start, s0, v0, pid, route, p):
    tm = TractionModel()
    kf = LongitudinalFilter(q_accel=0.0)
    kf.x[0] = s0
    kf.x[1] = max(0.0, v0)
    gp = SimpleNamespace(path_id=pid, route=route, p=p, _route_s=lambda: kf.x[0])
    step, fwd = p["substep_s"], p["epoch_forward_s"]
    t, out = None, [(t_start, s0, kf.v, 0.0)]
    for kind, hdr, val in events:
        if kind not in ("cmd", "front", "rear"):
            continue
        st = hdr * 1e-9
        if kind == "cmd":
            tm.on_controller(st, val)
        if st <= t_start:
            continue
        if t is None:
            t = t_start
        dt = st - t
        if dt <= 0.0:
            continue
        if dt > fwd:
            t = st
            continue
        while dt > 1e-9:
            h = min(step, dt)
            v = kf.v
            a_table = tm.table(t + h, v)
            a_grade = tm.grade_accel(Estimator._grade(gp))
            a = kf.accel(a_table, a_grade)
            hold = v < p["stop_speed_mps"] and a <= 0.0 and tm.notch_at(t + h) <= 0
            kf.predict(h, a_table, a_grade, hold=hold)
            t += h
            dt -= h
            out.append((t, kf.s, kf.v, a))
    return np.array(out, float)


def route_err(route, pid, d_off, ts, ss, rt, bx, by, max_gap):
    al, cr = np.full(len(rt), np.nan), np.full(len(rt), np.nan)
    if len(ts) < 2:
        return al, cr
    j = np.clip(np.searchsorted(ts, rt), 1, len(ts) - 1)
    ok = (rt >= ts[0]) & (rt <= ts[-1]) & (rt - ts[j - 1] <= max_gap) & (ts[j] - rt <= max_gap)
    ok &= np.isfinite(bx) & np.isfinite(by)
    s_at = np.interp(rt, ts, ss)
    for i in np.where(ok)[0]:
        x, y, _, yaw = route.pose(pid, float(s_at[i]) + d_off)
        ex, ey = x - bx[i], y - by[i]
        c, sn = math.cos(yaw), math.sin(yaw)
        al[i], cr[i] = ex * c + ey * sn, -ex * sn + ey * c
    return al, cr


def project_on(route, pid, xs, ys):
    p = route.paths[pid]
    s_o, l_o = np.full(len(xs), np.nan), np.full(len(xs), np.nan)
    miss = []
    for k, (x, y) in enumerate(zip(xs, ys)):
        best = None
        for q, i in route._candidates(float(x), float(y)):
            if q == pid:
                c = p.project(i, float(x), float(y))
                if best is None or c[0] < best[0]:
                    best = c
        if best is None:
            miss.append(k)
        else:
            s_o[k], l_o[k] = best[1], best[2]
    if miss:
        m = np.asarray(miss)
        s_o[m], l_o[m] = PathNP(p).project(np.asarray(xs)[m], np.asarray(ys)[m])
    return s_o, l_o


def route_xy(route, pid, d_off, ts, ss, tq):
    tq = tq[(tq >= ts[0]) & (tq <= ts[-1])] if len(ts) else tq[:0]
    s_at = np.interp(tq, ts, ss) if len(ts) else tq
    xy = np.array([route.pose(pid, float(s) + d_off)[:2] for s in s_at], float).reshape(-1, 2)
    return tq, s_at, xy[:, 0], xy[:, 1]


class PathNP:

    def __init__(self, p):
        self.x, self.y, self.s = np.asarray(p.x), np.asarray(p.y), np.asarray(p.s)
        self.D = np.c_[np.diff(self.x), np.diff(self.y)]
        self.L2 = (self.D ** 2).sum(1)
        self.length = float(p.length)

    def project(self, x, y, chunk=400):
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        A = np.c_[self.x[:-1], self.y[:-1]]
        s_o, l_o = np.full(len(x), np.nan), np.full(len(x), np.nan)
        for a in range(0, len(x), chunk):
            q = np.c_[x[a:a + chunk], y[a:a + chunk]]
            rel = q[:, None, :] - A[None]
            t = np.clip((rel * self.D[None]).sum(-1) / self.L2[None], 0, 1)
            foot = A[None] + t[..., None] * self.D[None]
            dist = np.hypot(q[:, None, 0] - foot[..., 0], q[:, None, 1] - foot[..., 1])
            j = dist.argmin(1)
            r = np.arange(len(q))
            cross = self.D[j, 0] * rel[r, j, 1] - self.D[j, 1] * rel[r, j, 0]
            s_o[a:a + chunk] = self.s[j] + t[r, j] * (self.s[j + 1] - self.s[j])
            l_o[a:a + chunk] = np.sign(cross) * dist[r, j]
        return s_o, l_o


def nearest(tref, tq, tol):
    if len(tref) < 2:
        return np.zeros(len(tq), int), np.zeros(len(tq), bool)
    idx = np.clip(np.searchsorted(tref, tq), 1, len(tref) - 1)
    near = np.where(np.abs(tref[idx] - tq) < np.abs(tref[idx - 1] - tq), idx, idx - 1)
    return near, np.abs(tref[near] - tq) <= tol


def episodes(t, flag, gap=0.3):
    out = []
    for ti, f in zip(t, flag):
        if not f:
            continue
        if out and ti - out[-1][1] <= gap:
            out[-1][1] = ti
        else:
            out.append([ti, ti])
    return out


def near_mask(t, events, near_s=2.0):
    ev = np.sort(np.asarray(events, float))
    if len(t) == 0 or len(ev) == 0:
        return np.zeros(len(t), bool)
    j = np.clip(np.searchsorted(ev, t), 0, len(ev) - 1)
    d = np.minimum(np.abs(t - ev[j]), np.abs(t - ev[np.maximum(j - 1, 0)]))
    return d <= near_s


def keep_mask(t, dt, events, near_s=2.0):
    if len(t) == 0:
        return np.zeros(0, bool)
    b = np.floor((t - t[0]) / dt).astype(np.int64)
    m = np.r_[True, b[1:] != b[:-1]] | near_mask(t, events, near_s)
    m[-1] = True
    return m


def bin_min(t, v, dt, keep_full):
    if len(t) == 0:
        return t, v
    b = np.floor((t - t[0]) / dt).astype(np.int64)
    ot, ov = [], []
    i, n = 0, len(t)
    while i < n:
        if keep_full[i]:
            ot.append(t[i])
            ov.append(v[i])
            i += 1
            continue
        j = i
        while j + 1 < n and b[j + 1] == b[i] and not keep_full[j + 1]:
            j += 1
        k = i + int(np.argmin(v[i:j + 1]))
        ot.append(t[k])
        ov.append(v[k])
        i = j + 1
    return np.asarray(ot), np.asarray(ov)


def bins_mean(s, r, width=0.5):
    ok = np.isfinite(s) & np.isfinite(r)
    s, r = s[ok], r[ok]
    if len(s) == 0:
        return np.array([]), np.array([])
    k = np.floor(s / width).astype(np.int64)
    u, inv = np.unique(k, return_inverse=True)
    sm = np.bincount(inv, r) / np.bincount(inv)
    return (u + 0.5) * width, sm


def rnd(a, nd):
    a = np.asarray(a, float)
    out = np.round(a, nd).tolist()
    return [None if (v != v or v in (float("inf"), float("-inf"))) else v for v in out]


def build(bag, data_dir, ref_dir, check=False):
    t_start = time.time()
    params, gnss_init = load_config()
    events = read_bag(bag, data_dir)
    t_read = time.time() - t_start
    L = Log()
    rows, extra, est, route, nc = replay(events, params, gnss_init, L)
    t_run = time.time() - t_start - t_read
    res = {"bag": bag, "read_s": round(t_read, 1), "replay_s": round(t_run, 1), "outputs": len(rows),
           "errors": nc.errors}
    if check:
        same, n_plain, n_instr = check_bitexact(events, params, gnss_init, rows)
        res["bitexact"] = {"identical": same, "outputs_plain": n_plain, "outputs_instrumented": n_instr,
                           "compared": "stamp, x, y, z, v, position published (every output)"}
    if not rows:
        raise SystemExit("no outputs")
    O = np.array([r[:5] for r in rows], float)
    t_ns = np.array([r[0] for r in rows], np.int64)
    pub = np.array([r[5] for r in rows], bool)
    X = np.array(extra, float)
    t0_ns = int(t_ns[0])
    srt = np.argsort(t_ns, kind="stable")
    O, t_ns, pub, X = O[srt], t_ns[srt], pub[srt], X[srt]
    t = (t_ns - t0_ns) * 1e-9
    s_route, pid, pss, wsc, yaw = X[:, 0], X[:, 1].astype(int), X[:, 2], X[:, 3], X[:, 4]
    x, y = O[:, 1], O[:, 2]
    v = O[:, 4]
    n_map = route._n_map
    d_off = est.p["output_along_offset_m"]

    wf = np.array([(h, val) for k, h, val in events if k == "front"], float).reshape(-1, 2)
    wr = np.array([(h, val) for k, h, val in events if k == "rear"], float).reshape(-1, 2)
    cm = np.array([(h, val) for k, h, val in events if k == "cmd"], float).reshape(-1, 2)
    gv = np.array([(h, math.hypot(val[0], val[1])) for k, h, val in events if k == "vel" and val[2] == "master"
                   and math.isfinite(val[0]) and math.isfinite(val[1])], float).reshape(-1, 2)
    rel = lambda a: (a - t0_ns) * 1e-9
    for a in (wf, wr, cm, gv):
        if len(a):
            a[:, 0] = rel(a[:, 0])
            o = np.argsort(a[:, 0], kind="stable")
            a[:] = a[o]

    ref_fn = os.path.join(ref_dir, bag + ".npz") if ref_dir else ""
    have_ref = os.path.isfile(ref_fn)
    e_al = e_cr = e_ab = np.full(len(t), np.nan)
    err_i = None
    e_A = np.full(0, float)
    t_A = np.full(0, float)
    ref_track = None
    s_true_fn = None
    ref = rt = None
    f_al_full = None
    if have_ref:
        ref = dict(np.load(ref_fn))
        rt = ref["t"] - t0_ns * 1e-9
        o = np.argsort(rt, kind="stable")
        for k in ("t", "blx_h", "bly_h", "mx", "my", "status"):
            ref[k] = ref[k][o]
        rt = rt[o]
        near, ok = nearest(rt, t, 0.05)
        ok &= pub
        ex, ey = x - ref["blx_h"][near], y - ref["bly_h"][near]
        c, s_ = np.cos(yaw), np.sin(yaw)
        e_al = np.where(ok, ex * c + ey * s_, np.nan)
        e_cr = np.where(ok, -ex * s_ + ey * c, np.nan)
        e_ab = np.where(ok, np.hypot(ex, ey), np.nan)
        pt, pxp, pyp, pyaw = t[pub], x[pub], y[pub], yaw[pub]
        if len(pt) >= 2:
            j = np.clip(np.searchsorted(pt, rt), 1, len(pt) - 1)
            t_a, t_b = pt[j - 1], pt[j]
            gi = (rt >= pt[0]) & (rt <= pt[-1]) & (rt - t_a <= 0.1) & (t_b - rt <= 0.1) & (t_b > t_a)
            wq = np.where(gi, (rt - t_a) / np.where(t_b > t_a, t_b - t_a, 1.0), 0.0)
            xi = pxp[j - 1] + wq * (pxp[j] - pxp[j - 1])
            yi = pyp[j - 1] + wq * (pyp[j] - pyp[j - 1])
            yw = np.where(wq < 0.5, pyaw[j - 1], pyaw[j])
            dx, dy = xi - ref["blx_h"], yi - ref["bly_h"]
            gi &= np.isfinite(dx) & np.isfinite(dy)
            err_i = (rt[gi], xi[gi], yi[gi], (dx * np.cos(yw) + dy * np.sin(yw))[gi], (-dx * np.sin(yw) + dy * np.cos(yw))[gi],
                     ref["status"][gi] == 2)
            f_al_full = np.where(gi, dx * np.cos(yw) + dy * np.sin(yw), np.nan)
        ref_track = (rt, ref["blx_h"], ref["bly_h"], ref["status"])
        paths = {k: PathNP(route.paths[k]) for k in range(n_map)}
        proj = {k: p.project(ref["mx"], ref["my"]) for k, p in paths.items()}
        route_mode = (X[:, 6] == 0) & np.isfinite(s_route)
        ot, os_, op = t[route_mode], s_route[route_mode], pid[route_mode]
        if len(ot) >= 2:
            j = np.clip(np.searchsorted(ot, rt), 1, len(ot) - 1)
            good = (rt >= ot[0]) & (rt <= ot[-1]) & (rt - ot[j - 1] <= 0.1) & (ot[j] - rt <= 0.1) & (op[j - 1] == op[j])
            good &= (ot[j] > ot[j - 1]) & (ref["status"] == 2) & (rt > ot[0] + 30.0)
            w = np.where(good, (rt - ot[j - 1]) / np.where(ot[j] > ot[j - 1], ot[j] - ot[j - 1], 1), 0)
            s_est = os_[j - 1] + w * (os_[j] - os_[j - 1])
            eA = np.full(len(rt), np.nan)
            for k, (sp, lp) in proj.items():
                sel = good & (op[j - 1] == k)
                m = sel & np.isfinite(sp) & (np.abs(lp) < 1.5) & (sp > 1.0) & (sp < paths[k].length - 1.0)
                eA[m] = s_est[m] - sp[m]
            mA = np.isfinite(eA)
            e_A, t_A = eA[mA], rt[mA]

        def s_true_fn(tq, k):
            if k not in proj:
                return np.full(len(tq), np.nan)
            sp, lp = proj[k]
            m = np.isfinite(sp) & (np.abs(lp) < 3.0)
            if m.sum() < 2:
                return np.full(len(tq), np.nan)
            out = np.interp(tq, rt[m], sp[m], left=np.nan, right=np.nan)
            jj = np.clip(np.searchsorted(rt[m], tq), 1, m.sum() - 1)
            gap = rt[m][jj] - rt[m][jj - 1]
            return np.where(gap < 1.0, out, np.nan)

    ev_rmse = float("nan")
    if len(gv) > 10:
        iv, okv = nearest(gv[:, 0], t, 0.05)
        ev_ = v[okv] - gv[iv[okv], 1]
        ev_rmse = float(np.sqrt(np.mean(ev_ ** 2))) if len(ev_) else float("nan")

    fx = L.fixes
    for d in fx:
        d["t_rel"] = d["t"] - t0_ns * 1e-9
        d["shift"] = d["s_after"] - d["s_before"]
        i = min(np.searchsorted(t, d["t_rel"]), len(t) - 1)
        d["x"], d["y"] = float(x[i]), float(y[i])
        d["xy_ok"] = bool(pub[i])
    fix_t = [d["t_rel"] for d in fx]

    TR = L.trust
    tr_t = np.array([a[0] for a in TR], float) - t0_ns * 1e-9
    tr_b = np.array([a[1] for a in TR], int)
    tr_wd = np.array([a[2] for a in TR], float)
    tr_wa = np.array([a[3] for a in TR], float)
    tr_sl = np.array([a[4] for a in TR], bool)
    tr_k = np.array([a[6] for a in TR], float)
    eps = []
    for b, name in ((0, "передняя"), (1, "задняя")):
        sel = tr_b == b
        for a0, a1 in episodes(tr_t[sel], tr_sl[sel]):
            eps.append({"t0": a0, "t1": max(a1, a0 + 0.1), "bogie": name})
    eps.sort(key=lambda e: e["t0"])
    ev_wide = [d["t_rel"] for d in fx if d["src"] != 0] + [e["t0"] for e in eps] + [e["t1"] for e in eps] \
        + list(np.arange(0.0, 10.0, 1.0))
    ev_fp = [d["t_rel"] for d in fx if d["src"] == 0]

    def keep(tt, dt):
        m = keep_mask(tt, dt, ev_wide, 2.0)
        return m | near_mask(tt, ev_fp, 0.3)

    km = keep(t, 0.25)
    err = {"t": [], "x": [], "y": [], "al": [], "cr": [], "rtk": []}
    if err_i is not None and len(err_i[0]):
        ke = keep(err_i[0], 0.25)
        err = {"t": rnd(err_i[0][ke], 3), "x": rnd(err_i[1][ke], 2), "y": rnd(err_i[2][ke], 2),
               "al": rnd(err_i[3][ke], 3), "cr": rnd(err_i[4][ke], 3), "rtk": [int(k) for k in err_i[5][ke]],
               "sp": rnd(np.interp(err_i[0][ke], t, np.sqrt(np.maximum(pss, 0.0))), 3)}
    series = {"t": rnd(t[km], 3), "v": rnd(v[km], 3),
              "w": rnd(wsc[km], 6), "sqrtP": rnd(np.sqrt(np.maximum(pss[km], 0)), 3),
              "x": rnd(np.where(pub[km], x[km], np.nan), 2), "y": rnd(np.where(pub[km], y[km], np.nan), 2),
              "s": rnd(s_route[km], 2)}
    trust = {}
    for b, name in ((0, "front"), (1, "rear")):
        sel = tr_b == b
        tt = tr_t[sel]
        full = near_mask(tt, ev_wide, 2.0) | near_mask(tt, ev_fp, 0.3)
        a_t, a_w = bin_min(tt, tr_wa[sel], 0.25, full)
        d_t, d_w = bin_min(tt, tr_wd[sel], 0.25, full)
        k_t, k_k = bin_min(tt, tr_k[sel], 0.25, full)
        trust[name] = {"t": rnd(a_t, 3), "w": rnd(a_w, 4), "td": rnd(d_t, 3), "wd": rnd(d_w, 4),
                       "tk": rnd(k_t, 3), "k": rnd(k_k, 4)}
    gk = keep(gv[:, 0], 0.2) if len(gv) else np.zeros(0, bool)
    kf_, kr_ = keep(wf[:, 0], 0.2), keep(wr[:, 0], 0.2)
    cmd_keep = np.r_[True, cm[1:, 1] != cm[:-1, 1]] if len(cm) else np.zeros(0, bool)
    if len(cm):
        cmd_keep |= np.r_[cmd_keep[1:], True]
    inputs = {"front": {"t": rnd(wf[kf_, 0], 3), "v": rnd(wf[kf_, 1] * KMH_TO_MPS, 3)},
              "rear": {"t": rnd(wr[kr_, 0], 3), "v": rnd(wr[kr_, 1] * KMH_TO_MPS, 3)},
              "gnss_v": {"t": rnd(gv[gk, 0], 3), "v": rnd(gv[gk, 1], 3)} if len(gv) else {"t": [], "v": []},
              "cmd": {"t": rnd(cm[cmd_keep, 0], 3), "p": [int(a) for a in cm[cmd_keep, 1]]}}

    srcd, src_stats, graw = None, None, {}
    for ant in ("master", "rover"):
        g = np.array([(h, *R.to_grid(val[0], val[1]), val[4]) for k, h, val in events if k == "fix" and val[3] == ant
                      and math.isfinite(val[0]) and math.isfinite(val[1]) and abs(val[0]) > 1.0], float).reshape(-1, 4)
        if len(g):
            g[:, 0] = rel(g[:, 0])
            g = g[np.argsort(g[:, 0], kind="stable")]
        graw[ant] = g
    al_last = L.align[-1] if L.align else None
    if al_last is not None and al_last[1] is not None and al_last[0] is not None:
        t_al_abs, pid_al, s0_al, v_al = al_last
        t_al = t_al_abs - t0_ns * 1e-9
        tracks = {}
        okf, okr = np.isfinite(wf[:, 1]), np.isfinite(wr[:, 1])
        if okf.sum() >= 2 and okr.sum() >= 2:
            tg = np.union1d(wf[okf, 0], wr[okr, 0])
            vm = 0.5 * (np.interp(tg, wf[okf, 0], wf[okf, 1]) + np.interp(tg, wr[okr, 0], wr[okr, 1])) * KMH_TO_MPS
            tw, sw = integrate_from(tg, vm, t_al)
            tracks["wo"] = (tw, s0_al + sw)
        MO = model_only(events, t_al_abs, s0_al, v_al, pid_al, route, est.p)
        MO[:, 0] -= t0_ns * 1e-9
        tracks["mo"] = (MO[:, 0], MO[:, 1])
        if len(gv) >= 2:
            td, sd = integrate_from(gv[:, 0], gv[:, 1], t_al)
            tracks["dop"] = (td, s0_al + sd)
        gm = graw.get("master", np.zeros((0, 4)))
        if len(gm) >= 2:
            sp_, lp_ = project_on(route, pid_al, gm[:, 1], gm[:, 2])
            tracks["gnss"] = (gm[:, 0], sp_)
        srcd = {"t_align": round(t_al, 3), "pid": int(pid_al), "s0": round(float(s0_al), 3), "v0": round(float(v_al), 3),
                "path": route.names[pid_al] if pid_al < n_map else "соединитель", "tracks": {}, "err": None,
                "model_v": {"t": [], "v": []}}
        for key in ("wo", "mo", "dop"):
            if key in tracks:
                ts_, ss_ = tracks[key]
                tq, sq, xq, yq = route_xy(route, pid_al, d_off, ts_, ss_, np.arange(t_al, ts_[-1] + 1e-9, 0.5))
                srcd["tracks"][key] = {"t": rnd(tq, 2), "x": rnd(xq, 2), "y": rnd(yq, 2)}
        mk = keep_mask(MO[:, 0], 0.25, [])
        srcd["model_v"] = {"t": rnd(MO[mk, 0], 3), "v": rnd(MO[mk, 2], 3)}
        if have_ref:
            bx_, by_ = ref["blx_h"], ref["bly_h"]
            errs = {}
            for key, (ts_, ss_) in tracks.items():
                errs[key] = route_err(route, pid_al, d_off, ts_, ss_, rt, bx_, by_, 0.25 if key == "gnss" else 1.0)[0]
            errs["f"] = f_al_full if f_al_full is not None else np.full(len(rt), np.nan)
            common = (ref["status"] == 2) & (rt >= t_al)
            for key in ("wo", "mo", "dop", "gnss", "f"):
                if key in errs:
                    common &= np.isfinite(errs[key])
            src_stats = {"epochs": int(common.sum()), "t_first": None, "t_last": None}
            if common.any():
                ic = np.where(common)[0]
                src_stats["t_first"], src_stats["t_last"] = round(float(rt[ic[0]]), 2), round(float(rt[ic[-1]]), 2)
                for key, e in errs.items():
                    src_stats[key] = {"along_rmse_m": round(float(np.sqrt(np.mean(e[common] ** 2))), 3),
                                      "final_along_m": round(float(e[ic[-1]]), 3),
                                      "max_abs_along_m": round(float(np.max(np.abs(e[common]))), 3)}
            ke = keep(rt, 0.25) & (rt >= t_al - 1.0)
            srcd["err"] = {"t": rnd(rt[ke], 3), "rtk": [int(k) for k in ref["status"][ke] == 2]}
            for key, e in errs.items():
                if key != "f":
                    srcd["err"][key] = rnd(e[ke], 3)
        srcd["stats"] = src_stats
    for d in fx:
        d["e_al"] = NAN
        if rt is None or d["pid"] is None or len(rt) < 2:
            continue
        tq = d["t_rel"]
        j = int(np.clip(np.searchsorted(rt, tq), 1, len(rt) - 1))
        if not (rt[j - 1] <= tq <= rt[j] and rt[j] - rt[j - 1] <= 0.25):
            continue
        f_ = (tq - rt[j - 1]) / (rt[j] - rt[j - 1]) if rt[j] > rt[j - 1] else 0.0
        bxq = ref["blx_h"][j - 1] + f_ * (ref["blx_h"][j] - ref["blx_h"][j - 1])
        byq = ref["bly_h"][j - 1] + f_ * (ref["bly_h"][j] - ref["bly_h"][j - 1])
        px_, py_, _, pyw = route.pose(d["pid"], d["s_fix"] + d_off)
        d["e_al"] = (px_ - bxq) * math.cos(pyw) + (py_ - byq) * math.sin(pyw)
    for ant, g in graw.items():
        if len(g):
            gk_ = keep_mask(g[:, 0], 0.5, []) | (g[:, 3] != 2)
            graw[ant] = {"t": rnd(g[gk_, 0], 2), "x": rnd(g[gk_, 1], 2), "y": rnd(g[gk_, 2], 2),
                         "st": [int(a) for a in g[gk_, 3]], "n": int(len(g))}
        else:
            graw[ant] = None

    routes = []
    for k in range(n_map):
        p = route.paths[k]
        routes.append({"name": route.names[k], "x": rnd(p.x, 2), "y": rnd(p.y, 2)})
    refd = None
    if ref_track is not None:
        rt, bx, by, st = ref_track
        rk = keep_mask(rt, 0.5, [])
        refd = {"t": rnd(rt[rk], 2), "x": rnd(bx[rk], 2), "y": rnd(by[rk], 2), "status": [int(a) for a in st[rk]]}
    pid_final = est.path_id
    stops_doc = json.load(open(os.path.join(MAPS, "stops.json"), encoding="utf-8"))
    plats = []
    if pid_final is not None and pid_final < n_map:
        for m_ in stops_doc["routes"].get(route.names[pid_final], []):
            px, py, _, _ = route.pose(pid_final, m_["s"])
            plats.append({"s": m_["s"], "std": m_["std"], "x": round(px, 2), "y": round(py, 2)})
    gnss_pts = [{"t": round(a[0] - t0_ns * 1e-9, 3), "x": round(a[1], 2), "y": round(a[2], 2), "ant": a[3],
                 "status": a[4]} for a in L.gnss]
    start = None
    if L.align:
        ta, pa, sa = L.align[-1][:3]
        if pa is not None:
            sx, sy, _, _ = route.pose(pa, sa)
            start = {"t": round(ta - t0_ns * 1e-9, 3) if ta is not None else None, "pid": pa,
                     "path": route.names[pa] if pa < n_map else "соединитель", "s0": round(sa, 2),
                     "x": round(sx, 2), "y": round(sy, 2), "n_align": len(L.align),
                     "offset_xy": [round(est._start_offset[0], 3), round(est._start_offset[1], 3)]}

    along = None
    mm = L.matcher_map
    if mm is not None and mm[0] is not None and est.matcher is not None:
        mid, off = mm
        fpfile = f"fingerprint_{route.names[mid].removesuffix('.json').removeprefix('route_')}.json"
        fpdoc = json.load(open(os.path.join(MAPS, fpfile), encoding="utf-8"))
        rr = np.asarray(fpdoc["r"], float)
        nn = np.asarray(fpdoc["n"], int)
        sm = fpdoc["s_first"] + np.arange(len(rr)) * fpdoc["bin_m"]
        known = nn >= est.matcher.p["min_bags"]
        C = np.array([(a[0], a[1], a[2], a[3]) for a in L.cells], float).reshape(-1, 4)
        if len(C):
            cs_map = C[:, 2] - off
            med = np.nanmedian(C[:, 1])
            sb, rb = bins_mean(cs_map, C[:, 1] - med, 0.5)
            ctime = C[:, 0] - t0_ns * 1e-9
            if s_true_fn is not None:
                st_ = np.full(len(C), np.nan)
                for k in np.unique(C[:, 3]).astype(int):
                    sel = C[:, 3] == k
                    if 0 <= k < n_map:
                        st_[sel] = s_true_fn(ctime[sel], k) - off
                sg, rg = bins_mean(st_, C[:, 1] - med, 0.5)
            else:
                sg, rg = np.array([]), np.array([])
        else:
            sb = rb = sg = rg = np.array([])
        lo = min(np.nanmin(sb) if len(sb) else 0, np.nanmin(sg) if len(sg) else 0) - 50
        hi = max(np.nanmax(sb) if len(sb) else 0, np.nanmax(sg) if len(sg) else 0) + 50
        msel = (sm >= lo) & (sm <= hi)
        mp = est.matcher.p
        along = {"map_name": route.names[mid], "map_file": fpfile, "offset": off, "bin_m": fpdoc["bin_m"],
                 "map_s0": round(float(sm[msel][0]), 3) if msel.any() else None,
                 "map_r": [int(v_) for v_ in np.round(rr[msel] * 1e5)], "map_known": [int(k_) for k_ in known[msel]],
                 "rec_s": rnd(sb, 2), "rec_r": [int(v_) for v_ in np.round(rb * 1e5)],
                 "gnss_s": rnd(sg, 2), "gnss_r": [int(v_) for v_ in np.round(rg * 1e5)], "r_scale": 1e-5,
                 "cells": int(len(C)), "median_removed": round(float(med), 6) if len(C) else None,
                 "p": {k_: mp[k_] for k_ in ("window_m", "check_m", "cell_m", "excl_m", "rho_min", "pr_min", "reacq_rho_min",
                                             "reacq_pr_min", "fix_sigma_m", "gate_sigma", "gate_min_m", "gate_max_m")}}

    fixes_out = []
    for i, d in enumerate(fx):
        o = {"i": i, "t": round(d["t_rel"], 3), "src": d["src"], "s_fix": round(d["s_fix"], 3),
             "s_before": round(d["s_before"], 3), "shift": round(d["shift"], 3), "sigma": round(math.sqrt(d["var"]), 3),
             "sqrtP0": round(math.sqrt(max(d["p0"], 0.0)), 3), "sqrtP1": round(math.sqrt(max(d["p1"], 0.0)), 3),
             "w0": round(d["w0"], 6), "w1": round(d["w1"], 6), "x": round(d["x"], 2), "y": round(d["y"], 2),
             "xy_ok": d["xy_ok"], "K": round(d["K"], 4) if math.isfinite(d["K"]) else None,
             "sqrtPk": round(math.sqrt(d["p_prior"]), 3) if math.isfinite(d["p_prior"]) and d["p_prior"] >= 0 else None,
             "e_al": round(d["e_al"], 3) if math.isfinite(d["e_al"]) else None}
        j = d.get("job")
        if j is not None:
            pos_cm = np.round(np.asarray(j["pos"]) * 100).astype(np.int64)
            o["job"] = {"pos_cm": [int(pos_cm[0])] + [int(a) for a in np.diff(pos_cm)] if len(pos_cm) else [],
                        "y": [int(a) for a in np.round(np.asarray(j["y"]) * 1e5)],
                        "rho": [int(a) for a in np.round(np.asarray(j["rho"]) * 1e3)], "i_lo": j["i_lo"], "h": j["h"],
                        "gate": round(j["gate"], 3), "delta": round(j["delta"], 3), "rho_best": round(j["rho_best"], 3),
                        "pr": round(j["pr"], 3) if math.isfinite(j["pr"]) else None, "reacq": j["reacq"],
                        "d_since": round(j["d_since"], 1)}
        fixes_out.append(o)

    okp = np.isfinite(e_ab)
    dist = float(np.nanmax(s_route) - np.nanmin(s_route)) if np.isfinite(s_route).any() else float("nan")
    i_last = None
    if okp.any():
        ii = np.where(okp)[0]
        i_last = int(ii[np.argmax(srt[ii])])
    stats = {
        "duration_s": round(float(t[-1]), 1), "distance_m": round(dist, 1) if math.isfinite(dist) else None,
        "xy_rmse_m": round(float(np.sqrt(np.mean(e_ab[okp] ** 2))), 3) if okp.any() else None,
        "along_rmse_m": round(float(np.sqrt(np.nanmean(e_al[okp] ** 2))), 3) if okp.any() else None,
        "cross_rmse_m": round(float(np.sqrt(np.nanmean(e_cr[okp] ** 2))), 3) if okp.any() else None,
        "max_abs_m": round(float(np.max(e_ab[okp])), 3) if okp.any() else None,
        "final_xy_m": round(float(e_ab[i_last]), 3) if i_last is not None else None,
        "along_A_rmse_m": round(float(np.sqrt(np.mean(e_A ** 2))), 3) if len(e_A) else None,
        "v_rmse_mps": round(ev_rmse, 4) if math.isfinite(ev_rmse) else None,
        "fp_fixes": sum(1 for d in fx if d["src"] == 0), "stop_fixes": sum(1 for d in fx if d["src"] == 1),
        "slip_episodes": len(eps), "slip_s": round(sum(e["t1"] - e["t0"] for e in eps), 1),
        "jobs": L.jobs, "matcher_stats": dict(est.matcher.stats) if est.matcher is not None else None,
        "core_errors": nc.errors, "outputs": len(rows), "published_positions": int(pub.sum()),
        "max_wheel_diff_mps": round(float(np.nanmax(np.abs(np.interp(wf[:, 0], wr[:, 0], wr[:, 1]) - wf[:, 1]))
                                          * KMH_TO_MPS), 3) if len(wf) > 2 and len(wr) > 2 else None,
    }
    if okp.any():
        tp = t[okp]
        stats["start_max_abs_m"] = round(float(np.max(e_ab[okp][tp <= tp[0] + 60.0])), 3)
        stats["end_max_abs_m"] = round(float(np.max(e_ab[okp][tp >= tp[-1] - 60.0])), 3)
    utc0 = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0_ns / 1e9)) + f".{(t0_ns % NS) // 1_000_000:03d} UTC"
    page = {
        "bag": bag, "utc0": utc0, "t0_ns": str(t0_ns), "gnss_init_s": gnss_init, "params": params,
        "along_offset_m": d_off, "stats": stats, "series": series, "trust": trust, "inputs": inputs,
        "routes": routes, "ref": refd, "platforms": plats, "gnss": gnss_pts, "start": start, "fixes": fixes_out,
        "episodes": [{"t0": round(e["t0"], 3), "t1": round(e["t1"], 3), "bogie": e["bogie"]} for e in eps],
        "along": along, "err": err, "eA": {"t": rnd(t_A[keep(t_A, 0.25)], 3) if len(t_A) else [],
                               "e": rnd(e_A[keep(t_A, 0.25)], 3) if len(t_A) else []},
        "path_final": route.names[pid_final] if pid_final is not None and pid_final < n_map else None,
        "src": srcd, "graw": graw,
        "have_ref": have_ref,
        "kmh_to_mps": KMH_TO_MPS,
        "model_delay_s": TractionModel().delay_s,
    }
    stats["sources"] = src_stats
    res["stats"] = stats
    return page, res


def write_html(page, out):
    tpl = open(TEMPLATE, encoding="utf-8").read()
    blob = json.dumps(page, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    html = tpl.replace("/*__DATA__*/null", blob).replace("__TITLE__", f"Одометрия: {page['bag']}")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    return os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser(description="Offline replay of the odometry core on one recording and an HTML page with its internals")
    ap.add_argument("bag")
    ap.add_argument("--out", default=None, help="HTML file (default <bag>.html in the current directory)")
    ap.add_argument("--data", default=DATA, help="directory of unpacked bags (<data>/<bag>/*.db3)")
    ap.add_argument("--ref", default=REF, help="directory of base_link reference npz (<ref>/<bag>.npz)")
    ap.add_argument("--check", action="store_true", help="also replay without instrumentation and compare bit for bit")
    ap.add_argument("--stats", default=None, help="write the run summary as JSON here")
    ap.add_argument("--note", default="", help="short text shown at the top of the page (why this recording)")
    a = ap.parse_args()
    page, res = build(a.bag, a.data, a.ref, check=a.check)
    page["note"] = a.note
    out = a.out or f"{a.bag}.html"
    res["html"] = os.path.abspath(out)
    res["html_bytes"] = write_html(page, out)
    if a.stats:
        with open(a.stats, "w", encoding="utf-8") as fh:
            json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    if a.check and not res["bitexact"]["identical"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
