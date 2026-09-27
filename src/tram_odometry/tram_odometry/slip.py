from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass

try:
    from .preprocess import BOGIES, FLAG_EPOCH, FLAG_GAP, FLAG_OUTLIER, FLAG_STALE
except ImportError:
    from preprocess import BOGIES, FLAG_EPOCH, FLAG_GAP, FLAG_OUTLIER, FLAG_STALE

NAN = math.nan

DEFAULTS = {
    "meas_sigma0": 0.03,
    "meas_sigma_rel": 0.01,
    "z_lo": 3.0,
    "z_hi": 6.0,
    "pair_max_dt": 0.3,
    "cross_abs": 0.15,
    "cross_rel": 0.03,
    "cross_low_speed": 0.5,
    "cross_low_speed_v": 1.0,
    "cross_dt_accel": 0.5,
    "cross_full": 2.0,
    "acc_window": 0.3,
    "acc_min_span": 0.15,
    "env_trac": 2.5,
    "env_trac_full": 4.0,
    "env_brake": 6.5,
    "env_brake_full": 10.0,
    "acc_dev": 1.0,
    "acc_dev_full": 2.5,
    "phase_eps": 0.1,
    "soft_gain": 0.7,
    "entry_soft_gain": 0.5,
    "slip_on": 0.5,
    "slip_off": 0.2,
    "hold_s": 0.5,
    "recover_s": 1.0,
    "slip_weight_cap": 0.05,
    "relax_s": 3.0,
    "relax_ramp_s": 1.0,
    "history_n": 16,
    "v_abs_max": 100.0,
}


@dataclass(frozen=True)
class WheelTrust:

    weight: float
    residual: float
    slip: bool
    z: float = NAN
    accel: float = NAN
    cross: float = NAN
    hard: float = 0.0
    soft: float = 0.0
    reason: str = ""
    common: bool = False


def _f(x):
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _ramp(x: float, lo: float, hi: float) -> float:
    if x <= lo:
        return 0.0
    if hi <= lo or x >= hi:
        return 1.0
    return (x - lo) / (hi - lo)


def _combine(a: float, b: float) -> float:
    return 1.0 - (1.0 - a) * (1.0 - b)


class _WheelState:
    __slots__ = ("hist", "latest", "last_stamp", "slip", "last_ev_t", "t_clear", "soft_since", "soft_last",
                 "weight")

    def __init__(self, n: int):
        self.hist = deque(maxlen=n)
        self.reset()

    def reset(self):
        self.hist.clear()
        self.latest = None
        self.last_stamp = None
        self.slip = False
        self.last_ev_t = None
        self.t_clear = None
        self.soft_since = None
        self.soft_last = None
        self.weight = 1.0


class SlipDetector:

    def __init__(self, params: dict | None = None, **kwargs):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        p.update(kwargs)
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown SlipDetector params: {sorted(unknown)}")
        self.p = {k: (int(v) if k == "history_n" else float(v)) for k, v in p.items()}
        n = max(4, self.p["history_n"])
        self._st = {b: _WheelState(n) for b in BOGIES}


    def state(self, bogie: str) -> dict:
        st = self._st.get(bogie)
        if st is None:
            return {}
        return {"slip": st.slip, "weight": st.weight, "last_stamp": st.last_stamp, "latest": st.latest}

    def any_slip(self) -> bool:
        return any(st.slip for st in self._st.values())

    def reset(self) -> None:
        for st in self._st.values():
            st.reset()


    def update(self, sample, v_pred, var_pred, a_model) -> WheelTrust:
        try:
            return self._update(sample, v_pred, var_pred, a_model)
        except Exception:
            return WheelTrust(weight=0.0, residual=NAN, slip=False, reason="error")

    def _update(self, sample, v_pred, var_pred, a_model) -> WheelTrust:
        p = self.p
        bogie = getattr(sample, "bogie", None)
        st = self._st.get(bogie) if isinstance(bogie, str) else None
        t = _f(getattr(sample, "stamp", None))
        v = _f(getattr(sample, "v", None))
        if st is None or t is None or v is None or abs(v) > p["v_abs_max"]:
            return WheelTrust(weight=0.0, residual=NAN, slip=False, reason="invalid")
        flags = getattr(sample, "flags", None)
        if not isinstance(flags, (frozenset, set, tuple, list)):
            flags = frozenset()
        other = self._st["rear" if bogie == "front" else "front"]

        if FLAG_EPOCH in flags or (st.last_stamp is not None and t < st.last_stamp):
            st.reset()
        if FLAG_GAP in flags:
            st.hist.clear()
        st.last_stamp = t

        vp = _f(v_pred)
        if vp is not None and abs(vp) > p["v_abs_max"]:
            vp = None
        vv = _f(var_pred)
        if vv is not None and vv < 0.0:
            vv = None
        am = _f(a_model)
        residual = v - vp if vp is not None else NAN

        if FLAG_STALE in flags:
            return WheelTrust(weight=0.0, residual=residual, slip=st.slip, reason="stale",
                              common=st.slip and other.slip)
        if FLAG_OUTLIER in flags:
            if st.slip:
                st.last_ev_t = t
            st.soft_since = None
            return WheelTrust(weight=0.0, residual=residual, slip=st.slip, hard=1.0, reason="outlier",
                              common=st.slip and other.slip)

        accel = NAN
        w_max = p["acc_window"] + 1e-6
        for (ti, vi) in st.hist:
            span = t - ti
            if span <= w_max:
                if span >= p["acc_min_span"]:
                    accel = (v - vi) / span
                break

        cross = NAN
        s_cross = 0.0
        if other.latest is not None and not other.slip:
            to, vo, ao = other.latest
            dto = t - to
            if abs(dto) <= p["pair_max_dt"]:
                if not (ao == ao and -p["env_brake"] <= ao <= p["env_trac"]):
                    ao = 0.0
                vo_adj = vo + ao * dto
                cross = v - vo_adj
                vref = 0.5 * abs(v + vo_adj)
                tol = p["cross_abs"] + p["cross_rel"] * vref + p["cross_dt_accel"] * abs(dto)
                if vref < p["cross_low_speed_v"]:
                    tol = max(tol, p["cross_low_speed"])
                s_x = _ramp(abs(cross) / tol, 1.0, p["cross_full"])
                if s_x > 0.0:
                    if am is not None:
                        a_phase = am
                    elif accel == accel and ao == ao:
                        a_phase = 0.5 * (accel + ao)
                    else:
                        a_phase = None
                    s_cross = s_x * self._blame(cross, v, vo_adj, vp, vv, a_phase)

        s_env = 0.0
        s_dev = 0.0
        if accel == accel:
            if accel >= 0.0:
                s_env = _ramp(accel, p["env_trac"], p["env_trac_full"])
            else:
                s_env = _ramp(-accel, p["env_brake"], p["env_brake_full"])
            if am is not None:
                e = accel - am
                if am > p["phase_eps"]:
                    dev = e
                elif am < -p["phase_eps"]:
                    dev = -e
                else:
                    dev = abs(e)
                s_dev = _ramp(dev, p["acc_dev"], p["acc_dev_full"])

        z = NAN
        s_res = 0.0
        if vp is not None and vv is not None:
            sm = p["meas_sigma0"] + p["meas_sigma_rel"] * abs(v)
            z = residual / math.sqrt(vv + sm * sm)
            s_res = _ramp(abs(z), p["z_lo"], p["z_hi"])

        hard = _combine(s_cross, s_env)
        soft = _combine(s_res, s_dev)

        if hard >= p["slip_off"]:
            st.soft_since = None
        elif soft > 0.0:
            if st.soft_since is None:
                st.soft_since = t
            st.soft_last = t
        elif st.soft_since is not None and (st.soft_last is None or t - st.soft_last > p["hold_s"]):
            st.soft_since = None
        if soft > 0.0 and st.soft_since is not None:
            run_s = t - st.soft_since
            if run_s > p["relax_s"]:
                rr = p["relax_ramp_s"]
                soft *= max(0.0, 1.0 - (run_s - p["relax_s"]) / rr) if rr > 0.0 else 0.0

        reason = ""
        if hard > 0.0 or soft > 0.0:
            reason = max((s_cross, "cross"), (s_env, "envelope"), (s_res * (soft > 0.0), "residual"),
                         (s_dev * (soft > 0.0), "a_model"))[1]

        entry = hard + p["entry_soft_gain"] * soft if hard > 0.0 else 0.0
        if entry >= p["slip_on"] and not st.slip:
            st.slip = True
            st.last_ev_t = t
        if st.slip:
            if hard >= p["slip_off"] or soft >= p["slip_off"]:
                st.last_ev_t = t
            elif st.last_ev_t is None or t - st.last_ev_t >= p["hold_s"]:
                st.slip = False
                st.t_clear = t
        cap = p["slip_weight_cap"]
        if st.slip:
            w_env = cap
        elif st.t_clear is not None and p["recover_s"] > 0.0:
            w_env = cap + (1.0 - cap) * min(1.0, max(0.0, (t - st.t_clear) / p["recover_s"]))
        else:
            w_env = 1.0
        w_inst = (1.0 - hard) * (1.0 - p["soft_gain"] * soft)
        weight = min(w_inst, w_env)
        if not weight >= 0.0:
            weight = 0.0
        elif weight > 1.0:
            weight = 1.0
        st.weight = weight

        st.hist.append((t, v))
        st.latest = (t, v, accel)
        common = (st.slip and other.slip and other.last_stamp is not None
                  and abs(t - other.last_stamp) <= p["pair_max_dt"])
        return WheelTrust(weight=weight, residual=residual, slip=st.slip, z=z, accel=accel, cross=cross,
                          hard=hard, soft=soft, reason=reason, common=common)

    def _blame(self, cross, v, vo, vp, vv, a_phase) -> float:
        am = a_phase
        eps = self.p["phase_eps"]
        if am is not None and am > eps:
            b_phys = 1.0 if cross > 0.0 else 0.0
        elif am is not None and am < -eps:
            b_phys = 1.0 if cross < 0.0 else 0.0
        else:
            b_phys = 0.5
        b = b_phys
        if vp is not None:
            d2 = cross * cross
            b_pred = 0.5 + (abs(v - vp) - abs(vo - vp)) / (abs(cross) + 1e-9)
            b_pred = 0.0 if b_pred < 0.0 else (1.0 if b_pred > 1.0 else b_pred)
            alpha = d2 / (d2 + 4.0 * vv) if vv is not None else 0.5
            b = alpha * b_pred + (1.0 - alpha) * b_phys
        return b
