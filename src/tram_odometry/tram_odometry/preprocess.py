from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass, field

KMH_TO_MS = 1.0 / 3.6

BOGIES = ("front", "rear")

FLAG_NONFINITE = "nonfinite"
FLAG_RANGE = "range"
FLAG_BACKSTEP = "backstep"
FLAG_DUPLICATE = "duplicate"
FLAG_FUTURE = "future"
FLAG_BOGIE = "bogie"
FLAG_ERROR = "error"
FLAG_OUTLIER = "outlier"
FLAG_STALE = "stale"
FLAG_GAP = "gap"
FLAG_PARTNER_STALE = "partner_stale"
FLAG_EPOCH = "epoch"

UNUSABLE_FLAGS = frozenset({FLAG_OUTLIER, FLAG_STALE})

DEFAULTS = {
    "v_max": 30.0,
    "gap_s": 0.5,
    "stale_s": 0.5,
    "median_n": 5,
    "jump_abs": 0.8,
    "jump_accel": 6.0,
    "resync_n": 50,
    "future_s": 2.0,
    "future_confirm_n": 2,
}


@dataclass(frozen=True)
class WheelSample:

    bogie: str
    stamp: float
    v: float
    flags: frozenset = field(default_factory=frozenset)
    raw: float = math.nan
    dt: float = math.nan

    @property
    def usable(self) -> bool:
        return not (self.flags & UNUSABLE_FLAGS)


def _as_float(x):
    if isinstance(x, bool) or not isinstance(x, numbers.Real):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return f if math.isfinite(f) else None


def _median(values) -> float:
    s = sorted(values)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else 0.5 * (s[m - 1] + s[m])


class _BogieState:
    __slots__ = ("last_stamp", "latest", "values", "stamps", "bad_run", "epoch_pending", "held_t", "held_n")

    def __init__(self, n: int):
        self.last_stamp = None
        self.latest = None
        self.values = deque(maxlen=n)
        self.stamps = deque(maxlen=n)
        self.bad_run = 0
        self.epoch_pending = False
        self.held_t = None
        self.held_n = 0

    def reset(self):
        self.last_stamp = None
        self.latest = None
        self.values.clear()
        self.stamps.clear()
        self.bad_run = 0
        self.epoch_pending = False
        self.held_t = None
        self.held_n = 0


class WheelPreprocessor:

    def __init__(self, params: dict | None = None, **kwargs):
        p = dict(DEFAULTS)
        if params:
            p.update(params)
        p.update(kwargs)
        unknown = set(p) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown WheelPreprocessor params: {sorted(unknown)}")
        self.v_max = float(p["v_max"])
        self.gap_s = float(p["gap_s"])
        self.stale_s = float(p["stale_s"])
        self.median_n = max(3, int(p["median_n"]))
        self.jump_abs = float(p["jump_abs"])
        self.jump_accel = float(p["jump_accel"])
        self.resync_n = max(1, int(p["resync_n"]))
        self.future_s = float(p["future_s"])
        self.future_confirm_n = max(0, int(p["future_confirm_n"]))
        self.params = p
        self._state = {b: _BogieState(self.median_n) for b in BOGIES}
        self.last_reject = None
        self.counts = {k: 0 for k in (FLAG_NONFINITE, FLAG_BOGIE, FLAG_RANGE, FLAG_BACKSTEP, FLAG_DUPLICATE,
                                      FLAG_FUTURE, FLAG_ERROR, FLAG_OUTLIER, FLAG_STALE, FLAG_GAP, FLAG_PARTNER_STALE,
                                      FLAG_EPOCH, "accepted")}


    def latest(self, bogie: str):
        st = self._state.get(bogie)
        return None if st is None else st.latest

    def age(self, bogie: str, stamp: float) -> float:
        st = self._state.get(bogie)
        t = _as_float(stamp)
        if st is None or st.last_stamp is None or t is None:
            return math.inf
        return t - st.last_stamp

    def reset(self) -> None:
        for st in self._state.values():
            st.reset()
        self.last_reject = None


    def push(self, bogie: str, stamp: float, raw: float):
        try:
            return self._push(bogie, stamp, raw)
        except Exception:
            return self._reject(FLAG_ERROR)

    def _reject(self, reason: str):
        self.last_reject = reason
        self.counts[reason] = self.counts.get(reason, 0) + 1
        return None

    def _push(self, bogie, stamp, raw):
        st = self._state.get(bogie) if isinstance(bogie, str) else None
        if st is None:
            return self._reject(FLAG_BOGIE)
        t = _as_float(stamp)
        r = _as_float(raw)
        if t is None or r is None:
            return self._reject(FLAG_NONFINITE)
        v = r * KMH_TO_MS
        if abs(v) > self.v_max:
            return self._reject(FLAG_RANGE)

        flags = set()
        other = self._state["rear" if bogie == "front" else "front"]
        if st.last_stamp is not None and t <= st.last_stamp:
            st.bad_run += 1
            if st.bad_run < self.resync_n:
                return self._reject(FLAG_BACKSTEP if t < st.last_stamp else FLAG_DUPLICATE)
            st.reset()
            if other.last_stamp is not None and other.last_stamp > t + self.stale_s:
                other.reset()
                other.epoch_pending = True
            flags.add(FLAG_EPOCH)
        else:
            ref = st.last_stamp
            if ref is not None and other.last_stamp is not None and other.last_stamp > ref:
                ref = other.last_stamp
            if ref is not None and t - ref > self.future_s:
                if st.held_t is not None and 0.0 < t - st.held_t <= self.gap_s:
                    st.held_n += 1
                else:
                    st.held_n = 1
                st.held_t = t
                confirmed = (other.held_t is not None and abs(t - other.held_t) <= self.stale_s) \
                    or st.held_n > self.future_confirm_n
                if not confirmed:
                    return self._reject(FLAG_FUTURE)
        st.bad_run = 0
        st.held_t = None
        st.held_n = 0
        if st.epoch_pending:
            st.epoch_pending = False
            flags.add(FLAG_EPOCH)

        dt = math.nan if st.last_stamp is None else t - st.last_stamp
        if dt == dt and dt > self.gap_s:
            flags.add(FLAG_GAP)
            st.values.clear()
            st.stamps.clear()

        if other.last_stamp is None or abs(t - other.last_stamp) > self.stale_s:
            if other.last_stamp is not None and other.last_stamp - t > self.stale_s:
                flags.add(FLAG_STALE)
            else:
                flags.add(FLAG_PARTNER_STALE)

        if len(st.values) >= 3:
            med = _median(st.values)
            t_med = _median(st.stamps)
            allowed = self.jump_abs + self.jump_accel * max(0.0, t - t_med)
            if abs(v - med) > allowed:
                flags.add(FLAG_OUTLIER)
        st.values.append(v)
        st.stamps.append(t)

        sample = WheelSample(bogie=bogie, stamp=t, v=v, flags=frozenset(flags), raw=r, dt=dt)
        st.last_stamp = t
        st.latest = sample
        self.last_reject = None
        self.counts["accepted"] += 1
        for f in flags:
            self.counts[f] = self.counts.get(f, 0) + 1
        return sample
