import bisect
import itertools
import json
import math
import os

X0, Y0, ZONE = 300000.0, 6100000.0, 37
K0, E0 = 0.9996, 500000.0
Z_OFFSET_M = 3.1
LOOP_FILE = "loop.json"
ROUTE_PREFIX = "route_"
DEPARTURE_SNAP_M = 60.0
CONNECT_MIN_M = 6.0
OFF_ROUTE_M = 50.0
DEPARTURES_FILE = "departures.json"
DEPARTURE_TIE_M = 0.5
BRANCHES_FILE = "branches.json"
MODEL_KIND = "route_envelope"
BRANCH_W_LO = 0.05
BRANCH_W_HI = 0.9
PATH_FILES = ("shchukinskaya-tallinskaya.json", "tallinskaya-shchukinskaya.json")
PARAMS_FILE = "map_params.json"
_CELL_M = 50.0
_A, _F = 6378137.0, 1.0 / 298.257223563


def _utm(lat: float, lon: float, zone: int = ZONE):
    e2 = _F * (2 - _F)
    ep2 = e2 / (1 - e2)
    phi, lam = math.radians(lat), math.radians(lon)
    lam0 = math.radians((zone - 1) * 6 - 180 + 3)
    sin_phi, cos_phi, tan_phi = math.sin(phi), math.cos(phi), math.tan(phi)
    n = _A / math.sqrt(1 - e2 * sin_phi * sin_phi)
    t = tan_phi * tan_phi
    c = ep2 * cos_phi * cos_phi
    a = cos_phi * (lam - lam0)
    e4, e6 = e2 * e2, e2 * e2 * e2
    m = _A * ((1 - e2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
              - (3 * e2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
              + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
              - (35 * e6 / 3072) * math.sin(6 * phi))
    east = K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * ep2) * a ** 5 / 120) + E0
    north = K0 * (m + n * tan_phi * (a * a / 2 + (5 - t + 9 * c + 4 * c * c) * a ** 4 / 24
                                     + (61 - 58 * t + t * t + 600 * c - 330 * ep2) * a ** 6 / 720))
    return east, north


def to_grid(lat: float, lon: float):
    east, north = _utm(lat, lon)
    return east - X0, north - Y0


def grid_scale(x: float) -> float:
    r = 6381000.0
    de = x + X0 - E0
    return K0 * (1.0 + de * de / (2.0 * K0 * K0 * r * r))


def _read_points(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    points = data["points"]
    paths = data.get("paths") or [{}]
    meta = paths[0]
    order = meta.get("point_indices") or list(range(len(points)))
    return [(float(points[i]["x"]), float(points[i]["y"]), float(points[i].get("z", 0.0))) for i in order], meta


def _read_departures(path: str, names, paths):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for d in data.get("departures", []):
        if d.get("route") not in names:
            continue
        pid, j = names.index(d["route"]), int(d["join_index"])
        base = paths[pid]
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in d["points"]]
        s_route = [float(v) for v in d.get("s_route", [])]
        if not 0 < j < len(base.x) - 1 or len(pts) < 2 or pts[-1][0] != base.x[j] or pts[-1][1] != base.y[j]:
            continue
        dp = _Path(pts, closed=False)
        if len(s_route) != len(dp.s) or any(b <= a for a, b in zip(s_route, s_route[1:])):
            continue
        out.append((pid, j, dp, s_route))
    return out


def _read_model(m):
    if not isinstance(m, dict) or m.get("kind") != MODEL_KIND:
        return None
    try:
        loc = [float(v) for v in m["route_loc"]]
        sd = [float(v) for v in m["route_scale"]]
        bin_m, nu, floor = float(m["bin_m"]), float(m["nu"]), float(m["scale_floor"])
        vmax, corr, prior = float(m["branch_v_max"]), float(m["corr_m"]), float(m["prior_branch"])
        e0, e1 = (float(v) for v in m["evidence_m"])
        r0, r1 = (float(v) for v in m["stop_revert_m"])
        stop_v, dead_end = float(m["stop_v"]), float(m["dead_end_m"])
    except (KeyError, TypeError, ValueError):
        return None
    vals = loc + sd + [bin_m, nu, floor, vmax, corr, prior, e0, e1, r0, r1, stop_v, dead_end]
    if (not loc or len(sd) != len(loc) or not all(math.isfinite(v) for v in vals) or min(sd) < 0.0
            or min(bin_m, nu, floor, vmax, corr) <= 0.0 or not 0.0 < prior < 1.0 or not 0.0 <= e0 < e1
            or r1 < r0 or stop_v < 0.0):
        return None
    log_c = math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu) - 0.5 * math.log(nu * math.pi)
    terms = tuple((-math.log(vmax) - log_c + math.log(max(floor, s)), mu, 1.0 / max(floor, s))
                  for mu, s in zip(loc, sd))
    return dict(bin_m=bin_m, terms=terms, half_nu1=0.5 * (nu + 1.0), inv_nu=1.0 / nu, v_max=vmax, inv_corr=1.0 / corr,
                prior_log_odds=math.log(prior / (1.0 - prior)), evidence=(e0, e1), stop=(r0, r1), stop_v=stop_v,
                dead_end=dead_end)


class _Branch:

    __slots__ = ("name", "pid", "fork_s", "path", "model", "bin_m", "terms", "half_nu1", "inv_nu", "v_max",
                 "inv_corr", "prior_log_odds", "evidence", "stop", "stop_v", "dead_end")

    def __init__(self, name, pid, fork_s, path, model):
        self.name, self.pid, self.fork_s, self.path = name, pid, fork_s, path
        self.model = model is not None
        m = model if model is not None else dict(
            bin_m=1.0, terms=(), half_nu1=0.0, inv_nu=0.0, v_max=0.0, inv_corr=0.0, prior_log_odds=-math.inf,
            evidence=(0.0, 0.0), stop=(math.inf, -math.inf), stop_v=0.0, dead_end=-math.inf)
        for k, v in m.items():
            setattr(self, k, v)

    def log_lr(self, j: int, v: float) -> float:
        a, mu, inv_s = self.terms[min(j, len(self.terms) - 1)]
        z = (min(max(v, 0.0), self.v_max) - mu) * inv_s
        return a + self.half_nu1 * math.log1p(z * z * self.inv_nu)


def _read_branches(path: str, names, paths):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for b in data.get("branches", []):
        if b.get("route") not in names:
            continue
        pid, j = names.index(b["route"]), int(b["fork_index"])
        base = paths[pid]
        pts = [(float(p[0]), float(p[1]), float(p[2])) for p in b["points"]]
        if not 0 < j < len(base.x) - 1 or len(pts) < 2 or pts[0][0] != base.x[j] or pts[0][1] != base.y[j]:
            continue
        full = _Path(list(zip(base.x[:j], base.y[:j], base.z[:j])) + pts, closed=False)
        out.append(_Branch(b.get("name", ""), pid, base.s[j], full, _read_model(b.get("model"))))
    return out


class BranchSelector:

    __slots__ = ("branch", "log_odds", "dead", "chosen", "w", "_d_prev")

    DEAD_STOP, DEAD_END = 1, 2

    def __init__(self, branch: _Branch):
        self.branch = branch
        self.reset()

    def reset(self) -> None:
        self.log_odds = self.branch.prior_log_odds
        self.dead = 0
        self.chosen = False
        self.w = 0.0
        self._d_prev = math.nan

    def update(self, s_route: float, v: float) -> bool:
        b = self.branch
        d = s_route - b.fork_s
        if not b.model or self.dead or not math.isfinite(d):
            self.chosen = False
            self.w = 0.0
            return False
        dp = self._d_prev
        if not d <= dp:
            self._d_prev = d
        if math.isfinite(v):
            lo, hi = b.evidence
            if dp > lo:
                lo = dp
            if d < hi:
                hi = d
            if hi > lo and dp == dp:
                w = b.bin_m
                j = int(lo / w)
                x = lo
                last = len(b.terms) - 1
                while x < hi:
                    edge = (j + 1) * w if j < last else hi
                    if edge > hi:
                        edge = hi
                    if edge > x:
                        self.log_odds += b.log_lr(j, v) * (edge - x) * b.inv_corr
                        x = edge
                    j += 1
            r0, r1 = b.stop
            if v < b.stop_v and r0 <= d <= r1:
                self.dead = self.DEAD_STOP
        if d > b.dead_end:
            self.dead = self.DEAD_END
        self.chosen = not self.dead and self.log_odds > 0.0
        lo = self.log_odds
        if self.dead:
            self.w = 0.0
        elif lo >= 0.0:
            self.w = 1.0 / (1.0 + math.exp(-lo))
        else:
            e = math.exp(lo)
            self.w = e / (1.0 + e)
        return self.chosen

    def weight(self) -> float:
        w = self.w
        if w < BRANCH_W_LO:
            return 0.0
        if w > BRANCH_W_HI:
            return 1.0
        return w


class _Path:

    def __init__(self, pts, closed: bool, meta=None):
        meta = meta or {}
        clean = [pts[0]]
        for p in pts[1:]:
            if math.hypot(p[0] - clean[-1][0], p[1] - clean[-1][1]) > 1e-6:
                clean.append(p)
        if closed and math.hypot(clean[0][0] - clean[-1][0], clean[0][1] - clean[-1][1]) > 1e-6:
            clean.append(clean[0])
        if len(clean) < 2:
            raise ValueError("a route path needs at least two distinct points")
        self.closed = closed
        self.x = [p[0] for p in clean]
        self.y = [p[1] for p in clean]
        self.z = [p[2] for p in clean]
        self.yaw = []
        self.s = [0.0]
        self.inc = []
        for i in range(len(clean) - 1):
            dx, dy = self.x[i + 1] - self.x[i], self.y[i + 1] - self.y[i]
            self.yaw.append(math.atan2(dy, dx))
            k = grid_scale(0.5 * (self.x[i] + self.x[i + 1]))
            d = math.hypot(dx, dy) / k
            self.inc.append(d)
            self.s.append(self.s[-1] + d)
        self.length = self.s[-1]
        k = grid_scale(self.x[0])
        self.departure_s = float(meta.get("departure_m", 0.0)) / k
        self.arrival_s = float(meta.get("arrival_m", 0.0)) / k

    @classmethod
    def prepended(cls, p, base: "_Path") -> "_Path":
        if base.closed or math.hypot(base.x[0] - p[0], base.y[0] - p[1]) <= 1e-6:
            return cls([p] + list(zip(base.x, base.y, base.z)), closed=False)
        path = cls.__new__(cls)
        dx, dy = base.x[0] - p[0], base.y[0] - p[1]
        path.closed = False
        path.x, path.y, path.z = [p[0]] + base.x, [p[1]] + base.y, [p[2]] + base.z
        path.yaw = [math.atan2(dy, dx)] + base.yaw
        path.inc = [math.hypot(dx, dy) / grid_scale(0.5 * (p[0] + base.x[0]))] + base.inc
        path.s = list(itertools.accumulate(path.inc, initial=0.0))
        path.length = path.s[-1]
        path.departure_s = path.arrival_s = 0.0
        return path

    def segment(self, s: float):
        if self.closed:
            s %= self.length
        i = bisect.bisect_right(self.s, s) - 1
        i = min(max(i, 0), len(self.s) - 2)
        return i, (s - self.s[i]) / (self.s[i + 1] - self.s[i])

    def pose(self, s: float):
        i, f = self.segment(s)
        x = self.x[i] + f * (self.x[i + 1] - self.x[i])
        y = self.y[i] + f * (self.y[i + 1] - self.y[i])
        fz = min(max(f, 0.0), 1.0)
        z = self.z[i] + fz * (self.z[i + 1] - self.z[i])
        return x, y, z, self.yaw[i]

    def project(self, i: int, x: float, y: float):
        ax, ay = self.x[i], self.y[i]
        dx, dy = self.x[i + 1] - ax, self.y[i + 1] - ay
        l2 = dx * dx + dy * dy
        t = ((x - ax) * dx + (y - ay) * dy) / l2
        t = min(max(t, 0.0), 1.0)
        fx, fy = ax + t * dx, ay + t * dy
        dist = math.hypot(x - fx, y - fy)
        side = dx * (y - ay) - dy * (x - ax)
        return dist, self.s[i] + t * (self.s[i + 1] - self.s[i]), math.copysign(dist, side)

    def behind_start(self, x: float, y: float, max_m: float):
        if self.closed:
            return None
        ax, ay = self.x[0], self.y[0]
        dx, dy = self.x[1] - ax, self.y[1] - ay
        l2 = dx * dx + dy * dy
        t = ((x - ax) * dx + (y - ay) * dy) / l2
        s = t * (self.s[1] - self.s[0])
        if t >= 0.0 or -s > max_m:
            return None
        return s, (dx * (y - ay) - dy * (x - ax)) / math.sqrt(l2)


class RouteMap:

    def __init__(self, paths, z_offset_m: float = Z_OFFSET_M, snap_departures: bool = False):
        self.paths = paths
        self._n_map = len(paths)
        self.snap_departures = snap_departures
        self.names = [f"path{i}" for i in range(len(paths))]
        self._dyn_base = None
        self.departures = []
        self._dep_paths = {}
        self._dyn_map = None
        self.branches = []
        self.z_offset_m = z_offset_m
        self._cells = {}
        for pid, p in enumerate(paths):
            for i in range(len(p.s) - 1):
                x0, x1 = sorted((p.x[i], p.x[i + 1]))
                y0, y1 = sorted((p.y[i], p.y[i + 1]))
                for cx in range(int(math.floor(x0 / _CELL_M)), int(math.floor(x1 / _CELL_M)) + 1):
                    for cy in range(int(math.floor(y0 / _CELL_M)), int(math.floor(y1 / _CELL_M)) + 1):
                        self._cells.setdefault((cx, cy), []).append((pid, i))

    @classmethod
    def load(cls, directory: str) -> "RouteMap":
        z_offset = Z_OFFSET_M
        params = os.path.join(directory, PARAMS_FILE)
        if os.path.isfile(params):
            with open(params, encoding="utf-8") as f:
                z_offset = float(json.load(f).get("z_offset_m", Z_OFFSET_M))
        loop = os.path.join(directory, LOOP_FILE)
        if os.path.isfile(loop):
            route = cls([_Path(*_read_points(loop)[:1], closed=True)], z_offset)
            route.names = [LOOP_FILE]
            return route
        names = sorted(n for n in os.listdir(directory) if n.startswith(ROUTE_PREFIX) and n.endswith(".json"))
        trips = bool(names)
        if not names:
            names = [n for n in PATH_FILES if os.path.isfile(os.path.join(directory, n))]
        if not names:
            raise FileNotFoundError(f"no route maps in {directory}")
        paths = []
        for n in names:
            pts, meta = _read_points(os.path.join(directory, n))
            paths.append(_Path(pts, closed=False, meta=meta))
        route = cls(paths, z_offset, snap_departures=trips)
        route.names = names
        dep = os.path.join(directory, DEPARTURES_FILE)
        if trips and os.path.isfile(dep):
            route.departures = _read_departures(dep, names, paths)
        br = os.path.join(directory, BRANCHES_FILE)
        if trips and os.path.isfile(br):
            route.branches = _read_branches(br, names, paths)
        return route

    def _candidates(self, x: float, y: float):
        cx, cy = int(math.floor(x / _CELL_M)), int(math.floor(y / _CELL_M))
        found, ring = [], 0
        while ring <= 4:
            for i in range(cx - ring, cx + ring + 1):
                for j in range(cy - ring, cy + ring + 1):
                    if max(abs(i - cx), abs(j - cy)) == ring:
                        found.extend(self._cells.get((i, j), ()))
            if found and ring >= 1:
                return found
            ring += 1
        return found or [(pid, i) for pid, p in enumerate(self.paths[:self._n_map]) for i in range(len(p.s) - 1)]

    def localize(self, x: float, y: float, heading=None):
        per_path, agree = {}, {}
        for pid, i in self._candidates(x, y):
            p = self.paths[pid]
            dist, s, lateral = p.project(i, x, y)
            if pid not in per_path or dist < per_path[pid][0]:
                per_path[pid] = (dist, s, lateral)
            if heading is not None and math.cos(p.yaw[i] - heading) > 0.1 and (pid not in agree or dist < agree[pid][0]):
                agree[pid] = (dist, s, lateral)
        nearest = min(c[0] for c in per_path.values())
        pool = per_path
        if heading is not None and agree and min(c[0] for c in agree.values()) <= nearest + 10.0:
            pool = agree
        pid = min(pool, key=lambda k: pool[k][0])
        dist, s, lateral = pool[pid]
        if dist > OFF_ROUTE_M:
            p = self.paths[pid]
            yaw = heading if heading is not None else p.pose(s)[3]
            z = p.pose(s)[2]
            return self._dynamic([(x, y, z), (x + math.cos(yaw), y + math.sin(yaw), z)], None), 0.0, 0.0
        path = self.paths[pid]
        if self.snap_departures and not path.closed and s > path.length - path.arrival_s:
            for other_id, c in sorted(pool.items(), key=lambda kv: kv[1][0]):
                other = self.paths[other_id]
                if other_id == pid or c[1] > other.departure_s or c[0] > DEPARTURE_SNAP_M:
                    continue
                if c[1] <= 0.5 and c[0] > CONNECT_MIN_M:
                    dyn = self._dynamic(_Path.prepended((x, y, other.z[0]), other), other_id)
                    return self._departure(other_id, x, y, 0.0) or (dyn, 0.0, 0.0)
                return self._departure(other_id, x, y, abs(c[2])) or ((other_id,) + self._on_extension(other, x, y, c[1], c[2]))
        return self._departure(pid, x, y, abs(lateral)) or ((pid,) + self._on_extension(path, x, y, s, lateral))

    def _on_extension(self, path, x: float, y: float, s: float, lateral: float):
        if self.snap_departures and s <= 0.0:
            behind = path.behind_start(x, y, CONNECT_MIN_M)
            if behind is not None:
                return behind
        return s, lateral

    def _departure(self, route_id: int, x: float, y: float, dist_route: float):
        best = None
        for k, (pid, j, dp, _sr) in enumerate(self.departures):
            if pid != route_id:
                continue
            for i in range(len(dp.s) - 1):
                c = dp.project(i, x, y)
                if best is None or c[0] < best[1][0]:
                    best = (k, c)
            behind = dp.behind_start(x, y, CONNECT_MIN_M)
            if behind is not None and (best is None or abs(behind[1]) < best[1][0]):
                best = (k, (abs(behind[1]), behind[0], behind[1]))
        if best is None or best[1][0] > dist_route + DEPARTURE_TIE_M:
            return None
        pid, j, dp, s_route = self.departures[best[0]]
        if best[1][1] >= dp.length:
            return None
        key = (best[0], id(dp))
        path = self._dep_paths.get(key)
        if path is None:
            base = self.paths[pid]
            pts = list(zip(dp.x, dp.y, dp.z)) + list(zip(base.x[j + 1:], base.y[j + 1:], base.z[j + 1:]))
            path = self._dep_paths[key] = _Path(pts, closed=False)
        dyn = self._dynamic(path, pid)
        self._dyn_map = (dp.s, s_route)
        self._dyn_base = (pid, s_route[-1] - dp.s[-1])
        return dyn, best[1][1], best[1][2]

    def _dynamic(self, pts, base_id=None) -> int:
        path = pts if isinstance(pts, _Path) else _Path(pts, closed=False)
        self._dyn_map = None
        if len(self.paths) > self._n_map:
            self.paths[self._n_map] = path
        else:
            self.paths.append(path)
        if base_id is None:
            self._dyn_base = None
        else:
            self._dyn_base = (base_id, -(path.length - self.paths[base_id].length))
        return self._n_map

    def base_of(self, path_id: int, s=None):
        if path_id is None:
            return None, 0.0
        if path_id < self._n_map:
            return path_id, 0.0
        if self._dyn_base is None:
            return None, 0.0
        if s is not None and self._dyn_map is not None and math.isfinite(s):
            ts, rs = self._dyn_map
            if s < ts[-1]:
                if s <= 0.0:
                    return self._dyn_base[0], rs[0]
                i = bisect.bisect_right(ts, s) - 1
                f = (s - ts[i]) / (ts[i + 1] - ts[i])
                return self._dyn_base[0], rs[i] + f * (rs[i + 1] - rs[i]) - s
        return self._dyn_base

    def branch_of(self, path_id: int):
        base, shift = self.base_of(path_id)
        for k, b in enumerate(self.branches):
            if base is not None and b.pid == base:
                return k, shift
        return None

    def branch_selector(self, k: int) -> BranchSelector:
        return BranchSelector(self.branches[k])

    def branch_pose(self, k: int, s: float):
        b = self.branches[k]
        if s < b.fork_s:
            return None
        x, y, z, yaw = b.path.pose(min(s, b.path.length))
        return x, y, z + self.z_offset_m, yaw

    def pose(self, path_id: int, s: float):
        x, y, z, yaw = self.paths[path_id].pose(s)
        return x, y, z + self.z_offset_m, yaw

    def length(self, path_id: int) -> float:
        return self.paths[path_id].length
