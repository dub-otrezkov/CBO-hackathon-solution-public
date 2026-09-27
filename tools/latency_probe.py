import os
import signal
import time
from collections import deque

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tram_vehicle_msgs.msg import VelocitySensor

try:
    from tram_vehicle_msgs.msg import DriverControllerCommand
except ImportError:
    DriverControllerCommand = None

INPUTS = ("/vehicle/front_bogie_velocity", "/vehicle/rear_bogie_velocity")
CONTROLLER = "/vehicle/driver_position_cmd"
OUTPUTS = ("/result/velocity", "/result/position")
MAX_INPUT_STAMPS = 20000


def stamp_ns(msg):
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def percentile(values, q):
    if not values:
        return float("nan")
    s = sorted(values)
    return s[min(len(s) - 1, int(round(q / 100.0 * (len(s) - 1))))]


def find_node_pid():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(a.endswith("/odometry_node") for a in argv[:2]):
            return int(pid)
    return None


class Probe(Node):
    def __init__(self):
        super().__init__("latency_probe")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1000, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.arrivals = {}
        self.order = deque()
        self.lat = {t: [] for t in OUTPUTS}
        self.unmatched = {t: 0 for t in OUTPUTS}
        self.recv = {t: [] for t in OUTPUTS}
        self.frames = {}
        self.cov = [0, 0]
        for t in INPUTS:
            self.create_subscription(VelocitySensor, t, self._input, qos)
        if DriverControllerCommand is not None:
            self.create_subscription(DriverControllerCommand, CONTROLLER, self._input, qos)
        self.create_subscription(VelocitySensor, OUTPUTS[0], lambda m: self._output(OUTPUTS[0], m), qos)
        self.create_subscription(Odometry, OUTPUTS[1], self._odometry, qos)
        self.pid = None
        self.last_pid = None
        self.cpu_samples = []
        self.rss_samples = []
        self._last_cpu = None
        self.create_timer(1.0, self._sample_resources)

    def _input(self, msg):
        now = time.monotonic_ns()
        s = stamp_ns(msg)
        if s not in self.arrivals:
            self.order.append(s)
            if len(self.order) > MAX_INPUT_STAMPS:
                self.arrivals.pop(self.order.popleft(), None)
        self.arrivals[s] = now

    def _output(self, topic, msg):
        now = time.monotonic_ns()
        self.recv[topic].append(now)
        self.frames.setdefault(topic, msg.header.frame_id)
        t_in = self.arrivals.get(stamp_ns(msg))
        if t_in is None or t_in > now:
            self.unmatched[topic] += 1
        else:
            self.lat[topic].append((now - t_in) / 1e6)

    def _odometry(self, msg):
        self._output(OUTPUTS[1], msg)
        self.frames.setdefault("child_frame_id", msg.child_frame_id)
        self.cov[0] += any(c != 0.0 for c in msg.pose.covariance)
        self.cov[1] += any(c != 0.0 for c in msg.twist.covariance)

    def _sample_resources(self):
        if self.pid is None:
            self.pid = find_node_pid()
            if self.pid is None:
                return
            self.last_pid = self.pid
        try:
            with open(f"/proc/{self.pid}/stat") as f:
                fields = f.read().rsplit(")", 1)[1].split()
            ticks = int(fields[11]) + int(fields[12])
            with open(f"/proc/{self.pid}/status") as f:
                rss_kib = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:"))
        except (OSError, StopIteration, ValueError):
            self.pid = None
            return
        now = time.monotonic()
        if self._last_cpu is not None:
            dt = now - self._last_cpu[0]
            if dt > 0:
                self.cpu_samples.append((ticks - self._last_cpu[1]) / os.sysconf("SC_CLK_TCK") / dt)
        self._last_cpu = (now, ticks)
        self.rss_samples.append(rss_kib / 1024.0)

    def report(self):
        lines = []
        for t in OUTPUTS:
            r = self.recv[t]
            dur = (r[-1] - r[0]) / 1e9 if len(r) > 1 else 0.0
            rate = (len(r) - 1) / dur if dur > 0 else float("nan")
            lat = self.lat[t]
            lines.append(
                f"{t}: n={len(r)}, rate={rate:.1f} Hz, frame_id={self.frames.get(t, '?')}, "
                f"latency p50={percentile(lat, 50):.2f} ms p95={percentile(lat, 95):.2f} ms "
                f"max={max(lat) if lat else float('nan'):.2f} ms (matched {len(lat)}, unmatched {self.unmatched[t]})")
        lines.append(f"/result/position child_frame_id={self.frames.get('child_frame_id', '?')}, "
                     f"messages with pose covariance {self.cov[0]}, with twist covariance {self.cov[1]}")
        if self.cpu_samples:
            lines.append(f"odometry_node pid {self.last_pid}: CPU p95={percentile(self.cpu_samples, 95):.2f} cores "
                         f"max={max(self.cpu_samples):.2f} cores; RSS max={max(self.rss_samples):.1f} MiB "
                         f"({len(self.cpu_samples)} samples, 1 s)")
        else:
            lines.append("odometry_node process not found in /proc (resources not measured)")
        return "\n".join(lines)


def main():
    rclpy.init()
    node = Probe()
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    last = time.monotonic()
    try:
        while rclpy.ok() and not stop["flag"]:
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.monotonic() - last > 30.0:
                last = time.monotonic()
                print(node.report(), flush=True)
            seen = node.recv[OUTPUTS[1]]
            if seen and time.monotonic_ns() - seen[-1] > 10_000_000_000:
                break
    finally:
        print("final:\n" + node.report(), flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
