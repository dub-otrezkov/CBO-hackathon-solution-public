import os
import sqlite3
import sys

import numpy as np

TOPIC = "/localization/kinematic_state"


def main():
    bag_dir, out_dir = sys.argv[1], sys.argv[2]
    bag = os.path.basename(os.path.normpath(bag_dir))
    db = next(os.path.join(bag_dir, f) for f in sorted(os.listdir(bag_dir)) if f.endswith(".db3"))
    try:
        from rosbags.typesys import Stores, get_typestore
        ts = get_typestore(Stores.ROS2_HUMBLE)
        decode = ts.deserialize_cdr
    except ImportError:
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        msg_type = get_message("nav_msgs/msg/Odometry")
        decode = lambda raw, typ: deserialize_message(bytes(raw), msg_type)
    rows = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        tid = {n: (i, t) for i, n, t in con.execute("SELECT id, name, type FROM topics")}
        if TOPIC not in tid:
            raise SystemExit(f"{TOPIC} not in {db}")
        i, typ = tid[TOPIC]
        for (data,) in con.execute("SELECT data FROM messages WHERE topic_id = ? ORDER BY id", (i,)):
            m = decode(data, typ)
            p = m.pose.pose.position
            rows.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, p.x, p.y, p.z))
    a = np.array(rows)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, bag + ".npz"), t=a[:, 0], blx_h=a[:, 1], bly_h=a[:, 2], blz_h=a[:, 3],
             status=np.full(len(a), 2), mx=a[:, 1], my=a[:, 2])
    print(f"{bag}: {len(a)} reference epochs -> {os.path.join(out_dir, bag + '.npz')}")


if __name__ == "__main__":
    main()
