"""
Network Traffic Simulator
Generates realistic mock traffic flows for live dashboard demonstration.
In production, replace this with actual packet capture (Scapy / Wireshark).
"""

import random
import time
import threading
import queue
import numpy as np
from datetime import datetime

FEATURE_COLS = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std',
    'Fwd IAT Total', 'Fwd IAT Mean', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Min Packet Length', 'Max Packet Length', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1',
    'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets',
    'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
    'act_data_pkt_fwd', 'min_seg_size_forward',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

PROTOCOLS = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'FTP']
SUBNETS   = ['192.168.1', '192.168.2', '10.0.0', '172.16.0', '10.10.10']


def _random_ip(subnet=None):
    if subnet is None:
        subnet = random.choice(SUBNETS)
    return f"{subnet}.{random.randint(1, 254)}"


def _gen_benign():
    n = len(FEATURE_COLS)
    base = np.abs(np.random.normal(0.3, 0.15, n))
    return dict(zip(FEATURE_COLS, base))


def _gen_apt_recon():
    base = _gen_benign()
    base['Flow Duration'] *= 5
    base['Flow Packets/s'] *= 0.15
    base['SYN Flag Count'] *= 3
    base['Total Fwd Packets'] *= 0.3
    return base


def _gen_apt_lateral():
    base = _gen_benign()
    base['Flow Duration'] *= 6
    base['Flow Packets/s'] *= 0.12
    base['Total Length of Fwd Packets'] *= 0.5
    base['Fwd Packets/s'] *= 0.2
    return base


def _gen_apt_exfil():
    base = _gen_benign()
    base['Total Length of Bwd Packets'] *= 8
    base['Bwd Packet Length Max'] *= 6
    base['Flow Bytes/s'] *= 3
    base['Flow Duration'] *= 4
    return base


def _gen_apt_c2():
    base = _gen_benign()
    base['Flow IAT Std'] *= 0.04   # very regular
    base['Flow IAT Mean'] *= 2
    base['Flow Packets/s'] *= 0.1
    base['ACK Flag Count'] *= 3
    return base


def _gen_ddos():
    base = _gen_benign()
    base['Flow Packets/s'] *= 12
    base['Total Fwd Packets'] *= 15
    base['SYN Flag Count'] *= 8
    base['Flow Bytes/s'] *= 10
    return base


def _gen_portscan():
    base = _gen_benign()
    base['SYN Flag Count'] *= 6
    base['RST Flag Count'] *= 5
    base['Flow Duration'] *= 0.05
    base['Total Fwd Packets'] *= 0.1
    return base


TRAFFIC_GENERATORS = {
    'BENIGN':               (_gen_benign,       0.60),
    'APT-Recon':            (_gen_apt_recon,    0.07),
    'APT-LateralMovement':  (_gen_apt_lateral,  0.06),
    'APT-Exfiltration':     (_gen_apt_exfil,    0.06),
    'APT-C2':               (_gen_apt_c2,       0.05),
    'DDoS':                 (_gen_ddos,         0.07),
    'PortScan':             (_gen_portscan,     0.05),
    'BruteForce':           (_gen_benign,       0.04),
}


def generate_traffic_flow():
    """Generate one synthetic network flow."""
    labels  = list(TRAFFIC_GENERATORS.keys())
    weights = [TRAFFIC_GENERATORS[l][1] for l in labels]
    label   = random.choices(labels, weights=weights, k=1)[0]
    gen_fn  = TRAFFIC_GENERATORS[label][0]
    features = gen_fn()

    # Clip to non-negative
    features = {k: max(0.0, float(v)) for k, v in features.items()}

    src_ip   = _random_ip()
    dst_ip   = _random_ip()
    src_port = random.randint(1024, 65535)
    dst_port = random.choice([80, 443, 22, 21, 53, 445, 3389, 8080, 8443,
                               random.randint(1, 1023)])
    protocol = random.choice(PROTOCOLS)

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'dst_port': dst_port,
        'protocol': protocol,
        'true_label': label,
        'features': features
    }


# ─────────────────────────────────────────────
#  Background traffic feed (producer thread)
# ─────────────────────────────────────────────
class TrafficFeed:
    """
    Runs in a background thread, continuously generating traffic flows
    and placing them into a thread-safe queue.
    """
    def __init__(self, interval=1.5, max_queue=500):
        self.interval = interval
        self.q = queue.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def get_flow(self, timeout=0.1):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        while not self._stop_event.is_set():
            flow = generate_traffic_flow()
            try:
                self.q.put_nowait(flow)
            except queue.Full:
                self.q.get_nowait()   # drop oldest
                self.q.put_nowait(flow)
            time.sleep(self.interval + random.uniform(-0.3, 0.3))
