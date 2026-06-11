"""
APT Detection System — Flask Web Application
Dashboard for real-time threat monitoring, manual analysis, and reporting.
"""

import os
import sys
import json
import time
import threading
from datetime import datetime
from collections import deque

from flask import Flask, render_template, jsonify, request

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.apt_model import APTDetector, FEATURE_COLS
from utils.traffic_simulator import TrafficFeed, generate_traffic_flow

# ─────────────────────────────────────────────
#  App Initialisation
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = 'apt-detection-secret-2024'

# Global state
detector     = APTDetector(model_dir='models')
traffic_feed = TrafficFeed(interval=2.0)
traffic_feed.start()

# In-memory circular buffers
MAX_ALERTS = 200
MAX_TRAFFIC = 500

alerts_log   = deque(maxlen=MAX_ALERTS)   # threat detections only
traffic_log  = deque(maxlen=MAX_TRAFFIC)  # all flows
stats = {
    'total_flows':    0,
    'total_threats':  0,
    'critical':       0,
    'high':           0,
    'medium':         0,
    'low_threats':    0,
    'attack_counts':  {},
    'protocol_counts': {},
    'hourly_traffic': [0] * 24,
    'started_at':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
}
stats_lock = threading.Lock()


# ─────────────────────────────────────────────
#  Background Monitor Thread
# ─────────────────────────────────────────────
def monitor_loop():
    """Continuously pulls from traffic feed, runs detection, updates state."""
    while True:
        flow = traffic_feed.get_flow(timeout=0.5)
        if flow is None:
            continue

        result = detector.predict(flow['features'])
        hour   = datetime.now().hour

        entry = {
            'id':          stats['total_flows'] + 1,
            'timestamp':   flow['timestamp'],
            'src_ip':      flow['src_ip'],
            'dst_ip':      flow['dst_ip'],
            'src_port':    flow['src_port'],
            'dst_port':    flow['dst_port'],
            'protocol':    flow['protocol'],
            'threat_level': result['threat_level'],
            'attack_type':  result['attack_type'],
            'score':        result['ensemble_score'],
            'is_threat':    result['is_threat'],
            'model_scores': result['model_scores'],
        }

        with stats_lock:
            stats['total_flows'] += 1
            stats['hourly_traffic'][hour] += 1
            proto = flow['protocol']
            stats['protocol_counts'][proto] = stats['protocol_counts'].get(proto, 0) + 1

            traffic_log.appendleft(entry)

            if result['is_threat']:
                stats['total_threats'] += 1
                lvl = result['threat_level']
                if lvl == 'CRITICAL':
                    stats['critical'] += 1
                elif lvl == 'HIGH':
                    stats['high'] += 1
                elif lvl == 'MEDIUM':
                    stats['medium'] += 1
                else:
                    stats['low_threats'] += 1

                atype = result['attack_type']
                stats['attack_counts'][atype] = stats['attack_counts'].get(atype, 0) + 1
                alerts_log.appendleft(entry)


monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
monitor_thread.start()


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def api_stats():
    with stats_lock:
        s = dict(stats)
        s['detection_rate'] = (
            round(s['total_threats'] / s['total_flows'] * 100, 1)
            if s['total_flows'] > 0 else 0
        )
    return jsonify(s)


@app.route('/api/alerts')
def api_alerts():
    limit = int(request.args.get('limit', 50))
    with stats_lock:
        data = list(alerts_log)[:limit]
    return jsonify(data)


@app.route('/api/traffic')
def api_traffic():
    limit = int(request.args.get('limit', 100))
    with stats_lock:
        data = list(traffic_log)[:limit]
    return jsonify(data)


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Manual single-flow analysis endpoint.
    Accepts JSON with feature key-values.
    """
    data = request.get_json(force=True)
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    result = detector.predict(data)
    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return jsonify(result)


@app.route('/api/simulate_attack', methods=['POST'])
def api_simulate_attack():
    """
    Injects a guaranteed attack flow for demo purposes.
    Body: { "attack_type": "APT-C2" }
    """
    body = request.get_json(force=True) or {}
    attack = body.get('attack_type', 'APT-Exfiltration')

    from utils.traffic_simulator import (
        _gen_apt_recon, _gen_apt_lateral, _gen_apt_exfil,
        _gen_apt_c2, _gen_ddos, _gen_portscan
    )
    gen_map = {
        'APT-Recon':           _gen_apt_recon,
        'APT-LateralMovement': _gen_apt_lateral,
        'APT-Exfiltration':    _gen_apt_exfil,
        'APT-C2':              _gen_apt_c2,
        'DDoS':                _gen_ddos,
        'PortScan':            _gen_portscan,
    }
    gen_fn = gen_map.get(attack, _gen_apt_exfil)
    features = {k: max(0.0, float(v)) for k, v in gen_fn().items()}
    result = detector.predict(features)
    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    result['simulated'] = True
    result['attack_type'] = attack
    return jsonify(result)


@app.route('/api/recent_chart')
def api_recent_chart():
    """Last 60 flows: timestamp + score for sparkline chart."""
    with stats_lock:
        data = [
            {'t': e['timestamp'][-8:], 'score': e['score'], 'threat': e['is_threat']}
            for e in list(traffic_log)[:60]
        ]
    return jsonify(list(reversed(data)))


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  APT Detection System — Starting Dashboard")
    print("  URL: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
