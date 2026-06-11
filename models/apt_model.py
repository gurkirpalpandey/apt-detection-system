"""
APT Detection Model Engine
Trains Random Forest, LSTM, and Isolation Forest models on network traffic data.
"""

import os
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# TensorFlow/Keras for LSTM
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    print("[WARNING] TensorFlow not found. LSTM model will be disabled.")

# ─────────────────────────────────────────────
#  Feature Columns (CICIDS-compatible)
# ─────────────────────────────────────────────
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

LABEL_COL = 'Label'

# Attack categories used in synthetic data generation
ATTACK_LABELS = [
    'BENIGN', 'APT-Recon', 'APT-LateralMovement',
    'APT-Exfiltration', 'APT-C2', 'DDoS', 'PortScan', 'BruteForce'
]


# ─────────────────────────────────────────────
#  Synthetic Data Generator (when no CICIDS)
# ─────────────────────────────────────────────
def generate_synthetic_data(n_samples=5000, random_state=42):
    """
    Generates synthetic but realistic-looking network traffic data
    for demonstration / initial training when CICIDS dataset is absent.
    """
    np.random.seed(random_state)
    records = []
    n_features = len(FEATURE_COLS)

    label_dist = {
        'BENIGN': 0.55,
        'APT-Recon': 0.08,
        'APT-LateralMovement': 0.07,
        'APT-Exfiltration': 0.07,
        'APT-C2': 0.06,
        'DDoS': 0.07,
        'PortScan': 0.05,
        'BruteForce': 0.05,
    }

    for label, frac in label_dist.items():
        n = int(n_samples * frac)
        if label == 'BENIGN':
            # Normal traffic: moderate flow, low variance
            base = np.abs(np.random.normal(loc=0.3, scale=0.15, size=(n, n_features)))
        elif label in ('APT-Recon', 'APT-LateralMovement'):
            # Slow, low-volume, spread out in time
            base = np.abs(np.random.normal(loc=0.15, scale=0.1, size=(n, n_features)))
            base[:, 0] *= 5   # long flow duration
            base[:, 12] *= 0.2  # very low packets/s
        elif label == 'APT-Exfiltration':
            # Large payload, few connections
            base = np.abs(np.random.normal(loc=0.4, scale=0.2, size=(n, n_features)))
            base[:, 3] *= 8   # large fwd bytes
            base[:, 6] *= 0.1  # small min packet length
        elif label == 'APT-C2':
            # Periodic beaconing: very regular IAT
            base = np.abs(np.random.normal(loc=0.2, scale=0.05, size=(n, n_features)))
            base[:, 14] *= 0.05  # very low IAT std (regular)
        elif label == 'DDoS':
            base = np.abs(np.random.normal(loc=0.8, scale=0.3, size=(n, n_features)))
            base[:, 12] *= 10  # very high packets/s
        elif label == 'PortScan':
            base = np.abs(np.random.normal(loc=0.1, scale=0.05, size=(n, n_features)))
            base[:, 33] *= 5   # high SYN flags
        elif label == 'BruteForce':
            base = np.abs(np.random.normal(loc=0.5, scale=0.2, size=(n, n_features)))
            base[:, 36] *= 4   # high ACK flags

        df_chunk = pd.DataFrame(base, columns=FEATURE_COLS)
        df_chunk[LABEL_COL] = label
        records.append(df_chunk)

    df = pd.concat(records, ignore_index=True).sample(frac=1, random_state=random_state)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.clip(lower=0)
    return df


# ─────────────────────────────────────────────
#  Data Loading
# ─────────────────────────────────────────────
def load_cicids_data(data_dir: str):
    """Load and combine CICIDS CSV files from a directory."""
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in csv_files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        dfs.append(df)
        print(f"  Loaded: {f} → {len(df):,} rows")

    return pd.concat(dfs, ignore_index=True)


def preprocess(df: pd.DataFrame, scaler=None, le=None, fit=True):
    """
    Cleans, encodes labels, scales features.
    Returns: X (array), y (array), scaler, label_encoder
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Keep only available feature columns
    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) < 10:
        raise ValueError(f"Too few feature columns found ({len(available)}). "
                         "Check that your CSV matches CICIDS format.")

    # Fill missing features with 0
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0

    X_df = df[FEATURE_COLS].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_df = X_df.clip(lower=0)

    # Binary label: BENIGN=0, else=1
    if LABEL_COL in df.columns:
        y_raw = df[LABEL_COL].astype(str).str.strip()
        y = (y_raw != 'BENIGN').astype(int).values
    else:
        y = None

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X_df.values)
    else:
        X = scaler.transform(X_df.values)

    return X, y, scaler


# ─────────────────────────────────────────────
#  Random Forest
# ─────────────────────────────────────────────
def train_random_forest(X_train, y_train):
    print("\n[RF] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    rf.fit(X_train, y_train)
    print("[RF] Done.")
    return rf


# ─────────────────────────────────────────────
#  Isolation Forest (anomaly detection)
# ─────────────────────────────────────────────
def train_isolation_forest(X_train):
    print("\n[IF] Training Isolation Forest (unsupervised)...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.15,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_train)
    print("[IF] Done.")
    return iso


# ─────────────────────────────────────────────
#  LSTM
# ─────────────────────────────────────────────
def build_lstm(input_dim: int):
    model = Sequential([
        LSTM(128, input_shape=(1, input_dim), return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_lstm(X_train, y_train, model_path='models/lstm_model.keras'):
    if not LSTM_AVAILABLE:
        print("[LSTM] Skipping — TensorFlow not installed.")
        return None

    print("\n[LSTM] Training LSTM model...")
    X_3d = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

    model = build_lstm(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_3d, y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.15,
        callbacks=[es],
        verbose=1
    )
    model.save(model_path)
    print(f"[LSTM] Saved to {model_path}")
    return model


# ─────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name} Evaluation")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall   : {rec*100:.2f}%")
    print(f"  F1-Score : {f1*100:.2f}%")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=['BENIGN', 'ATTACK']))

    return {'name': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1': f1, 'confusion_matrix': cm.tolist()}


# ─────────────────────────────────────────────
#  Full Training Pipeline
# ─────────────────────────────────────────────
def train_all(data_dir=None, use_synthetic=True, model_save_dir='models'):
    os.makedirs(model_save_dir, exist_ok=True)

    # 1. Load data
    if data_dir and os.path.isdir(data_dir):
        print(f"\n[DATA] Loading CICIDS data from: {data_dir}")
        df = load_cicids_data(data_dir)
    else:
        print("\n[DATA] No dataset directory given — generating synthetic data...")
        df = generate_synthetic_data(n_samples=8000)

    print(f"[DATA] Total records: {len(df):,}")
    print(f"[DATA] Label distribution:\n{df[LABEL_COL].value_counts()}\n")

    # 2. Preprocess
    X, y, scaler = preprocess(df, fit=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[DATA] Train: {X_train.shape}, Test: {X_test.shape}")

    results = []

    # 3. Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results.append(evaluate_model("Random Forest", y_test, rf_preds))

    # 4. Isolation Forest
    iso = train_isolation_forest(X_train)
    iso_raw = iso.predict(X_test)
    iso_preds = (iso_raw == -1).astype(int)
    results.append(evaluate_model("Isolation Forest", y_test, iso_preds))

    # 5. LSTM
    lstm_model = train_lstm(X_train, y_train,
                            model_path=os.path.join(model_save_dir, 'lstm_model.keras'))
    if lstm_model is not None:
        X_test_3d = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
        lstm_raw = lstm_model.predict(X_test_3d, verbose=0).flatten()
        lstm_preds = (lstm_raw > 0.5).astype(int)
        results.append(evaluate_model("LSTM", y_test, lstm_preds))

    # 6. Save artefacts
    with open(os.path.join(model_save_dir, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(model_save_dir, 'iso_model.pkl'), 'wb') as f:
        pickle.dump(iso, f)
    with open(os.path.join(model_save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    print("\n[SAVE] All models saved to:", model_save_dir)
    return results


# ─────────────────────────────────────────────
#  Inference (used by Flask app)
# ─────────────────────────────────────────────
class APTDetector:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.rf = None
        self.iso = None
        self.lstm = None
        self.scaler = None
        self._load_models()

    def _load_models(self):
        try:
            with open(os.path.join(self.model_dir, 'rf_model.pkl'), 'rb') as f:
                self.rf = pickle.load(f)
            with open(os.path.join(self.model_dir, 'iso_model.pkl'), 'rb') as f:
                self.iso = pickle.load(f)
            with open(os.path.join(self.model_dir, 'scaler.pkl'), 'rb') as f:
                self.scaler = pickle.load(f)
            lstm_path = os.path.join(self.model_dir, 'lstm_model.keras')
            if LSTM_AVAILABLE and os.path.exists(lstm_path):
                self.lstm = load_model(lstm_path)
            print("[APTDetector] All models loaded successfully.")
        except FileNotFoundError:
            print("[APTDetector] Models not found. Run train_all() first.")

    def predict(self, feature_dict: dict) -> dict:
        """
        Accepts a dict of feature_name → value.
        Returns ensemble threat score + individual model predictions.
        """
        row = {col: float(feature_dict.get(col, 0)) for col in FEATURE_COLS}
        X_raw = np.array([[row[c] for c in FEATURE_COLS]])
        X_scaled = self.scaler.transform(X_raw)

        scores = {}

        # RF probability
        if self.rf:
            rf_prob = self.rf.predict_proba(X_scaled)[0][1]
            scores['random_forest'] = float(rf_prob)

        # Isolation Forest: score_samples returns negative values; closer to -1 = anomaly
        if self.iso:
            iso_score = self.iso.score_samples(X_scaled)[0]
            # Normalise to 0-1 threat probability (roughly)
            iso_threat = float(np.clip((iso_score * -1 + 0.5), 0, 1))
            scores['isolation_forest'] = iso_threat

        # LSTM
        if self.lstm:
            X_3d = X_scaled.reshape((1, 1, X_scaled.shape[1]))
            lstm_prob = float(self.lstm.predict(X_3d, verbose=0)[0][0])
            scores['lstm'] = lstm_prob

        # Ensemble: weighted average
        weights = {'random_forest': 0.5, 'isolation_forest': 0.2, 'lstm': 0.3}
        total_w = sum(weights[k] for k in scores)
        ensemble = sum(scores[k] * weights[k] for k in scores) / total_w if total_w else 0

        threat_level = (
            'CRITICAL' if ensemble > 0.85 else
            'HIGH'     if ensemble > 0.65 else
            'MEDIUM'   if ensemble > 0.40 else
            'LOW'
        )

        # Best-guess attack type from RF
        attack_type = 'BENIGN'
        if self.rf and ensemble > 0.40:
            # Use feature patterns to approximate attack category
            flow_dur = row.get('Flow Duration', 0)
            pkt_rate = row.get('Flow Packets/s', 0)
            syn_flag = row.get('SYN Flag Count', 0)
            iat_std  = row.get('Flow IAT Std', 1)
            bwd_bytes = row.get('Total Length of Bwd Packets', 0)

            if pkt_rate > 0.7:
                attack_type = 'DDoS'
            elif syn_flag > 0.3 and pkt_rate > 0.3:
                attack_type = 'PortScan'
            elif flow_dur > 1.5 and pkt_rate < 0.2 and bwd_bytes > 0.3:
                attack_type = 'APT-Exfiltration'
            elif iat_std < 0.05:
                attack_type = 'APT-C2'
            elif flow_dur > 1.0 and pkt_rate < 0.15:
                attack_type = 'APT-LateralMovement'
            else:
                attack_type = 'APT-Recon'

        return {
            'ensemble_score': round(ensemble, 4),
            'threat_level': threat_level,
            'attack_type': attack_type,
            'model_scores': {k: round(v, 4) for k, v in scores.items()},
            'is_threat': ensemble > 0.40
        }


if __name__ == '__main__':
    print("=== APT Detection System — Training Mode ===")
    results = train_all(use_synthetic=True, model_save_dir='models')
    print("\n=== Training Complete ===")
