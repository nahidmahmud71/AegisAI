# ==========================================
# AegisAI v3.1 - FastAPI Cyber Defense, XAI & Live OS Socket Engine
# ==========================================
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import subprocess
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import xgboost as xgb
import numpy as np
import pandas as pd
import os

app = FastAPI(
    title="AegisAI Cyber Defense Engine v3.1",
    description="Hybrid ML, PyTorch Deep Learning, TreeSHAP XAI & Real-Time OS Socket Scanner",
    version="3.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AegisDeepNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(AegisDeepNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_PATH = os.path.join(BASE_DIR, "aegis_artifacts.pkl")
DL_MODEL_PATH = os.path.join(BASE_DIR, "aegis_dl_model.pth")

def build_compatible_local_models():
    print("⚙️ Building native models compatible with local Python & XGBoost version...")
    train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
    ]

    train_df = pd.read_csv(train_url, names=columns)
    test_df = pd.read_csv(test_url, names=columns)
    full_df = pd.concat([train_df, test_df], ignore_index=True)

    dos_attacks = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop', 'apache2', 'udpstorm', 'processtable', 'worm', 'mailbomb']
    probe_attacks = ['satan', 'ipsweep', 'nmap', 'portsweep', 'mscan', 'saint']
    r2l_attacks = ['guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster', 'warezclient', 'spy', 'xlock', 'xsnoop', 'snmpguess', 'snmpgetattack', 'httptunnel', 'sendmail', 'named']
    u2r_attacks = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl', 'sqlattack', 'xterm', 'ps']

    def map_attack(label):
        if label == 'normal': return 'Normal'
        if label in dos_attacks: return 'DoS / DDoS'
        if label in probe_attacks: return 'Port Scan / Probe'
        if label in r2l_attacks: return 'Brute Force / R2L'
        if label in u2r_attacks: return 'Root Exploit / U2R'
        return 'Other Attack'

    full_df['attack_category'] = full_df['label'].apply(map_attack)

    selected_features = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'logged_in', 'count',
        'srv_count', 'serror_rate', 'same_srv_rate', 'diff_srv_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate'
    ]
    categorical_cols = ['protocol_type', 'service', 'flag']

    X_all = full_df[selected_features].copy()
    feature_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_all[col] = le.fit_transform(X_all[col].astype(str))
        feature_encoders[col] = le

    target_encoder = LabelEncoder()
    y_all = target_encoder.fit_transform(full_df['attack_category'])

    X_train_bal, X_test_bal, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test_bal)

    rf_model = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)
    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled)) * 100

    xgb_model = XGBClassifier(n_estimators=80, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    xgb_model.fit(X_train_scaled, y_train)
    xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled)) * 100

    dl_model = AegisDeepNet(input_dim=len(selected_features), num_classes=len(target_encoder.classes_))
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dl_model.parameters(), lr=0.002)

    dl_model.train()
    for _ in range(6):
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(dl_model(bx), by)
            loss.backward()
            optimizer.step()

    dl_model.eval()
    with torch.no_grad():
        dl_preds = torch.argmax(dl_model(torch.tensor(X_test_scaled, dtype=torch.float32)), dim=1).numpy()
    dl_acc = accuracy_score(y_test, dl_preds) * 100

    sample_presets = {}
    for class_idx, class_name in enumerate(target_encoder.classes_):
        matching_rows = X_test_bal[y_test == class_idx]
        if len(matching_rows) > 0:
            row_dict = matching_rows.iloc[0].to_dict()
            for cat_col in categorical_cols:
                row_dict[cat_col] = str(feature_encoders[cat_col].inverse_transform([int(row_dict[cat_col])])[0])
            sample_presets[class_name] = row_dict

    artifacts_bundle = {
        'xgb_model': xgb_model,
        'rf_model': rf_model,
        'scaler': scaler,
        'feature_encoders': feature_encoders,
        'target_encoder': target_encoder,
        'selected_features': selected_features,
        'categorical_cols': categorical_cols,
        'dl_architecture': {
            'input_dim': len(selected_features),
            'num_classes': len(target_encoder.classes_)
        },
        'model_metrics': {
            'random_forest_acc': round(rf_acc, 2),
            'xgboost_acc': round(xgb_acc, 2),
            'deep_learning_acc': round(dl_acc, 2),
            'total_trained_records': int(len(X_train_bal)),
            'total_tested_records': int(len(X_test_bal))
        },
        'sample_presets': sample_presets
    }

    joblib.dump(artifacts_bundle, ARTIFACTS_PATH)
    torch.save(dl_model.state_dict(), DL_MODEL_PATH)
    print("✅ Local Models Built & Saved Successfully!")
    return artifacts_bundle, dl_model

try:
    artifacts = joblib.load(ARTIFACTS_PATH)
    dl_arch = artifacts["dl_architecture"]
    dl_model = AegisDeepNet(input_dim=dl_arch["input_dim"], num_classes=dl_arch["num_classes"])
    dl_model.load_state_dict(torch.load(DL_MODEL_PATH, map_location=torch.device("cpu")))
    dl_model.eval()
    print("✅ Pre-trained AegisAI Models Loaded Successfully!")
except Exception as load_err:
    print(f"⚠️ Version mismatch detected ({load_err}). Rebuilding models natively...")
    artifacts, dl_model = build_compatible_local_models()

xgb_model = artifacts["xgb_model"]
rf_model = artifacts["rf_model"]
scaler = artifacts["scaler"]
feature_encoders = artifacts["feature_encoders"]
target_encoder = artifacts["target_encoder"]
selected_features = artifacts["selected_features"]
categorical_cols = artifacts["categorical_cols"]
model_metrics = artifacts["model_metrics"]
sample_presets = artifacts["sample_presets"]

MITIGATION_PLAYBOOK = {
    "Normal": {
        "severity": "Low",
        "status": "Safe",
        "mitre_id": "NONE (Benign Flow)",
        "mitre_tactic": "Verified Baseline Activity",
        "risk_score": 4,
        "summary": "Traffic flow matches verified benign baseline behavior with zero protocol anomalies.",
        "firewall_rule": "# No block rule needed. Allow standard flow:\niptables -A INPUT -p tcp -m state --state ESTABLISHED,RELATED -j ACCEPT",
        "snort_rule": "# Benign traffic signature - no alert triggered",
        "actions": [
            "No immediate firewall intervention required.",
            "Continue routine packet telemetry indexing in SIEM."
        ]
    },
    "DoS / DDoS": {
        "severity": "Critical",
        "status": "Attack Detected",
        "mitre_id": "T1498 / T1499",
        "mitre_tactic": "Impact • Network Denial of Service",
        "risk_score": 98,
        "summary": "High-volume volumetric SYN flood or connection exhaustion attack targeting service availability.",
        "firewall_rule": "iptables -A INPUT -p tcp --syn -m limit --limit 1/s --limit-burst 3 -j ACCEPT\niptables -A INPUT -p tcp --syn -j DROP",
        "snort_rule": "alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:\"AEGIS-AI DoS/SYN Flood Detected\"; flags:S; threshold:type both, track by_src, count 100, seconds 5; sid:900001; rev:1;)",
        "actions": [
            "Enable SYN cookies and rate-limiting on edge ingress routers immediately.",
            "Quarantine offending source IP subnet via automated BGP blackhole rule.",
            "Redirect ingress traffic through automated DDoS scrubbing pipeline."
        ]
    },
    "Port Scan / Probe": {
        "severity": "High",
        "status": "Attack Detected",
        "mitre_id": "T1046 / T1595",
        "mitre_tactic": "Discovery • Network Service Scanning",
        "risk_score": 78,
        "summary": "Reconnaissance activity sweeping open ports and enumerating active network daemons.",
        "firewall_rule": "iptables -A INPUT -p tcp --tcp-flags ALL NONE -j DROP\niptables -A INPUT -p icmp --icmp-type echo-request -m limit --limit 1/s -j ACCEPT",
        "snort_rule": "alert ip $EXTERNAL_NET any -> $HOME_NET any (msg:\"AEGIS-AI Reconnaissance Port Sweep\"; detection_filter:track by_src, count 30, seconds 10; sid:900002; rev:1;)",
        "actions": [
            "Drop incoming unauthorized ICMP echo and TCP SYN probe packets.",
            "Isolate scanning host IP and trigger SIEM reconnaissance alert.",
            "Audit exposed ports and obfuscate service version banners."
        ]
    },
    "Brute Force / R2L": {
        "severity": "High",
        "status": "Attack Detected",
        "mitre_id": "T1110 / T1078",
        "mitre_tactic": "Credential Access • Brute Force & R2L",
        "risk_score": 86,
        "summary": "Unauthorized remote-to-local authentication or credential stuffing attempt over network service.",
        "firewall_rule": "iptables -A INPUT -p tcp --dport 21:22 -m recent --name brute --set\niptables -A INPUT -p tcp --dport 21:22 -m recent --name brute --update --seconds 60 --hitcount 4 -j DROP",
        "snort_rule": "alert tcp $EXTERNAL_NET any -> $HOME_NET 21:22 (msg:\"AEGIS-AI R2L Credential Stuffing\"; flow:to_server,established; threshold:type both, track by_src, count 5, seconds 30; sid:900003; rev:1;)",
        "actions": [
            "Enforce immediate account lockout and step-up Multi-Factor Authentication (MFA).",
            "Terminate active suspicious FTP/SSH sessions on target host.",
            "Blacklist offending IP address in Intrusion Prevention System (IPS)."
        ]
    },
    "Root Exploit / U2R": {
        "severity": "Critical",
        "status": "Attack Detected",
        "mitre_id": "T1068 / T1548",
        "mitre_tactic": "Privilege Escalation • Root Exploit (U2R)",
        "risk_score": 99,
        "summary": "Critical privilege escalation attempt trying to spawn an unauthorized root or administrative shell.",
        "firewall_rule": "iptables -A INPUT -s <COMPROMISED_HOST_IP> -j DROP\niptables -A OUTPUT -d <COMPROMISED_HOST_IP> -j DROP",
        "snort_rule": "alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:\"AEGIS-AI U2R Root Shell Exploit Signature\"; content:\"/bin/sh\"; sid:900004; rev:1;)",
        "actions": [
            "Quarantine affected host from internal VLAN segment immediately.",
            "Revoke compromised user tokens and capture volatile memory dump for forensics.",
            "Audit SUID binaries and patch vulnerable local service daemons."
        ]
    }
}

class NetworkTrafficInput(BaseModel):
    features: Dict[str, Any]

class BatchTrafficInput(BaseModel):
    packets: List[Dict[str, Any]]

class AdversarialTestInput(BaseModel):
    features: Dict[str, Any]
    noise_level: float = 0.15

def preprocess_single_record(raw_features: Dict[str, Any]) -> np.ndarray:
    row = {}
    for feat in selected_features:
        val = raw_features.get(feat, 0)
        if feat in categorical_cols:
            le = feature_encoders[feat]
            val_str = str(val)
            if val_str in le.classes_:
                row[feat] = int(le.transform([val_str])[0])
            else:
                row[feat] = 0
        else:
            try:
                row[feat] = float(val)
            except (ValueError, TypeError):
                row[feat] = 0.0

    df = pd.DataFrame([row], columns=selected_features)
    return scaler.transform(df)

@app.get("/")
def root():
    return {
        "platform": "AegisAI Cyber Defense Engine v3.1",
        "status": "Online",
        "models_loaded": ["XGBoost", "Random Forest", "Deep Neural Network (PyTorch)", "TreeSHAP XAI", "Live OS Socket Scanner"]
    }

@app.get("/api/overview")
def get_system_overview():
    categorical_options = {
        col: list(feature_encoders[col].classes_) for col in categorical_cols
    }
    return {
        "metrics": model_metrics,
        "selected_features": selected_features,
        "categorical_cols": categorical_cols,
        "categorical_options": categorical_options,
        "classes": list(target_encoder.classes_),
        "sample_presets": sample_presets
    }

@app.post("/api/analyze")
def analyze_network_traffic(payload: NetworkTrafficInput):
    try:
        scaled_input = preprocess_single_record(payload.features)

        xgb_probs = xgb_model.predict_proba(scaled_input)[0]
        xgb_pred_idx = int(np.argmax(xgb_probs))
        xgb_pred_label = str(target_encoder.inverse_transform([xgb_pred_idx])[0])
        xgb_confidence = round(float(xgb_probs[xgb_pred_idx]) * 100, 2)

        rf_probs = rf_model.predict_proba(scaled_input)[0]
        rf_pred_idx = int(np.argmax(rf_probs))
        rf_pred_label = str(target_encoder.inverse_transform([rf_pred_idx])[0])
        rf_confidence = round(float(rf_probs[rf_pred_idx]) * 100, 2)

        with torch.no_grad():
            tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
            logits = dl_model(tensor_input)
            dl_probs = torch.softmax(logits, dim=1).numpy()[0]
            dl_pred_idx = int(np.argmax(dl_probs))
            dl_pred_label = str(target_encoder.inverse_transform([dl_pred_idx])[0])
            dl_confidence = round(float(dl_probs[dl_pred_idx]) * 100, 2)

        dmatrix = xgb.DMatrix(scaled_input, feature_names=selected_features)
        shap_contribs = xgb_model.get_booster().predict(dmatrix, pred_contribs=True)[0]

        if shap_contribs.shape[0] == len(target_encoder.classes_):
            class_shap_values = shap_contribs[xgb_pred_idx, :-1]
        else:
            class_shap_values = shap_contribs[:-1, xgb_pred_idx]

        shap_explanation = []
        for idx, feat_name in enumerate(selected_features):
            impact = float(class_shap_values[idx])
            shap_explanation.append({
                "feature": feat_name,
                "raw_value": payload.features.get(feat_name, 0),
                "shap_impact": round(impact, 4),
                "direction": "Increases Risk" if impact > 0 else "Decreases Risk"
            })

        shap_explanation.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        class_probabilities = {
            str(target_encoder.classes_[i]): round(float(xgb_probs[i]) * 100, 2)
            for i in range(len(target_encoder.classes_))
        }

        playbook = MITIGATION_PLAYBOOK.get(xgb_pred_label, MITIGATION_PLAYBOOK["Normal"])

        return {
            "primary_prediction": xgb_pred_label,
            "confidence": xgb_confidence,
            "severity": playbook["severity"],
            "status": playbook["status"],
            "mitre_id": playbook["mitre_id"],
            "mitre_tactic": playbook["mitre_tactic"],
            "risk_score": playbook["risk_score"],
            "summary": playbook["summary"],
            "firewall_rule": playbook["firewall_rule"],
            "snort_rule": playbook["snort_rule"],
            "recommended_actions": playbook["actions"],
            "model_consensus": {
                "xgboost": {"prediction": xgb_pred_label, "confidence": xgb_confidence},
                "random_forest": {"prediction": rf_pred_label, "confidence": rf_confidence},
                "deep_neural_network": {"prediction": dl_pred_label, "confidence": dl_confidence}
            },
            "class_probabilities": class_probabilities,
            "xai_top_features": shap_explanation[:8]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/adversarial-test")
def run_adversarial_stress_test(payload: AdversarialTestInput):
    try:
        base_scaled = preprocess_single_record(payload.features)
        noise_std = max(0.01, min(float(payload.noise_level), 1.0))

        rng = np.random.default_rng(42)
        perturbation = rng.normal(0, noise_std, size=base_scaled.shape)
        for idx, feat in enumerate(selected_features):
            if feat in categorical_cols:
                perturbation[0, idx] = 0.0

        adv_scaled = base_scaled + perturbation

        orig_xgb_probs = xgb_model.predict_proba(base_scaled)[0]
        orig_label = str(target_encoder.inverse_transform([int(np.argmax(orig_xgb_probs))])[0])
        orig_conf = round(float(np.max(orig_xgb_probs)) * 100, 2)

        adv_xgb_probs = xgb_model.predict_proba(adv_scaled)[0]
        adv_xgb_label = str(target_encoder.inverse_transform([int(np.argmax(adv_xgb_probs))])[0])
        adv_xgb_conf = round(float(np.max(adv_xgb_probs)) * 100, 2)

        adv_rf_probs = rf_model.predict_proba(adv_scaled)[0]
        adv_rf_label = str(target_encoder.inverse_transform([int(np.argmax(adv_rf_probs))])[0])
        adv_rf_conf = round(float(np.max(adv_rf_probs)) * 100, 2)

        with torch.no_grad():
            logits = dl_model(torch.tensor(adv_scaled, dtype=torch.float32))
            adv_dl_probs = torch.softmax(logits, dim=1).numpy()[0]
            adv_dl_label = str(target_encoder.inverse_transform([int(np.argmax(adv_dl_probs))])[0])
            adv_dl_conf = round(float(np.max(adv_dl_probs)) * 100, 2)

        models_resisted = sum([
            1 if adv_xgb_label == orig_label else 0,
            1 if adv_rf_label == orig_label else 0,
            1 if adv_dl_label == orig_label else 0,
        ])

        return {
            "noise_level_pct": round(noise_std * 100, 1),
            "original_prediction": orig_label,
            "original_confidence": orig_conf,
            "ensemble_robustness_score": round((models_resisted / 3.0) * 100, 1),
            "evasion_blocked": models_resisted >= 2,
            "perturbed_results": {
                "xgboost": {"prediction": adv_xgb_label, "confidence": adv_xgb_conf, "resisted": adv_xgb_label == orig_label},
                "random_forest": {"prediction": adv_rf_label, "confidence": adv_rf_conf, "resisted": adv_rf_label == orig_label},
                "deep_neural_network": {"prediction": adv_dl_label, "confidence": adv_dl_conf, "resisted": adv_dl_label == orig_label},
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adversarial test failed: {str(e)}")

@app.post("/api/batch-scan")
def batch_scan_packets(payload: BatchTrafficInput):
    try:
        results = []
        category_counts: Dict[str, int] = {}
        threats_detected = 0

        for idx, pkt in enumerate(payload.packets[:100]):
            scaled = preprocess_single_record(pkt)
            probs = xgb_model.predict_proba(scaled)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = str(target_encoder.inverse_transform([pred_idx])[0])
            conf = round(float(probs[pred_idx]) * 100, 2)

            playbook = MITIGATION_PLAYBOOK.get(pred_label, MITIGATION_PLAYBOOK["Normal"])
            if pred_label != "Normal":
                threats_detected += 1

            category_counts[pred_label] = category_counts.get(pred_label, 0) + 1

            results.append({
                "packet_index": idx + 1,
                "protocol": str(pkt.get("protocol_type", "tcp")),
                "service": str(pkt.get("service", "http")),
                "src_bytes": float(pkt.get("src_bytes", 0)),
                "count": int(float(pkt.get("count", 1))),
                "prediction": pred_label,
                "confidence": conf,
                "severity": playbook["severity"],
                "mitre_id": playbook["mitre_id"]
            })

        total = len(results)
        return {
            "total_scanned": total,
            "threats_detected": threats_detected,
            "benign_count": total - threats_detected,
            "category_breakdown": category_counts,
            "packets": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch scan failed: {str(e)}")

@app.get("/api/live-os-scan")
def scan_live_os_connections():
    try:
        output = subprocess.check_output(["netstat", "-an"], text=True, errors="ignore")
        lines = output.splitlines()
        raw_sockets = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0].upper() in ("TCP", "UDP"):
                proto = parts[0].lower()
                local_addr = parts[1]
                remote_addr = parts[2]
                state = parts[3].upper() if proto == "tcp" and len(parts) >= 4 else "ESTABLISHED"
                if "0.0.0.0" in remote_addr or "*:*" in remote_addr or "[::]" in remote_addr:
                    continue
                raw_sockets.append({
                    "proto": proto,
                    "local": local_addr,
                    "remote": remote_addr,
                    "state": state
                })

        packets = []
        category_counts: Dict[str, int] = {}
        threats_detected = 0

        for idx, sock in enumerate(raw_sockets[:30]):
            port_str = sock["remote"].split(":")[-1] if ":" in sock["remote"] else "80"
            try:
                port = int(port_str)
            except ValueError:
                port = 80

            if port in (80, 443, 3000, 8000, 8080):
                service = "http"
            elif port == 53:
                service = "domain_u"
            elif port == 22:
                service = "ssh"
            elif port == 21:
                service = "ftp_data"
            else:
                service = "other"

            flag = "SF"
            serror = 0.0
            if "SYN" in sock["state"]:
                flag = "S0"
                serror = 1.0
            elif "FIN" in sock["state"] or "CLOSE" in sock["state"]:
                flag = "RSTR"

            pkt_features = {
                "duration": 0,
                "protocol_type": sock["proto"],
                "service": service,
                "flag": flag,
                "src_bytes": 240 if flag == "SF" else 0,
                "dst_bytes": 4200 if flag == "SF" else 0,
                "logged_in": 1 if sock["state"] == "ESTABLISHED" else 0,
                "count": 6,
                "srv_count": 6,
                "serror_rate": serror,
                "same_srv_rate": 1.0,
                "diff_srv_rate": 0.0,
                "dst_host_count": 25,
                "dst_host_srv_count": 240,
                "dst_host_same_srv_rate": 1.0
            }

            scaled = preprocess_single_record(pkt_features)
            probs = xgb_model.predict_proba(scaled)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = str(target_encoder.inverse_transform([pred_idx])[0])
            conf = round(float(probs[pred_idx]) * 100, 2)

            playbook = MITIGATION_PLAYBOOK.get(pred_label, MITIGATION_PLAYBOOK["Normal"])
            if pred_label != "Normal":
                threats_detected += 1

            category_counts[pred_label] = category_counts.get(pred_label, 0) + 1

            packets.append({
                "packet_index": idx + 1,
                "protocol": f"{sock['proto'].upper()} ({sock['local']} → {sock['remote']})",
                "service": f"{service} [{sock['state']}]",
                "src_bytes": pkt_features["src_bytes"],
                "count": pkt_features["count"],
                "prediction": pred_label,
                "confidence": conf,
                "severity": playbook["severity"],
                "mitre_id": playbook["mitre_id"]
            })

        total = len(packets)
        return {
            "total_scanned": total,
            "threats_detected": threats_detected,
            "benign_count": total - threats_detected,
            "category_breakdown": category_counts,
            "packets": packets
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live OS scan failed: {str(e)}")