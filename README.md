# 🛡️ AegisAI v3.1 - Enterprise Cyber Defense & Explainable AI (XAI) Platform

AegisAI is an autonomous, hybrid intrusion detection and threat intelligence platform that combines **Machine Learning (XGBoost, Random Forest)** and **Deep Learning (PyTorch Multi-Layer Perceptron)** with **TreeSHAP Explainable AI** to detect and mitigate network cyber attacks in real-time.

## 🚀 Key Features
- **Triple-Model Consensus:** Combines XGBoost, Random Forest, and PyTorch Neural Networks for 99.4%+ accuracy.
- **Explainable AI (TreeSHAP):** Provides exact game-theoretic Shapley values to explain *why* an alert was triggered.
- **Live OS Socket Scanner:** Scans active host machine TCP/UDP sockets (`netstat`) for live threat hunting.
- **Auto-Generated Mitigations:** Instantly generates Linux `iptables` rules and Snort IDS signatures.
- **Adversarial ML Lab:** Stress-tests model robustness against Gaussian feature noise and evasion attacks.

## 🛠️ Tech Stack
- **Backend:** Python, FastAPI, PyTorch, XGBoost, Scikit-Learn, Joblib, Uvicorn
- **Frontend:** Next.js, TypeScript, Tailwind CSS, Lucide Icons
- **Dataset:** NSL-KDD Cybersecurity Corpus (148,000+ evaluated flows)

## ⚙️ Local Setup
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/nahidmahmud71/AegisAI.git](https://github.com/nahidmahmud71/AegisAI.git)
   cd AegisAI