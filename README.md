# SIGMA
SIGMA (Smart Intelligent Grid Management Architecture) .SIGMA is an AI framework designed to bring intelligence, resilience, and automation to modern power grids. This project integrates simulation data, machine learning models, and real‑time monitoring pipelines to detect, classify, and respond to faults across multiple layers of the grid.

# SIGMA – Smart Intelligent Grid Management Architecture

SIGMA is an AI framework for **real‑time smart grid fault detection, anomaly analysis, and resilient infrastructure management**.  
It integrates **simulation data, machine learning models, fusion logic, and alerting services** into a modular pipeline that demonstrates how AI can strengthen critical infrastructure.

---

## 📌 Overview

Modern power grids face increasing complexity due to renewable integration, distributed generation, and rising demand.  
Faults can occur at multiple levels from topology and connectivity issues to equipment degradation.  
SIGMA addresses this challenge by combining **simulation tools (MATLAB/Simulink, pandapower)** with **machine learning models (PyTorch, PyTorch Geometric, scikit‑learn)** to build a layered detection and prediction system.



## 🎯 Project Goals

- **Topology & Connectivity Faults** → Detect feeder outages, islanding, and network disruptions using Graph Neural Networks (GNNs).  
- **Grid Instability & Power Quality** → Identify voltage sags, frequency deviations, and harmonics with LSTMs and temporal models.  
- **Electrical Fault Classification** → Recognize SLG, LL, DLG, 3‑phase, and high‑impedance faults using Autoencoders and anomaly detection.  
- **Equipment Degradation & RUL** → Predict Remaining Useful Life (RUL) for transformers, cables, and switchgear.  
- **Fusion & Alerts** → Merge outputs into a unified decision engine and trigger real‑time alerts for corrective actions.



## 🛠️ Tech Stack

- **Simulation** → MATLAB / Simulink, pandapower, NetworkX  
- **Machine Learning** → PyTorch, PyTorch Lightning, PyTorch Geometric  
- **Data Handling** → Pandas, NumPy  
- **Visualization & Alerts** → Matplotlib, Seaborn, custom alert service  
- **Collaboration** → GitHub (branch‑based workflow, PR reviews)



## 🚀 Getting Started

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/SIGMA.git
   cd SIGMA

2.**Install dependencies**
   pip install -r requirements.txt



