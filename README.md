# TaskFree-Federated-Continual-Forecasting

# 🧠 Task-Free Federated Continual Forecasting with Two-Tier Replay

This repository provides a reproducible notebook implementation for the conference paper:

**Mitigating Catastrophic Forgetting in Task-Free Federated Continual Forecasting via Two-Tier Replay under Concept Drift**

All experiments are performed using the **Beijing Multi-Site Air Quality Dataset**, where each monitoring station acts as an independent federated client.

---

## 📘 Overview

Federated time-series forecasting is challenging when data arrive as a continuous non-stationary stream. In this setting, models must adapt to concept drift without losing the historical knowledge learned during the offline base phase.

This repository implements a task-free federated continual-learning benchmark in which drift is detected from prediction-window RMSE rather than from predefined task labels. The proposed **TwoTierReplay** method separates replay memory into:

- a fixed offline tier containing representative base-period samples, and
- an online FIFO tier containing recent revealed stream samples.

This separation is designed to improve the stability-plasticity trade-off during drift-aware adaptation.

---

## 🧩 Implemented Methods

| Method | Description |
|---|---|
| 🧱 Static | Offline FedAvg model evaluated on the stream without continual adaptation. |
| 🔄 Naive CL | Drift-triggered continual learning using only recent stream samples. |
| 🔁 NormalReplay | Single unified replay buffer initialized from base samples and updated with stream samples. |
| 🧠 TwoTierReplay | Proposed two-tier memory with fixed offline representative memory and online FIFO stream memory. |
| 🧮 EWC | Elastic Weight Consolidation using Fisher-based regularization. |
| 🧑‍🏫 LwF | Learning without Forgetting using teacher-student prediction distillation. |
| 🔗 SI | Synaptic Intelligence using importance-weighted parameter regularization. |

---

## 📊 Dataset & Task Definition

- **Dataset:** Beijing Multi-Site Air Quality Dataset
- **Clients:** 12 monitoring stations
- **Targets:** `TEMP`, `PM2.5`, `WSPM`
- **Forecasting horizons:** 6, 12, and 24 hours
- **Input window:** past 12 time steps
- **Base/offline period:** 2013-03-01 to 2014-03-01 23:00:00
- **Continual stream period:** 2014-03-02 to 2017-03-03 23:00:00
- **Normalization:** robust global scaling computed from the base period only
- **Federated setting:** each station is treated as one client

Place the 12 CSV files inside the `AirQuality/` folder before running the notebook.

---

## ⚙️ Experimental Configuration

The main notebook uses the final article configuration:

```python
CONFIG = {
    "targets": ["TEMP", "PM2.5", "WSPM"],
    "horizons": [6, 12, 24],
    "seeds": [42, 43, 44],
    "n_lags": 12,
    "base_rounds": 100,
    "base_lr": 1e-4,
    "warmup_windows": 1,
    "k_mad": 1,
    "persistence_M": 2,
    "buffer_capacity": 20000,
    "sample_fraction": 0.10,
    "online_capacity": 300,
    "R": 5,
    "E": 1,
    "cl_lr": 1e-3,
    "replay_fraction": 0.30,
    "lambda_current": 0.5,
    "lambda_replay": 2,
    "p_offline": 0.8,
}




🧪 Reproducibility & Usage
1. Clone the repository
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
2. Install dependencies
pip install -r requirements.txt
3. Add the dataset

Place the 12 Beijing air-quality CSV files inside:

AirQuality/

The notebook expects the filenames listed in AirQuality/README.md.

4. Run the notebook

Open and run:

FLTA2026-Task-Free-Federated-Continual-Forecasting.ipynb






