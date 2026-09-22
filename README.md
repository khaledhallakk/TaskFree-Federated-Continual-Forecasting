# Task-Free Federated Continual Forecasting with Two-Tier Replay

This repository accompanies the conference paper:

> **Mitigating Catastrophic Forgetting in Task-Free Federated Continual Forecasting via Two-Tier Replay under Concept Drift**  
> Khaled Hallak and Oudom Kem

The repository provides the implementation and supplementary material for a drift-aware federated continual-learning framework for multistep time-series forecasting. The framework detects concept drift from prediction-window error and triggers model adaptation without predefined task labels or known regime boundaries.

## Overview

Federated time-series forecasting must address both client heterogeneity and temporal non-stationarity. A model trained during an offline period may become inaccurate as local data distributions evolve, while continual adaptation may overwrite previously learned knowledge.

The proposed **TwoTierReplay** method separates replay memory according to temporal function:

- a **fixed offline tier** containing representative samples from the base period; and
- a **bounded online FIFO tier** containing recent revealed stream samples.

The offline tier supports long-term retention, whereas the online tier supports adaptation to recent distributional changes. Model updates are event-driven: clients monitor forecasting error, and federated adaptation is triggered only after persistent threshold violations.

## Main contributions

1. A task-free federated forecasting protocol in which adaptation is triggered by persistent forecasting-error degradation rather than predefined task boundaries.
2. A two-tier replay mechanism that protects representative base-period samples from being overwritten by incoming stream samples.
3. A controlled comparison with Static, Naive continual learning, NormalReplay, Elastic Weight Consolidation (EWC), Learning without Forgetting (LwF), and Synaptic Intelligence (SI).
4. An evaluation across 12 federated clients, three forecasting targets, three prediction horizons, and three random seeds.

## Implemented methods

| Method | Description |
|---|---|
| **Static** | Evaluates the offline FedAvg model on the stream without continual adaptation. |
| **Naive CL** | Performs drift-triggered adaptation using recent stream samples without an explicit forgetting-mitigation mechanism. |
| **NormalReplay** | Uses one replay buffer shared by representative base samples and incoming stream samples. |
| **TwoTierReplay** | Uses a fixed offline representative memory and a separate bounded online FIFO memory. |
| **EWC** | Penalizes changes to parameters considered important for the base model. |
| **LwF** | Regularizes the adapted model toward predictions produced by the previous teacher model. |
| **SI** | Uses path-based parameter-importance weights to reduce forgetting during adaptation. |

## Dataset and forecasting task

Experiments use the [Beijing Multi-Site Air-Quality Dataset](https://doi.org/10.24432/C5RK5G) from the UCI Machine Learning Repository.

| Component | Setting |
|---|---|
| Federated clients | 12 monitoring stations |
| Targets | `TEMP`, `PM2.5`, and `WSPM` |
| Forecasting horizons | 6, 12, and 24 hours |
| Input length | Previous 12 hourly observations |
| Offline/base period | 2013-03-01 00:00 to 2014-03-01 23:00 |
| Online stream period | 2014-03-02 00:00 to 2017-02-28 23:00 |
| Stream order | Chronological; no stream shuffling |

Each station is treated as an independent federated client. Missing numerical values are handled using the time-aware interpolation and filling procedure implemented in the notebook. Wind direction is represented using sine and cosine components, and hour, day, and month variables are cyclically encoded. Feature-scaling bounds are computed from the offline base period and then held fixed during online evaluation.

## Forecasting model

The experiments use a one-layer long short-term memory network with:

- hidden dimension: 64;
- linear output head;
- Adam optimizer; and
- batch size: 32.

The continual-learning framework is not restricted to this architecture; the LSTM is used as a common forecasting model so that the comparison focuses on the adaptation strategy.

## Drift detection and adaptation

For client `k`, the drift threshold is calibrated from base-period prediction-window errors:

```text
tau_k = median(E_k) + 1.4826 * MAD(E_k)
```

A drift alarm is confirmed after `M = 2` consecutive threshold violations. When at least one client confirms drift, a federated adaptation event is initiated. Clients perform method-specific local adaptation, and the server aggregates the updated model parameters using FedAvg. If no client confirms drift, the model continues forecasting without communication or retraining.

## TwoTierReplay

The proposed memory contains two distinct components:

1. **Offline memory:** representative base-period forecasting samples selected using KMeans. This tier is fixed after initialization and cannot be overwritten by stream samples.
2. **Online memory:** a FIFO queue containing at most 300 recent stream samples.

During replay, the offline tier is selected with probability `p_off = 0.8`, and the online tier is selected with probability `0.2` when both tiers are nonempty. This corresponds to an expected offline-to-online replay ratio of `4:1`.

## Main experimental configuration

| Parameter | Value |
|---|---:|
| Random seeds | 42, 43, 44 |
| Input length | 12 hours |
| Base FedAvg rounds | 100 |
| Base local epochs per round | 1 |
| Base learning rate | 1e-4 |
| Continual-adaptation rounds | 5 |
| Continual local epochs per round | 1 |
| Continual learning rate | 1e-3 |
| Warm-up windows | 1 |
| MAD sensitivity coefficient | 1 |
| MAD normal-consistency factor | 1.4826 |
| Drift persistence `M` | 2 |
| Representative base-sample fraction | 0.10 |
| NormalReplay buffer capacity | 20,000 samples |
| Online FIFO capacity | 300 samples |
| Replay fraction | 0.30 |
| Current-data loss weight | 0.5 |
| Replay-loss weight | 2.0 |
| Offline replay probability `p_off` | 0.8 |
| Batch size | 32 |

The notebook configuration is the authoritative source for method-specific regularization and distillation weights and for any target-horizon-specific settings selected during tuning. All methods use the same data split, seeds, streaming order, drift protocol, and forecasting architecture.

## Evaluation metrics

The evaluation considers plasticity on the online stream and retention on the base distribution.

The base-retention forgetting measure is:

```text
AF_base = RMSE_base(final) - RMSE_base(initial)
```

where `RMSE_base(initial)` is the offline model's base-period RMSE and `RMSE_base(final)` is the final adapted model's RMSE on the same base evaluation data.

The principal stability-plasticity score is:

```text
SP = RMSE_stream + AF_base
```

Lower `SP` indicates a better balance between online adaptation and base-period retention. Positive `AF_base` indicates forgetting, whereas negative `AF_base` indicates improved performance on the base evaluation data after adaptation.

## Main results

The following values are the mean stability-plasticity scores over three random seeds. Lower values are better.

| Method | TEMP-6 | TEMP-12 | TEMP-24 | PM2.5-6 | PM2.5-12 | PM2.5-24 | WSPM-6 | WSPM-12 | WSPM-24 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Static | 0.2413 | 0.2430 | 0.2503 | 0.1582 | 0.1620 | 0.1660 | 0.1504 | 0.1569 | 0.1637 | 0.1880 |
| Naive | 0.2139 | 0.2292 | 0.2432 | 0.1614 | 0.1642 | 0.1686 | 0.1767 | 0.1678 | 0.1723 | 0.1886 |
| EWC | 0.2043 | 0.2285 | 0.2425 | 0.1532 | 0.1561 | 0.1619 | 0.1512 | 0.1580 | 0.1638 | 0.1800 |
| LwF | 0.2139 | 0.2292 | 0.2434 | 0.1567 | 0.1606 | 0.1631 | 0.1742 | 0.1741 | 0.1801 | 0.1898 |
| SI | 0.2139 | 0.2293 | 0.2430 | 0.1588 | 0.1605 | 0.1639 | 0.1553 | 0.1580 | 0.1632 | 0.1841 |
| NormalReplay | 0.1966 | 0.2099 | 0.2186 | 0.1502 | 0.1562 | 0.1637 | 0.1577 | 0.1627 | 0.1680 | 0.1760 |
| **TwoTierReplay** | **0.1827** | **0.1939** | **0.2006** | **0.1083** | **0.1352** | **0.1515** | **0.1316** | **0.1448** | **0.1546** | **0.1559** |

TwoTierReplay obtains the lowest mean score in all nine target-horizon configurations. Its overall mean score of 0.1559 improves on NormalReplay, the strongest baseline, by approximately 11.4%.

## Supplementary diagnostics and reproducibility records

The supplementary result files associated with this repository are intended to report, for every method, target, horizon, and random seed:

- stream RMSE;
- base-retention forgetting `AF_base`;
- stability-plasticity score;
- detected local drift events;
- global federated-adaptation events;
- CPU time;
- per-seed values; and
- the mean and standard deviation across seeds.

The supplementary material should also retain the complete method-specific hyperparameter configurations and tuning protocol. These files must be generated directly from the final notebook runs so that the reported values remain consistent with the paper. Numerical standard deviations and CPU-time values are not reproduced in this README because they are not reported in the main paper.

## Repository setup

Clone the repository:

```bash
git clone https://github.com/khaledhallakk/TaskFree-Federated-Continual-Forecasting.git
cd TaskFree-Federated-Continual-Forecasting
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Download the Beijing Multi-Site Air-Quality Dataset from the UCI repository and place its 12 station CSV files in:

```text
AirQuality/
```

The expected station filenames are documented in `AirQuality/README.md`.

Then open and run the notebook from the first cell to the last:

```text
FLTA2026-Task-Free-Federated-Continual-Forecasting.ipynb
```

The notebook should be executed separately for the required target-horizon configurations and seeds, using the configuration cells included in the notebook.

## Reproducibility checklist

Before comparing a reproduced run with the paper, verify that:

- the 12 station files are available in `AirQuality/`;
- the chronological offline-online split is unchanged;
- feature-scaling bounds are fitted using only the offline base period;
- the stream is not shuffled;
- predictions are generated before their corresponding targets are revealed;
- adaptation occurs only after the targets become available;
- seeds 42, 43, and 44 are used;
- all methods use the same forecasting architecture and stream protocol; and
- the reported means and standard deviations are computed from the corresponding per-seed outputs.

## Citation

If you use this implementation, please cite:

> K. Hallak and O. Kem, “Mitigating Catastrophic Forgetting in Task-Free Federated Continual Forecasting via Two-Tier Replay under Concept Drift,” accepted conference paper, 2026.

The final proceedings citation and BibTeX entry should replace this provisional citation once the official publication metadata are available.

## Contact

- **Khaled Hallak**, American University in Dubai (AUD): `khallak@aud.edu`
- **Oudom Kem**, CEA LIST: `oudom.kem@cea.fr`
