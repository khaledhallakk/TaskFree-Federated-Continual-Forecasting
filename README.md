# Corrected PERC-free federated continual-forecasting baseline

This is a clean research baseline derived from the FLTA/Article 2027 code audit. It is **not** a continuation of the concatenated notebook and contains no PERC or PERC-Lite implementation. The seven retained methods are Static, Naive, NormalReplay, TwoTierReplay, EWC, LwF, and SI.

No training starts when a module or notebook is imported. Journal-scale execution is also locked behind an explicit command-line confirmation.

## What is corrected

| Risk in the previous notebook | Corrected behavior |
|---|---|
| One client's drift alarm updated every client | Only locally drifting clients participate and receive an aggregate; all other client models remain untouched |
| A multi-horizon target could enter replay before its full horizon elapsed | A window issued at origin `o` is released only at `o + horizon`; target values are not materialized in prediction records before then |
| Future/backward imputation and whole-period preprocessing could leak information | Causal forward fill is used, fallback medians and scaling bounds are fitted on base-training rows only |
| Quantile scaling clipped stream extremes and could hide drift | Scaling remains linear outside the fitted quantile interval; stream targets are never clipped |
| Base retention was measured on training data | Base train, validation, and untouched retention splits are chronological and disjoint; boundary-crossing targets are excluded |
| NormalReplay and TwoTierReplay could have unequal effective memory | Both begin with identical representatives and use the same effective total capacity; only their retention policy differs |
| Forecast horizon also implicitly controlled several unrelated mechanisms | `horizon`, prediction `stride`, detector persistence, recent capacity, and online capacity are independent settings |
| Hidden or unseeded random generators changed replay samples | Dataset generation, selection, batching, replay, model training, and every grid context have derived deterministic seeds |
| Duplicated definitions and a bottom-of-file full-grid call made behavior order-dependent | One importable implementation per component, a small CLI, a safe notebook launcher, and no import-time execution |
| Result path variables could be undefined after a run | Every result path is created in one typed output manifest and printed only after successful completion |

## Causal online protocol

At raw stream step `s`, the engine performs these operations in order:

1. Release all training windows whose complete targets are now observable (`available_step <= s`).
2. Score forecasts maturing at `s` and update each client's own persistent detector.
3. If a client triggers, adapt only the triggering client set and return an aggregate only to that set.
4. Issue forecasts scheduled at origin `s` using the newly available history and current model.

With `prediction_stride: "horizon"`, evaluated forecasts are non-overlapping, while every causally mature stream window remains available for training. Set the stride to `1` for rolling forecasts. The event and label traces make the ordering auditable.

The default continual aggregation is `weighted_delta`: participant updates are averaged, then the average delta is applied to each participating client's own starting model. This remains meaningful after client models diverge. `fedavg_state` is available only as an explicit ablation. Neither mode writes to a non-participant.

## Project layout

```text
fcl_baseline/
  config.py           typed grid and validation
  data.py             Beijing, generic CSV, and synthetic adapters
  models.py           LSTM/GRU registry
  memory.py           deterministic FIFO and two-tier memories
  drift.py            robust threshold calibration and persistence
  federated.py        base FedAvg, aggregation, Fisher, SI artifacts
  methods.py          seven PERC-free strategies
  engine.py           shared causal prequential engine
  evaluation.py       stream, retention, lead, client, and cost metrics
  experiment.py       grid orchestration and atomic checkpoints
configs/
  synthetic_verification.json
  beijing_smoke.json
  beijing_full_template.json
tests/
Corrected_Baseline_Quickstart.ipynb
run_experiments.py
```

## Safe start

Create an isolated Python environment and install the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Run the verification suite:

```bash
python -m unittest discover -s tests -v
python run_experiments.py --config configs/synthetic_verification.json --validate-only
python run_experiments.py --config configs/synthetic_verification.json
```

The last command is a tiny synthetic engineering test. It must not be reported as an experimental finding.

For Beijing, first correct `datasets[0].path` in a copy of `configs/beijing_smoke.json`, then validate it:

```bash
python run_experiments.py --config configs/my_beijing_smoke.json --validate-only
```

The full template has `execution_mode: "full"`; the runner refuses it unless `--confirm-full-run` is supplied. Do not unlock it until the real-data smoke outputs and protocol choices have been inspected.

## Extending the baseline without editing the engine

### Seeds, targets, horizons, and ablations

These are configuration lists. A method ablation is another method entry with the same `name`, a unique `label`, and different `params`. Random order is shared across ablations of the same context. Method parameters can be overridden through `params_by_dataset`, `params_by_target`, or `params_by_horizon`.

Example:

```json
{
  "name": "ewc",
  "label": "EWC-lambda10",
  "params": {"regularization": 10.0},
  "params_by_horizon": {"24": {"regularization": 30.0}}
}
```

Two-tier hyperparameters currently live in the shared `replay` block. For a clean ablation, use separate configuration files or add a small strategy parameter only after the baseline is frozen.

### Another forecasting architecture

Register a factory in `models.py` (or a new imported model module) under a new name. It must accept `[batch, lags, features]` and return `[batch, horizon]`. LSTM and GRU already demonstrate this contract; the engine and metrics need no changes.

### Another dataset or feature set

- For numeric CSVs, use `generic_csv_clients` and configure timestamp, target, feature, and optional client-ID columns.
- For domain-specific categorical variables or irregular sampling, add a dedicated adapter through the dataset registry.
- Each dataset entry owns its dates, lags, features, targets, horizons, scaling scope, and client semantics.

Multiple dataset entries can coexist in one grid. Dataset-specific targets and horizons override the global fallbacks.

### A new continual-learning method

Subclass `ContinualStrategy`, implement only its memory or extra loss behavior, and register it. Do not duplicate the stream loop. This guarantees that causality, local participation, drift detection, metrics, and communication accounting stay identical across methods.

## Outputs and primary metrics

Each completed run writes atomic/checkpointed files for:

- raw context/seed summaries;
- per-client metrics and worst-client performance;
- per-lead horizon metrics;
- drift calibration statistics;
- an auditable event/label trace;
- mean and standard deviation across seeds;
- method ranking and an optional Excel workbook.

The primary lower-is-better score is

\[
\text{SP} = \operatorname{RMSE}_{stream,norm}
 + \lambda \max(0,\operatorname{RMSE}_{retain,final}-\operatorname{RMSE}_{retain,base}).
\]

The signed conference-style companion is also reported. Using positive forgetting for the primary score prevents improvement on the base-retention set from numerically cancelling poor stream forecasting. Rankings across heterogeneous targets use normalized metrics; original-unit RMSE/MAE are retained for interpretation.

## Scientific boundaries of this baseline

- The synthetic run verifies software behavior only; no Beijing or journal-scale experiment has been run here.
- Hyperparameters in the templates are starting points, not validated optima.
- EWC uses an empirical squared-gradient diagonal on base-training data.
- SI uses a documented federated path-integral approximation accumulated during base training; it should be described as such rather than as an exact centralized SI implementation.
- Federated partitioning alone is not a privacy guarantee. Add privacy/security claims only if the corresponding mechanisms and threat model are evaluated.
- `global_base_train` scaling computes pooled simulation statistics. If raw-statistic sharing conflicts with the deployment threat model, use client-local scaling or implement an explicitly evaluated privacy-preserving aggregation protocol.
- Freeze this baseline and its protocol before introducing the proposed journal method. Any later correction must be rerun for every comparator.

See `AUDIT_AND_VERIFICATION.md` for the implementation evidence and remaining pre-experiment checks.
