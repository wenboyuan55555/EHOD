# normal_demand

Paper table: `tab:regular-forecasting`. Evaluation: `full_od`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table normal_demand --city nyc
python scripts/run_experiment.py --table normal_demand --city chicago
python scripts/run_experiment.py --table normal_demand --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table normal_demand --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `ha` | `standalone` | HA |
| `xgboost` | `standalone` | XGBoost |
| `lstm` | `standalone` | LSTM |
| `duet` | `standalone` | DUET |
| `stid` | `standalone` | STID |
| `stidgcn` | `standalone` | STIDGCN |
| `patchstg` | `standalone` | PatchSTG |
| `stdn` | `standalone` | STDN |
| `lightst` | `standalone` | LightST |
| `strep` | `standalone` | ST-ReP |
| `stan` | `standalone` | STAN |
| `cmod` | `standalone` | CMOD |
| `bootcf` | `standalone` | BootCF |
| `stpro` | `standalone` | STPro |
| `odmixer` | `standalone` | ODMixer |
| `hcf` | `full` | HCF |
