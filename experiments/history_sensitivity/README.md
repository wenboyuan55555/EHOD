# history_sensitivity

Paper table: `tab:history-sensitivity`. Evaluation: `full_od`.

Histories: 1, 3, 6; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table history_sensitivity --city nyc
python scripts/run_experiment.py --table history_sensitivity --city chicago
python scripts/run_experiment.py --table history_sensitivity --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table history_sensitivity --city nyc
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

The runner saves all method-level results needed for the shorter-history comparison. Lead is relative MAE reduction against the strongest baseline at the same history; rank compares all 16 methods.
