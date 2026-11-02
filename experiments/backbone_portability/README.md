# backbone_portability

Paper table: `tab:backbone-portability`. Evaluation: `event_cells`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table backbone_portability --city nyc
python scripts/run_experiment.py --table backbone_portability --city chicago
python scripts/run_experiment.py --table backbone_portability --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table backbone_portability --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `ha` | `frozen` | HA |
| `ha` | `with_eadt` | HA + EADT |
| `xgboost` | `frozen` | XGBoost |
| `xgboost` | `with_eadt` | XGBoost + EADT |
| `lstm` | `frozen` | LSTM |
| `lstm` | `with_eadt` | LSTM + EADT |
| `duet` | `frozen` | DUET |
| `duet` | `with_eadt` | DUET + EADT |
| `stid` | `frozen` | STID |
| `stid` | `with_eadt` | STID + EADT |
| `stidgcn` | `frozen` | STIDGCN |
| `stidgcn` | `with_eadt` | STIDGCN + EADT |
| `patchstg` | `frozen` | PatchSTG |
| `patchstg` | `with_eadt` | PatchSTG + EADT |
| `stdn` | `frozen` | STDN |
| `stdn` | `with_eadt` | STDN + EADT |
| `lightst` | `frozen` | LightST |
| `lightst` | `with_eadt` | LightST + EADT |
| `strep` | `frozen` | ST-ReP |
| `strep` | `with_eadt` | ST-ReP + EADT |
| `stan` | `frozen` | STAN |
| `stan` | `with_eadt` | STAN + EADT |
| `cmod` | `frozen` | CMOD |
| `cmod` | `with_eadt` | CMOD + EADT |
| `bootcf` | `frozen` | BootCF |
| `bootcf` | `with_eadt` | BootCF + EADT |
| `stpro` | `frozen` | STPro |
| `stpro` | `with_eadt` | STPro + EADT |
| `odmixer` | `frozen` | ODMixer |
| `odmixer` | `with_eadt` | ODMixer + EADT |
| `hcf` | `frozen` | HCF |
| `hcf` | `with_eadt` | HCF + EADT (EHOD) |
