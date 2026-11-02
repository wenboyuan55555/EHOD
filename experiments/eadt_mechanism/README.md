# eadt_mechanism

Paper table: `tab:ablation-mechanism`. Evaluation: `event_cells`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table eadt_mechanism --city nyc
python scripts/run_experiment.py --table eadt_mechanism --city chicago
python scripts/run_experiment.py --table eadt_mechanism --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table eadt_mechanism --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `eadt` | `full` | EHOD (full) |
| `eadt` | `without_end_anchor` | w/o end anchor |
| `eadt` | `without_od_specificity` | w/o OD specificity |
| `eadt` | `without_time_alignment` | w/o time alignment |

Saved results contain absolute metrics. For the paper's delta columns, compare each seed with its paired full EHOD run: `100 * (ablated / full - 1)` for MAE/RMSE; `ablated - full` for PCC.
