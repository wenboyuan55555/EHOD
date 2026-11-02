# eadt_attributes

Paper table: `tab:reviewer-question-ablations`. Evaluation: `event_cells`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table eadt_attributes --city nyc
python scripts/run_experiment.py --table eadt_attributes --city chicago
python scripts/run_experiment.py --table eadt_attributes --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table eadt_attributes --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `eadt` | `full` | EHOD (full) |
| `eadt` | `without_type` | w/o type identity |
| `eadt` | `without_venue` | w/o venue identity |
| `eadt` | `without_magnitude_update` | w/o magnitude update |
| `eadt` | `without_activation_update` | w/o activation update |

Saved results contain absolute metrics. For the paper's delta columns, compare each seed with its paired full EHOD run: `100 * (ablated / full - 1)` for MAE/RMSE; `ablated - full` for PCC.
