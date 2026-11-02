# event_prediction

Paper table: `tab:event-baselines`. Evaluation: `event_cells`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table event_prediction --city nyc
python scripts/run_experiment.py --table event_prediction --city chicago
python scripts/run_experiment.py --table event_prediction --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table event_prediction --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `t3` | `origin` | T3 |
| `t3` | `with_hcf` | T3 + HCF |
| `fuse` | `origin` | FUSE |
| `fuse` | `with_hcf` | FUSE + HCF |
| `concat` | `origin` | Direct concat |
| `concat` | `with_hcf` | Direct concat + HCF |
| `ss_moe` | `origin` | SS-MoE |
| `ss_moe` | `with_hcf` | SS-MoE + HCF |
| `egaf` | `origin` | EGAF |
| `egaf` | `with_hcf` | EGAF + HCF |
| `qa_stgacn` | `origin` | QA-STGACN |
| `qa_stgacn` | `with_hcf` | QA-STGACN + HCF |
| `rast` | `origin` | RAST |
| `rast` | `with_hcf` | RAST + HCF |
| `cdgon` | `origin` | CDGON |
| `cdgon` | `with_hcf` | CDGON + HCF |
| `event_matrix` | `origin` | Event matrix |
| `event_matrix` | `with_hcf` | Event matrix + HCF |
| `hcf` | `frozen` | Frozen HCF |
| `ehod` | `full` | EHOD (HCF + EADT) |
