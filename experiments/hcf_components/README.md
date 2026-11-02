# hcf_components

Paper table: `tab:hcf-components`. Evaluation: `full_od`.

Histories: 12; seeds: 0, 1, 2.

```bash
python scripts/run_experiment.py --table hcf_components --city nyc
python scripts/run_experiment.py --table hcf_components --city chicago
python scripts/run_experiment.py --table hcf_components --city nyc --seeds 0 --dry-run
python scripts/summarize_results.py --table hcf_components --city nyc
```

Run from the repository root. Filter rows with `--methods` and `--variants`.

| Method ID | Variant ID | Paper row |
| --- | --- | --- |
| `hcf` | `without_count_fusion` | w/o count fusion |
| `hcf` | `without_hurdle` | w/o hurdle factorization |
| `hcf` | `without_short_state` | w/o short-term state |
| `hcf` | `full` | HCF |
