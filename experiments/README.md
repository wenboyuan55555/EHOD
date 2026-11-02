# Paper experiments

Run commands from the repository root. Each directory contains the ordered rows and module settings for one table.

| Directory | Paper reference | Evaluation | Histories |
| --- | --- | --- | --- |
| [backbone_portability](backbone_portability/) | `tab:backbone-portability` | `event_cells` | 12 |
| [eadt_attributes](eadt_attributes/) | `tab:reviewer-question-ablations` | `event_cells` | 12 |
| [eadt_mechanism](eadt_mechanism/) | `tab:ablation-mechanism` | `event_cells` | 12 |
| [event_prediction](event_prediction/) | `tab:event-baselines` | `event_cells` | 12 |
| [hcf_components](hcf_components/) | `tab:hcf-components` | `full_od` | 12 |
| [history_sensitivity](history_sensitivity/) | `tab:history-sensitivity` | `full_od` | 1, 3, 6 |
| [normal_demand](normal_demand/) | `tab:regular-forecasting` | `full_od` | 12 |

```bash
python scripts/run_experiment.py --list
python scripts/run_experiment.py --table event_prediction --city nyc --dry-run
```

Event `with_hcf` means **origin+HCF**; normal-backbone `with_eadt` attaches EADT to frozen predictions. **HCF + EADT = EHOD**. Common intermediate artifacts are reused across tables.
