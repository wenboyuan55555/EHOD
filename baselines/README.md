# Baselines and module combinations

## Normal-demand baselines

| Paper category | Method | Source directory |
| --- | --- | --- |
| Statistical | HA | [regular/ha](regular/ha/) |
| Machine learning | XGBoost | [regular/xgboost](regular/xgboost/) |
| Time series | LSTM | [regular/lstm](regular/lstm/) |
| Time series | DUET | [regular/duet](regular/duet/) |
| Spatiotemporal | STID | [regular/stid](regular/stid/) |
| Spatiotemporal | STIDGCN | [regular/stidgcn](regular/stidgcn/) |
| Spatiotemporal | PatchSTG | [regular/patchstg](regular/patchstg/) |
| Spatiotemporal | STDN | [regular/stdn](regular/stdn/) |
| Spatiotemporal | LightST | [regular/lightst](regular/lightst/) |
| Spatiotemporal | ST-ReP | [regular/strep](regular/strep/) |
| OD-specific | STAN | [regular/stan](regular/stan/) |
| OD-specific | CMOD | [regular/cmod](regular/cmod/) |
| OD-specific | BootCF | [regular/bootcf](regular/bootcf/) |
| OD-specific | STPro | [regular/stpro](regular/stpro/) |
| OD-specific | ODMixer | [regular/odmixer](regular/odmixer/) |

`standalone` predicts full OD matrices. `with_eadt` freezes that predictor and trains EADT at H=12.
Five marginal predictors (DUET, PatchSTG, STDN, LightST, ST-ReP) use training-only seasonal priors and MIPF. Their standalone raw-count reconstruction is evaluated independently of the activation/magnitude interface used for EADT; the latter adds the shared gate and selects its threshold on validation event cells.

## Event baselines

| Paper method | Directory | Origin model ID | HCF mechanism ID |
| --- | --- | --- | --- |
| T3 | [event/t3](event/t3/) | `t3_stid` | `t3_dual` |
| FUSE | [event/fuse](event/fuse/) | `fuse_d2stgnn` | `fuse_cross_attention` |
| Direct concat | [event/concat](event/concat/) | `fuse_concat` | `concat` |
| SS-MoE | [event/ss_moe](event/ss_moe/) | `ss_moe` | `semantic_spatial_moe` |
| EGAF | [event/egaf](event/egaf/) | `egaf` | `legacy_attention` |
| QA-STGACN | [event/qa_stgacn](event/qa_stgacn/) | `qa_stgacn` | `qa_query` |
| RAST | [event/rast](event/rast/) | `rast` | `retrieval` |
| CDGON | [event/cdgon](event/cdgon/) | `cdgon` | `cdgon_ode` |
| Event matrix | [event/event_matrix](event/event_matrix/) | `odpf` | `matrix` |

`origin` retains the source architecture adapted to this OD task. `with_hcf` is the paper's `origin+HCF` comparison: the mechanism consumes frozen HCF activation and magnitude states. HCF is trained once per city/history/seed and shared across these comparisons. HCF + EADT is EHOD.

Model definitions and per-method `run.py` / `config.json` live together. Shared trainers live in `_shared/`; upstream files and licenses are under each method's `upstream/`. Source architectures are task adaptations, not claims of reproducing the original paper datasets or reported scores. See [THIRD_PARTY.md](../THIRD_PARTY.md).
