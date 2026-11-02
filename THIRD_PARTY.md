# Baseline sources and adaptations

The benchmark includes the task adapters used by the formal experiments.
Adaptation to hourly taxi OD prediction changes the input/output task;
these implementations do not claim to reproduce the source papers' original
datasets or reported results.

## Upstream model files

`baselines/sources_lock.json` records SHA-256 hashes of the exact included
upstream files. Only required model modules are included; upstream datasets,
papers, logs, and pretrained checkpoints are omitted.

| Directory | Source | Use in EHOD |
| --- | --- | --- |
| `baselines/regular/stidgcn/upstream` | STIDGCN author model supplied with the experiment source | Dynamic graph tree and GLU; OD wrapper in the corresponding method directory |
| `baselines/regular/cmod/upstream` | CMOD author implementation | Continuous memory and decoder modules; strict-window OD adapter in the corresponding method directory |
| `baselines/regular/bootcf/upstream` | BootCF author implementation | Message functions and OD predictor; strict-window encoder in the corresponding method directory |
| `baselines/regular/duet/upstream` | [DUET](https://github.com/decisionintelligence/DUET) | Author DUET core, predicting node marginals |
| `baselines/regular/stdn/upstream` | [STDN](https://github.com/roarer008/STDN) | Author model with independent origin/destination networks |
| `baselines/regular/patchstg/upstream` | [PatchSTG](https://github.com/LMissher/PatchSTG) | Author patch model and spatial partition utilities |
| `baselines/regular/strep/upstream` | [ST-ReP](https://github.com/zhuoshu/ST-ReP) | Author representation encoder; Ridge downstream forecasting |
| `baselines/regular/lightst/upstream` | [LightST](https://github.com/lizzyhku/TP) | Author teacher/student architecture with training on benchmark data |

Available upstream license files are retained alongside the corresponding
sources. CMOD uses Apache-2.0; the included DUET and PatchSTG license files
use MIT. Other supplied snapshots did not contain a license file. EHOD's
MIT license does not grant additional rights to these upstream files.

## Compatibility changes

Upstream files in each `upstream/` directory retain their supplied contents. Runtime
compatibility changes are explicit in the corresponding `baselines/regular/<method>/model.py`:

- DUET: remove a dispatch `squeeze(1)` that deletes the time axis when H=1;
  load only the declared core rather than unrelated TFB baselines.
- STDN: select the requested device instead of a module-level CUDA constant;
  an unused demonstration GCN import has a stub if PyG is absent. The
  original H-step decoder is retained and its first step is evaluated.
- PatchSTG: keep integer index tensors on the current device; provide the
  compatible `Attention`/`Mlp` interfaces when timm is unavailable.
- LightST: register projection layers originally constructed inside forward,
  so they are optimized and device portable. The supplied upstream pretrained
  checkpoint is not used.
- ST-ReP: preserve representation pretraining and use the original Ridge
  forecasting approach on the benchmark's node marginals.

Five marginal baselines are reconstructed into OD matrices with
training-only hour-of-week priors and iterative proportional fitting.
The implementation is in `baselines/regular/_shared/mipf.py`, `baselines/regular/_shared/marginal_data.py`, and the
corresponding trainers.

## Other normal-demand baselines

HA, XGBoost, and LSTM implementations are in
their `baselines/regular/{ha,xgboost,lstm}/` directories. STID is a task implementation following
[Spatial-Temporal Identity](https://github.com/zezhishao/STID), not a copy
of that full repository. STPro and ODPF are paper-based task implementations.
STAN and ODMixer preserve the supplied source architecture in OD adapters.
Their model definitions are under the corresponding baseline directories.

## Event baseline adaptations

Definitions and training are in each `baselines/event/<method>/model.py`,
`baselines/event/_shared/data.py`, and `baselines/event/_shared/train_origin.py`.
The trainer's `SOURCE` and `DEVIATIONS` dictionaries record the source paper,
available repository/commit, and task changes for every method.

| Mechanism | Reference source |
| --- | --- |
| T3 | [EventTrafficForecasting](https://github.com/2448845600/EventTrafficForecasting), commit `0d8ce0b959b7232345cc592c5c874c5219317489` |
| FUSE / direct concatenation | [FUSE-Traffic](https://github.com/GeoAICenter/FUSE-Traffic_Sigspatial2025), commit `46abd153192b9ea378b16e3cc69f8ea8bbfe64c5` |
| RAST | [RAST](https://github.com/RWLinno/RAST), commit `4b48a47a002dd8d4151afe47be8f436f0c028a67` |
| CDGON | [CDGON](https://github.com/tsinghua-fib-lab/CDGON-KDD24), commit `5bf8d114a485cacc886a5645119edce35479f4a4` |
| SS-MoE, EGAF, QA-STGACN | Paper-based implementations; details in the trainer metadata |
| Event matrix | Native three-channel scheduled-event ODPF adaptation |

`origin+HCF` applies each event mechanism to frozen HCF states using
`baselines/event/_shared/train_with_hcf.py` or `baselines/event/rast/train_with_hcf.py`. It is a separate
matched-backbone comparison from the `origin` model.

Frozen event text embeddings are derived using
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
Its model weights are not included. Raw public-data and schedule provenance
are described in `data/README.md` and the schedule CSV files.
