# EHOD

Processed datasets, baseline adaptations, and the HCF + EADT implementation for
hourly origin–destination demand forecasting. Experiments are organized by the
comparison tables in the paper.

## Download the processed datasets

The complete NYC and Chicago datasets are published in
[EHOD data v1](https://github.com/wenboyuan55555/EHOD/releases/tag/data-v1).
The Git checkout includes the data manifest and small metadata files; download
the arrays before running data checks or experiments:

```bash
python scripts/download_data.py --city all
```

The downloader requires only Python's standard library. It downloads one ZIP per
city, checks each archive's SHA-256, restores the original directory structure,
and verifies every extracted file against `data/manifest.json`. Existing valid
data files are reused. Use `--city nyc` or `--city chicago` to download one city.
Allow about 15 GB of disk space for the full extracted dataset and downloads.
See [data/README.md](data/README.md) for direct links and manual extraction.

## Directory layout

```text
EHOD/
├── baselines/
│   ├── regular/                # 15 normal-demand baselines
│   │   ├── ha/  xgboost/  lstm/  duet/
│   │   ├── stid/  stidgcn/  patchstg/  stdn/  lightst/  strep/
│   │   ├── stan/  cmod/  bootcf/  stpro/  odmixer/
│   │   └── _shared/            # Shared training, reconstruction and evaluation
│   ├── event/                  # 9 event mechanisms
│   │   ├── t3/  fuse/  concat/  ss_moe/  egaf/
│   │   ├── qa_stgacn/  rast/  cdgon/  event_matrix/
│   │   └── _shared/            # Matched event inputs and trainers
│   └── sources_lock.json       # Hashes of pinned third-party files
├── ehod/
│   ├── hcf/                    # Activation, count fusion, order state, training
│   ├── eadt/                   # Event correction, schedule slots, state caches
│   ├── data/                   # Shared dataset loaders and preprocessing
│   ├── evaluation/             # Metrics and event-cluster bootstrap
│   ├── common/                 # Paths, devices, isolated upstream loading
│   └── reproduction/           # Table registry, dependency runner, collection
├── experiments/                # One directory per comparison/component table
├── configs/benchmark.json      # Splits, histories, seeds and module defaults
├── data/{nyc,chicago}/         # Processed demand, events and order-state arrays
└── scripts/                    # Run, collect, data checks and smoke checks
```

Each baseline directory contains `config.json` and `run.py`, together with its
model/adapter or method-specific trainer. Event baselines keep the source-task
adaptation in `model.py` and the frozen-HCF mechanism in `fusion.py`; RAST uses
retrieval code for the latter. Included author model files and licenses live in
that method's `upstream/` directory. Shared optimizers, dataset code and metrics
are reused across methods.

## Setup

Use Python 3.8 and PyTorch 2.4.1 with a compatible CUDA installation for training.
The CPU smoke check can run without a GPU. Run commands from this directory.

```bash
python -m pip install -e .
python scripts/download_data.py --city all
python scripts/check_data.py --city all
python scripts/run_experiment.py --list
```

With dependencies already installed, `python -m pip install -e . --no-deps --no-build-isolation`
uses the current environment, including older setuptools. The editable install
keeps table configurations and data paths attached to this checkout.

`requirements.txt` records the tested dependencies. `requirements-events.txt`
contains optional text-encoding/retrieval dependencies; frozen event text
embeddings are already included. Data preprocessing from raw Parquet additionally
uses `pyarrow`.

Set `EHOD_DATA_ROOT` or `EHOD_OUTPUT_ROOT` to relocate data or generated artifacts.
All paths default to this repository. `--device cuda:0` selects a visible GPU;
CUDA device indices follow `CUDA_VISIBLE_DEVICES`.

## Benchmark protocol

| Dataset | Year | Zones | Hourly OD tensor | Scheduled events |
| --- | --- | ---: | --- | ---: |
| NYC | 2023 | 263 | 8,760 × 263 × 263 | 470 |
| Chicago | 2023 | 77 | 8,760 × 77 × 77 | 413 |

A target hour `t` uses only `[t-H,t)` observations, with `H ∈ {1,3,6,12}`.
The supervised target intervals are training `[12,6720)`, validation
`[6720,7728)`, and test `[7728,8760)`. The normalizer is fitted to `[0,6720)`.
Chicago invalid target hours `1682,8757,8758,8759` retain their original array
positions and are excluded under the released protocol.

Formal stochastic experiments use seeds `0,1,2`; HA is deterministic and the
shared activation classifier uses seed 0. Event experiments fix `H=12`.
Normal-demand tables evaluate complete OD matrices; event tables evaluate the
released event-cell mask. See [data/README.md](data/README.md) for file schemas,
schedule provenance and temporal conventions.

## Reproduce the comparison tables

| Experiment directory / `--table` | Paper table | Rows / combinations |
| --- | --- | --- |
| `normal_demand` | `tab:regular-forecasting` | 15 standalone demand baselines + HCF, H=12 |
| `event_prediction` | `tab:event-baselines` | 9 mechanisms × (`origin`, `with_hcf`) + frozen HCF + EHOD |
| `backbone_portability` | `tab:backbone-portability` | 15 backbones + HCF, each frozen and with EADT |
| `history_sensitivity` | `tab:history-sensitivity` | All 16 normal predictors, H=1/3/6 |
| `hcf_components` | `tab:hcf-components` | Full HCF and three component removals |
| `eadt_mechanism` | `tab:ablation-mechanism` | End anchor, OD specificity and time alignment |
| `eadt_attributes` | `tab:reviewer-question-ablations` | Type, venue, magnitude update and activation update |

Each directory has an ordered `config.json` and a README with commands. For example:

```bash
# Complete main event comparison; automatic gate, backbone and cache preparation.
python scripts/run_experiment.py --table event_prediction --city nyc

# Independent normal-demand comparison.
python scripts/run_experiment.py --table normal_demand --city chicago

# EADT on all frozen backbones, with paired before/after results.
python scripts/run_experiment.py --table backbone_portability --city nyc

# HCF and EADT component studies.
python scripts/run_experiment.py --table hcf_components --city chicago
python scripts/run_experiment.py --table eadt_mechanism --city nyc
python scripts/run_experiment.py --table eadt_attributes --city chicago

# Inspect dependency commands without starting training.
python scripts/run_experiment.py --table event_prediction --city nyc --dry-run
```

Add `--seeds 0` for one run, `--methods t3,fuse` to select methods, or
`--variants origin,with_hcf` to select combinations. The table configurations
and `--list` provide the available IDs. `--histories` selects a subset of the
histories belonging to that table. Completed dependencies are reused.

### Run from a specific baseline directory

```bash
# STIDGCN standalone, or STIDGCN followed by EADT.
python -m baselines.regular.stidgcn.run --city nyc --variant standalone
python -m baselines.regular.stidgcn.run --city nyc --variant with_eadt

# Normal-demand short histories.
python -m baselines.regular.duet.run --city chicago --histories 1,3,6

# T3 origin architecture and its event mechanism on frozen HCF.
python -m baselines.event.t3.run --city nyc --variant both
python -m baselines.event.fuse.run --city chicago --variant with_hcf

# Our complete framework, or HCF alone.
python scripts/run_experiment.py --table event_prediction --city nyc --methods ehod
python scripts/run_experiment.py --table normal_demand --city nyc --methods hcf
```

`origin` means the source architecture adapted to this OD task. `with_hcf`
corresponds to the paper's **origin+HCF**: its event mechanism acts on a frozen
HCF state. `with_eadt` attaches EADT to a frozen normal-demand predictor;
**HCF + EADT = EHOD**. The five marginal predictors use training-only seasonal
priors and iterative proportional fitting to reconstruct OD matrices. Their
standalone evaluation and their activation/magnitude interface are distinct
configurations, as documented in [baselines/README.md](baselines/README.md).

## Results

```bash
python scripts/summarize_results.py --table event_prediction --city nyc
```

Table-row records are saved under
`outputs/reproduction/<table>/<city>/h<H>/<method>/<variant>/seed<S>.json`.
Collected CSVs are written to `outputs/tables/<table>/<city>.csv`, preserving
paper row order, evaluation scope, completed seeds, means and sample standard
deviations. Missing seeds are marked `incomplete`; one observed seed does not
produce a reported three-seed standard deviation.

Training checkpoints and intermediate caches are shared under
`outputs/<city>/{runs,cache}/`. Some internal artifact names retain the original
experiment identifiers so that dependency loading stays consistent. Reproduction
records point to these artifacts without duplicating checkpoints or predictions.

## Checks

```bash
python scripts/check_data.py --city all --full
python scripts/smoke_check.py --city chicago
```

`--full` verifies all data checksums. The smoke check uses real demand inputs to
exercise HCF, EADT and the OD model interfaces; it is separate from formal
training. Packaging checks also covered all event architecture and HCF-fusion
forwards, the five marginal adapters at H=1/H=12, and small CUDA training/evaluation runs for HA, XGBoost, LSTM, STIDGCN,
HCF, EADT, T3 origin and T3 + HCF.
Full three-seed experiments are not rerun during directory organization.

## Data sources and license

NYC trip records are from the
[NYC Taxi and Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
Chicago records are from the
[City of Chicago Taxi Trips data](https://data.cityofchicago.org/Transportation/Taxi-Trips-2013-2023-/wrvz-psew).
Event schedule CSVs retain source and timing-provenance columns. The processed
release includes aggregated demand and derived state, without trip identifiers
or passenger/driver identifiers.

EHOD-authored code uses [MIT](LICENSE). Third-party implementations and underlying
data retain their respective terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
