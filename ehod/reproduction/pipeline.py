#!/usr/bin/env python3
"""Run formal EHOD experiments in dependency order; no server paths needed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = json.loads((ROOT / "configs/benchmark.json").read_text())
from ehod.reproduction.registry import regular_configs, event_configs
NORMAL = tuple(regular_configs())
DISCRETE = {"stid", "stidgcn", "stpro", "odmixer", "stan"}
MARGINAL = {k: (v["training"]["run_suffix"], v["training"]["batch_size"],
                v["training"]["eval_batch_size"], v["training"]["learning_rate"], v["training"]["epochs"])
            for k, v in regular_configs().items() if v["output_type"] == "node_marginals_mipf"}
EVENT = {k: (v["origin_model"], v["hcf_mechanism"]) for k, v in event_configs().items()}


class Runner:
    def __init__(self, args):
        self.args = args
        self.output = Path(os.environ.get("EHOD_OUTPUT_ROOT", str(ROOT / "outputs"))).expanduser().resolve() / args.city
        self.data = Path(os.environ.get("EHOD_DATA_ROOT", str(ROOT / "data"))).expanduser().resolve() / args.city
        self.env = os.environ.copy()
        self.env.update(EHOD_CITY=args.city, PYTHONPATH=str(ROOT),
                        EHOD_DATA_ROOT=str(self.data.parent), EHOD_OUTPUT_ROOT=str(self.output.parent),
                        EHOD_CUDA_DEVICE=args.device, EHOD_N_NODES=str(PROTOCOL["cities"][args.city]["zones"]),
                        EHOD_INVALID_TARGET_HOURS=",".join(map(str, PROTOCOL["cities"][args.city]["invalid_target_hours"])))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.env.setdefault(name, "4")
        self.scheduled = set()

    def run(self, module, options=(), result=None, script=False):
        command = [sys.executable]
        command += [str(ROOT / module)] if script else ["-m", module]
        command += list(map(str, options))
        key = tuple(command)
        if key in self.scheduled:
            return
        self.scheduled.add(key)
        if result is not None and result.is_file():
            print("Completed:", result.relative_to(self.output), flush=True)
            return
        print(shlex.join(command), flush=True)
        if not self.args.dry_run:
            subprocess.run(command, cwd=ROOT, env=self.env, check=True)
            if result is not None and not result.is_file():
                raise RuntimeError("Command finished without producing %s" % result)

    def gate(self, history):
        self.run("ehod.hcf.train_activation", ["--history-hours", history, "--seed", 0, "--device", self.args.device],
                 self.output / ("runs/activation/h%d_seed0/summary.json" % history))

    def normal_result(self, name, history, seed):
        if name in DISCRETE:
            return self.output / ("runs/%s/h%d_hurdle_seed%d_formal/result.json" % (name, history, seed))
        if name in MARGINAL:
            return self.output / ("runs/seven_baselines/%s_mipf/h%d_seed%d_%s/result.json" %
                                  (name, history, seed, MARGINAL[name][0]))
        if name in {"cmod", "bootcf"}:
            return self.output / ("runs/%s/h%d_legacy_training_seed%d_formal/hurdle_metrics.json" % (name, history, seed))
        return self.output / ("runs/simple_baselines/%s/h%d_seed%d/result.json" %
                              (name, history, 0 if name == "ha" else seed))

    def normal(self, name, history, seed):
        if name not in MARGINAL:
            self.gate(history)
        result = self.normal_result(name, history, seed)
        opts = ["--history-hours", history, "--seed", seed, "--device", self.args.device]
        if name in DISCRETE:
            self.run("baselines.regular._shared.train_discrete", ["--model", name, "--variant", "hurdle"] + opts, result)
        elif name in MARGINAL:
            _, batch, evaluation_batch, lr, epochs = MARGINAL[name]
            opts += ["--batch-size", batch, "--eval-batch-size", evaluation_batch,
                     "--learning-rate", lr, "--epochs", epochs]
            if name != "strep":
                opts += ["--method", name + "_mipf", "--teacher-epochs", epochs]
            self.run("baselines.regular.strep.train" if name == "strep" else "baselines.regular._shared.train_marginal", opts, result)
        elif name in {"ha", "xgboost", "lstm"}:
            self.run("baselines.regular._shared.simple_data", ["--city", self.args.city, "--models", name,
                     "--horizons", history, "--seed", seed, "--device", self.args.device], result)
        else:
            opts = ["--input-hours", history, "--seed", seed, "--device", self.args.device,
                    "--cache-dir", self.data / "continuous_cache", "--od-matrix", self.data / "core_od/od_matrix.npy",
                    "--output-dir", result.parent]
            if name == "cmod":
                opts += ["--epochs", 8 if self.args.city == "chicago" else 20,
                         "--patience", 4 if self.args.city == "chicago" else 5,
                         "--batch-size", 4 if self.args.city == "chicago" else 1]
            self.run("baselines.regular.%s.train" % name, opts, result)
        return result

    def hcf_result(self, history, seed):
        return self.output / ("runs/continuous_fusion_od/h%d_seed%d_hierarchical_no_stats_v1/result.json" % (history, seed))

    def hcf(self, history, seed):
        self.normal("stidgcn", history, seed)
        result = self.hcf_result(history, seed)
        self.run("ehod.hcf.train", ["--history-hours", history, "--seed", seed,
                 "--device", self.args.device, "--continuous-mode", "hierarchical",
                 "--feature-ablation", "no_statistical_priors", "--magnitude-loss", "decision",
                 "--run-name", "hierarchical_no_stats_v1"], result)
        return result

    def event_features(self):
        self.gate(12)
        cache = self.output / "cache"
        self.run("ehod.eadt.prepare_slots", result=cache / "event_transition_slots/metadata.json")
        self.run("baselines.event._shared.prepare_features", result=cache / "event_structured_features.json")
        self.run("baselines.event.event_matrix.prepare_features", result=cache / "event_matrix_features.json")
        self.run("ehod.eadt.cache_activation", ["--history-hours", 12, "--device", self.args.device],
                 cache / "event_backbones/h12/strict_activation_probability.json")

    def threshold(self, path):
        if self.args.dry_run and not path.is_file():
            return "<validation-selected-threshold>"
        payload = json.loads(path.read_text())
        return payload["evaluation"]["threshold"] if "evaluation" in payload else payload["threshold"]

    def hcf_cache(self, seed):
        result = self.hcf(12, seed)
        self.event_features()
        key = "count_fusion_short_v1_seed%d" % seed
        self.run("ehod.hcf.cache_event_state", ["--history-hours", 12, "--seed", seed, "--device", self.args.device,
                 "--continuous-run-name", "hierarchical_no_stats_v1", "--continuous-mode", "hierarchical",
                 "--feature-ablation", "no_statistical_priors", "--cache-key", key],
                 self.output / ("cache/event_backbones/h12/%s.json" % key))
        return key, self.threshold(result)

    def eadt(self, name, key, threshold, seed, run_name="validated_v5_complete_grid", extra=()):
        self.run("ehod.eadt.train", ["--backbone", name, "--cache-key", key,
                 "--base-threshold", threshold, "--seed", seed, "--backbone-seed", seed,
                 "--device", self.args.device, "--loss-preset", "validated_three_term",
                 "--selection", "balanced", "--run-name", run_name] + list(extra),
                 self.output / ("runs/event_transport/h12/%s/%s_seed%d/result.json" % (name, run_name, seed)))

        return self.output / ("runs/event_transport/h12/%s/%s_seed%d/result.json" % (name, run_name, seed))

    def event_origin(self, name, seed):
        if name == "event_matrix":
            self.gate(12)
            source = "h12_hurdle_seed%d_paired_event_matrix_v1" % seed
            output = "paired_event_matrix_v1_seed%d" % seed
            self.run("baselines.regular._shared.train_discrete", ["--model", "odpf", "--variant", "hurdle", "--history-hours", 12,
                     "--seed", seed, "--device", self.args.device, "--run-name", source],
                     self.output / ("runs/odpf/%s/result.json" % source))
            self.run("baselines.event.event_matrix.evaluate", ["--model", "odpf", "--seed", seed,
                     "--device", self.args.device, "--source-run-name", source, "--output-run-name", output],
                     self.output / ("runs/event_adapters/h12/odpf_native/%s/result.json" % output))
        else:
            method = EVENT[name][0]
            self.run("baselines.event._shared.train_origin", ["--model", method, "--history-hours", 12,
                     "--seed", seed, "--device", self.args.device],
                     self.output / ("runs/original_event_baselines/%s/h12_seed%d_formal/result.json" % (method, seed)))

        if name == "event_matrix":
            return self.output / ("runs/event_adapters/h12/odpf_native/%s/result.json" % output)
        return self.output / ("runs/original_event_baselines/%s/h12_seed%d_formal/result.json" % (method, seed))

    def event_hcf(self, name, seed):
        key, threshold = self.hcf_cache(seed)
        method = EVENT[name][1]
        run_name = ("paired_v1_" if name in {"qa_stgacn", "cdgon"} else "fixed_hcf_v1_") + method
        module = "baselines.event.rast.train_with_hcf" if method == "retrieval" else "baselines.event._shared.train_with_hcf"
        self.run(module, ["--backbone", "count_fusion", "--history-hours", 12, "--method", method,
                 "--cache-key", key, "--base-threshold", threshold, "--seed", seed,
                 "--device", self.args.device, "--run-name", run_name],
                 self.output / ("runs/event_adapters/h12/count_fusion/%s_seed%d/result.json" % (run_name, seed)))

        return self.output / ("runs/event_adapters/h12/count_fusion/%s_seed%d/result.json" % (run_name, seed))

    def portability(self, name, seed):
        result = self.normal(name, 12, seed)
        self.event_features()
        base = self.output / "cache/event_backbones/h12"
        if name in MARGINAL:
            threshold = "<validation-selected-interface-threshold>"
            internal = name + "_mipf"
            key = "seven_" + internal + ("" if seed == 0 else "_seed%d" % seed)
            self.run("ehod.eadt.cache_marginal", ["--method", internal, "--seed", seed,
                     "--device", self.args.device], base / (key + "_magnitude.json"))
            if not self.args.dry_run:
                threshold = json.loads((base / (key + "_magnitude.json")).read_text())["interface"]["selected_threshold"]
            return self.eadt(internal, key, threshold, seed, "validated_v3_seven_baseline")
        else:
            threshold = self.threshold(result)
            if name in {"ha", "xgboost", "lstm"}:
                key = "supplemental_" + name + ("_seed%d" % seed if seed and name != "ha" else "")
            else:
                key = name + "_seed%d" % seed
                cache_options = ["--model", name, "--history-hours", 12, "--cache-key", key,
                                 "--source-dir", result.parent, "--device", self.args.device]
                if name not in {"cmod", "bootcf"}:
                    cache_options += ["--seed", seed]
                self.run("ehod.eadt.cache_continuous" if name in {"cmod", "bootcf"} else "ehod.eadt.cache_discrete",
                         cache_options,
                         base / (key + "_magnitude.json"))
            return self.eadt(name, key, threshold, seed)
