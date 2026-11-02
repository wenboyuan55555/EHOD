"""Collect completed paper rows in manuscript order, with sample standard deviations."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics

from ehod.reproduction.registry import ROOT, tables


def select_metrics(payload, selector=None, scope="full_od"):
    if selector:
        for key in selector.split("."):
            payload = payload[key]
        return payload
    if scope != "full_od":
        raise ValueError("Event-cell results require an explicit metric selector")
    if "evaluation" in payload:
        return payload["evaluation"]["test"]
    value = payload["test"]
    return value.get("od", value)


def main():
    registry = tables()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--table", required=True, choices=tuple(registry))
    p.add_argument("--city", required=True, choices=("nyc", "chicago"))
    args = p.parse_args()
    root = Path(os.environ.get("EHOD_OUTPUT_ROOT", str(ROOT / "outputs"))).resolve()
    table = registry[args.table]
    rows = []
    for history in table["histories"]:
        reference = {}
        if args.table.startswith("eadt_"):
            for seed in table["seeds"]:
                path = root / "reproduction" / args.table / args.city / ("h%d" % history) / "eadt/full" / ("seed%d.json" % seed)
                if path.is_file():
                    record = json.loads(path.read_text())
                    reference[seed] = select_metrics(json.loads((root / record["result"]).read_text()), record["selector"], table["scope"])
        for row in table["rows"]:
            samples = []
            for seed in table["seeds"]:
                path = root / "reproduction" / args.table / args.city / ("h%d" % history) / row["method"] / row["variant"] / ("seed%d.json" % seed)
                if not path.is_file():
                    continue
                record = json.loads(path.read_text())
                if record["scope"] != table["scope"]:
                    raise ValueError("Mismatched evaluation scope: " + str(path))
                payload = json.loads((root / record["result"]).read_text())
                metrics = select_metrics(payload, record["selector"], table["scope"])
                samples.append((seed, metrics))
            values = {"method": row["method"], "variant": row["variant"], "label": row["label"],
                      "city": args.city, "history": history, "scope": table["scope"],
                      "completed_seeds": ",".join(str(s) for s, _ in samples),
                      "status": "complete" if len(samples) == len(table["seeds"]) else "incomplete"}
            metrics = ["mae", "rmse", "pcc"]
            if args.table == "hcf_components":
                metrics += ["activation_f1", "positive_mae", "total_flow_absolute_percentage_error"]
            for metric in metrics:
                x = [float(m[metric]) for _, m in samples if metric in m]
                values[metric + "_mean"] = statistics.mean(x) if x else ""
                values[metric + "_sample_std"] = statistics.stdev(x) if len(x) > 1 else ""
            if args.table.startswith("eadt_"):
                for metric in ("mae", "rmse", "pcc"):
                    x = [(float(m[metric]) - float(reference[s][metric])) *
                         (1.0 if metric == "pcc" else 100.0 / float(reference[s][metric]))
                         for s, m in samples if s in reference]
                    values[metric + "_delta_mean"] = statistics.mean(x) if x else ""
                    values[metric + "_delta_sample_std"] = statistics.stdev(x) if len(x) > 1 else ""
            rows.append(values)
    dest = root / "tables" / args.table / (args.city + ".csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dest)


if __name__ == "__main__":
    main()
