"""Intentionally fit all 32 existing Gate 1.1 training states; no new seeds."""

import argparse
import json

import torch

from .e2_dev_stack import model_outputs
from .train_material_a35 import collect
from .train_material_a35_gate11 import OUT, _collect_challenge
from .value_model import VariableSetValueNet, decision_score, quantile_huber_loss


def _evaluate(model, examples):
    correct = 0
    regret = 0.0
    by_stratum = {}
    with torch.no_grad():
        for item in examples:
            mission, quantiles = model_outputs(model, item)
            action = int(torch.argmin(decision_score(mission, quantiles)))
            truth = (item["mission"] + item["risk"][:, :, 0])[0]
            oracle = int(torch.argmin(truth))
            value = float(truth[action] - truth[oracle])
            correct += action == oracle
            regret += value
            stratum = item["phase"].split("_")[0]
            entry = by_stratum.setdefault(stratum, {"count": 0, "correct": 0,
                                                     "regret_sum": 0.0})
            entry["count"] += 1
            entry["correct"] += action == oracle
            entry["regret_sum"] += value
    return {"count": len(examples), "exact_action_rate": correct/len(examples),
            "mean_regret": regret/len(examples), "by_stratum": by_stratum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.manual_seed(2701)
    torch.set_num_threads(1)
    rows = json.loads((OUT/"train_manifest.json").read_text())["rows"]
    examples, collection = [], []
    for row in rows:
        if row["stratum"] == "Natural":
            items, record = collect(row["group"])
            for item in items:
                item["phase"] = "Natural_" + item["phase"]
        else:
            items, record = _collect_challenge(row)
        examples.extend(items)
        collection.append(record)
        print("collect", record, flush=True)
    if len(examples) != 32:
        raise RuntimeError("Gate 1.1 training rows did not reproduce 32 examples")
    buckets = {key: [x for x in examples if x["phase"].split("_")[0] == key]
               for key in ("Natural", "Complementarity", "Moving-A5")}
    device = torch.device(args.device)
    moved = {}
    for key, bucket in buckets.items():
        moved[key] = []
        for original in bucket:
            item = dict(original)
            item["inputs"] = tuple(x.to(device) for x in original["inputs"])
            for name in ("candidates", "sets", "mission", "risk"):
                item[name] = original[name].to(device)
            moved[key].append(item)
    model = VariableSetValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    curve = []
    passed = False
    for step in range(1, args.max_steps+1):
        optimizer.zero_grad()
        stratum_losses = []
        for bucket in moved.values():
            losses = []
            for item in bucket:
                mission, quantiles = model_outputs(model, item)
                losses.append((mission-item["mission"]).square().mean() +
                              quantile_huber_loss(quantiles, item["risk"]))
            stratum_losses.append(torch.stack(losses).mean())
        loss = torch.stack(stratum_losses).mean()
        loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.max_steps:
            evaluation = _evaluate(model, [x for bucket in moved.values() for x in bucket])
            point = {"step": step, "loss": float(loss.detach()), **evaluation}
            curve.append(point)
            print("fit", {"step": step, "loss": point["loss"],
                          "exact_action_rate": evaluation["exact_action_rate"],
                          "mean_regret": evaluation["mean_regret"]}, flush=True)
            if evaluation["exact_action_rate"] >= .98:
                passed = True
                break
    report = {"status": "passed" if passed else "failed",
              "scope": "all 32 deterministic Gate 1.1 training states only",
              "architecture": "VariableSetValueNet", "optimizer": "Adam lr=3e-4",
              "loss": "equal-stratum absolute mission MSE + risk quantile Huber",
              "arguments": vars(args), "collection": collection,
              "curve": curve, "validation_or_test_opened": False}
    (OUT/"micro_overfit_full_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"], "final": curve[-1]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
