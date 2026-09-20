"""Gate 1.2 preflight: intentionally fit the fixed Gate 1.1 memory switch pair.

The two rows are deterministically replayed from the existing training
manifest. No validation or test state is opened. Architecture, optimizer, and
value/quantile loss match train_material_a35_gate11.py.
"""

import argparse
import json

import torch

from .e2_dev_stack import model_outputs
from .train_material_a35_gate11 import OUT, _collect_challenge
from .value_model import VariableSetValueNet, decision_score, quantile_huber_loss


ROW = {"stratum": "Moving-A5", "parent_seed": 1100000,
       "world_bits": [False, False, True, True]}


def evaluate(model, items):
    results = []
    with torch.no_grad():
        for item in items:
            mission, quantiles = model_outputs(model, item)
            action = int(torch.argmin(decision_score(mission, quantiles)))
            truth = (item["mission"] + item["risk"][:, :, 0])[0]
            results.append({"phase": item["phase"], "action": action,
                            "oracle": int(torch.argmin(truth)),
                            "regret": float(truth[action] - truth.min())})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    torch.manual_seed(2701)
    torch.set_num_threads(1)
    manifest = json.loads((OUT / "train_manifest.json").read_text())
    if ROW not in manifest["rows"]:
        raise RuntimeError("fixed micro-overfit row absent from Gate 1.1 train manifest")
    items, collection = _collect_challenge(ROW)
    if len(items) != 2 or not collection["memory_optimum_changed"]:
        raise RuntimeError("frozen training switch pair did not replay")
    device = torch.device(args.device)
    model = VariableSetValueNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    moved = []
    for original in items:
        item = dict(original)
        item["inputs"] = tuple(t.to(device) for t in original["inputs"])
        for key in ("candidates", "sets", "mission", "risk"):
            item[key] = original[key].to(device)
        moved.append(item)
    curve = []
    passed = False
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad()
        losses = []
        for item in moved:
            mission, quantiles = model_outputs(model, item)
            losses.append((mission-item["mission"]).square().mean() +
                          quantile_huber_loss(quantiles, item["risk"]))
        loss = torch.stack(losses).mean()
        loss.backward(); optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.max_steps:
            decisions = evaluate(model, moved)
            record = {"step": step, "loss": float(loss.detach()),
                      "actions": [x["action"] for x in decisions],
                      "regrets": [x["regret"] for x in decisions]}
            curve.append(record)
            print(record, flush=True)
            if all(x["regret"] < 1e-3 for x in decisions):
                passed = True
                break
    report = {"status": "passed" if passed else "failed",
              "scope": "same deterministic Gate 1.1 training switch pair only",
              "architecture": "VariableSetValueNet", "optimizer": "Adam lr=3e-4",
              "loss": "absolute mission MSE + risk quantile Huber (unchanged)",
              "arguments": vars(args), "collection": collection,
              "curve": curve, "final_decisions": evaluate(model, moved),
              "validation_or_test_opened": False}
    (OUT / "micro_overfit_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"status": report["status"],
                      "final_decisions": report["final_decisions"]}, indent=2),
          flush=True)


if __name__ == "__main__":
    main()
