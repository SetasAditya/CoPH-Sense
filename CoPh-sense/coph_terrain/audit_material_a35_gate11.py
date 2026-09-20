"""Read-only diagnostic of legal-input sufficiency after the frozen Gate 1.1 run."""

import hashlib
import json
from pathlib import Path

import torch

from .train_material_a35_gate11 import OUT, _collect_challenge
from .value_model import VariableSetValueNet, decision_score
from .e2_dev_stack import model_outputs


def fingerprint(item):
    digest = hashlib.sha256()
    for tensor in (*item["inputs"], item["candidates"], item["sets"]):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    checkpoint = OUT / "a3_a5_candidate_failed.pt"
    payload = torch.load(checkpoint, map_location="cpu")
    model = VariableSetValueNet()
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    rows = []
    for split, seed in (("train", 1100000), ("validation", 1200000)):
        group = {"stratum": "Moving-A5", "parent_seed": seed,
                 "world_bits": [False, False, True, True]}
        items, collection = _collect_challenge(group)
        details = []
        for item in items:
            truth = (item["mission"] + item["risk"][:,:,0])[0]
            with torch.no_grad():
                mission, quantiles = model_outputs(model, item)
                selected = int(torch.argmin(decision_score(mission, quantiles)))
            details.append({"phase": item["phase"],
                            "legal_input_sha256": fingerprint(item),
                            "exact_action": int(torch.argmin(truth)),
                            "learner_action": selected,
                            "true_costs": [float(x) for x in truth],
                            "selected_regret": float(truth[selected]-truth.min())})
        rows.append({"split": split, "collection": collection,
                     "states": details,
                     "paired_inputs_differ": len(details) == 2 and
                     details[0]["legal_input_sha256"] != details[1]["legal_input_sha256"]})
        print(split, json.dumps(rows[-1]), flush=True)
    report = {"scope": "targeted frozen-row sufficiency audit; no new fitting or seed selection",
              "rows": rows,
              "interpretation_limit": "Only the predeclared train switch corner and matching validation corner are replayed; zero pair-optimal validation states cannot test pair-action identifiability."}
    (OUT/"gate11_sufficiency_audit.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
