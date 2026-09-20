"""Belief-conditional sufficiency audit for frozen Lookahead-v1 A4.

This is the hard-stop audit for selective communication.  It estimates the
SEND value available to the deployed sender, rather than the realized-world
hindsight value.  In this finite static task the complete legal history has a
known sufficient reduction:

* the innovation Gaussian factor summarizes the sender's local region scan;
* public progress/trajectory is represented by start, age and current public
  kinematics;
* prior sends and ACKs determine sender-known recipient overlap; and
* timestamps determine message age/actionability.

Receiver-private scans and hidden terrain labels are never inputs.  Hidden
worlds and continuation seeds are averaged within each legal context.  The
script does not fit or select an A4 policy.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np


WORLDS = {"safe": .04, "risky": .95}


def normal_cdf(x):
    return .5 * (1. + math.erf(x / math.sqrt(2.)))


def posterior_risky(row):
    """Recover p(W=risky | the transmitted Gaussian innovation factor)."""
    mean = float(row["actor_features"][1])
    variance = float(row["actor_features"][2]) ** 2
    post_precision = 1. / variance
    delta_precision = max(post_precision - 4., 1e-12)
    delta_weighted = mean * post_precision - 4. * .35
    observation = delta_weighted / delta_precision
    log_weights = []
    for name in ("safe", "risky"):
        error = observation - WORLDS[name]
        log_weights.append(-.5 * delta_precision * error * error)
    shift = max(log_weights)
    weights = np.exp(np.asarray(log_weights) - shift)
    return float(weights[1] / weights.sum()), observation, 1. / delta_precision


def legal_context(row):
    # private_direct_overlap is deliberately not included: the sender does not
    # know it. ACK-derived overlap is legal and is explicitly represented.
    acked = bool(row["recipient_features"][4] or
                 row["recipient_features"][5])
    return (int(row["scout_start_x"]), int(row["age_steps"]), acked)


def summarize(values):
    a = np.asarray(values, dtype=float)
    mean = float(a.mean())
    se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None
    # One-sided 95% normal bound. The report also exposes all raw counts and
    # split replication; this is an uncertainty summary, not a theorem.
    upper = mean + 1.6448536269514722 * se if se is not None else None
    return {"n": len(a), "mean": mean, "standard_error": se,
            "one_sided_95_upper": upper,
            "realized_positive_fraction": float(np.mean(a > 0.))}


def split_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(legal_context(row), row["risk"])].append(
            float(row["decision_value"]))
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    directory = args.dataset_dir.resolve()
    splits = {name: json.loads((directory / f"{name}.json").read_text())["rows"]
              for name in ("train", "validation", "test")}
    grouped = {name: split_groups(rows) for name, rows in splits.items()}

    # Train-world resampling estimates define V(H). Validation and test are
    # independent confirmation samples under the same legal contexts.
    train = grouped["train"]
    predictions = {}
    posterior_diagnostics = []
    for split, rows in splits.items():
        predicted = []
        for row in rows:
            context = legal_context(row)
            p_risky, observation, obs_var = posterior_risky(row)
            safe = float(np.mean(train[(context, "safe")]))
            risky = float(np.mean(train[(context, "risky")]))
            value = (1.-p_risky) * safe + p_risky * risky
            predicted.append(value)
            posterior_diagnostics.append({
                "split": split, "risk_evaluator_only": row["risk"],
                "posterior_risky": p_risky, "observation": observation,
                "observation_variance": obs_var})
        predictions[split] = {
            "rows": len(rows), "positive_conditional_predictions": int(
                np.sum(np.asarray(predicted) > 0)),
            "maximum_conditional_prediction": float(np.max(predicted)),
            "mean_conditional_prediction": float(np.mean(predicted))}

    contexts = sorted({context for groups in grouped.values()
                       for context, _ in groups})
    table = []
    for context in contexts:
        for world in ("safe", "risky"):
            row = {"scout_start_x": context[0], "age_steps": context[1],
                   "sender_visible_ack_overlap": context[2], "world": world}
            for split in ("train", "validation", "test"):
                row[split] = summarize(grouped[split][(context, world)])
            # Reproducibility gate: positive conditional value must appear in
            # validation and test means, not merely in realized individual worlds.
            row["replicated_positive_mean"] = bool(
                row["validation"]["mean"] > 0 and row["test"]["mean"] > 0)
            table.append(row)

    report = {
        "scope": "post hoc finite-task Bayes sufficiency/hard-stop audit",
        "environment_changed": False,
        "teacher_changed": False,
        "legal_history_factorization": {
            "local_measurements": "Gaussian innovation factor reconstructed from sender packet",
            "public_trajectory": "public start/progress schedule and current kinematics",
            "prior_sends_and_acks": "sender-visible ACK overlap only",
            "timestamps": "acquisition age and public simulation time",
            "excluded": ["receiver-private scans", "hidden world label",
                         "future continuation randomness", "seed identifier"]},
        "estimator": "equal-prior two-world posterior from the legal innovation; independent seed averages for each legal public context/world",
        "stopping_question": "Does any sender-identifiable context have positive mean SEND value replicated on validation and test?",
        "answer": "yes" if any(row["replicated_positive_mean"] for row in table) else "no",
        "train_based_conditional_predictions": predictions,
        "conditional_table": table,
        "posterior_diagnostics": {
            "rows": len(posterior_diagnostics),
            "mean_p_risky_given_safe": float(np.mean([
                x["posterior_risky"] for x in posterior_diagnostics
                if x["risk_evaluator_only"] == "safe"])),
            "mean_p_risky_given_risky": float(np.mean([
                x["posterior_risky"] for x in posterior_diagnostics
                if x["risk_evaluator_only"] == "risky"]))},
        "interpretation_limit": "This answers the finite declared two-world Lookahead-v1 task. It is not a universal impossibility result for other communication information structures."
    }
    path = directory / "full_history_bayes_sufficiency_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)
    print("answer", report["answer"])
    print("predictions", json.dumps(predictions, indent=2))
    best = sorted(table, key=lambda x: x["test"]["mean"], reverse=True)[:5]
    print("best test contexts")
    for row in best:
        print(row)


if __name__ == "__main__":
    main()
