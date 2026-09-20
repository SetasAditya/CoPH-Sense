#!/usr/bin/env python3
"""A3→A4 exact finite continuation-consistency evaluation."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from coph_fork.critic import SUBSETS, SetCostCritic  # noqa: E402
from coph_fork.nested_oracle import nested_values, pair_ordering  # noqa: E402
from coph_fork.oracle import ExactOracleConfig, default_prior  # noqa: E402
from coph_fork.share_oracle import ShareState  # noqa: E402
from coph_fork.train_critic import sample_context  # noqa: E402
from coph_fork.train_share_critic import build_features_v2  # noqa: E402


OUTPUT = HERE / "results" / "nested_a3_a4"


def load_a3(seed, state, candidates):
    path = HERE / "results" / "a3"
    path = path if seed == 1701 else path / ("seed" + str(seed))
    model = SetCostCritic(len(state), candidates.shape[-1], "pairwise")
    model.load_state_dict(torch.load(
        path / "pairwise.pth", map_location="cpu", weights_only=True
    ))
    model.eval()
    return model


def load_a4(seed):
    example_state, example_candidates = build_features_v2(
        ExactOracleConfig(), default_prior(), ShareState(acquired=())
    )
    model = SetCostCritic(len(example_state), example_candidates.shape[-1], "pairwise")
    path = HERE / "results" / "a4_v2" / ("seed" + str(seed)) / "pairwise_weighted.pth"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def selected_a3(model, state, candidates):
    with torch.no_grad():
        scores = model(
            torch.tensor(state)[None], torch.tensor(candidates)[None]
        )[0].sum(-1).numpy()
    return int(scores.argmin())


def summarize_contexts(rows):
    def scalar(name):
        values = np.asarray([row[name] for row in rows], dtype=float)
        return {
            "mean": float(values.mean()),
            "p90": float(np.quantile(values, 0.9)),
            "max": float(values.max()),
        }
    positive = [row for row in rows if row["new_oracle_pair_ordering"]]
    return {
        "count": len(rows),
        "mean_gap_over_all_sensing_sets": scalar("mean_continuation_gap"),
        "max_gap_over_sensing_sets": scalar("max_continuation_gap"),
        "gap_at_A3_selected_set": scalar("gap_at_a3_set"),
        "A3_regret_under_original_A2": scalar("a3_regret_a2"),
        "A3_regret_under_new_exact_A4": scalar("a3_regret_new_oracle"),
        "nested_regret_under_learned_A4": scalar("nested_regret"),
        "original_A2_vs_new_exact_A4_mean_abs_value_shift": scalar("mean_abs_a2_bridge_shift"),
        "A2_and_new_exact_A4_optimum_disagree": sum(row["a2_optimum"] != row["new_oracle_optimum"] for row in rows),
        "new_exact_and_learned_A4_optimum_disagree": sum(row["new_oracle_optimum"] != row["learned_continuation_optimum"] for row in rows),
        "A3_choice_equals_learned_continuation_optimum": sum(row["a3_selected"] == row["learned_continuation_optimum"] for row in rows),
        "new_oracle_pair_ordering_count": len(positive),
        "learned_A4_preserves_pair_ordering_count": sum(row["learned_pair_ordering"] for row in positive),
    }


def main():
    torch.set_num_threads(2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    by_seed = {}
    for a3_seed, a4_seed in ((1701, 1801), (1702, 1802), (1703, 1803)):
        rng = np.random.RandomState(a3_seed + 2)
        # First held-out state determines the unchanged A3 model dimensions.
        state, candidates, labels, _, config, prior = sample_context(rng, True)
        a3_model = load_a3(a3_seed, state, candidates)
        a4_model = load_a4(a4_seed)
        contexts = [(state, candidates, labels, config, prior)]
        contexts.extend(
            (s, c, l, cfg, pr)
            for s, c, l, _, cfg, pr in
            (sample_context(rng, True) for _ in range(119))
        )
        seed_rows = []
        for index, (state, candidates, labels, config, prior) in enumerate(contexts):
            q_a2 = labels.sum(-1).astype(float)
            q_exact, q_learned = nested_values(config, prior, a4_model)
            gap = q_learned - q_exact
            if np.min(gap) < -1e-8:
                raise AssertionError("learned continuation beat exact A4 oracle")
            a3_selected = selected_a3(a3_model, state, candidates)
            row = {
                "a3_seed": a3_seed,
                "a4_seed": a4_seed,
                "context_index": index,
                "public_state": state.tolist(),
                "a2_optimum": int(q_a2.argmin()),
                "new_oracle_optimum": int(q_exact.argmin()),
                "learned_continuation_optimum": int(q_learned.argmin()),
                "a3_selected": a3_selected,
                "a3_regret_a2": float(q_a2[a3_selected] - q_a2.min()),
                "a3_regret_new_oracle": float(q_exact[a3_selected] - q_exact.min()),
                "nested_regret": float(q_learned[a3_selected] - q_learned.min()),
                "mean_abs_a2_bridge_shift": float(np.abs(q_exact - q_a2).mean()),
                "mean_continuation_gap": float(gap.mean()),
                "max_continuation_gap": float(gap.max()),
                "gap_at_a3_set": float(gap[a3_selected]),
                "new_oracle_pair_ordering": pair_ordering(q_exact),
                "learned_pair_ordering": pair_ordering(q_learned),
                "q_a2": q_a2.tolist(),
                "q_new_oracle": q_exact.tolist(),
                "q_learned_continuation": q_learned.tolist(),
            }
            seed_rows.append(row)
        by_seed[str(a3_seed)] = summarize_contexts(seed_rows)
        all_rows.extend(seed_rows)
        print(a3_seed, json.dumps(by_seed[str(a3_seed)]), flush=True)
    canonical = {}
    for a4_seed in (1801, 1802, 1803):
        canonical_exact, canonical_learned = nested_values(
            ExactOracleConfig(), default_prior(), load_a4(a4_seed)
        )
        canonical[str(a4_seed)] = {
            "new_exact_Q": canonical_exact.tolist(),
            "learned_A4_Q": canonical_learned.tolist(),
            "new_exact_pair_ordering": pair_ordering(canonical_exact),
            "learned_A4_pair_ordering": pair_ordering(canonical_learned),
        }
    report = {
        "scope": "finite route-cost diagnostic; A3 pairwise checkpoint paired with A4-v2 weighted pairwise checkpoint",
        "protocol": "A4 no-silence-inference sharing, charged evidence and expected ACK attempts; no initial recipient ACKs",
        "original_A2_labels_are_not_same_protocol": "A2 uses content-conditional send rules and omits ACK cost; bridge shift is reported separately",
        "subsets": [list(subset) for subset in SUBSETS],
        "by_seed": by_seed,
        "aggregate": summarize_contexts(all_rows),
        "canonical": canonical,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "contexts.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    print("saved", OUTPUT)


if __name__ == "__main__":
    main()
