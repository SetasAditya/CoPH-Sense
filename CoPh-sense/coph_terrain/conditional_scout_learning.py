"""Train and evaluate an ex-ante conditional-scout dispatch-value model.

The exact admission benchmark remains the source of labels.  This module
randomizes the admission cases, predicts the *value of dispatch* before the
scout moves, and evaluates selective dispatch on held-out randomized cases.
It is intentionally small: the first learned quantity is the carrier-side
value gate, not an end-to-end two-agent MARL policy.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .conditional_scout_benchmark import (
    HiddenMode,
    ScoutNecessityCase,
    evaluate_case,
)

KINDS = ("static_block", "local_known", "moving_blocker", "moving_cost")
FEATURE_NAMES = (
    "kind_static_block", "kind_local_known", "kind_moving_blocker", "kind_moving_cost",
    "commit_time", "arrival_time", "scout_travel_time", "scout_dwell_time",
    "communication_delay", "single_agent_remote_time", "dispatch_cost",
    "top_base_cost", "bottom_cost", "failure_top_cost", "dynamic_radius",
    "moving_cost_amplitude", "carrier_knows_mode", "prior_top_mean", "prior_top_std",
    "report_slack", "single_query_slack",
)


def _weighted_top_stats(case: ScoutNecessityCase) -> Tuple[float, float]:
    probs = np.asarray([m.probability for m in case.modes], dtype=np.float64)
    probs = probs / probs.sum()
    from .conditional_scout_benchmark import top_route_cost
    vals = np.asarray([top_route_cost(case, m) for m in case.modes], dtype=np.float64)
    mean = float((probs * vals).sum())
    var = float((probs * (vals - mean) ** 2).sum())
    return mean, float(np.sqrt(max(var, 0.0)))


def case_features(case: ScoutNecessityCase) -> np.ndarray:
    kind = [1.0 if case.kind == k else 0.0 for k in KINDS]
    mean, std = _weighted_top_stats(case)
    numeric = [
        case.commit_time,
        case.carrier_route_arrival_time,
        case.scout_travel_time,
        case.scout_dwell_time,
        case.communication_delay,
        case.single_agent_remote_time,
        case.dispatch_cost,
        case.top_base_cost,
        case.bottom_cost,
        case.failure_top_cost,
        case.dynamic_radius,
        case.moving_cost_amplitude,
        float(case.carrier_knows_mode),
        mean,
        std,
        case.commit_time - case.report_time,
        case.commit_time - case.single_agent_remote_time,
    ]
    return np.asarray(kind + numeric, dtype=np.float32)


def sample_case(rng: np.random.Generator, index: int) -> ScoutNecessityCase:
    """Randomized legal-history case; hidden realized mode is never a feature."""
    kind = str(rng.choice(KINDS, p=[.30, .15, .30, .25]))
    p = float(rng.uniform(.25, .75))
    arrival = float(rng.uniform(6.0, 11.0))
    commit = float(rng.uniform(2.2, 6.5))
    travel = float(rng.uniform(1.0, 3.4))
    dwell = float(rng.uniform(.2, .8))
    delay = float(rng.uniform(.1, .9))
    single_remote = float(rng.uniform(4.8, 9.0))
    dispatch_cost = float(rng.uniform(.12, .85))
    top_base = float(rng.uniform(.7, 1.5))
    bottom = float(rng.uniform(2.1, 4.2))
    failure = float(rng.uniform(5.0, 9.0))
    radius = float(rng.uniform(.20, .55))
    amplitude = float(rng.uniform(3.0, 7.0))

    if kind in ("static_block", "local_known"):
        modes = (HiddenMode("clear", p, blocked=False),
                 HiddenMode("blocked", 1.0 - p, blocked=True))
    elif kind == "moving_blocker":
        # Construct one likely crossing trajectory and one clearing trajectory,
        # with modest random perturbations.  The sender sees the trajectory only
        # after dispatch; the carrier sees only the prior statistics here.
        v_cross = float(rng.uniform(-.30, -.10))
        phase_cross = float(-v_cross * arrival + rng.normal(0, radius * .20))
        v_clear = float(rng.uniform(.10, .30))
        phase_clear = float(rng.uniform(.7, 1.8))
        modes = (HiddenMode("crossing", p, phase=phase_cross, velocity=v_cross),
                 HiddenMode("clearing", 1.0 - p, phase=phase_clear, velocity=v_clear))
    else:
        v_bad = float(rng.uniform(.10, .30))
        phase_bad = float(-v_bad * arrival + rng.normal(0, radius * .15))
        v_good = float(rng.uniform(.10, .30))
        phase_good = float(rng.uniform(.8, 2.0))
        modes = (HiddenMode("pulse_at_crossing", p, phase=phase_bad, velocity=v_bad),
                 HiddenMode("pulse_away", 1.0 - p, phase=phase_good, velocity=v_good))

    return ScoutNecessityCase(
        name=f"sample_{index:06d}",
        kind=kind,
        modes=modes,
        commit_time=commit,
        carrier_route_arrival_time=arrival,
        scout_travel_time=travel,
        scout_dwell_time=dwell,
        communication_delay=delay,
        single_agent_remote_time=single_remote,
        dispatch_cost=dispatch_cost,
        top_base_cost=top_base,
        bottom_cost=bottom,
        failure_top_cost=failure,
        dynamic_radius=radius,
        moving_cost_amplitude=amplitude,
        carrier_knows_mode=(kind == "local_known"),
    )


def make_dataset(count: int, seed: int):
    rng = np.random.default_rng(seed)
    cases = [sample_case(rng, i) for i in range(int(count))]
    x = np.stack([case_features(c) for c in cases], axis=0)
    rows = [evaluate_case(c) for c in cases]
    y = np.asarray([r["scout_value"] for r in rows], dtype=np.float32)
    label = np.asarray([r["dispatch"] for r in rows], dtype=np.int64)
    return cases, x, y, label, rows


class DispatchValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def _standardize(x, mean, std):
    return (x - mean) / std


def train_model(outdir: Path, n_train=12000, n_val=2500, epochs=120,
                batch_size=256, lr=2e-3, seed=1701, device="cpu"):
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, xtr, ytr, ltr, _ = make_dataset(n_train, seed)
    _, xva, yva, lva, _ = make_dataset(n_val, seed + 1)
    mean, std = _standardize_fit(xtr)
    xtr_n = _standardize(xtr, mean, std)
    xva_n = _standardize(xva, mean, std)

    ds = TensorDataset(torch.from_numpy(xtr_n), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = DispatchValueMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss(beta=.15)
    best = float("inf")
    history = []
    ckpt = outdir / "dispatch_value_best.pt"

    xva_t = torch.from_numpy(xva_n).to(device)
    yva_t = torch.from_numpy(yva).to(device)
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        seen = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item()) * len(xb)
            seen += len(xb)
        model.eval()
        with torch.no_grad():
            pv = model(xva_t)
            vl = float(loss_fn(pv, yva_t).item())
            dispatch_hat = (pv > 0).cpu().numpy().astype(np.int64)
        acc = float((dispatch_hat == lva).mean())
        row = {"epoch": epoch, "train_loss": running / max(seen, 1),
               "val_loss": vl, "val_dispatch_accuracy": acc}
        history.append(row)
        if vl < best:
            best = vl
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": int(xtr.shape[1]),
                "hidden": 96,
                "feature_names": FEATURE_NAMES,
                "mean": mean,
                "std": std,
                "seed": seed,
                "best_val_loss": best,
            }, ckpt)

    (outdir / "train_history.json").write_text(json.dumps(history, indent=2) + "\n")
    _plot_history(history, outdir / "train_history.png")
    return ckpt, history[-1]


def load_model(path: Path, device="cpu"):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch 1.13 compatibility
        payload = torch.load(path, map_location=device)
    model = DispatchValueMLP(payload["input_dim"], payload.get("hidden", 96)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def predict_value(model, payload, cases: Sequence[ScoutNecessityCase], device="cpu"):
    x = np.stack([case_features(c) for c in cases], axis=0)
    xn = _standardize(x, np.asarray(payload["mean"]), np.asarray(payload["std"]))
    with torch.no_grad():
        p = model(torch.from_numpy(xn).to(device)).cpu().numpy()
    return p.astype(np.float64)


def evaluate_model(ckpt: Path, outdir: Path, n_test=5000, seed=2701,
                   margin=0.0, device="cpu"):
    outdir.mkdir(parents=True, exist_ok=True)
    cases, _, y, labels, rows = make_dataset(n_test, seed)
    model, payload = load_model(ckpt, device)
    pred = predict_value(model, payload, cases, device)
    pred_dispatch = pred > float(margin)
    truth_dispatch = y > 0.0
    tp = int(np.sum(pred_dispatch & truth_dispatch))
    tn = int(np.sum(~pred_dispatch & ~truth_dispatch))
    fp = int(np.sum(pred_dispatch & ~truth_dispatch))
    fn = int(np.sum(~pred_dispatch & truth_dispatch))
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    acc = float(np.mean(pred_dispatch == truth_dispatch))
    recall = float(tp / max(tp + fn, 1))
    false_dispatch = float(fp / max(fp + tn, 1))

    # Cost regret under predicted dispatch versus exact selective dispatch.
    learned_cost = []
    oracle_cost = []
    idle_cost = []
    for p, row in zip(pred_dispatch, rows):
        learned_cost.append(row["dispatch_cost_total"] if p else row["idle_cost"])
        oracle_cost.append(row["conditional_cost"])
        idle_cost.append(row["idle_cost"])
    learned_cost = np.asarray(learned_cost)
    oracle_cost = np.asarray(oracle_cost)
    idle_cost = np.asarray(idle_cost)
    metrics = {
        "count": int(n_test), "seed": int(seed), "margin": float(margin),
        "value_mae": mae, "value_rmse": rmse,
        "dispatch_accuracy": acc,
        "useful_scout_recall": recall,
        "false_dispatch_rate": false_dispatch,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "mean_idle_cost": float(idle_cost.mean()),
        "mean_oracle_conditional_cost": float(oracle_cost.mean()),
        "mean_learned_conditional_cost": float(learned_cost.mean()),
        "mean_learned_regret_to_oracle": float((learned_cost - oracle_cost).mean()),
    }
    (outdir / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _plot_prediction(y, pred, outdir / "value_prediction.png")
    return metrics


def _plot_history(history, path):
    epochs = [r["epoch"] for r in history]
    train = [r["train_loss"] for r in history]
    val = [r["val_loss"] for r in history]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(epochs, train, label="train")
    ax.plot(epochs, val, label="validation")
    ax.set(xlabel="epoch", ylabel="SmoothL1 loss", title="Dispatch-value training")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_prediction(y, pred, path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), constrained_layout=True)
    ax.scatter(y, pred, s=8, alpha=.25)
    lo = float(min(np.min(y), np.min(pred)))
    hi = float(max(np.max(y), np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.axhline(0, linewidth=.8)
    ax.axvline(0, linewidth=.8)
    ax.set(xlabel="Exact scout value", ylabel="Predicted scout value",
           title="Held-out dispatch-value prediction")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--outdir", type=Path, default=Path("coph_terrain/results/conditional_scout_learning"))
    p.add_argument("--n-train", type=int, default=12000)
    p.add_argument("--n-val", type=int, default=2500)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("eval")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("coph_terrain/results/conditional_scout_learning/test"))
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--seed", type=int, default=2701)
    p.add_argument("--margin", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.command == "train":
        ckpt, last = train_model(args.outdir, args.n_train, args.n_val,
                                 args.epochs, args.batch_size, args.lr,
                                 args.seed, args.device)
        print(json.dumps({"checkpoint": str(ckpt), "last": last}, indent=2))
    else:
        metrics = evaluate_model(args.ckpt, args.outdir, args.n_test,
                                 args.seed, args.margin, args.device)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
