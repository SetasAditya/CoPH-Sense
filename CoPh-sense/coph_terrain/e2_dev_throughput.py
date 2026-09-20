"""Measure E2-dev candidate generation and critic inference separately."""

import json
import time

import torch

from .e2_dev_stack import OUT, TRAIN, acquisition_menu, local_inputs
from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition
from .value_model import VariableSetValueNet, decision_score


def main():
    torch.set_num_threads(1)
    group = json.loads(TRAIN.read_text())["groups"][3]
    seed = group["parent_seed"]
    env = CoPHTerrainEnv(group["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, 120)
    observation = env.observations()["carrier"]
    model = VariableSetValueNet()
    model.load_state_dict(torch.load(OUT / "stack_v1_a3_a5_disposable.pt",
                                     map_location="cpu"))
    model.eval()
    samples = 30
    generated = []
    for _ in range(samples):
        before = time.perf_counter()
        candidates, _, sets = acquisition_menu(env, "carrier")
        generated.append(time.perf_counter() - before)
    features = torch.tensor([candidate.features() for candidate in candidates],
                            dtype=torch.float32)[None]
    sets_tensor = torch.tensor(sets, dtype=torch.long)
    inputs = local_inputs(observation, "carrier")
    # Warm-up before measuring the stable inference path.
    with torch.no_grad():
        for _ in range(5):
            decision_score(*model(*inputs, features, sets_tensor)).argmin()
        inferred = []
        for _ in range(samples):
            before = time.perf_counter()
            decision_score(*model(*inputs, features, sets_tensor)).argmin()
            inferred.append(time.perf_counter() - before)
    report = {"status": "engineering_benchmark", "device": "cpu",
              "samples": samples, "candidate_count": len(candidates),
              "set_count": len(sets),
              "candidate_median_ms": sorted(generated)[samples // 2] * 1000,
              "critic_median_ms": sorted(inferred)[samples // 2] * 1000,
              "scope": "single-state public candidate + one critic decision; excludes physical continuation"}
    path = OUT / "throughput_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
