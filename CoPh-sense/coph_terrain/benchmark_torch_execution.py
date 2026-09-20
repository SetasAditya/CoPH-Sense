"""Measure batched pH-kernel throughput without pretending it is full E2 throughput."""

import argparse
import json
import time

import numpy as np
import torch

from .environment import CoPHTerrainEnv
from .execution import belief_risk_grid
from .torch_execution import ph_step_batch


def benchmark(device, batch, steps):
    env = CoPHTerrainEnv("open", 660000, 0, 13)
    obs = env.observations()["carrier"]
    dtype = torch.float32 if device.startswith("cuda") else torch.float64
    q = torch.tensor(np.repeat([[-5., -.25]], batch, axis=0),
                     device=device, dtype=dtype)
    v = torch.zeros_like(q)
    target = torch.tensor(np.repeat([[-4., -.25]], batch, axis=0),
                          device=device, dtype=dtype)
    geometry = torch.tensor(np.repeat(obs["known_geometry"][None], batch, 0),
                            device=device, dtype=dtype)
    risk = torch.tensor(np.repeat(belief_risk_grid(obs)[None], batch, 0),
                        device=device, dtype=dtype)
    teammate = torch.tensor(np.repeat([[-5., .25]], batch, axis=0),
                             device=device, dtype=dtype)
    mass = torch.full((batch,), 1.8, device=device, dtype=dtype)
    radius = torch.full((batch,), .10, device=device, dtype=dtype)
    speed = torch.full((batch,), .55, device=device, dtype=dtype)
    traction = torch.full((batch,), .8, device=device, dtype=dtype)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            q, v, _ = ph_step_batch(q, v, target, geometry, risk,
                                    teammate, mass, radius, speed, traction, .05)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return {"device": device, "batch": batch, "steps": steps,
            "elapsed_seconds": elapsed,
            "branch_steps_per_second": batch * steps / elapsed,
            "gpu_peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 2**20)
            if device.startswith("cuda") else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=(1, 16, 137))
    args = parser.parse_args()
    for batch in args.batches:
        print(json.dumps(benchmark(args.device, batch, args.steps)), flush=True)


if __name__ == "__main__":
    main()
