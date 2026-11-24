# Scenario baseline admission

This directory records the source/reproduction gate for the two new external comparisons. It does not implement or claim to reproduce either method.

| Baseline | Pinned primary source | Native status | Comparative status |
|---|---|---|---|
| DistGP | [arXiv:2603.07351v1](https://arxiv.org/html/2603.07351v1) | Blocked: official implementation/native example not located | Not run |
| UGV-conditioned exposure planning | [arXiv:2606.12306v3](https://arxiv.org/html/2606.12306v3) | Blocked: official implementation/native example not located | Not run |

Checked on 2026-11-24. The manuscripts, author sites, public author repository catalogs, and exact-name/title GitHub searches were examined. This is a source-availability finding, not a claim that the authors have no implementation. `source_capture.json` records the consulted URLs, hashes, and repository-query results. No author was contacted.

DistGP's lead-author site announces the paper but links no source. Its public repository catalog contains the different `gbp_learning` method, which was not substituted. The exposure-planning coauthor [publication page](https://smoggy-p.github.io/publications/) lists this paper with a Paper link only; adjacent papers explicitly provide Code links. The other active-perception/exploration repositories implement different methods and were not substituted. Exact software commits, licenses, dependency versions, native inputs, and run commands remain unavailable rather than guessed.

## Reproduction order

1. Obtain an official source link or author-supplied source snapshot and usage/license terms.
2. Pin the exact source revision; install only into an isolated environment.
3. Run an unmodified official small example and preserve its declared checks, inputs, environment, output log, and hashes.
4. Validate the CoPH-facing adapter on identical inputs before freezing matched evaluation settings.
5. Freeze the completed matched protocol before any comparative outcomes. Keep original and adapted methods separately named.

The source search is complete for the currently discoverable sources. No native or comparative run was executed because admission is blocked. This stop preserves the requested native-first evaluation contract.

## Comparison intent

`matched_protocol_intent.json` seals the intended data, endpoints, and fairness requirements. It is **not a finalized executable comparison protocol**: native settings and adapters cannot be fixed faithfully without the source. The seal preserves this distinction; no comparative results were used to select it.

- **DistGP / scenario 5:** first use identical four-reading batches and supported path pairs from the existing saved-state study. That isolates representation, not distributed-fusion benefits. A later distributed comparison must give both systems the same local measurement streams and opportunities and count actual messages/work.
- **Exposure planning / scenarios 2–3:** reproduce its original one-UGV/aerial-scout roles first. A controlled-scout adapter can then compare belief/route use with identical legal observations and the same downstream execution. Full acquisition comparison additionally requires matched aerial mobility, sensing actions and planning budgets. The existing planar reciprocal maze is not an unmodified native setting for that framework.

Do not force a positive result. DistGP already supplies collaborative reconstruction and uncertainty; the exposure planner already supplies task-directed sensing and hazard-aware replanning. Any CoPH advantage must be measured for a stated endpoint and resource regime.

## Commands

From the repository root:

```bash
python benchmarks/scenario_baselines/audit.py status
python benchmarks/scenario_baselines/audit.py verify
python -m unittest discover -s benchmarks/scenario_baselines -p 'test_*.py' -v
```

The `freeze` command requires a passing source-pinned, unadapted native record and refuses blocked records. It does not run or certify the baseline automatically. Once official sources become available, complete the record and resolve the adapter details before using it to publish a final protocol. The existing phMARL reproduction remains in `../phmarl/` and is unaffected.
