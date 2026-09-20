# CoPH-Sense confirmatory campaign

The campaign runner fixes six method identifiers:

1. `independent`
2. `raw_broadcast`
3. `compact_broadcast`
4. `novelty_mi`
5. `need_request_match`
6. `full_coph`

Every method receives the same training seed, map group, terrain realization,
and episode seed. Reporting rejects incomplete paired blocks. Required episode
metrics are success, team cost, material-risk exposure, and charged bytes.
Duplicate sensing, disjoint explored support, and effective look-ahead gain are
optional only when a benchmark cannot define them.

The runner writes append-only JSONL so interrupted jobs can resume, then emits
JSON, CSV, Markdown, and PNG summaries. It computes empirical risk CVaR.95 and
paired block-bootstrap differences between full CoPH and each rival. The source
manifest hashes the evaluator, environment, material-pH adapter, calibration,
map manifests, and frozen A4 manifest used by a launch.

`confirmatory_smoke_adapter.py` contains synthetic metrics solely for plumbing
tests. Its artifacts must never be reported as experimental evidence. Replace
it with the final episode adapter after all six complete policies are frozen.
