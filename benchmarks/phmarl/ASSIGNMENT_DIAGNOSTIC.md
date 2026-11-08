# Reset-only assignment diagnostic

The installed VMAS `simple_spread` reward sums each landmark's distance to its nearest agent. It does not prescribe an agent-to-landmark assignment. The original native CoPH adapter nevertheless targets landmark `i` with agent `i`.

This diagnostic compares that assignment with a centralized minimum-total-squared-distance one-to-one assignment, computed once from native reset observations. All 24 permutations are enumerated; exact ties use lexicographic ordering. The assignment remains fixed throughout the episode. The controller and protection layer receive the same explicit goal indices; observations, native rewards and executed-input identification are unchanged.

Run from the repository root in the pinned native environment:

```bash
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python -m unittest discover -s benchmarks/phmarl -p 'test_*.py'
CoPh-sense_wrong/external/envs/phmarl-py38/bin/python benchmarks/phmarl/assignment_campaign.py --out CoPH-Stagewise/results/native_assignment_v1 --workers 4
```

The runner refuses existing output directories. Use a new directory for reproduction. It freezes source and archived-input hashes before running 60 index-assignment episodes and checks their trajectories, controls, learning decisions and work against the saved native campaign. Only after that gate passes does it run 60 minimum-distance episodes. Damping modes are frozen, known-native and online; seeds are 82000–82019; every episode has 400 steps.

Artifacts include episode arrays, decoded protection and update traces, assignment records, archive parity, paired summaries and intervals, a complete incomplete/contact inventory, and synchronized animations of previously observed failure seeds 82000, 82011 and 82019. Compressed gzip byte differences are reported independently of decoded-record parity.

For reassigned episodes, use `assigned_goal_stall_agent_steps`, not the legacy worker's index-goal stall count. `stationary_agent_steps_with_incomplete_team_coverage` additionally measures stationary agents while anonymous team coverage is incomplete. These are diagnostics, not distinct incident counts. Existing initial overlaps remain in the records and are distinguished from new contact onsets.

Centralized assignment uses all initial agent positions. It is a task-allocation diagnostic rather than a decentralized-policy superiority result. No recovery waypoints or revised safety rules are introduced. Receiver reconstruction and historical replay are not part of this native adapter.
