# Five-second staggered-start demonstration

Both agents start at the same point and navigate to the same goal. A0 starts at 0 s; A1 activates at exactly 5 s, with no sensing or communication before activation. Each receives its full active horizon. Communication remains reciprocal fixed-rate. Controller and learner sources are unchanged.

Two development scenes (slalom passage and long obstacle corridor), two seeds, and communication off/on produced **16/16 agent goals across eight episodes**, with zero collisions or timeouts. These demonstrations are separate from held-out validation.

- [Slalom animation](results/staggered_demo_v1/figures/staggered_slalom_passage.gif)
- [Long corridor animation](results/staggered_demo_v1/figures/staggered_long_corridor.gif)
- [Full results](results/staggered_demo_v1/REPORT.md)
- [Settings and reproduction](STAGGERED_DEMO.md)

The synchronized panels show off/on trajectories, delivery arrows and anchors, communication timestamps, and isolated deliver/withhold model and control differences. A delivery does not imply acceptance or an immediate control effect. The first eligible leader-to-follower packet in the illustrated slalom run arrives at 8.72 s; earlier leader packets lack receiver support.

Across reciprocal episodes, 381 packets were delivered, 296 were eligible, and 289 produced accepted updates. In 32/58 isolated audits, the next applied control changed by more than the reporting tolerance of 1e-10 N. This tolerance distinguishes recorded numerical differences; it is not a practical-benefit threshold. Audits use identical retained receiver history and an update in both branches, without mutating operational state.

**Communication benefit remains unproven here.** For the two illustrated seed-73000 pairs, maximum off/on position differences were below 0.00041 units, and finish times were identical. The animations show these small effects numerically without magnification. The head start makes timing clearer but does not remove receiver-support restrictions.

Four targeted tests passed. Artifact checks verified all 55 frozen source fingerprints, exact launch times, active horizons, cost sums, delivery ages, byte limits, audit isolation, GIF frame records, and failure accounting. Maximum local-update work was 614/1024 flows; maximum physical-energy residual was 1.66e-15. Every episode completed; the failure gallery is empty. Original results and final-test namespaces are preserved.
