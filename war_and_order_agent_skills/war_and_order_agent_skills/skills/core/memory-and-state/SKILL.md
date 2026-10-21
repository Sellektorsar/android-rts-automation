---
name: memory-and-state
description: Maintain an auditable local state model and action history using only confirmed observations.
---
# Memory and State
Store device profile/calibration version, last confirmed screen, visible resource readings, queues, timers, hero/army facts, current goal, allowed actions and verification history. Every fact needs timestamp, source frame, confidence and status: `confirmed`, `stale`, `uncertain` or `unknown`. Never use stale values for cost checks. Timers are estimates; observe the screen before acting when a timer should finish. Log action, rationale, pre-state, expected/actual result and evidence. Do not store passwords, tokens or payment information.
