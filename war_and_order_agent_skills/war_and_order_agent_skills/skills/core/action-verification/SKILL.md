---
name: action-verification
description: Verify the expected in-game result after every Android input.
---
# Action Verification
Before acting define precondition, expected postcondition, timeout and visual evidence. Capture pre-state, execute one action, wait, capture a new frame and compare. Return `confirmed`, `failed` or `unknown`. `unknown` means do not repeat the action: reobserve once and diagnose. Update game memory only after confirmation. Examples: tab opened → new header visible; upgrade started → queue/timer/level visibly changed; march sent → march/target/timer visible. Never infer success merely because no error appeared or because the tap command returned successfully.
