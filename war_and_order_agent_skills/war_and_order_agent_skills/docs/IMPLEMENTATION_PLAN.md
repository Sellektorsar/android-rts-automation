# Implementation and acceptance plan

## 1. Device connection
- Install Platform Tools, enable USB debugging, unlock phone and authorize the host.
- `adb devices` must show exactly one intended device as `device`.
- Run `python tools/adb_smoke_test.py`.
Acceptance: screenshot saved, resolution known, no taps issued.

## 2. Observe-only perception
Collect real screenshots of city, world map, hero/army, research, training, alliance, events, battle, loading and dialogs. Label screen type and critical element bounding boxes. Use a conservative initial confidence threshold (0.90 for screen, 0.93 for critical controls); tune using real validation data.
Acceptance: unknown screens stay unknown; unreadable resources remain null.

## 3. Calibrate actions
Use current-frame bounding boxes; one action at a time; obtain a new frame after each action.
Acceptance: no stale-coordinate taps; orientation/resolution changes block input.

## 4. Low-risk pilot
Start with navigation and reading state. Then allow explicitly approved free claims and queue inspection. No premium currency, purchases, player PvP, teleport, rare items or social actions.
Acceptance: 20 scenarios verified by postconditions; no blind retries.

## 5. Bounded autonomy
Limit session duration and action count; provide an immediate stop command; stop on disconnect, unknown modal, low confidence or two consecutive unverified actions. Produce a session report.
Acceptance: emergency stop tested; no infinite loops; disconnected ADB means no further actions.

## 6. Add game workflows
Add one workflow at a time. Each needs preconditions, risk class, expected postcondition, visual evidence, timeout, recovery path and replayable screenshot tests.
