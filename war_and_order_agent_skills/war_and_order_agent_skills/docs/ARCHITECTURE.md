# Architecture

```text
User goal → Planner → State memory → Screen perception
          → Risk/policy gate → one action via Android adapter
          → fresh screenshot → postcondition verifier
             ├ confirmed: update memory and continue
             ├ unknown: reobserve once, do not repeat action
             └ failed: recovery skill, stop if unresolved
```

## Adapter contract
Implement these functions in your host process:
- `get_device_info()`: authorized device serial, resolution, orientation, foreground app if available.
- `capture_screen() -> image`
- `tap(x, y)`
- `swipe(x1, y1, x2, y2, duration_ms)`
- `press_back()`
- `wait(seconds)`
- `stop_all()`

Coordinates must be derived from a fresh frame and current resolution. Prefer detector bounding boxes over hard-coded points. Invalidate screen coordinates after zoom, pan, orientation or resolution changes.

## Perception schema
Return structured observations such as:
```json
{
  "screen": "city|world_map|hero_or_army|research|training|alliance|event|battle|dialog|loading|unknown",
  "screen_confidence": 0.0,
  "elements": [{"name":"unknown","bbox":[0,0,0,0],"text":null,"confidence":0.0}],
  "resources": {"food":{"value":null,"confidence":0.0}},
  "dialogs": [],
  "frame_id": "unique-id",
  "uncertainties": []
}
```
This is a schema example, not pre-calibrated game data. Use `null` rather than inventing unreadable values.

## Risk gate
Require a fresh frame, sufficient confidence, a known screen, valid target bounds, an expected postcondition, and policy approval before each action. Keep purchases, premium currency, rare items, player PvP, teleportation, alliance changes and messages behind explicit owner approval.

## Verification
Every action declares preconditions, expected postcondition, timeout and visual evidence. The command being sent is not proof of in-game success. On `unknown`, reobserve once; never automatically repeat potentially costly actions.

## Memory
Persist observations with timestamp, source frame, confidence and status (`confirmed`, `stale`, `uncertain`, `unknown`). Track resource readings, queues, timers, current objective, actions and verification evidence. Do not store passwords, tokens or payment data.
