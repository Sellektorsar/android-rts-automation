# War and Order Agent Skills

Starter pack for a custom visual agent controlling an owner-authorized physical Android phone. It is a skill pack and implementation blueprint, not a ready-to-run game bot: screen layouts and coordinates must be calibrated on your device.

## Architecture
Observe screenshot → classify screen/OCR → update state → choose one skill/action → execute through ADB → capture new screenshot → verify postcondition → update memory. If uncertain, stop rather than guess or repeat.

## Skills
- `skills/core/android-device-control/SKILL.md`
- `skills/core/screen-perception/SKILL.md`
- `skills/core/action-verification/SKILL.md`
- `skills/core/recovery-and-safety/SKILL.md`
- `skills/core/memory-and-state/SKILL.md`
- `skills/war-and-order/city-economy/SKILL.md`
- `skills/war-and-order/research-and-training/SKILL.md`
- `skills/war-and-order/hero-and-army/SKILL.md`
- `skills/war-and-order/world-map-and-monsters/SKILL.md`
- `skills/war-and-order/alliance-and-events/SKILL.md`
- `skills/war-and-order/daily-planner/SKILL.md`

## Quick start
1. Install Android Platform Tools; connect phone by USB; enable Developer options → USB debugging; unlock and authorize the computer.
2. Run `adb devices`; ensure exactly your intended device has status `device`.
3. Run `python tools/adb_smoke_test.py` to capture a screenshot without tapping.
4. Open War and Order manually and collect screenshots of city, map, research, training, hero/army, alliance, events, battle and common dialogs.
5. Calibrate `config/game-profile.template.yaml` from actual screenshots. Never guess coordinates.
6. Connect the SKILL.md files to your agent's skill loader and implement the adapter interface in `docs/ARCHITECTURE.md`.
7. Start in `observe_only`, then enable only low-risk verified actions.

## Safety
No CAPTCHA/anti-cheat bypass, process memory access, packet tampering or client modification. No purchases, premium currency, rare-item use, player PvP, teleportation, alliance changes or messaging without explicit owner approval. Check the game's current terms before unattended automation.
