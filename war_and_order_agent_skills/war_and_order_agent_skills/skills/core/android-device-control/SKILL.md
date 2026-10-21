---
name: android-device-control
description: Control an owner-authorized physical Android device through the provided ADB adapter.
---
# Android Device Control
1. Confirm authorized device, unlocked phone, current resolution/orientation and fresh screenshot.
2. Select targets only from current-frame vision bounding boxes. Check bounds.
3. Execute exactly one tap/swipe/back action.
4. Wait for animation and capture a new frame.
5. Pass the result to `action-verification`.
6. Stop on disconnect, system dialog, orientation change, unknown screen or low confidence.
Never hide ADB, bypass protections, enter credentials/2FA/payment data, or reuse stale coordinates. A command being sent is not proof of game success. Do not automatically select among multiple connected devices.
Return device status, frame ID, target bbox, action, execution result and verification status.
