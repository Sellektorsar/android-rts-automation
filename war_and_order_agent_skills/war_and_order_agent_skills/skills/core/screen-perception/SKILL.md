---
name: screen-perception
description: Classify War and Order screens and extract visible UI facts from a fresh screenshot using vision and OCR.
---
# Screen Perception
Classify only when evidence supports one of: `city`, `world_map`, `hero_or_army`, `research`, `training`, `alliance`, `event`, `battle`, `dialog`, `loading`, `unknown`.
Extract visible labels, bounding boxes, enabled state, OCR text and confidence. Store unreadable values as `null`, never guess. Detect system overlays, keyboard, modal dialogs and loading screens. Initial conservative thresholds: screen confidence >= 0.90; critical control >= 0.93. Below threshold, do not act. Reobserve after zoom, pan, orientation/resolution change or modal transition. Return screen, confidence, elements, resources, timers, dialogs, frame ID and uncertainties.
