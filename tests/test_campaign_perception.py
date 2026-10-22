"""Регрессия восприятия карты кампании и брифинга этапа."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from src.fsm import GameState
from src.perception import DEFAULT_PROFILES, ScreenClassifier
from src.vision import Vision


def _profile_has_template(profile, template: str) -> bool:
    return any(anchor.template == template for group in profile.groups for anchor in group)


class CampaignPerceptionTest(unittest.TestCase):
    def test_briefing_has_priority_over_campaign_map(self):
        """Модалка этапа рисуется поверх карты и должна выигрывать."""
        briefing = next(
            profile
            for profile in DEFAULT_PROFILES
            if profile.state == GameState.STAGE_BRIEFING
            and _profile_has_template(profile, "btn_stage_challenge")
        )
        campaign = next(
            profile
            for profile in DEFAULT_PROFILES
            if profile.state == GameState.CAMPAIGN_MAP
            and _profile_has_template(profile, "title_campaign")
        )
        self.assertLess(briefing.priority, campaign.priority)

    @unittest.skipUnless(
        Path("logs/inspect/campaign_map_probe.png").exists(),
        "нет локального эталона карты кампании",
    )
    def test_campaign_map_screenshot(self):
        vision = Vision(templates_dir="assets")
        classifier = ScreenClassifier(vision, ocr_scale=0.5, allow_ocr=False)
        frame = cv2.imread("logs/inspect/campaign_map_probe.png")
        result = classifier.classify(frame, allow_ocr=False)
        self.assertEqual(result.state, GameState.CAMPAIGN_MAP)
        self.assertGreaterEqual(result.confidence, 0.80)

    @unittest.skipUnless(
        Path("logs/inspect/campaign_stage7_probe.png").exists(),
        "нет локального эталона брифинга этапа 7",
    )
    def test_stage7_briefing_screenshot(self):
        vision = Vision(templates_dir="assets")
        classifier = ScreenClassifier(vision, ocr_scale=0.5, allow_ocr=False)
        frame = cv2.imread("logs/inspect/campaign_stage7_probe.png")
        result = classifier.classify(frame, allow_ocr=False)
        self.assertEqual(result.state, GameState.STAGE_BRIEFING)
        self.assertGreaterEqual(result.confidence, 0.80)


if __name__ == "__main__":
    unittest.main()
