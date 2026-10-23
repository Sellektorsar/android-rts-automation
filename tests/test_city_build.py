"""Тесты безопасной постройки: типы карусели, требования и цвет кнопки."""

import unittest

import numpy as np

from src.tasks.city_build import CityBuildTask


class CityBuildTypeMappingTest(unittest.TestCase):
    def setUp(self):
        self.task = CityBuildTask(client=None, vision=None, plots=[], interval_sec=1)

    def test_live_ocr_variants_map_to_types(self):
        cases = {
            "pepMa": "farm",
            "NeconMAka": "sawmill",
            "yyebHbin Aarepb": "camp",
            "MeAMuMHcKa naaTKa": "tent",
            "KaMeHHan WaxTa": "quarry",
            "XKere3Hbli pyAHMK": "iron",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                index = self.task._type_index(text)
                self.assertEqual(self.task.TYPE_ORDER[index], expected)

    def test_unknown_title(self):
        self.assertEqual(self.task._type_index("Требуется"), -1)

    def test_cycle_covers_four_secondary_types(self):
        self.assertEqual(
            set(self.task.BUILD_CYCLE), {"farm", "sawmill", "camp", "tent"}
        )


class CityBuildRequirementsTest(unittest.TestCase):
    def setUp(self):
        self.task = CityBuildTask(client=None, vision=None, plots=[], interval_sec=1)

    def test_numbers_from_ocr_block(self):
        numbers = self.task._row_numbers(
            "5 yp. / 1 yp. 3aMOK 166,000 / 200"
        )
        self.assertEqual(numbers, [5, 1, 166000, 200])

    def test_leading_letter_word_is_not_a_number(self):
        numbers = self.task._row_numbers("5 yp. / 4 yp. 3aMOK 166,000 / 245")
        self.assertEqual(numbers, [5, 4, 166000, 245])

    def test_slash_misread_as_one_does_not_glue_to_resource(self):
        numbers = self.task._row_numbers("5 yp. / 1 yp. 3aMOK 136,321 1/200")
        self.assertEqual(numbers, [5, 1, 136321, 200])

    def test_synthetic_green_ticks(self):
        zone = np.zeros((100, 120, 3), dtype=np.uint8)
        zone[20:40, 20:40] = (40, 220, 60)
        zone[60:80, 20:40] = (40, 220, 60)
        self.assertEqual(self.task._green_ticks(zone), 2)


class CityBuildBlueButtonTest(unittest.TestCase):
    def setUp(self):
        self.task = CityBuildTask(client=None, vision=None, plots=[], interval_sec=1)

    def test_active_blue_button(self):
        frame = np.full((2436, 1080, 3), 30, dtype=np.uint8)
        frame[2314:2374, 770:930] = (200, 120, 40)  # BGR: синий
        saturation, blue_minus_red = self.task._blue_button_color(frame, (850, 2344, 0.99))
        self.assertGreaterEqual(saturation, 60.0)
        self.assertGreaterEqual(blue_minus_red, 10.0)

    def test_grey_inactive_button(self):
        frame = np.full((2436, 1080, 3), 30, dtype=np.uint8)
        frame[2314:2374, 770:930] = (90, 90, 90)
        saturation, blue_minus_red = self.task._blue_button_color(frame, (850, 2344, 0.99))
        self.assertLess(saturation, 60.0)
        self.assertLess(blue_minus_red, 10.0)


class CityBuildPlotFilterTest(unittest.TestCase):
    def setUp(self):
        self.task = CityBuildTask(client=None, vision=None, plots=[], interval_sec=1)

    def test_excludes_built_and_attempted(self):
        built = {(818, 2140)}
        self.assertFalse(self.task._usable((818, 2140), 1080, 2436, built))
        self.task.attempted.add((728, 1856))
        self.assertFalse(self.task._usable((728, 1856), 1080, 2436, set()))
        self.assertTrue(self.task._usable((878, 1759), 1080, 2436, built))

    def test_excludes_hud_zones(self):
        self.assertFalse(self.task._usable((1000, 1650), 1080, 2436, set()))
        self.assertFalse(self.task._usable((540, 2350), 1080, 2436, set()))
        self.assertFalse(self.task._usable((60, 1500), 1080, 2436, set()))
        # Настоящая ячейка у левого края квартала не должна отбрасываться.
        self.assertTrue(self.task._usable((418, 1486), 1080, 2436, set()))


if __name__ == "__main__":
    unittest.main()
