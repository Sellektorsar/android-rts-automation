import unittest

from src.time_parser import parse_duration


class DurationParserTest(unittest.TestCase):
    def test_minutes_seconds(self):
        self.assertEqual(parse_duration("11:52"), 712)

    def test_hours_minutes_seconds(self):
        self.assertEqual(parse_duration("1:02:03"), 3723)

    def test_ocr_dot_instead_of_colon(self):
        self.assertEqual(parse_duration("11.52"), 712)

    def test_russian_units(self):
        self.assertEqual(parse_duration("11 мин 52 сек"), 712)
        self.assertEqual(parse_duration("1 ч 2 мин 3 с"), 3723)

    def test_compact_digits(self):
        self.assertEqual(parse_duration("1152"), 712)

    def test_invalid_seconds_rejected(self):
        self.assertIsNone(parse_duration("11:75"))
        self.assertIsNone(parse_duration("1:75:00"))

    def test_non_duration_text_rejected(self):
        self.assertIsNone(parse_duration("Детали"))
        self.assertIsNone(parse_duration("110,890"))

    def test_zero_duration(self):
        self.assertEqual(parse_duration("00:00"), 0)
        self.assertEqual(parse_duration("0:05"), 5)


if __name__ == "__main__":
    unittest.main()
