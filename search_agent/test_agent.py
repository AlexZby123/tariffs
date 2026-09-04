# -*- coding: utf-8 -*-
"""
Testy jednostkowe i integracyjne dla modułu search_agent.
"""
from __future__ import annotations

import unittest
from models import ProductQuery, RawDimensions
from normalizer import (
    normalize_length_to_mm,
    normalize_weight_to_g,
    normalize_raw_dimensions,
    parse_combined_dimensions_string,
    _parse_fraction,
)
from search_provider import MockDatasheetProvider
from extractor_agent import DimensionExtractorAgent
from pipeline import DimensionWorkflow


class TestDimensionNormalizer(unittest.TestCase):

    def test_length_conversion(self):
        # Milimetry
        self.assertEqual(normalize_length_to_mm("120 mm"), 120.0)
        self.assertEqual(normalize_length_to_mm("12.5mm"), 12.5)

        # Centymetry i metry
        self.assertEqual(normalize_length_to_mm("15 cm"), 150.0)
        self.assertEqual(normalize_length_to_mm("1.2 m"), 1200.0)

        # Cale (inches)
        self.assertEqual(normalize_length_to_mm("1 in"), 25.4)
        self.assertEqual(normalize_length_to_mm('4"'), 101.6)
        self.assertEqual(normalize_length_to_mm("3.5 inches"), 88.9)

        # Ułamki calowe (fractions)
        self.assertEqual(normalize_length_to_mm("1 1/2 in"), 38.1)
        self.assertEqual(normalize_length_to_mm("3/8 in"), 9.52)
        self.assertEqual(normalize_length_to_mm("1/4 in"), 6.35)

        # Stopy (feet)
        self.assertEqual(normalize_length_to_mm("2 ft"), 609.6)

    def test_weight_conversion(self):
        # Gramy i kilogramy
        self.assertEqual(normalize_weight_to_g("350 g"), 350.0)
        self.assertEqual(normalize_weight_to_g("2.5 kg"), 2500.0)
        self.assertEqual(normalize_weight_to_g("500 mg"), 0.5)

        # Funty (lbs) i uncje (oz)
        self.assertAlmostEqual(normalize_weight_to_g("1 lb"), 453.59, delta=0.1)
        self.assertAlmostEqual(normalize_weight_to_g("2.5 lbs"), 1133.98, delta=0.1)
        self.assertAlmostEqual(normalize_weight_to_g("10 oz"), 283.50, delta=0.1)

        # Złożone: lbs + oz
        self.assertAlmostEqual(normalize_weight_to_g("2 lbs 4 oz"), 1020.58, delta=0.1)

    def test_combined_dimensions_string(self):
        l, w, h = parse_combined_dimensions_string("182 x 35 x 30 mm")
        self.assertEqual(l, 182.0)
        self.assertEqual(w, 35.0)
        self.assertEqual(h, 30.0)

        l_in, w_in, h_in = parse_combined_dimensions_string("10 x 5 x 2 in")
        self.assertEqual(l_in, 254.0)
        self.assertEqual(w_in, 127.0)
        self.assertEqual(h_in, 50.8)

    def test_normalize_raw_dimensions_object(self):
        raw = RawDimensions(
            raw_length="182 mm",
            raw_width="35 mm",
            raw_height="30 mm",
            raw_weight="490 g",
            confidence=0.95
        )
        metric = normalize_raw_dimensions(raw)
        self.assertEqual(metric.length_mm, 182.0)
        self.assertEqual(metric.width_mm, 35.0)
        self.assertEqual(metric.height_mm, 30.0)
        self.assertEqual(metric.weight_g, 490.0)
        self.assertEqual(metric.weight_kg, 0.49)
        self.assertIsNotNone(metric.volume_cm3)
        self.assertEqual(metric.dimension_string_mm, "182.0 x 35.0 x 30.0 mm")


class TestDimensionWorkflowIntegration(unittest.TestCase):

    def setUp(self):
        self.workflow = DimensionWorkflow(use_mock=True)

    def test_end_to_end_wheel_speed_sensor(self):
        q = ProductQuery(part_number="0265005303", brand="Bosch", title="Wheel speed sensor")
        res = self.workflow.process(q)

        self.assertTrue(res.found)
        self.assertIsNotNone(res.metric)
        # Z mock bazy: 65 x 24 x 18 mm, 78 g
        self.assertEqual(res.metric.length_mm, 65.0)
        self.assertEqual(res.metric.width_mm, 24.0)
        self.assertEqual(res.metric.height_mm, 18.0)
        self.assertEqual(res.metric.weight_g, 78.0)
        self.assertGreaterEqual(res.confidence, 0.8)

    def test_end_to_end_diesel_injector(self):
        q = ProductQuery(part_number="0445110189", brand="Bosch", title="Common rail injector")
        res = self.workflow.process(q)

        self.assertTrue(res.found)
        self.assertIsNotNone(res.metric)
        self.assertEqual(res.metric.length_mm, 182.0)
        self.assertEqual(res.metric.weight_g, 490.0)

    def test_end_to_end_bracket_imperial_units(self):
        q = ProductQuery(part_number="0000000000", brand="Generic", title="Mounting bracket")
        res = self.workflow.process(q)

        self.assertTrue(res.found)
        self.assertIsNotNone(res.metric)
        # W mock: 8.5 in x 4.25 in x 1.75 in, 2 lbs 3 oz
        self.assertAlmostEqual(res.metric.length_mm, 215.9, delta=0.2)
        self.assertAlmostEqual(res.metric.width_mm, 107.95, delta=0.2)
        self.assertAlmostEqual(res.metric.weight_g, 992.23, delta=0.5)


if __name__ == "__main__":
    unittest.main()
