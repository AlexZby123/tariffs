# -*- coding: utf-8 -*-
"""
Testy jednostkowe i integracyjne dla modułu search_agent.
"""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from models import ProductQuery, RawDimensions
from normalizer import (
    normalize_length_to_mm,
    normalize_weight_to_g,
    normalize_raw_dimensions,
    parse_combined_dimensions_string,
    parse_numeric_value,
    _parse_fraction,
)
from search_utils import generate_pn_variants, mentions_part_number, score_title_match, extract_relevant_lines
from search_provider import (
    MockDatasheetProvider,
    CatalogDatasheetProvider,
    HybridDatasheetProvider,
    CachedSearchProvider,
    BaseDatasheetProvider,
)
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

    def test_length_conversion_no_space_before_unit(self):
        # Regresja: liczba "sklejona" z jednostką (bez spacji) wcześniej
        # cichym błędem trafiała w domyślny współczynnik 1.0 (jakby już
        # była w mm), np. "10cm" dawało 10.0 zamiast 100.0.
        self.assertEqual(normalize_length_to_mm("10cm"), 100.0)
        self.assertEqual(normalize_length_to_mm("8.5in"), 215.9)
        self.assertEqual(normalize_length_to_mm("1.2m"), 1200.0)
        self.assertEqual(normalize_length_to_mm("65mm"), 65.0)

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

    def test_weight_conversion_no_space_before_unit(self):
        self.assertEqual(normalize_weight_to_g("2kg"), 2000.0)
        self.assertAlmostEqual(normalize_weight_to_g("1lb"), 453.59, delta=0.1)
        self.assertEqual(normalize_weight_to_g("78g"), 78.0)

    def test_decimal_comma_vs_thousands_separator(self):
        # Europejski przecinek dziesiętny
        self.assertEqual(parse_numeric_value("12,5"), 12.5)
        self.assertEqual(parse_numeric_value("3,75"), 3.75)
        # Amerykański separator tysięcy
        self.assertEqual(parse_numeric_value("1,200"), 1200.0)
        # Oba naraz (US: przecinek tysięcy + kropka dziesiętna)
        self.assertEqual(parse_numeric_value("1,234.5"), 1234.5)

    def test_parse_fraction_directly(self):
        self.assertEqual(_parse_fraction("3/8 in"), 0.375)
        self.assertEqual(_parse_fraction("1 1/2 in"), 1.5)
        self.assertIsNone(_parse_fraction("120 mm"))

    def test_combined_dimensions_string(self):
        l, w, h = parse_combined_dimensions_string("182 x 35 x 30 mm")
        self.assertEqual(l, 182.0)
        self.assertEqual(w, 35.0)
        self.assertEqual(h, 30.0)

        l_in, w_in, h_in = parse_combined_dimensions_string("10 x 5 x 2 in")
        self.assertEqual(l_in, 254.0)
        self.assertEqual(w_in, 127.0)
        self.assertEqual(h_in, 50.8)

        # Bez spacji wokół jednostki - to już wcześniej działało poprawnie,
        # bo ten parser wyodrębnia jednostkę jako osobną grupę zamiast \b.
        l2, w2, h2 = parse_combined_dimensions_string("182x35x30mm")
        self.assertEqual((l2, w2, h2), (182.0, 35.0, 30.0))

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


class TestSearchUtils(unittest.TestCase):

    def test_generate_pn_variants_covers_both_grouping_conventions(self):
        variants = generate_pn_variants("0265005303")
        self.assertIn("0265005303", variants)
        self.assertIn("0 265 005 303", variants)       # 1-3-3-3
        self.assertIn("0.265.005.303", variants)
        self.assertIn("0265.005.303", variants)          # 4-3-3
        self.assertIn("0265 005 303", variants)
        # bez duplikatów
        self.assertEqual(len(variants), len(set(variants)))

    def test_mentions_part_number_ignores_formatting(self):
        self.assertTrue(mentions_part_number("Bosch part 0 265 005 303 datasheet", "0265005303"))
        self.assertTrue(mentions_part_number("PN: 0265.005.303", "0265005303"))
        self.assertFalse(mentions_part_number("Totally unrelated content", "0265005303"))

    def test_score_title_match_prefers_exact_pn_in_title(self):
        variants = generate_pn_variants("0265005303")
        score_exact = score_title_match("Bosch 0265005303 Datasheet", variants)
        score_generic = score_title_match("General sensor overview", variants)
        self.assertGreater(score_exact, score_generic)

    def test_extract_relevant_lines_keeps_dimension_mentions_beyond_head_cutoff(self):
        lines = [f"filler line {i}" for i in range(200)]
        lines[150] = "Weight: 78 g"
        result = extract_relevant_lines(lines, max_lines=50)
        self.assertIn("Weight: 78 g", result)
        self.assertLessEqual(len(result), 50)


class TestCatalogDatasheetProviderSelector(unittest.TestCase):
    """Regresja dla zbyt szerokiego selektora CSS, który dopasowywał KAŻDY <tr>
    na stronie (łącznie z niepowiązanymi tabelami typu 'podobne produkty')."""

    def test_specs_table_class_does_not_leak_unrelated_comparison_table(self):
        html = """
        <html><body>
        <h1>Bosch Wheel Speed Sensor 0265005303</h1>
        <table class="specs-table">
          <tr><td>Length</td><td>65 mm</td></tr>
          <tr><td>Weight</td><td>78 g</td></tr>
        </table>
        <h2>Customers also viewed</h2>
        <table class="compare-table">
          <tr><td>Alt. Sensor XYZ-200 - Length</td><td>210 mm</td></tr>
        </table>
        </body></html>
        """

        class FakeResp:
            status_code = 200
            text = html

        with patch("requests.get", return_value=FakeResp()):
            provider = CatalogDatasheetProvider()
            result = provider.search_and_fetch_text(ProductQuery(part_number="0265005303", brand="Bosch"))

        self.assertTrue(result["has_data"])
        self.assertIn("65 mm", result["text"])
        self.assertNotIn("Alt. Sensor", result["text"])

    def test_fallback_selector_skips_related_products_heading(self):
        # Strona bez klas "specs-table"/"tech-details" - sprawdzamy heurystykę
        # fallbackową opartą o nagłówki (_looks_like_unrelated_section).
        html = """
        <html><body>
        <h1>Bosch Wheel Speed Sensor 0265005303</h1>
        <table>
          <tr><td>Length</td><td>65 mm</td></tr>
          <tr><td>Weight</td><td>78 g</td></tr>
        </table>
        <h2>Related products</h2>
        <table>
          <tr><td>Alt Sensor - Height</td><td>210 mm</td></tr>
        </table>
        </body></html>
        """

        class FakeResp:
            status_code = 200
            text = html

        with patch("requests.get", return_value=FakeResp()):
            provider = CatalogDatasheetProvider()
            result = provider.search_and_fetch_text(ProductQuery(part_number="0265005303", brand="Bosch"))

        self.assertTrue(result["has_data"])
        self.assertIn("65 mm", result["text"])
        self.assertNotIn("Alt Sensor", result["text"])


class TestHybridDatasheetProviderFallback(unittest.TestCase):
    """Regresja: Hybrid wcześniej pytał WSZYSTKIE źródła zawsze, niezależnie
    od tego, czy wcześniejsze już zwróciło dane - wbrew własnemu docstringowi."""

    def test_stops_after_first_successful_source(self):
        calls = []

        class StubDocupedia(BaseDatasheetProvider):
            def search_and_fetch_text(self, query):
                calls.append("docupedia")
                return {"text": "authoritative", "source_url": "https://docupedia/x", "has_data": True}

        class StubCatalog(BaseDatasheetProvider):
            def search_and_fetch_text(self, query):
                calls.append("catalog")
                return {"text": "noisy", "source_url": "https://findpart/x", "has_data": True}

        class StubWeb(BaseDatasheetProvider):
            def search_and_fetch_text(self, query):
                calls.append("web")
                return {"text": "noisy", "source_url": "https://web/x", "has_data": True}

        hybrid = HybridDatasheetProvider(docupedia=StubDocupedia(), catalog=StubCatalog(), web=StubWeb())
        result = hybrid.search_and_fetch_text(ProductQuery(part_number="0265005303", brand="Bosch"))

        self.assertEqual(calls, ["docupedia"])
        self.assertEqual(result["source_type"], "docupedia")

    def test_falls_through_when_earlier_sources_have_no_data(self):
        class StubEmpty(BaseDatasheetProvider):
            def search_and_fetch_text(self, query):
                return {"text": "", "source_url": "", "has_data": False}

        class StubWeb(BaseDatasheetProvider):
            def search_and_fetch_text(self, query):
                return {"text": "found it", "source_url": "https://web/x", "has_data": True}

        hybrid = HybridDatasheetProvider(docupedia=StubEmpty(), catalog=StubEmpty(), web=StubWeb())
        result = hybrid.search_and_fetch_text(ProductQuery(part_number="X", brand="Bosch"))
        self.assertTrue(result["has_data"])
        self.assertEqual(result["source_type"], "web")


class TestCachedSearchProviderTTL(unittest.TestCase):

    def test_expired_cache_is_refetched(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = {"n": 0}

            class CountingProvider(BaseDatasheetProvider):
                def search_and_fetch_text(self, query):
                    calls["n"] += 1
                    return {"text": f"call {calls['n']}", "source_url": "https://x", "has_data": True}

            cached = CachedSearchProvider(CountingProvider(), cache_dir=Path(tmp), ttl_seconds=1)
            q = ProductQuery(part_number="TTLTEST", brand="Bosch")

            cached.search_and_fetch_text(q)
            self.assertEqual(calls["n"], 1)

            cached.search_and_fetch_text(q)  # nadal świeże -> z cache
            self.assertEqual(calls["n"], 1)

            time.sleep(1.2)
            cached.search_and_fetch_text(q)  # TTL minął -> nowe zapytanie
            self.assertEqual(calls["n"], 2)

    def test_refresh_flag_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = {"n": 0}

            class CountingProvider(BaseDatasheetProvider):
                def search_and_fetch_text(self, query):
                    calls["n"] += 1
                    return {"text": f"call {calls['n']}", "source_url": "https://x", "has_data": True}

            inner = CountingProvider()
            q = ProductQuery(part_number="REFRESHTEST", brand="Bosch")

            CachedSearchProvider(inner, cache_dir=Path(tmp)).search_and_fetch_text(q)
            self.assertEqual(calls["n"], 1)

            CachedSearchProvider(inner, cache_dir=Path(tmp), refresh=True).search_and_fetch_text(q)
            self.assertEqual(calls["n"], 2)


class TestExtractorAgentFallbackVisibility(unittest.TestCase):
    """Regresja: brak realnego klienta LLM w trybie nie-mock wcześniej
    prowadził do CICHEGO przełączenia na parser regexowy, bez żadnego ostrzeżenia."""

    def test_warns_when_llm_client_unavailable_in_live_mode(self):
        import extractor_agent
        original = extractor_agent.LLMClient
        extractor_agent.LLMClient = None  # symulujemy brak możliwości importu llm_client
        try:
            with self.assertLogs("extractor_agent", level="WARNING") as cm:
                agent = extractor_agent.DimensionExtractorAgent(use_mock=False)
            self.assertIsNone(agent.client)
            self.assertTrue(any("regex" in msg.lower() for msg in cm.output))
        finally:
            extractor_agent.LLMClient = original

    def test_mock_extraction_sets_regex_fallback_method(self):
        agent = DimensionExtractorAgent(use_mock=True)
        raw = agent.extract(ProductQuery(part_number="X", brand="Bosch"), "Length: 100 mm, Weight: 50 g")
        self.assertEqual(raw.extraction_method, "regex_fallback")


class TestMockDatasheetProviderMatching(unittest.TestCase):
    """Regresja: dopasowanie 'substring' (zamiast dokładnego) pozwalało
    krótkiemu/zdegenerowanemu numerowi części (np. '0', powstałemu np. przez
    utratę wiodących zer w innym miejscu pipeline'u) trafić w PIERWSZY klucz
    zawierający ten znak - czyli praktycznie dowolną część w bazie."""

    def test_does_not_fuzzy_match_degenerate_part_number(self):
        provider = MockDatasheetProvider()
        result = provider.search_and_fetch_text(ProductQuery(part_number="0", brand="Generic"))
        self.assertNotIn("Wheel Speed Sensor", result.get("title", ""))
        self.assertNotIn("Injector", result.get("title", ""))

    def test_exact_match_still_works(self):
        provider = MockDatasheetProvider()
        result = provider.search_and_fetch_text(ProductQuery(part_number="0265005303", brand="Bosch"))
        self.assertIn("Wheel Speed Sensor", result["title"])


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
        self.assertEqual(res.source_type, "mock")

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


class TestDocupediaIntegration(unittest.TestCase):

    def test_docupedia_client_initialization(self):
        from docupedia_provider import DocupediaClient
        client = DocupediaClient()
        if not client.is_configured:
            self.skipTest("Docupedia credentials not available in this environment (expected outside the Bosch network).")
        self.assertTrue(client.is_configured)

    def test_docupedia_search_and_fetch(self):
        from docupedia_provider import DocupediaClient
        client = DocupediaClient()
        if not client.is_configured:
            self.skipTest("Docupedia credentials not available in this environment (expected outside the Bosch network).")
        pages = client.search_internal_knowledge_base("sensor", limit=2)
        self.assertIsInstance(pages, list)
        if pages:
            self.assertIn("id", pages[0])
            self.assertIn("title", pages[0])


if __name__ == "__main__":
    unittest.main()
