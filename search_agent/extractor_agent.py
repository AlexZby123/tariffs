# -*- coding: utf-8 -*-
"""
Agent LLM do precyzyjnej ekstrakcji wymiarów i wagi z kart katalogowych i specyfikacji.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any

from models import ProductQuery, RawDimensions

# Umożliwienie importu LLMClient z katalogu agentic/
_AGENTIC_DIR = Path(__file__).parent.parent / "agentic"
if str(_AGENTIC_DIR) not in sys.path:
    sys.path.append(str(_AGENTIC_DIR))

try:
    from llm_client import LLMClient, wczytaj_config
except ImportError:
    LLMClient = None
    wczytaj_config = None


SYSTEM_PROMPT = """You are an expert Technical Specification Extraction Agent for Bosch automotive and industrial components.
Your task is to extract physical dimensions (Length, Width, Height, Diameter) and Weight/Mass from the provided technical datasheet or search text.

CRITICAL INSTRUCTIONS:
1. Extract the EXACT raw value and unit as stated in the text (e.g. '120 mm', '4.5 in', '15.2 cm', '350 g', '2 lbs 4 oz', '0.078 kg').
2. If overall dimensions are given as a compound string like '182 x 35 x 30 mm', break it down:
   raw_length: '182 mm', raw_width: '35 mm', raw_height: '30 mm'.
3. If the part is cylindrical (e.g. pin, rod, needle, bolt, shaft, o-ring cord, circular sensor body) and only diameter and length are given, set raw_diameter and raw_length.
4. Provide a direct short quote (source_snippet) confirming where the numbers came from.
5. Provide confidence:
   - 0.9 to 1.0: Exact datasheet match for this specific part number.
   - 0.6 to 0.8: Belongs to the same part family or clear spec.
   - < 0.5: Uncertain or deduced.
   - 0.0: No dimensions or weight found in the text.
6. If any attribute is not mentioned in the text, return null for that field.

Return ONLY a JSON object matching this schema:
{
  "found": true/false,
  "raw_length": "string with unit or null",
  "raw_width": "string with unit or null",
  "raw_height": "string with unit or null",
  "raw_diameter": "string with unit or null",
  "raw_weight": "string with unit or null",
  "confidence": 0.0-1.0,
  "source_snippet": "short quote from text"
}
"""


def _mock_extract_dimensions(text: str) -> Dict[str, Any]:
    """Heurystyczny parser regexowy (do testów offline bez aktywnego tokenu LLM)."""
    res = {
        "found": False,
        "raw_length": None,
        "raw_width": None,
        "raw_height": None,
        "raw_diameter": None,
        "raw_weight": None,
        "confidence": 0.0,
        "source_snippet": None,
    }

    # 1. Szukaj potrójnego wymiaru: np. 182 x 35 x 30 mm
    match_triple = re.search(
        r"(\d+(?:\.\d+)?)\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*([a-zA-Z\"'µ]+)",
        text
    )
    if match_triple:
        unit = match_triple.group(4)
        res["raw_length"] = f"{match_triple.group(1)} {unit}"
        res["raw_width"] = f"{match_triple.group(2)} {unit}"
        res["raw_height"] = f"{match_triple.group(3)} {unit}"
        res["found"] = True
        res["confidence"] = 0.9
        res["source_snippet"] = match_triple.group(0)

    # 2. Szukaj długości
    if not res["raw_length"]:
        m_l = re.search(r"(?:length|długość|dlugosc)[^\d\n]*?(\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inches?|\"|ft))\b", text, re.I)
        if m_l:
            res["raw_length"] = m_l.group(1)
            res["found"] = True
            res["confidence"] = max(res["confidence"], 0.85)

    # 3. Szukaj szerokości
    if not res["raw_width"]:
        m_w = re.search(r"(?:width|szerokość|szerokosc)[^\d\n]*?(\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inches?|\"|ft))\b", text, re.I)
        if m_w:
            res["raw_width"] = m_w.group(1)
            res["found"] = True

    # 4. Szukaj wysokości
    if not res["raw_height"]:
        m_h = re.search(r"(?:height|wysokość|wysokosc)[^\d\n]*?(\d+(?:\.\d+)?\s*(?:mm|cm|m|in|inches?|\"|ft))\b", text, re.I)
        if m_h:
            res["raw_height"] = m_h.group(1)
            res["found"] = True

    # 5. Szukaj średnicy
    m_d = re.search(r"(?:diameter|średnica|srednica|Ø|OD|ID)[^\d\n]*?(\d+(?:\.\d+)?\s*(?:mm|cm|in|\"))\b", text, re.I)
    if m_d:
        res["raw_diameter"] = m_d.group(1)
        res["found"] = True

    # 6. Szukaj wagi / masy
    # Najpierw sprawdź złożoną wagę (np. 2 lbs 3 oz)
    m_wg = re.search(r"(\d+(?:\.\d+)?\s*lbs?\s*\d+(?:\.\d+)?\s*oz)", text, re.I)
    if not m_wg:
        m_wg = re.search(r"(?:weight|waga|masa|mass)[^\d\n]*?(\d+(?:\.\d+)?\s*(?:lbs?|pounds?|oz|ounces?|kg|kilograms?|g|grams?|mg))\b", text, re.I)
    if m_wg:
        res["raw_weight"] = m_wg.group(1)
        res["found"] = True
        res["confidence"] = max(res["confidence"], 0.85)

    return res


class DimensionExtractorAgent:
    """Agent wyciągający wymiary za pomocą LLM (lub trybu offline)."""

    def __init__(self, llm_client: Optional[Any] = None, use_mock: bool = False):
        self.use_mock = use_mock
        self.client = llm_client

        if not self.use_mock and self.client is None and LLMClient is not None:
            # Próba wczytania konfiguracji z agentic/config.yaml
            cfg_path = _AGENTIC_DIR / "config.yaml"
            if cfg_path.exists():
                try:
                    cfg = wczytaj_config(cfg_path)
                    self.client = LLMClient(cfg)
                except Exception:
                    self.use_mock = True
            else:
                self.use_mock = True

    def extract(self, query: ProductQuery, source_text: str, source_url: str = "") -> RawDimensions:
        """Główna metoda ekstrakcji."""
        if self.use_mock or self.client is None:
            raw_dict = _mock_extract_dimensions(source_text)
            return RawDimensions(
                raw_length=raw_dict.get("raw_length"),
                raw_width=raw_dict.get("raw_width"),
                raw_height=raw_dict.get("raw_height"),
                raw_diameter=raw_dict.get("raw_diameter"),
                raw_weight=raw_dict.get("raw_weight"),
                confidence=raw_dict.get("confidence", 0.0),
                source_snippet=raw_dict.get("source_snippet") or source_text[:200],
                source_url=source_url,
            )

        # Uruchomienie modelu LLM
        user_prompt = (
            f"TARGET PART IDENTIFIERS:\n"
            f"- Brand: {query.brand}\n"
            f"- Part Number: {query.part_number}\n"
            f"- Title / Description: {query.title or 'N/A'}\n"
            f"- Category: {query.category or 'N/A'}\n\n"
            f"DATASHEET / SPECIFICATION TEXT:\n"
            f"\"\"\"\n{source_text}\n\"\"\"\n"
        )

        try:
            resp_obj = self.client.chat_json(SYSTEM_PROMPT, user_prompt)
            return RawDimensions(
                raw_length=resp_obj.get("raw_length"),
                raw_width=resp_obj.get("raw_width"),
                raw_height=resp_obj.get("raw_height"),
                raw_diameter=resp_obj.get("raw_diameter"),
                raw_weight=resp_obj.get("raw_weight"),
                confidence=float(resp_obj.get("confidence", 0.0)),
                source_snippet=resp_obj.get("source_snippet"),
                source_url=source_url,
            )
        except Exception as e:
            # Fallback na mock extractor w razie awarii API
            fallback = _mock_extract_dimensions(source_text)
            return RawDimensions(
                raw_length=fallback.get("raw_length"),
                raw_width=fallback.get("raw_width"),
                raw_height=fallback.get("raw_height"),
                raw_diameter=fallback.get("raw_diameter"),
                raw_weight=fallback.get("raw_weight"),
                confidence=fallback.get("confidence", 0.0),
                source_snippet=f"(Fallback parser - error: {str(e)[:60]})",
                source_url=source_url,
            )
