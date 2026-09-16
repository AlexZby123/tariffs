# -*- coding: utf-8 -*-
"""
Agent LLM do precyzyjnej ekstrakcji wymiarów i wagi z kart katalogowych i specyfikacji.
"""
from __future__ import annotations

import base64
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List

from models import ProductQuery, RawDimensions

logger = logging.getLogger(__name__)

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
Your task is to identify and provide the physical dimensions (Length, Width, Height, Diameter) and Weight/Mass of the target part.

RULES:
1. You may receive textual datasheets AND/OR images of schematics, technical drawings, or catalogs. Extract the EXACT raw values and units (e.g. '120 mm', '4.5 in', '15.2 cm', '350 g', '0.18 kg').
2. Pay close attention to engineering drawings: look for dimension lines, labels, arrows, and tables.
3. SOURCE PRIORITY: the provided text may be labeled by its origin (e.g. "BOSCH DOCUPEDIA INTERNAL DOCUMENTATION", "AFTERMARKET CATALOG SPECS", "TECHNICAL SEARCH RESULTS" / web snippets, or a page explicitly flagged as not clearly mentioning the target part number). When sources disagree, trust internal Bosch/Docupedia documentation and official OEM/manufacturer sources over generic aftermarket catalogs or web search snippets, and treat flagged/unverified pages with extra caution. If sources meaningfully disagree, prefer the more authoritative one and LOWER your confidence score rather than averaging the values or picking one silently.
4. If the provided data (text or images) is completely generic and does NOT contain dimensions/weight, you MAY use your automotive technical domain knowledge regarding this exact part number (and its product title) to provide a plausible estimate - but you MUST mark it as such (see rule 7). Never present a guess as a verified reading.
5. For cylindrical or ring-shaped parts (e.g. steering torque sensor, o-ring, shaft), provide raw_diameter (outer diameter) and raw_height (thickness) or raw_length.
6. In 'source_snippet', state either the direct quote from the text, OR a description of where you found it in the drawing (e.g. 'Found in table on schematic page 2'), OR if using domain knowledge: 'Estimated from domain knowledge'.
7. Set "extraction_method" to exactly one of:
   - "extracted": the values came from real text or image content you were given.
   - "estimated": you had no real data and used general domain knowledge (rule 4). NEVER label a guess as "extracted" - this field is what lets downstream systems (e.g. customs/tariff classification) tell verified data from guesses apart.
8. Set confidence:
   - 0.90 to 1.0: exact, verified specification from a datasheet/schematic/internal documentation.
   - 0.70 to 0.89: reliable specification based on part family/standard form factor, or from a source that isn't fully authoritative.
   - < 0.60: approximate/deduced from domain knowledge, or from a low-trust/conflicting source.
   - 0.0: unknown part and no data found.

Return ONLY a JSON object:
{
  "found": true,
  "raw_length": "string with unit or null",
  "raw_width": "string with unit or null",
  "raw_height": "string with unit or null",
  "raw_diameter": "string with unit or null",
  "raw_weight": "string with unit or null",
  "confidence": 0.0-1.0,
  "extraction_method": "extracted" or "estimated",
  "source_snippet": "quote or technical reference note"
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
                except Exception as e:
                    logger.warning("Failed to initialize LLMClient from %s (%s) - falling back to the regex extractor.", cfg_path, e)
                    self.use_mock = True
            else:
                logger.warning("agentic/config.yaml not found at %s - falling back to the regex extractor.", cfg_path)
                self.use_mock = True

        if not self.use_mock and self.client is None:
            # LLMClient module itself could not be imported (llm_client not on sys.path /
            # agentic/ directory not where expected). This used to fail SILENTLY - a
            # non-mock ("--live") run would quietly extract with the crude regex parser
            # instead of the LLM, with no indication anything was wrong.
            logger.warning(
                "Non-mock extraction was requested but no LLM client is available "
                "(llm_client module could not be imported). Every extract() call will "
                "silently use the basic regex fallback instead of the LLM. Check that "
                "agentic/llm_client.py and agentic/config.yaml are reachable relative to "
                "this module's location, or pass an explicit llm_client= instance."
            )

    def extract(self, query: ProductQuery, source_text: str, source_url: str = "", attachments: Optional[List[str]] = None) -> RawDimensions:
        """Główna metoda ekstrakcji (wspiera analizę wizyjną z załączników PDF/obrazów)."""
        attachments = attachments or []

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
                extraction_method="regex_fallback",
            )

        # 1. PRZYGOTOWANIE TEKSTU I INTELIGENTNE WYSZUKIWANIE W PDF
        pdf_docs = []
        pdf_filtered_text = ""
        
        if attachments:
            try:
                import pymupdf
            except ImportError:
                pymupdf = None

            for att_path in attachments:
                path = Path(att_path)
                if not path.exists():
                    continue

                if pymupdf and path.suffix.lower() == ".pdf":
                    try:
                        doc = pymupdf.open(str(path))
                        pdf_docs.append((path, doc))
                        
                        keywords = ["mm", "cm", "kg", " g", "lbs", "weight", "mass", "length", "width", "height", "dimension", "size", "wymiar", "waga", "masa", "diameter", "grubo", "srednica"]
                        pn = query.part_number.lower() if query.part_number else ""
                        
                        extracted = []
                        for page in doc:
                            text = page.get_text()
                            for line in text.splitlines():
                                lower = line.lower()
                                if any(k in lower for k in keywords) or (pn and pn in lower):
                                    if len(line.strip()) > 3:
                                        extracted.append(line.strip())
                        
                        if extracted:
                            pdf_filtered_text += f"\n\n--- EXTRACTED TEXT FROM PDF: {path.name} ---\n"
                            seen = set()
                            for line in extracted:
                                if line not in seen:
                                    pdf_filtered_text += line + "\n"
                                    seen.add(line)
                    except Exception as e:
                        print(f"Error reading PDF text {path.name}: {e}")

        full_source_text = source_text
        if pdf_filtered_text:
            full_source_text += pdf_filtered_text

        user_text_prompt = (
            f"TARGET PART IDENTIFIERS:\n"
            f"- Brand: {query.brand}\n"
            f"- Part Number: {query.part_number}\n"
            f"- Title / Description: {query.title or 'N/A'}\n"
            f"- Category: {query.category or 'N/A'}\n\n"
            f"DATASHEET / SPECIFICATION TEXT:\n"
            f"\"\"\"\n{full_source_text}\n\"\"\"\n"
        )

        try:
            # ETAP 1: Wywołanie tekstowe (Tanie, szybkie)
            resp_obj = self.client.chat_json(SYSTEM_PROMPT, user_text_prompt)
            extraction_method = resp_obj.get("extraction_method")
            if extraction_method not in ("extracted", "estimated"):
                extraction_method = "estimated" if float(resp_obj.get("confidence", 0.0)) < 0.6 else "extracted"
            conf = float(resp_obj.get("confidence", 0.0))

            # Jeśli sukces (znaleziono w tekście), ZWRÓĆ WYNIK
            if extraction_method == "extracted" and conf >= 0.7:
                return RawDimensions(
                    raw_length=resp_obj.get("raw_length"),
                    raw_width=resp_obj.get("raw_width"),
                    raw_height=resp_obj.get("raw_height"),
                    raw_diameter=resp_obj.get("raw_diameter"),
                    raw_weight=resp_obj.get("raw_weight"),
                    confidence=conf,
                    source_snippet=resp_obj.get("source_snippet"),
                    source_url=source_url,
                    extraction_method=extraction_method,
                )
            
            # Jeśli brak sukcesu, ale nie ma załączników, zwróć co mamy
            if not attachments:
                return RawDimensions(
                    raw_length=resp_obj.get("raw_length"),
                    raw_width=resp_obj.get("raw_width"),
                    raw_height=resp_obj.get("raw_height"),
                    raw_diameter=resp_obj.get("raw_diameter"),
                    raw_weight=resp_obj.get("raw_weight"),
                    confidence=conf,
                    source_snippet=resp_obj.get("source_snippet"),
                    source_url=source_url,
                    extraction_method=extraction_method,
                )

            # ETAP 2: Fallback na VISION (droższe wywołanie obrazkowe, jeśli tekst zawiódł)
            content_blocks = [{"type": "text", "text": user_text_prompt}]
            has_images = False

            for path, doc in pdf_docs:
                for i, page in enumerate(doc):
                    if i >= 10:
                        break
                    try:
                        pix = page.get_pixmap(dpi=100)
                        img_data = pix.tobytes("jpeg")
                        b64_img = base64.b64encode(img_data).decode('utf-8')
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
                        has_images = True
                    except Exception as e:
                        print(f"Error rendering PDF {path.name}: {e}")

            for att_path in attachments:
                path = Path(att_path)
                if path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    try:
                        b64_img = base64.b64encode(path.read_bytes()).decode('utf-8')
                        content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                        })
                        has_images = True
                    except Exception as e:
                        print(f"Error reading image {path.name}: {e}")

            if not has_images:
                # Zwróć wynik tekstowy jeśli ostatecznie brak obrazków
                return RawDimensions(
                    raw_length=resp_obj.get("raw_length"),
                    raw_width=resp_obj.get("raw_width"),
                    raw_height=resp_obj.get("raw_height"),
                    raw_diameter=resp_obj.get("raw_diameter"),
                    raw_weight=resp_obj.get("raw_weight"),
                    confidence=conf,
                    source_snippet=resp_obj.get("source_snippet"),
                    source_url=source_url,
                    extraction_method=extraction_method,
                )

            # Ostatnie wywołanie LLM z obrazkami
            resp_obj_vision = self.client.chat_json(SYSTEM_PROMPT, content_blocks)
            extraction_method_v = resp_obj_vision.get("extraction_method")
            if extraction_method_v not in ("extracted", "estimated"):
                extraction_method_v = "estimated" if float(resp_obj_vision.get("confidence", 0.0)) < 0.6 else "extracted"

            return RawDimensions(
                raw_length=resp_obj_vision.get("raw_length"),
                raw_width=resp_obj_vision.get("raw_width"),
                raw_height=resp_obj_vision.get("raw_height"),
                raw_diameter=resp_obj_vision.get("raw_diameter"),
                raw_weight=resp_obj_vision.get("raw_weight"),
                confidence=float(resp_obj_vision.get("confidence", 0.0)),
                source_snippet=resp_obj_vision.get("source_snippet"),
                source_url=source_url,
                extraction_method=extraction_method_v,
            )
        except Exception as e:
            logger.warning("LLM extraction call failed (%s: %s) - falling back to the regex extractor for this query.", type(e).__name__, str(e)[:200])
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
                extraction_method="regex_fallback",
            )
