# -*- coding: utf-8 -*-
"""
Dostawcy wyszukiwania i pobierania kart katalogowych (datasheet / technical specs).

Obsługuje:
1. WebDatasheetProvider - odpytuje publiczne wyszukiwarki i pobiera fragmenty stron / kart produktów.
2. MockDatasheetProvider - realistyczne specyfikacje części Bosch do testów offline i szybkiego prototypowania.
3. CachedSearchProvider - warstwa buforująca wyniki zapytań na dysku.
"""
from __future__ import annotations

import json
import re
import urllib.parse
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

from models import ProductQuery

CACHE_DIR = Path(__file__).parent / "cache"


class BaseDatasheetProvider(ABC):
    """Interfejs dostawcy danych technicznych."""

    @abstractmethod
    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        """Zwraca słownik: {'text': str, 'source_url': str, 'title': str}."""
        pass


# ============================================================
#  1. Mock Provider (Offline / Szybkie testy / Benchmark)
# ============================================================

_MOCK_SPECS_DATABASE = {
    "0265005303": {
        "title": "Bosch Wheel Speed Sensor 0 265 005 303 (Active ABS Sensor)",
        "source_url": "https://www.bosch-automotive-catalog.com/en/product-detail/-/product/0265005303",
        "text": (
            "Technical Data Sheet - Bosch Wheel Speed Sensor\n"
            "Product Number: 0 265 005 303\n"
            "Part Type: Active Hall Sensor\n"
            "Design: Flange mount with cable\n"
            "Length of sensor head: 65 mm\n"
            "Diameter of sensor pole: 18.0 mm\n"
            "Cable length: 850 mm\n"
            "Overall housing dimensions: 65 x 24 x 18 mm\n"
            "Net Weight: 78 g (0.078 kg)\n"
            "Gross Weight: 95 g\n"
            "Operating Temperature: -40 deg C to +150 deg C\n"
        )
    },
    "0445110189": {
        "title": "Bosch Common Rail Diesel Injector CRI 0 445 110 189",
        "source_url": "https://www.bosch-mobility.com/en/products/powertrain/diesel/injector-cri2/",
        "text": (
            "Bosch Diesel Injector Technical Specification Sheet\n"
            "PN: 0 445 110 189\n"
            "Type: Solenoid Common Rail Injector (CRI2)\n"
            "Total Length: 182 mm\n"
            "Nozzle diameter: 7.0 mm\n"
            "Body diameter: 17.0 mm\n"
            "Overall dimensions: 182 x 35 x 30 mm\n"
            "Weight: 490 grams\n"
            "System pressure: up to 1600 bar\n"
        )
    },
    "F00N200098": {
        "title": "Bosch Gasket / O-Ring F 00N 200 098",
        "source_url": "https://www.bosch-repair-service.com/o-ring-f00n200098",
        "text": (
            "Sealing O-Ring Technical Data\n"
            "Part Number: F 00N 200 098 (F00N200098)\n"
            "Material: FKM (Fluorocarbon rubber, 80 Shore A)\n"
            "Inner Diameter (ID): 14.5 mm\n"
            "Outer Diameter (OD): 19.5 mm\n"
            "Cross Section / Cord Thickness: 2.5 mm\n"
            "Dimensions: 19.5 x 19.5 x 2.5 mm\n"
            "Weight: 1.2 g\n"
        )
    },
    "0000000000": {
        "title": "Generic Automotive Bracket Assembly",
        "source_url": "https://spec-sheet.internal/part/bracket-generic",
        "text": (
            "Heavy Duty Mounting Bracket - Steel Galvanized\n"
            "Length: 8.5 in\n"
            "Width: 4.25 in\n"
            "Height: 1.75 in\n"
            "Thickness: 3.0 mm\n"
            "Weight: 2 lbs 3 oz\n"
        )
    }
}


class MockDatasheetProvider(BaseDatasheetProvider):
    """Zwraca realistyczne karty katalogowe dla celów testowych."""

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        pn_clean = re.sub(r"[\s\-_]", "", query.part_number).upper()
        # Sprawdź czy mamy w bazie dokładny lub częściowy numer
        for key, spec in _MOCK_SPECS_DATABASE.items():
            if key in pn_clean or pn_clean in key:
                return spec

        # Domyślny fallback heurystyczny jeśli numer jest inny
        cat = (query.category or query.title or "PART").upper()
        return {
            "title": f"Datasheet for {query.brand} {query.part_number}",
            "source_url": f"https://mock-catalog.example.com/item/{query.part_number}",
            "text": (
                f"Component Specification: {query.brand} {query.part_number}\n"
                f"Designation: {cat}\n"
                "Standard dimensions:\n"
                "Length: 110 mm\n"
                "Width: 45 mm\n"
                "Height: 35 mm\n"
                "Weight: 215 g\n"
            )
        }


# ============================================================
#  2. Web Datasheet Provider (Live Online Search)
# ============================================================

class WebDatasheetProvider(BaseDatasheetProvider):
    """Wyszukuje karty produktów w internecie za pomocą requests i DuckDuckGo HTML."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        import requests

        pn = str(query.part_number).strip()
        pn_clean = re.sub(r"[\s\-_]", "", pn)

        # Warianty frazy wyszukiwania
        # Dla części Bosch np. 0265005166 -> '0 265 005 166'
        queries_to_try = [
            f'{query.brand} "{pn}" dimensions weight',
            f'{query.brand} {pn} datasheet',
        ]
        if len(pn_clean) == 10:
            spaced = f"{pn_clean[0]} {pn_clean[1:4]} {pn_clean[4:7]} {pn_clean[7:]}"
            queries_to_try.insert(0, f'{query.brand} "{spaced}" dimensions weight')
            queries_to_try.append(f'{query.brand} "{spaced}"')

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9,pl;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        all_snippets = []
        found_url = ""

        for q_str in queries_to_try:
            try:
                resp = requests.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": q_str},
                    headers=headers,
                    timeout=self.timeout
                )
                if resp.status_code == 200 and "result-snippet" in resp.text:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(resp.text, "html.parser")
                    for td in soup.find_all("td", class_="result-snippet"):
                        text = td.get_text(strip=True)
                        if text and text not in all_snippets:
                            all_snippets.append(text)
                    for a in soup.find_all("a", class_="result-link"):
                        href = a.get("href")
                        if href and not found_url and not href.startswith("/"):
                            found_url = href
                if len(all_snippets) >= 4:
                    break
            except Exception:
                continue

        if not all_snippets:
            return {
                "title": f"Search results for {query.brand} {pn}",
                "source_url": found_url or "https://duckduckgo.com",
                "text": f"No online specifications found for {query.brand} {pn}.",
                "has_data": False,
            }

        combined_text = "\n---\n".join(all_snippets[:8])
        return {
            "title": f"Web specs for {query.brand} {pn}",
            "source_url": found_url or "https://duckduckgo.com",
            "text": f"TECHNICAL SEARCH RESULTS FOR {query.brand} {pn} ({query.title or ''}):\n\n{combined_text}",
            "has_data": True,
        }


# ============================================================
#  3. Cached Search Provider
# ============================================================

class CachedSearchProvider(BaseDatasheetProvider):
    """Buforuje pobrane karty katalogowe w plikach JSON, by unikać powtarzanych zapytań sieciowych."""

    def __init__(self, inner: BaseDatasheetProvider, cache_dir: Optional[Path] = None):
        self.inner = inner
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        pn_clean = re.sub(r"[^\w\-_]", "", query.part_number)
        cache_file = self.cache_dir / f"datasheet_{pn_clean}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Użyj cache TYLKO jeśli zawiera realne dane (nie błędy)
                    if data.get("has_data") or ("No online specifications found" not in data.get("text", "") and "No detailed snippets" not in data.get("text", "")):
                        return data
            except Exception:
                pass

        result = self.inner.search_and_fetch_text(query)
        # Zapisuj do cache TYLKO jeśli znaleziono realne dane
        if result.get("has_data") or ("No online specifications found" not in result.get("text", "") and "No detailed snippets" not in result.get("text", "")):
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        return result
