# -*- coding: utf-8 -*-
"""
Dostawcy wyszukiwania i pobierania kart katalogowych (datasheet / technical specs).

Obsługuje:
1. WebDatasheetProvider     - odpytuje wyszukiwarki (Serper -> DuckDuckGo) i pobiera
                              strony/PDF-y kart produktów, z generyczną ekstrakcją specyfikacji.
2. CatalogDatasheetProvider - przeszukuje PRAWDZIWE katalogi części przez wyszukiwarkę
                              z filtrem site: (bez hardcodowanych schematów URL martwych domen).
3. MockDatasheetProvider    - realistyczne specyfikacje części Bosch do testów offline.
4. CachedSearchProvider     - warstwa buforująca wyniki na dysku (TTL + opcjonalny cache negatywny).
5. HybridDatasheetProvider  - fallback Docupedia -> katalogi -> web, z dociąganiem
                              brakujących danych (waga vs wymiary) z kolejnych źródeł.

Zmiany vs poprzednia wersja:
- USUNIĘTO scraping findpart.org (domena przejęta przez spam - nie jest już katalogiem części).
- Wspólny SearchBackend (Serper z fallbackiem do DDG Lite) dla Catalog i Web.
- Generyczna, wielojęzyczna (EN/DE/PL) ekstrakcja specyfikacji z tabel, <dl> i linii "Klucz: Wartość".
- Ekstrakcja tekstu z PDF-ów (karty katalogowe to często PDF).
- Sesja requests z retry, blokada domen śmieciowych, limity rozmiaru stron.
- Logowanie zamiast `except: pass` - błędy sieci/parsowania są widoczne w logach.
- Hybrid: jeśli pierwsze źródło ma np. tylko wagę bez wymiarów, dociąga kolejne i scala.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from models import ProductQuery
from search_utils import generate_pn_variants, mentions_part_number, extract_relevant_lines

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
DEFAULT_CACHE_TTL_SECONDS = 14 * 24 * 60 * 60  # 14 dni

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8,pl;q=0.7",
}

# Domeny, których nigdy nie pobieramy: martwe/przejęte katalogi i serwisy
# bez wartości technicznej. findpart.org zostało przejęte przez stronę
# spamową (kasyno) - zwraca 200 OK, ale treść nie ma nic wspólnego z częściami.
BLOCKED_DOMAINS = (
    "findpart.org",
    "pinterest.",
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "youtube.com",
    "x.com/",
    "twitter.com",
)

# Realne katalogi aftermarket/OEM z danymi fizycznymi (waga/wymiary) per numer
# części. Lista jest konfigurowalna w konstruktorze CatalogDatasheetProvider.
DEFAULT_CATALOG_DOMAINS = (
    "boschaftermarket.com",
    "autodoc.co.uk",
    "autodoc.de",
    "onlinecarparts.co.uk",
    "partsouq.com",
    "amayama.com",
)

# Wielojęzyczne klucze specyfikacji fizycznych (EN / DE / PL). Dopasowanie
# dotyczy ETYKIET w HTML-u zwróconym przez stronę (nie języka zapytania) -
# europejskie katalogi serwują tabelki w różnych językach zależnie od locale.
SPEC_KEY_TERMS = (
    # EN
    "weight", "net weight", "gross weight", "length", "width", "height",
    "depth", "diameter", "dimensions", "thickness",
    # DE
    "gewicht", "nettogewicht", "bruttogewicht", "länge", "breite", "höhe",
    "tiefe", "durchmesser", "abmessungen", "maße", "stärke",
    # PL
    "waga", "masa", "długość", "szerokość", "wysokość", "głębokość",
    "średnica", "wymiary", "grubość",
)

_WEIGHT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:kg|kilograms?|g|grams?|lbs?|pounds?|oz|ounces?)\b", re.I
)
_DIM_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mm|cm|in|inch(?:es)?)\b"
    r"|\b\d+(?:[.,]\d+)?\s*[x×*]\s*\d+(?:[.,]\d+)?",
    re.I,
)

_UNRELATED_SECTION_KEYWORDS = (
    "related", "similar", "compare", "customers also", "you might",
    "recommended", "podobne", "polecane", "klienci", "alternative",
    "ähnliche", "empfohlen", "cross reference", "zamienniki",
)


# ============================================================
#  Helpery współdzielone
# ============================================================

def _is_blocked(url: str) -> bool:
    u = (url or "").lower()
    return any(b in u for b in BLOCKED_DOMAINS)


def _has_weight(text: str) -> bool:
    return bool(_WEIGHT_RE.search(text or ""))


def _has_dimensions(text: str) -> bool:
    return bool(_DIM_RE.search(text or ""))


def _is_complete(text: str) -> bool:
    """Czy tekst zawiera zarówno wagę, jak i wymiary liniowe (heurystyka)."""
    return _has_weight(text) and _has_dimensions(text)


def _make_session(total_retries: int = 2, backoff: float = 0.5):
    """Sesja requests z retry na błędy przejściowe (429/5xx) i wspólnymi nagłówkami."""
    import requests
    from requests.adapters import HTTPAdapter

    try:
        from urllib3.util.retry import Retry
        retry = Retry(
            total=total_retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        adapter = HTTPAdapter(max_retries=retry)
    except Exception:  # starsze urllib3
        adapter = HTTPAdapter()

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(DEFAULT_HEADERS)
    return session


def _looks_like_unrelated_section(tag) -> bool:
    """
    Heurystyka: sprawdza nagłówki poprzedzające dany tag (np. tabelę), żeby
    odróżnić prawdziwą sekcję specyfikacji od sekcji typu "podobne produkty"
    / "klienci oglądali także". Takie sekcje mogą zawierać wiersze pasujące
    do tych samych słów kluczowych (np. "height"), ale dotyczące zupełnie
    INNEGO produktu.
    """
    if tag is None:
        return False
    try:
        for sib in tag.find_all_previous(["h1", "h2", "h3", "h4"], limit=2):
            text = sib.get_text(strip=True).lower()
            if any(k in text for k in _UNRELATED_SECTION_KEYWORDS):
                return True
    except Exception:
        pass
    return False


_KEY_VALUE_LINE_RE = re.compile(r"^([^:]{2,40}):\s*(.+)$")


def extract_specs_from_soup(soup) -> Dict[str, str]:
    """
    Generyczna ekstrakcja specyfikacji fizycznych z dowolnej strony HTML.
    Nie zakłada żadnych konkretnych klas CSS (te i tak różnią się między
    katalogami i zmieniają w czasie). Zbiera pary klucz->wartość z:
      1) tabel dwukolumnowych,
      2) list definicji <dl><dt><dd>,
      3) krótkich linii tekstu w formacie "Klucz: Wartość" (<li>, <p>).
    Wartość musi zawierać cyfrę (odsiewa np. "Size: L").
    """
    specs: Dict[str, str] = {}

    def _maybe_add(key: str, val: str) -> None:
        key = key.strip().strip(":").lower()
        val = " ".join((val or "").split())
        if not key or not val or len(val) > 120:
            return
        if not any(ch.isdigit() for ch in val):
            return
        if any(term in key for term in SPEC_KEY_TERMS):
            specs.setdefault(key, val)

    for table in soup.find_all("table"):
        if _looks_like_unrelated_section(table):
            continue
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) == 2:
                _maybe_add(
                    cells[0].get_text(" ", strip=True),
                    cells[1].get_text(" ", strip=True),
                )

    for dl in soup.find_all("dl"):
        if _looks_like_unrelated_section(dl):
            continue
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
            _maybe_add(dt.get_text(" ", strip=True), dd.get_text(" ", strip=True))

    for el in soup.find_all(["li", "p"]):
        text = el.get_text(" ", strip=True)
        if not text or len(text) > 160:
            continue
        m = _KEY_VALUE_LINE_RE.match(text)
        if m:
            _maybe_add(m.group(1), m.group(2))

    return specs


def _format_specs(specs: Dict[str, str]) -> str:
    return "\n".join(f"- {k.capitalize()}: {v}" for k, v in specs.items())


def _extract_pdf_text(content: bytes, max_pages: int = 6, max_chars: int = 20000) -> str:
    """Karty katalogowe to często PDF-y - wyciągamy tekst z pierwszych stron."""
    try:
        import io
        try:
            from pypdf import PdfReader
        except ImportError:  # starsze środowiska
            from PyPDF2 import PdfReader  # type: ignore

        reader = PdfReader(io.BytesIO(content))
        parts: List[str] = []
        for page in reader.pages[:max_pages]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n".join(parts)[:max_chars]
    except Exception as exc:
        logger.debug("PDF text extraction failed: %s", exc)
        return ""


# ============================================================
#  Wspólny backend wyszukiwania (Serper -> DuckDuckGo Lite)
# ============================================================

class SearchBackend:
    """
    Jedno miejsce odpowiedzialne za "zapytaj wyszukiwarkę, oddaj wyniki".
    Najpierw Serper (Google) jeśli jest token, w przeciwnym razie / przy
    awarii - DuckDuckGo Lite. Zwraca listę {'title','link','snippet'}.
    """

    def __init__(self, timeout: int = 15, session=None):
        self.timeout = timeout
        self._session = session

    @property
    def session(self):
        if self._session is None:
            self._session = _make_session()
        return self._session

    @staticmethod
    def _get_serper_token() -> Optional[str]:
        # Najpierw zmienna środowiskowa, dopiero potem agentic/config.yaml -
        # dzięki temu token można wstrzyknąć w kontenerze/CI bez commitowania
        # pliku konfiguracyjnego.
        token = os.environ.get("SERPER_DEV_TOKEN") or os.environ.get("SERPER_API_KEY")
        if token:
            return token
        try:
            import yaml
            cfg_path = Path(__file__).parent.parent / "agentic" / "config.yaml"
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            # 'sarper_dev_token' - literówka historycznie obecna w niektórych
            # configach; czytamy oba klucze dla kompatybilności wstecznej.
            return cfg.get("serper_dev_token") or cfg.get("sarper_dev_token")
        except Exception:
            return None

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        token = self._get_serper_token()
        if token:
            results = self._serper(query, token, max_results)
            if results is not None:
                return results
            logger.warning("Serper search failed for %r - falling back to DuckDuckGo", query)
        return self._ddg(query, max_results)

    def _serper(self, query: str, token: str, max_results: int) -> Optional[List[Dict[str, str]]]:
        try:
            resp = self.session.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": token, "Content-Type": "application/json"},
                json={"q": query},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning("Serper returned HTTP %s for %r", resp.status_code, query)
                return None
            data = resp.json()
            out: List[Dict[str, str]] = []
            for res in data.get("organic", [])[:max_results]:
                out.append({
                    "title": res.get("title", ""),
                    "link": res.get("link", ""),
                    "snippet": res.get("snippet", ""),
                })
            return out
        except Exception as exc:
            logger.warning("Serper request error for %r: %s", query, exc)
            return None

    def _ddg(self, query: str, max_results: int) -> List[Dict[str, str]]:
        try:
            from bs4 import BeautifulSoup
            from urllib.parse import parse_qs, urlparse, unquote

            resp = self.session.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": query},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                logger.warning("DDG returned HTTP %s for %r", resp.status_code, query)
                return []

            soup = BeautifulSoup(resp.text, "html.parser")

            links: List[str] = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # DDG Lite czasem zwraca linki-przekierowania /l/?uddg=<url>
                if "duckduckgo.com/l/" in href:
                    q = parse_qs(urlparse(href).query).get("uddg")
                    href = unquote(q[0]) if q else ""
                if href.startswith("http") and "duckduckgo" not in href:
                    if href not in links:
                        links.append(href)

            snippets = [
                td.get_text(strip=True)
                for td in soup.find_all("td", class_="result-snippet")
            ]

            out: List[Dict[str, str]] = []
            for i, link in enumerate(links[:max_results]):
                out.append({
                    "title": "",
                    "link": link,
                    "snippet": snippets[i] if i < len(snippets) else "",
                })
            return out
        except Exception as exc:
            logger.warning("DDG request error for %r: %s", query, exc)
            return []


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
        ),
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
        ),
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
        ),
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
        ),
    },
}


class MockDatasheetProvider(BaseDatasheetProvider):
    """
    Zwraca realistyczne karty katalogowe dla celów testowych.

    fallback_enabled=False wyłącza syntetyczny fallback dla nieznanych numerów -
    zalecane wszędzie poza benchmarkami, bo zmyślone wymiary (110x45x35 / 215 g)
    trafiające do klasyfikacji taryfowej to cichy, groźny błąd.
    """

    def __init__(self, fallback_enabled: bool = True):
        self.fallback_enabled = fallback_enabled

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        pn_clean = re.sub(r"[\s\-_]", "", query.part_number).upper()
        # DOKŁADNE dopasowanie (nie luźne "substring") - dopasowanie
        # częściowe wcześniej pozwalało np. numerowi części "0" (skróconemu
        # przez utratę wiodących zer gdzie indziej w pipeline) trafić w
        # PIERWSZY klucz zawierający znak "0", czyli praktycznie dowolny -
        # co po cichu podstawiało dane zupełnie innej części.
        if pn_clean in _MOCK_SPECS_DATABASE:
            spec = _MOCK_SPECS_DATABASE[pn_clean]
            return {**spec, "has_data": True, "source_type": "mock"}

        if not self.fallback_enabled:
            return {
                "title": f"No mock specs for {query.brand} {query.part_number}",
                "source_url": "",
                "text": f"No mock datasheet defined for part number {query.part_number}.",
                "has_data": False,
                "source_type": "mock",
            }

        # Domyślny fallback heurystyczny jeśli numer jest inny - JAWNIE
        # oznaczony jako dane syntetyczne, żeby nigdy nie udawał realnych.
        cat = (query.category or query.title or "PART").upper()
        return {
            "title": f"[MOCK] Datasheet for {query.brand} {query.part_number}",
            "source_url": f"https://mock-catalog.example.com/item/{query.part_number}",
            "text": (
                f"SYNTHETIC MOCK DATA (not a real datasheet)\n"
                f"Component Specification: {query.brand} {query.part_number}\n"
                f"Designation: {cat}\n"
                "Standard dimensions:\n"
                "Length: 110 mm\n"
                "Width: 45 mm\n"
                "Height: 35 mm\n"
                "Weight: 215 g\n"
            ),
            "has_data": True,
            "source_type": "mock",
        }


# ============================================================
#  2. Catalog Provider (realne katalogi OEM / aftermarket)
# ============================================================

class CatalogDatasheetProvider(BaseDatasheetProvider):
    """
    Przeszukuje realne katalogi części przez wyszukiwarkę z filtrem site:
    (Serper/DDG), zamiast zgadywać schematy URL konkretnych domen.

    Dlaczego tak: poprzednia wersja odpytywała hardcodowany URL
    findpart.org/part/<pn> - domena została przejęta przez stronę spamową,
    a selektory CSS (.specs-table) i tak były niezweryfikowanym zgadywaniem.
    Podejście "wyszukaj numer w obrębie znanych katalogów -> pobierz stronę ->
    generyczna ekstrakcja" jest odporne na zmiany struktury HTML i pozwala
    dodawać/usuwać katalogi jednym wpisem w konfiguracji.
    """

    def __init__(
        self,
        timeout: int = 10,
        domains: Tuple[str, ...] = DEFAULT_CATALOG_DOMAINS,
        max_variants: int = 2,
        max_pages_to_fetch: int = 5,
        backend: Optional[SearchBackend] = None,
    ):
        self.timeout = timeout
        self.domains = tuple(domains)
        self.max_variants = max_variants
        self.max_pages_to_fetch = max_pages_to_fetch
        self.backend = backend or SearchBackend(timeout=timeout)

    def _build_queries(self, variants: List[str]) -> List[str]:
        queries: List[str] = []
        # Dzielimy domeny na paczki po 3, żeby zapytania "site:a OR site:b OR
        # site:c" pozostały krótkie i skuteczne (długie łańcuchy OR bywają
        # ignorowane przez wyszukiwarki).
        chunks = [self.domains[i:i + 3] for i in range(0, len(self.domains), 3)]
        for v in variants[: self.max_variants]:
            for chunk in chunks:
                sites = " OR ".join(f"site:{d}" for d in chunk)
                queries.append(f'"{v}" ({sites})')
        return list(dict.fromkeys(queries))

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        from bs4 import BeautifulSoup

        pn = str(query.part_number).strip()
        variants = generate_pn_variants(pn)
        session = self.backend.session

        candidate_links: List[str] = []
        for q_str in self._build_queries(variants):
            for res in self.backend.search(q_str, max_results=6):
                link = res.get("link", "")
                if not link or _is_blocked(link):
                    continue
                if not any(d in link for d in self.domains):
                    continue
                if link not in candidate_links:
                    candidate_links.append(link)
            if len(candidate_links) >= self.max_pages_to_fetch:
                break

        for url in candidate_links[: self.max_pages_to_fetch]:
            try:
                resp = session.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    logger.debug("Catalog page %s returned HTTP %s", url, resp.status_code)
                    continue
                if "text/html" not in resp.headers.get("Content-Type", ""):
                    continue

                soup = BeautifulSoup(resp.text[:800_000], "html.parser")
                page_text = soup.get_text(" ", strip=True)[:50_000]

                # Zabezpieczenie przed stroną "trafioną obok": specyfikacja
                # musi dotyczyć NASZEGO numeru części, nie ogólnej kategorii
                # ani zamiennika.
                if not mentions_part_number(page_text, pn):
                    logger.debug("Catalog page %s does not mention PN %s - skipping", url, pn)
                    continue

                specs = extract_specs_from_soup(soup)
                if not specs:
                    continue

                title_el = soup.find("h1")
                title = title_el.get_text(strip=True) if title_el else f"Catalog Part {pn}"

                return {
                    "title": title,
                    "source_url": url,
                    "text": (
                        f"AFTERMARKET CATALOG SPECS FOR {pn} (source: {url}):\n\n"
                        f"{_format_specs(specs)}"
                    ),
                    "has_data": True,
                    "source_type": "catalog",
                }
            except Exception as exc:
                logger.warning("Catalog fetch failed for %s: %s", url, exc)
                continue

        return {
            "title": f"No catalog specs for {query.brand} {pn}",
            "source_url": "",
            "text": f"No data found in aftermarket catalogs for {pn}.",
            "has_data": False,
            "source_type": "catalog",
        }


# ============================================================
#  3. Web Datasheet Provider (Live Online Search)
# ============================================================

class WebDatasheetProvider(BaseDatasheetProvider):
    """Wyszukuje karty produktów w internecie (Serper -> DuckDuckGo Lite) i pobiera strony/PDF-y."""

    def __init__(
        self,
        timeout: int = 15,
        max_queries: int = 8,
        max_fetch: int = 6,
        backend: Optional[SearchBackend] = None,
    ):
        self.timeout = timeout
        self.max_queries = max_queries
        self.max_fetch = max_fetch
        self.backend = backend or SearchBackend(timeout=timeout)

    def _build_queries(self, query: ProductQuery, pn: str) -> List[str]:
        pn_variants = generate_pn_variants(pn)[:3]
        clean_title = re.sub(r"[^\w\s]", " ", query.title).strip() if query.title else ""

        queries: List[str] = []

        # 1. NAJWYŻSZY PRIORYTET: marka + numer + opis + "dimensions"/"weight" -
        # dokładnie tak, jak człowiek wpisuje w Google:
        # "BOSCH 0204123733 Brake Master Cylinder dimensions"
        if clean_title:
            queries.append(f"{query.brand} {pn} {clean_title} dimensions")
            queries.append(f'{query.brand} "{pn}" {clean_title} dimensions weight')
            queries.append(f'"{pn}" {clean_title} dimensions')

        # 2. Bezpośrednie poszukiwanie wymiarów i wagi dla numeru części
        queries.append(f"{query.brand} {pn} dimensions weight")
        queries.append(f'"{pn}" dimensions weight')

        # 3. Warianty numeru w cudzysłowie + wykluczenie AGD (Bosch to też zmywarki...)
        for v in pn_variants:
            queries.append(
                f'{query.brand} "{v}" dimensions weight '
                f'-dishwasher -refrigerator -fridge -dryer -"washing machine" -"home connect"'
            )
            if clean_title:
                queries.append(f'"{v}" {clean_title}')
            queries.append(f'"{v}" datasheet specifications')

        return list(dict.fromkeys(queries))[: self.max_queries]

    @staticmethod
    def _has_complete_snippet_info(snippets: List[str]) -> bool:
        has_w = any(_has_weight(s) for s in snippets)
        has_d = any(_has_dimensions(s) for s in snippets)
        return has_w and has_d

    def _fetch_candidate(self, session, url: str, pn: str):
        """Pobiera stronę/PDF i zwraca (url, page_text, relevant, score) lub None."""
        from bs4 import BeautifulSoup
        import markdownify

        try:
            resp = session.get(url, timeout=8)
        except Exception as exc:
            logger.debug("Fetch failed for %s: %s", url, exc)
            return None
        if resp.status_code != 200:
            return None

        content_type = resp.headers.get("Content-Type", "")
        page_text = ""
        structured = ""

        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            pdf_text = _extract_pdf_text(resp.content)
            if pdf_text:
                lines = [ln.strip() for ln in pdf_text.splitlines() if ln.strip()]
                page_text = "\n".join(extract_relevant_lines(lines, max_lines=150))
        elif "text/html" in content_type:
            page_soup = BeautifulSoup(resp.text[:800_000], "html.parser")

            # Generyczna ekstrakcja tabel specyfikacji - LLM-owi dużo łatwiej
            # pracować na gotowych parach klucz: wartość niż na surowym markdownie.
            specs = extract_specs_from_soup(page_soup)
            if specs:
                structured = "STRUCTURED SPECS (auto-extracted):\n" + _format_specs(specs)

            md_text = markdownify.markdownify(
                str(page_soup), strip=["script", "style", "nav", "footer", "header"]
            )
            lines = [line.strip() for line in md_text.splitlines() if line.strip()]
            page_text = "\n".join(extract_relevant_lines(lines, max_lines=150))
        else:
            return None

        if structured:
            page_text = structured + "\n\n" + page_text
        if not page_text.strip():
            return None

        relevant = mentions_part_number(page_text, pn)
        score = (
            (2 if relevant else 0)
            + (1 if _has_weight(page_text) else 0)
            + (1 if _has_dimensions(page_text) else 0)
        )
        return (url, page_text, relevant, score)

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        pn = str(query.part_number).strip()
        session = self.backend.session

        all_snippets: List[str] = []
        top_links: List[str] = []
        found_url = ""

        for q_str in self._build_queries(query, pn):
            for res in self.backend.search(q_str, max_results=10):
                snippet = res.get("snippet", "")
                if snippet and snippet not in all_snippets:
                    all_snippets.append(snippet)
                href = res.get("link", "")
                if href and not _is_blocked(href) and href not in top_links:
                    top_links.append(href)
                    if not found_url:
                        found_url = href
            # Przerywamy wcześnie TYLKO wtedy, gdy mamy już zarówno wagę,
            # jak i wymiary liniowe w snippetach.
            if len(top_links) >= 3 and len(all_snippets) >= 12 and self._has_complete_snippet_info(all_snippets):
                break

        # Pobranie treści z najbardziej obiecujących linków. Zamiast ślepo brać
        # pierwsze 2 linki, pobieramy kilku kandydatów i punktujemy: wzmianka
        # o numerze części (najważniejsze), obecność wagi, obecność wymiarów.
        candidates = []
        for url in top_links:
            if len(candidates) >= self.max_fetch:
                break
            cand = self._fetch_candidate(session, url, pn)
            if cand:
                candidates.append(cand)
                # Jeśli mamy już 2 strony, które wspominają numer części ORAZ
                # mają komplet danych - nie ma po co pobierać dalej.
                full = [c for c in candidates if c[2] and c[3] >= 4]
                if len(full) >= 2:
                    break

        candidates.sort(key=lambda c: c[3], reverse=True)

        # Weryfikacja: jeśli ŻADEN snippet ani ŻADNA pobrana strona nie zawiera
        # numeru części, wyszukiwarka zwróciła przypadkowe wyniki niepowiązane
        # z częścią (np. zmywarki Bosch). Nie wolno ich uznawać za znalezione dane!
        has_pn_in_snippets = any(mentions_part_number(s, pn) for s in all_snippets)
        has_pn_in_pages = any(c[2] for c in candidates)
        if not (has_pn_in_snippets or has_pn_in_pages):
            return {
                "title": f"No online specifications found for {query.brand} {pn}",
                "source_url": found_url or "https://duckduckgo.com",
                "text": f"No online specifications or datasheets found mentioning part number {pn}.",
                "has_data": False,
                "source_type": "web",
            }

        fetched_text = ""
        for url, page_text, relevant, _score in candidates[:2]:
            flag = "" if relevant else " [NOTE: this page does not clearly mention the target part number - verify carefully]"
            fetched_text += f"\n\n--- Content from {url}{flag} ---\n{page_text}"

        if not all_snippets and not fetched_text:
            return {
                "title": f"Search results for {query.brand} {pn}",
                "source_url": found_url or "https://duckduckgo.com",
                "text": f"No online specifications found for {query.brand} {pn}.",
                "has_data": False,
                "source_type": "web",
            }

        combined_text = "=== SEARCH ENGINE SNIPPETS ===\n"
        combined_text += "\n---\n".join(all_snippets[:25])
        if fetched_text:
            combined_text += "\n\n=== FULL PAGE EXTRACTS ===\n" + fetched_text

        return {
            "title": f"Web specs for {query.brand} {pn}",
            "source_url": found_url or "https://duckduckgo.com",
            "text": f"TECHNICAL SEARCH RESULTS FOR {query.brand} {pn} ({query.title or ''}):\n\n{combined_text}",
            "has_data": True,
            "source_type": "web",
        }


# ============================================================
#  4. Cached Search Provider
# ============================================================

class CachedSearchProvider(BaseDatasheetProvider):
    """
    Buforuje pobrane karty katalogowe w plikach JSON, by unikać powtarzanych
    zapytań sieciowych.

    negative_ttl_seconds: opcjonalne cache'owanie wyników NEGATYWNYCH ("nic nie
    znaleziono") z krótszym TTL. Przy przetwarzaniu wsadowym tysięcy części
    oszczędza to mnóstwo czasu na numerach, których i tak nigdzie nie ma -
    a krótki TTL gwarantuje, że po kilku godzinach/dniach spróbujemy ponownie.
    None (domyślnie) = zachowanie jak dotychczas, negatywów nie cache'ujemy.
    """

    # Frazy używane przez dostawców do oznaczenia braku danych - używane jako
    # dodatkowe (poza has_data) zabezpieczenie przed uznaniem pustego wyniku za dane.
    _NO_DATA_PHRASES = (
        "No online specifications found",
        "No data found in aftermarket catalogs",
        "No documentation found in Docupedia",
        "No internal Docupedia documentation found",
        "SYNTHETIC MOCK DATA",
    )

    def __init__(
        self,
        inner: BaseDatasheetProvider,
        cache_dir: Optional[Path] = None,
        ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECONDS,
        refresh: bool = False,
        negative_ttl_seconds: Optional[int] = None,
    ):
        self.inner = inner
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # ttl_seconds=None oznacza "cache nigdy nie wygasa" (poprzednie zachowanie,
        # dostępne jawnie jako opt-in). Domyślnie: DEFAULT_CACHE_TTL_SECONDS.
        self.ttl_seconds = ttl_seconds
        self.refresh = refresh
        self.negative_ttl_seconds = negative_ttl_seconds

    def _has_real_data(self, data: Dict[str, Any]) -> bool:
        if not data.get("has_data"):
            return False
        text = data.get("text", "")
        return not any(phrase in text for phrase in self._NO_DATA_PHRASES)

    def _is_fresh(self, cache_file: Path, ttl: Optional[int]) -> bool:
        if ttl is None:
            return True
        age = time.time() - cache_file.stat().st_mtime
        return age < ttl

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        pn_clean = re.sub(r"[^\w\-_]", "", query.part_number)
        provider_name = type(self.inner).__name__.replace("DatasheetProvider", "").lower()
        cache_file = self.cache_dir / f"datasheet_{provider_name}_{pn_clean}.json"

        if cache_file.exists() and not self.refresh:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if self._has_real_data(data):
                    if self._is_fresh(cache_file, self.ttl_seconds):
                        return data
                elif self.negative_ttl_seconds is not None:
                    if self._is_fresh(cache_file, self.negative_ttl_seconds):
                        return data
            except Exception as exc:
                logger.warning("Cache read failed for %s: %s", cache_file, exc)

        result = self.inner.search_and_fetch_text(query)

        should_cache = self._has_real_data(result) or (
            self.negative_ttl_seconds is not None and not result.get("has_data")
        )
        if should_cache:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                logger.warning("Cache write failed for %s: %s", cache_file, exc)

        return result


# ============================================================
#  5. Hybrid Datasheet Provider (Docupedia + Catalog + Web)
# ============================================================

class HybridDatasheetProvider(BaseDatasheetProvider):
    """
    Dostawca hybrydowy - strategia fallback z dociąganiem braków:
    1. Najpierw wewnętrzna baza Bosch Docupedia (jeśli skonfigurowana).
    2. Jeśli brak danych - katalogi części (CatalogDatasheetProvider).
    3. Jeśli nadal brak - publiczne źródła internetowe.

    Nowość: require_complete=True (domyślnie) sprawdza KOMPLETNOŚĆ danych, nie
    tylko ich obecność. Jeśli pierwsze źródło zwróci np. samą wagę bez wymiarów
    (typowe dla stron sklepowych), pipeline dociąga kolejne źródło i scala oba
    teksty - każdy w wyraźnie oznaczonej sekcji, żeby LLM wiedział, skąd co
    pochodzi. Zatrzymujemy się na pierwszym momencie, w którym łączny kontekst
    zawiera i wagę, i wymiary liniowe.

    Cele bez zmian:
    - nie mieszać wiarygodnych danych wewnętrznych z szumem, gdy nie trzeba,
    - nie wykonywać zbędnych zapytań sieciowych, gdy odpowiedź już jest.
    """

    def __init__(
        self,
        docupedia: Optional[BaseDatasheetProvider] = None,
        catalog: Optional[BaseDatasheetProvider] = None,
        web: Optional[BaseDatasheetProvider] = None,
        require_complete: bool = True,
    ):
        self.docupedia = docupedia
        self.catalog = catalog or CatalogDatasheetProvider()
        self.web = web or WebDatasheetProvider()
        self.require_complete = require_complete

    def search_and_fetch_text(self, query: ProductQuery) -> Dict[str, Any]:
        collected: List[Dict[str, Any]] = []

        for provider, label in (
            (self.docupedia, "docupedia"),
            (self.catalog, "catalog"),
            (self.web, "web"),
        ):
            if provider is None:
                continue
            try:
                result = provider.search_and_fetch_text(query)
            except Exception:
                logger.exception("Provider %s raised - continuing with next source", label)
                continue

            if not result.get("has_data"):
                continue

            result.setdefault("source_type", label)
            collected.append(result)

            combined = "\n".join(r.get("text", "") for r in collected)
            if not self.require_complete or _is_complete(combined):
                break

        if not collected:
            return {
                "title": f"No specifications found for {query.brand} {query.part_number}",
                "source_url": "",
                "text": (
                    f"No documentation found in Docupedia, Catalogs, nor on the Web "
                    f"for {query.brand} {query.part_number}."
                ),
                "has_data": False,
                "attachments": [],
                "source_type": "hybrid",
            }

        if len(collected) == 1:
            result = collected[0]
            result.setdefault("attachments", [])
            return result

        # Scalanie danych z wielu źródeł - każde w oznaczonej sekcji.
        merged_text = "\n\n".join(
            f"=== SOURCE: {r['source_type'].upper()} ({r.get('source_url', '')}) ===\n{r.get('text', '')}"
            for r in collected
        )
        attachments: List[Any] = []
        for r in collected:
            attachments.extend(r.get("attachments") or [])

        first = collected[0]
        return {
            "title": first.get("title", f"Specs for {query.brand} {query.part_number}"),
            "source_url": first.get("source_url", ""),
            "text": merged_text,
            "has_data": True,
            "attachments": attachments,
            "source_type": "hybrid",
        }