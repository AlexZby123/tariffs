# -*- coding: utf-8 -*-
"""
Wspólne narzędzia współdzielone przez wszystkich dostawców danych
(Docupedia, Web, Catalog): generowanie wariantów numeru części,
ocena trafności wyników oraz "inteligentne" (nie tylko od początku)
obcinanie długich dokumentów.

Wydzielone z docupedia_provider.py / search_provider.py, gdzie ta sama
logika (warianty numeru części, obcinanie tekstu) była wcześniej
zduplikowana i lekko niespójna między dostawcami.
"""
from __future__ import annotations

import re
from typing import Iterable, List

# Słowa kluczowe / wzorce sugerujące, że linia tekstu dotyczy wymiarów lub wagi.
_DIM_KEYWORDS_RE = re.compile(
    r"(?:\bmm\b|\bcm\b|\binch(?:es)?\b|\bweight\b|\bwaga\b|\bmasa\b|\bgewicht\b|"
    r"\bdimension\w*\b|\bwymiar\w*\b|\blength\b|\bwidth\b|\bheight\b|\bdiameter\b|"
    r"\bd(?:ł|l)ugo(?:ś|s)(?:ć|c)\w*\b|\bszeroko(?:ś|s)(?:ć|c)\w*\b|\bwysoko(?:ś|s)(?:ć|c)\w*\b|"
    r"\b(?:ś|s)rednica\b|\d+(?:[.,]\d+)?\s*(?:mm|cm|kg|lbs?|oz)\b)",
    re.IGNORECASE,
)


def generate_pn_variants(part_number: str) -> List[str]:
    """
    Generuje warianty formatowania numeru części Bosch, które w praktyce
    pojawiają się w różnych źródłach (Docupedia, katalogi, strony producentów).

    Dla 10-cyfrowego numeru (np. '0265005303') generuje zarówno grupowanie
    1-3-3-3 ('0 265 005 303' / '0.265.005.303'), jak i 4-3-3
    ('0265.005.303' / '0265 005 303') - obie konwencje są w praktyce używane,
    a wcześniej każdy dostawca próbował tylko jednej z nich.
    """
    pn = str(part_number).strip()
    pn_clean = re.sub(r"[\s\-_.]", "", pn)

    variants = [pn]
    if pn_clean and pn_clean != pn:
        variants.append(pn_clean)

    if len(pn_clean) == 10 and pn_clean.isalnum():
        variants.append(f"{pn_clean[0]} {pn_clean[1:4]} {pn_clean[4:7]} {pn_clean[7:]}")
        variants.append(f"{pn_clean[0]}.{pn_clean[1:4]}.{pn_clean[4:7]}.{pn_clean[7:]}")
        variants.append(f"{pn_clean[:4]}.{pn_clean[4:7]}.{pn_clean[7:]}")
        variants.append(f"{pn_clean[:4]} {pn_clean[4:7]} {pn_clean[7:]}")

    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def mentions_part_number(text: str, part_number: str) -> bool:
    """
    Sprawdza, czy tekst faktycznie wspomina dany numer części, niezależnie
    od spacji/myślników/kropek użytych w oryginalnym formatowaniu.
    Używane do odrzucania (lub oznaczania) wyników wyszukiwania, które
    trafiły na stronę niezwiązaną z szukaną częścią.
    """
    if not text or not part_number:
        return False
    t_clean = re.sub(r"[\s\-_.]", "", text).upper()
    pn_clean = re.sub(r"[\s\-_.]", "", str(part_number)).upper()
    return bool(pn_clean) and pn_clean in t_clean


def score_title_match(title: str, pn_variants: Iterable[str]) -> int:
    """
    Prosta heurystyka trafności dla wyników wyszukiwania: strony, których
    TYTUŁ zawiera numer części, są dużo bardziej wiarygodne niż strony
    dopasowane tylko przez pełnotekstowe wyszukiwanie (gdzie numer może
    pojawić się przypadkiem, np. w tabeli "podobne produkty").
    """
    if not title:
        return 0
    title_upper = title.upper()
    score = 0
    for v in pn_variants:
        v = str(v).strip().upper()
        if v and v in title_upper:
            score += 10
    lower_title = title.lower()
    for kw in ("datasheet", "spec", "specification", "karta katalogowa", "dane techniczne", "technical data", "pflichtenheft", "lastenheft", "zeichnung", "drawing", "scs"):
        if kw in lower_title:
            score += 3
    return score


def extract_relevant_lines(lines: Iterable[str], max_lines: int = 120, context: int = 2) -> List[str]:
    """
    Zamiast obcinać długi dokument do pierwszych N linii (co gubi tabelę
    wymiarów, jeśli znajduje się dalej w dokumencie), zachowuje:
      - początek dokumentu (tytuł, nagłówki) dla ogólnego kontekstu,
      - linie w pobliżu wzmianek o wymiarach/wadze.

    Zwraca linie w oryginalnej kolejności.
    """
    lines = list(lines)
    if len(lines) <= max_lines:
        return lines

    keep = set(range(min(30, len(lines))))
    for i, line in enumerate(lines):
        if _DIM_KEYWORDS_RE.search(line):
            for j in range(max(0, i - context), min(len(lines), i + context + 1)):
                keep.add(j)
        if len(keep) >= max_lines:
            break

    ordered = sorted(keep)[:max_lines]
    return [lines[i] for i in ordered]
