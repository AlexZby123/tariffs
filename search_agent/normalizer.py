# -*- coding: utf-8 -*-
"""
Moduł deterministycznej normalizacji jednostek miar do standardu metrycznego (SI).

Standardy docelowe:
- Długość, szerokość, wysokość, średnica: milimetry [mm] (z opcją zaokrąglenia)
- Masa / waga: gramy [g] oraz kilogramy [kg]
- Objętość: centymetry sześcienne [cm3]
"""
from __future__ import annotations

import re
from typing import Optional, Tuple
from models import RawDimensions, MetricDimensions

# Współczynniki przeliczeniowe na milimetry [mm]
_LENGTH_UNITS = {
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "milimetr": 1.0,
    "milimetry": 1.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "centymetr": 10.0,
    "centymetry": 10.0,
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "metr": 1000.0,
    "metry": 1000.0,
    "in": 25.4,
    "inch": 25.4,
    "inches": 25.4,
    '"': 25.4,
    "cal": 25.4,
    "cale": 25.4,
    "ft": 304.8,
    "foot": 304.8,
    "feet": 304.8,
    "'": 304.8,
    "stopa": 304.8,
    "stopy": 304.8,
    "yd": 914.4,
    "yard": 914.4,
    "yards": 914.4,
    "um": 0.001,
    "µm": 0.001,
    "micrometer": 0.001,
}

# Współczynniki przeliczeniowe na gramy [g]
_WEIGHT_UNITS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "gm": 1.0,
    "gramy": 1.0,
    "gramow": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "kilo": 1000.0,
    "kilogramy": 1000.0,
    "mg": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
    "miligram": 0.001,
    "miligramy": 0.001,
    "lb": 453.59237,
    "lbs": 453.59237,
    "pound": 453.59237,
    "pounds": 453.59237,
    "funt": 453.59237,
    "funty": 453.59237,
    "oz": 28.349523,
    "ounce": 28.349523,
    "ounces": 28.349523,
    "uncja": 28.349523,
    "uncje": 28.349523,
    "t": 1_000_000.0,
    "tonne": 1_000_000.0,
    "tonnes": 1_000_000.0,
    "tona": 1_000_000.0,
    "tony": 1_000_000.0,
}


def _parse_fraction(text: str) -> Optional[float]:
    """Konwertuje ułamki zwykłe np. '1/2', '3/8', '1 1/4' na liczbę zmiennoprzecinkową."""
    # Usuwamy litery i cudzysłowy, aby wyodrębnić sam zapis liczbowy
    t = re.sub(r"[a-zA-Z\"'µ]+", "", text).strip()
    # Postać mieszana: np. "1 1/2"
    mixed_match = re.match(r"^(\d+)\s+(\d+)/(\d+)$", t)
    if mixed_match:
        whole = float(mixed_match.group(1))
        num = float(mixed_match.group(2))
        den = float(mixed_match.group(3))
        return whole + (num / den) if den != 0 else None

    # Zwykły ułamek: np. "3/8"
    frac_match = re.match(r"^(\d+)/(\d+)$", t)
    if frac_match:
        num = float(frac_match.group(1))
        den = float(frac_match.group(2))
        return num / den if den != 0 else None

    return None


def parse_numeric_value(val_str: str) -> Optional[float]:
    """Wyciąga pojedynczą wartość liczbową (w tym ułamki i przecinki dziesiętne)."""
    if not val_str:
        return None
    cleaned = val_str.strip().replace(",", ".")

    # Sprawdź czy to ułamek (np. 1/2 in, 3/8", 1 1/4)
    frac = _parse_fraction(cleaned)
    if frac is not None:
        return frac

    # Sprawdź czy to zakres (np. 10 - 12 -> średnia 11)
    range_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|–|to|\.\.)\s*(\d+(?:\.\d+)?)", cleaned)
    if range_match:
        v1 = float(range_match.group(1))
        v2 = float(range_match.group(2))
        return (v1 + v2) / 2.0

    # Zwykła liczba dziesiętna
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        return float(match.group(1))
    return None


def normalize_length_to_mm(raw_str: Optional[str]) -> Optional[float]:
    """Konwertuje dowolny ciąg wymiaru liniowego na milimetry [mm]."""
    if not raw_str or not raw_str.strip():
        return None

    raw_clean = raw_str.strip().lower()

    # Dopasuj jednostkę
    unit = None
    # Sortujemy klucze malejąco po długości, aby 'millimeters' dopasowało się przed 'mm'
    for u in sorted(_LENGTH_UNITS.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(u) + r"\b", raw_clean) or (u in ('"', "'") and u in raw_clean):
            unit = u
            break

    # Jeśli nie wykryto jednostki, ale jest liczba, zakładamy domyślnie mm (branża automotive)
    factor = _LENGTH_UNITS.get(unit, 1.0)
    num_val = parse_numeric_value(raw_clean)

    if num_val is None or num_val <= 0:
        return None

    mm = num_val * factor
    return round(mm, 2)


def normalize_weight_to_g(raw_str: Optional[str]) -> Optional[float]:
    """Konwertuje dowolny ciąg masy/wagi na gramy [g]."""
    if not raw_str or not raw_str.strip():
        return None

    raw_clean = raw_str.strip().lower()

    # Obsługa złożonych formatów np. "2 lbs 4 oz"
    compound_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\s*(\d+(?:\.\d+)?)\s*(?:oz|ounces?)", raw_clean)
    if compound_match:
        lbs = float(compound_match.group(1))
        oz = float(compound_match.group(2))
        total_g = lbs * _WEIGHT_UNITS["lb"] + oz * _WEIGHT_UNITS["oz"]
        return round(total_g, 2)

    # Standardowe wykrywanie jednostki
    unit = None
    for u in sorted(_WEIGHT_UNITS.keys(), key=len, reverse=True):
        if re.search(r"\b" + re.escape(u) + r"\b", raw_clean):
            unit = u
            break

    factor = _WEIGHT_UNITS.get(unit, 1.0)  # Domyślnie gramy
    num_val = parse_numeric_value(raw_clean)

    if num_val is None or num_val <= 0:
        return None

    g = num_val * factor
    return round(g, 2)


def parse_combined_dimensions_string(text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Rozpoznaje formaty potrójne np. '120 x 80 x 45 mm', '120x80x45mm', '10 x 5 x 2 in'."""
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*[xX*×]\s*(\d+(?:\.\d+)?)\s*([a-zA-Z\"'µ]+)?",
        re.IGNORECASE
    )
    m = pattern.search(text)
    if not m:
        return None, None, None

    l_val, w_val, h_val = float(m.group(1)), float(m.group(2)), float(m.group(3))
    raw_unit = (m.group(4) or "mm").strip().lower()

    factor = _LENGTH_UNITS.get(raw_unit, 1.0)
    for u in sorted(_LENGTH_UNITS.keys(), key=len, reverse=True):
        if u in raw_unit:
            factor = _LENGTH_UNITS[u]
            break

    return round(l_val * factor, 2), round(w_val * factor, 2), round(h_val * factor, 2)


def normalize_raw_dimensions(raw: RawDimensions) -> MetricDimensions:
    """Główna funkcja normalizująca obiekt RawDimensions do MetricDimensions."""
    length_mm = normalize_length_to_mm(raw.raw_length)
    width_mm = normalize_length_to_mm(raw.raw_width)
    height_mm = normalize_length_to_mm(raw.raw_height)
    diameter_mm = normalize_length_to_mm(raw.raw_diameter)

    # Sprawdź czy długość nie zawierała pełnego ciągu 'L x W x H'
    if raw.raw_length and (width_mm is None or height_mm is None):
        c_l, c_w, c_h = parse_combined_dimensions_string(raw.raw_length)
        if c_l is not None:
            length_mm, width_mm, height_mm = c_l, c_w, c_h

    # Jeśli część ma średnicę i brak szerokości/wysokości (część cylindryczna)
    if diameter_mm and not width_mm and not height_mm:
        width_mm = diameter_mm
        height_mm = diameter_mm

    weight_g = normalize_weight_to_g(raw.raw_weight)
    weight_kg = round(weight_g / 1000.0, 4) if weight_g is not None else None

    # Wylicz objętość bryły w cm3 jeśli mamy 3 wymiary
    volume_cm3 = None
    dim_string = None
    if length_mm and width_mm and height_mm:
        # 1 cm3 = 1000 mm3
        volume_cm3 = round((length_mm * width_mm * height_mm) / 1000.0, 2)
        dim_string = f"{length_mm} x {width_mm} x {height_mm} mm"
    elif diameter_mm and length_mm:
        dim_string = f"Ø {diameter_mm} x {length_mm} mm"

    return MetricDimensions(
        length_mm=length_mm,
        width_mm=width_mm,
        height_mm=height_mm,
        diameter_mm=diameter_mm,
        weight_g=weight_g,
        weight_kg=weight_kg,
        volume_cm3=volume_cm3,
        dimension_string_mm=dim_string,
    )
