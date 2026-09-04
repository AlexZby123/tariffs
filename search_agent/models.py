# -*- coding: utf-8 -*-
"""
Modele danych dla modułu search_agent (ekstrakcja i normalizacja wymiarów).
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ProductQuery(BaseModel):
    """Identyfikatory produktu przekazywane na wejściu."""
    part_number: str = Field(..., description="Numer katalogowy / Product Number (np. 0265005303)")
    brand: str = Field(default="Bosch", description="Marka / producent")
    title: Optional[str] = Field(default=None, description="Tytuł / krótki opis materiału")
    category: Optional[str] = Field(default=None, description="Kategoria lub typ funkcjonalny (np. SENSOR, PUMP)")
    extra_context: Optional[dict[str, Any]] = Field(default=None, description="Dodatkowe dane (np. kod HS, BU)")

    def search_query(self) -> str:
        """Buduje zwięzłą frazę do wyszukiwarki technicznej."""
        parts = [self.brand, self.part_number]
        if self.title:
            parts.append(self.title)
        parts.append("datasheet OR specifications OR dimensions OR weight")
        return " ".join(parts)


class RawDimensions(BaseModel):
    """Surowe wartości wyciągnięte przez LLM ze źródła przed normalizacją."""
    raw_length: Optional[str] = Field(default=None, description="Surowy odczyt długości (np. '120 mm', '4.5 in')")
    raw_width: Optional[str] = Field(default=None, description="Surowy odczyt szerokości")
    raw_height: Optional[str] = Field(default=None, description="Surowy odczyt wysokości")
    raw_weight: Optional[str] = Field(default=None, description="Surowy odczyt wagi (np. '250 g', '1.2 lbs')")
    raw_diameter: Optional[str] = Field(default=None, description="Surowy odczyt średnicy (jeśli część jest cylindryczna/okrągła)")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Ocena pewności odczytu przez model (0-1)")
    source_snippet: Optional[str] = Field(default=None, description="Cytat ze specyfikacji potwierdzający wymiary")
    source_url: Optional[str] = Field(default=None, description="URL lub nazwa źródła/karty katalogowej")


class MetricDimensions(BaseModel):
    """Znormalizowane wymiary w standardowych jednostkach metrycznych (SI)."""
    length_mm: Optional[float] = Field(default=None, description="Długość w milimetrach [mm]")
    width_mm: Optional[float] = Field(default=None, description="Szerokość w milimetrach [mm]")
    height_mm: Optional[float] = Field(default=None, description="Wysokość w milimetrach [mm]")
    diameter_mm: Optional[float] = Field(default=None, description="Średnica w milimetrach [mm]")
    weight_g: Optional[float] = Field(default=None, description="Masa / waga w gramach [g]")
    weight_kg: Optional[float] = Field(default=None, description="Masa / waga w kilogramach [kg]")
    volume_cm3: Optional[float] = Field(default=None, description="Objętość prostopadłościanu bryły w cm3")
    dimension_string_mm: Optional[str] = Field(default=None, description="Sformatowany ciąg L x W x H mm")


class DimensionExtractionResult(BaseModel):
    """Pełny wynik procesu ekstrakcji i normalizacji."""
    query: ProductQuery
    found: bool = Field(default=False, description="Czy znaleziono specyfikację techniczną z wymiarami")
    raw: Optional[RawDimensions] = None
    metric: Optional[MetricDimensions] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_url: Optional[str] = None
    notes: Optional[str] = None
    processing_time_s: Optional[float] = None
