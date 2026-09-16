# -*- coding: utf-8 -*-
"""
Pakiet search_agent - agentowa ekstrakcja i normalizacja wymiarów i wagi produktów.
"""
from __future__ import annotations

from .models import (
    ProductQuery,
    RawDimensions,
    MetricDimensions,
    DimensionExtractionResult,
)
from .normalizer import (
    normalize_length_to_mm,
    normalize_weight_to_g,
    normalize_raw_dimensions,
    parse_combined_dimensions_string,
)
from .search_provider import (
    BaseDatasheetProvider,
    WebDatasheetProvider,
    MockDatasheetProvider,
    CachedSearchProvider,
    HybridDatasheetProvider,
)
from .docupedia_provider import (
    DocupediaClient,
    DocupediaDatasheetProvider,
)
from .extractor_agent import DimensionExtractorAgent
from .pipeline import DimensionWorkflow

__all__ = [
    "ProductQuery",
    "RawDimensions",
    "MetricDimensions",
    "DimensionExtractionResult",
    "normalize_length_to_mm",
    "normalize_weight_to_g",
    "normalize_raw_dimensions",
    "parse_combined_dimensions_string",
    "BaseDatasheetProvider",
    "WebDatasheetProvider",
    "MockDatasheetProvider",
    "CachedSearchProvider",
    "HybridDatasheetProvider",
    "DocupediaClient",
    "DocupediaDatasheetProvider",
    "DimensionExtractorAgent",
    "DimensionWorkflow",
]
