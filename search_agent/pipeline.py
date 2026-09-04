# -*- coding: utf-8 -*-
"""
Główny orkiestrator workflow agentowego dla search_agent.

Przepływ:
1. Przyjęcie ProductQuery (numer części, marka, tytuł, kategoria)
2. Pobranie danych technicznych z wyszukiwarki / kart katalogowych (search_provider)
3. Ekstrakcja surowych wymiarów i wagi przez agenta LLM (extractor_agent)
4. Deterministyczna normalizacja jednostek do standardu metrycznego (normalizer)
5. Zwrócenie ustrukturyzowanego obiektu DimensionExtractionResult
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any

from models import ProductQuery, DimensionExtractionResult, MetricDimensions
from normalizer import normalize_raw_dimensions
from search_provider import (
    BaseDatasheetProvider,
    CachedSearchProvider,
    MockDatasheetProvider,
    WebDatasheetProvider,
)
from extractor_agent import DimensionExtractorAgent


class DimensionWorkflow:
    """Kompletny pipeline agentowy do wyszukiwania, ekstrakcji i normalizacji wymiarów."""

    def __init__(
        self,
        provider: Optional[BaseDatasheetProvider] = None,
        extractor: Optional[DimensionExtractorAgent] = None,
        use_mock: bool = False,
        enable_cache: bool = True,
    ):
        self.use_mock = use_mock
        if provider is not None:
            self.provider = provider
        elif use_mock:
            self.provider = MockDatasheetProvider()
        else:
            base_p = WebDatasheetProvider()
            self.provider = CachedSearchProvider(base_p) if enable_cache else base_p

        self.extractor = extractor or DimensionExtractorAgent(use_mock=use_mock)

    def process(self, query_or_dict: Union[ProductQuery, Dict[str, Any]]) -> DimensionExtractionResult:
        """Przetwarza pojedynczy produkt."""
        start_t = time.time()
        if isinstance(query_or_dict, dict):
            query = ProductQuery(**query_or_dict)
        else:
            query = query_or_dict

        # 1. Pobierz tekst specyfikacji
        data = self.provider.search_and_fetch_text(query)
        source_text = data.get("text", "")
        source_url = data.get("source_url", "")

        if not source_text or not source_text.strip():
            return DimensionExtractionResult(
                query=query,
                found=False,
                confidence=0.0,
                notes="Brak tekstu specyfikacji lub brak wyników wyszukiwania.",
                processing_time_s=round(time.time() - start_t, 3),
            )

        # 2. Wyciągnij surowe wymiary przez agenta
        raw = self.extractor.extract(query, source_text, source_url=source_url)

        # 3. Znormalizuj do jednostek metrycznych
        metric: Optional[MetricDimensions] = None
        has_any_dim = bool(raw.raw_length or raw.raw_width or raw.raw_height or raw.raw_diameter or raw.raw_weight)

        if has_any_dim:
            metric = normalize_raw_dimensions(raw)

        duration = round(time.time() - start_t, 3)

        return DimensionExtractionResult(
            query=query,
            found=has_any_dim,
            raw=raw,
            metric=metric,
            confidence=raw.confidence if has_any_dim else 0.0,
            source_url=source_url,
            notes=f"Processed in {duration}s",
            processing_time_s=duration,
        )

    def process_batch(
        self,
        queries: List[Union[ProductQuery, Dict[str, Any]]],
        concurrency: int = 4
    ) -> List[DimensionExtractionResult]:
        """Przetwarza listę produktów równolegle z użyciem puli wątków."""
        results: List[DimensionExtractionResult] = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self.process, q): i for i, q in enumerate(queries)
            }
            ordered_results = [None] * len(queries)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered_results[idx] = future.result()
                except Exception as e:
                    q = queries[idx]
                    pq = q if isinstance(q, ProductQuery) else ProductQuery(**q)
                    ordered_results[idx] = DimensionExtractionResult(
                        query=pq,
                        found=False,
                        confidence=0.0,
                        notes=f"Batch error: {str(e)}",
                    )
        return ordered_results
