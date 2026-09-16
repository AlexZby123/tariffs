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

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

from models import ProductQuery, DimensionExtractionResult, MetricDimensions
from normalizer import normalize_raw_dimensions
from search_provider import (
    BaseDatasheetProvider,
    CachedSearchProvider,
    MockDatasheetProvider,
    WebDatasheetProvider,
    HybridDatasheetProvider,
    DEFAULT_CACHE_TTL_SECONDS,
)
from docupedia_provider import DocupediaDatasheetProvider, DocupediaClient
from extractor_agent import DimensionExtractorAgent

logger = logging.getLogger(__name__)


class DimensionWorkflow:
    """Kompletny pipeline agentowy do wyszukiwania, ekstrakcji i normalizacji wymiarów."""

    def __init__(
        self,
        provider: Optional[Union[BaseDatasheetProvider, List[BaseDatasheetProvider]]] = None,
        extractor: Optional[DimensionExtractorAgent] = None,
        use_mock: bool = False,
        source: str = "hybrid",  # "hybrid" | "docupedia" | "web" | "mock"
        enable_cache: bool = True,
        cache_ttl_seconds: Optional[int] = DEFAULT_CACHE_TTL_SECONDS,
        refresh_cache: bool = False,
    ):
        self.use_mock = use_mock
        self.enable_cache = enable_cache
        self.extractor = extractor or DimensionExtractorAgent(use_mock=use_mock)

        def _maybe_cache(inner: BaseDatasheetProvider) -> BaseDatasheetProvider:
            if not enable_cache:
                return inner
            return CachedSearchProvider(inner, ttl_seconds=cache_ttl_seconds, refresh=refresh_cache)

        if provider is not None:
            self.providers = provider if isinstance(provider, list) else [provider]
        elif use_mock or source == "mock":
            self.providers = [MockDatasheetProvider()]
        elif source == "docupedia":
            self.providers = [_maybe_cache(DocupediaDatasheetProvider())]
        elif source == "web":
            self.providers = [_maybe_cache(WebDatasheetProvider())]
        else:
            # Tryb hybrydowy - Late-Binding Fallback pętla w pipeline
            self.providers = []
            if DocupediaClient().is_configured:
                self.providers.append(_maybe_cache(DocupediaDatasheetProvider()))
            from search_provider import CatalogDatasheetProvider
            self.providers.append(_maybe_cache(CatalogDatasheetProvider()))
            self.providers.append(_maybe_cache(WebDatasheetProvider()))

    def process(self, query_or_dict: Union[ProductQuery, Dict[str, Any]]) -> DimensionExtractionResult:
        """Przetwarza pojedynczy produkt, z inteligentnym fallbackiem opartym na jakości ekstrakcji."""
        start_t = time.time()
        if isinstance(query_or_dict, dict):
            query = ProductQuery(**query_or_dict)
        else:
            query = query_or_dict

        best_raw = None
        best_source_type = ""
        best_source_url = ""
        best_metric = None

        for provider_inst in self.providers:
            data = provider_inst.search_and_fetch_text(query)
            source_text = data.get("text", "")
            source_url = data.get("source_url", "")
            
            # Unpack CachedSearchProvider name if wrapped
            actual_provider = provider_inst.inner if isinstance(provider_inst, CachedSearchProvider) else provider_inst
            fallback_label = type(actual_provider).__name__.replace("DatasheetProvider", "").lower()
            source_type = data.get("source_type", fallback_label)
            attachments = data.get("attachments", [])

            if not data.get("has_data") and not attachments:
                continue

            # 2. Wyciągnij surowe wymiary przez agenta dla TEGO KONKRETNEGO ŹRÓDŁA
            raw = self.extractor.extract(query, source_text, source_url=source_url, attachments=attachments)

            # Posprzątaj pobrane załączniki po ekstrakcji tylko, jeśli cache jest wyłączony
            if not self.enable_cache:
                for att_path in attachments:
                    try:
                        Path(att_path).unlink(missing_ok=True)
                    except Exception:
                        pass

            # Oceniamy, czy wynik jest twardym faktem, czy tylko zgadywaniem
            is_extracted = (raw.extraction_method == "extracted")
            has_any_dim = bool(raw.raw_length or raw.raw_width or raw.raw_height or raw.raw_diameter or raw.raw_weight)
            
            print(f"[DEBUG] Source: {source_type} | Extracted: {is_extracted} | Confidence: {raw.confidence} | Has Dim: {has_any_dim}")
            
            # Funkcja sprawdzająca, czy mamy komplet: wagę ORAZ wymiary liniowe
            def _is_complete_dims(r: RawDimensions) -> bool:
                has_w = bool(r.raw_weight)
                has_l = bool(
                    (r.raw_length and (r.raw_width or r.raw_height or r.raw_diameter))
                    or (r.raw_diameter and (r.raw_height or r.raw_length))
                )
                return has_w and has_l

            # Fuzja atrybutów między źródłami: jeśli poprzednie źródło znalazło tylko wagę,
            # a kolejne źródło ma wymiary liniowe (lub odwrotnie), połącz je!
            if best_raw is not None and is_extracted:
                best_is_extracted = (best_raw.extraction_method == "extracted")
                if not best_is_extracted:
                    # Nowe źródło jest wyekstrahowane z realnego dokumentu, zastąp dotychczasowe zgadywanie
                    best_raw = raw
                    best_source_type = source_type
                    best_source_url = source_url
                else:
                    # Oba źródła są twardo wyekstrahowane - uzupełnij brakujące pola
                    if not best_raw.raw_length and raw.raw_length:
                        best_raw.raw_length = raw.raw_length
                    if not best_raw.raw_width and raw.raw_width:
                        best_raw.raw_width = raw.raw_width
                    if not best_raw.raw_height and raw.raw_height:
                        best_raw.raw_height = raw.raw_height
                    if not best_raw.raw_diameter and raw.raw_diameter:
                        best_raw.raw_diameter = raw.raw_diameter
                    if not best_raw.raw_weight and raw.raw_weight:
                        best_raw.raw_weight = raw.raw_weight
                    if source_url and source_url not in best_source_url:
                        best_source_url = f"{best_source_url}, {source_url}"
            elif best_raw is None or (is_extracted and best_raw.extraction_method != "extracted") or (raw.confidence > best_raw.confidence):
                best_raw = raw
                best_source_type = source_type
                best_source_url = source_url

            # BINGO: Przerywamy pętlę TYLKO wtedy, gdy mamy komplet (wagę i wymiary liniowe) z twardego źródła!
            if best_raw and _is_complete_dims(best_raw) and (best_raw.extraction_method == "extracted") and best_raw.confidence >= 0.7:
                break

        if best_raw is None:
            return DimensionExtractionResult(
                query=query,
                found=False,
                confidence=0.0,
                source_type="hybrid",
                notes="Brak tekstu specyfikacji lub wyników wyszukiwania.",
                processing_time_s=round(time.time() - start_t, 3),
            )

        has_any_dim = bool(best_raw.raw_length or best_raw.raw_width or best_raw.raw_height or best_raw.raw_diameter or best_raw.raw_weight)
        if has_any_dim:
            best_metric = normalize_raw_dimensions(best_raw)

        duration = round(time.time() - start_t, 3)
        logger.info(
            "Processed %s %s: found=%s source_type=%s extraction_method=%s confidence=%.2f (%.3fs)",
            query.brand, query.part_number, has_any_dim, best_source_type, best_raw.extraction_method, best_raw.confidence, duration,
        )

        return DimensionExtractionResult(
            query=query,
            found=has_any_dim,
            raw=best_raw,
            metric=best_metric,
            confidence=best_raw.confidence if has_any_dim else 0.0,
            source_url=best_source_url,
            source_type=best_source_type,
            notes=f"Processed in {duration}s",
            processing_time_s=duration,
        )

    def process_batch(
        self,
        queries: List[Union[ProductQuery, Dict[str, Any]]],
        concurrency: int = 4
    ) -> List[DimensionExtractionResult]:
        """
        Przetwarza listę produktów równolegle z użyciem puli wątków.

        Uwaga o limitach czasu: każde faktyczne wywołanie sieciowe niżej w
        stosie ma już swój własny timeout (Confluence: domyślnie 75s w
        bibliotece atlassian-python-api; requests.get/post w search_provider.py:
        jawne timeout=). To właśnie te limity - a nie limit na poziomie tej
        metody - realnie zapobiegają zawieszeniu się pojedynczego zapytania w
        nieskończoność. ThreadPoolExecutor i tak czeka (join) na wszystkie
        wątki robocze przy wyjściu z bloku `with`, więc "wrapper" z timeoutem
        tutaj dawałby fałszywe poczucie bezpieczeństwa, nie realne przerwanie
        zawieszonego wątku - stąd świadomie z niego zrezygnowano na rzecz
        pilnowania limitów czasu tam, gdzie faktycznie mają znaczenie.
        """
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self.process, q): i for i, q in enumerate(queries)
            }
            ordered_results: List[Optional[DimensionExtractionResult]] = [None] * len(queries)
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
