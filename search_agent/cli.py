# -*- coding: utf-8 -*-
"""
Wiersz poleceń (CLI) dla modułu search_agent.

Przykłady użycia:
  python cli.py --pn 0265005303 --mock
  python cli.py --pn 0445110189 --brand Bosch --title "Common rail injector" --mock
  python cli.py --pn 0265005303 --live
  python cli.py --pn 0265005303 --live --refresh-cache
  python cli.py --csv-in ../data_sample.csv --csv-out enriched.csv --mock
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from models import ProductQuery
from pipeline import DimensionWorkflow
from docupedia_provider import DocupediaClient


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic Product Dimension & Weight Extractor (Web + Docupedia)")
    p.add_argument("--pn", "--part-number", type=str, help="Product Number / Part Number")
    p.add_argument("--brand", type=str, default="Bosch", help="Marka (domyślnie Bosch)")
    p.add_argument("--title", type=str, default="", help="Tytuł / krótki opis materiału")
    p.add_argument("--category", type=str, default="", help="Kategoria części (np. SENSOR, VALVE)")
    p.add_argument("--source", choices=["hybrid", "docupedia", "web", "mock"], default="hybrid",
                   help="Źródło kart danych: hybrid (Docupedia + Web), docupedia, web, mock")
    p.add_argument("--docupedia", action="store_true", help="Szukaj wyłącznie w Docupedii (Bosch Confluence)")
    p.add_argument("--web", action="store_true", help="Szukaj wyłącznie w otwartym internecie")
    p.add_argument("--mock", action="store_true", default=False, help="Wymuś tryb offline (Mock)")
    p.add_argument("--live", action="store_true", default=False, help="Wyszukiwanie aktywne (Web/Docupedia)")
    p.add_argument("--json", action="store_true", help="Zwróć wynik jako czysty JSON")
    p.add_argument("--no-cache", action="store_true", help="Wyłącz warstwę cache - zawsze pytaj źródła na żywo")
    p.add_argument("--refresh-cache", action="store_true", help="Zignoruj istniejący cache dla tego zapytania i nadpisz go świeżym wynikiem")
    p.add_argument("--search-docupedia", type=str, help="Bezpośrednie wyszukanie stron w Docupedii")
    p.add_argument("--read-docupedia", type=str, help="Pobranie i wyświetlenie treści strony z Docupedii po ID")
    p.add_argument("--csv-in", type=Path, help="Ścieżka do pliku CSV z listą części do wzbogacenia")
    p.add_argument("--csv-out", type=Path, help="Ścieżka zapisu wzbogaconego pliku CSV")
    p.add_argument("--limit", type=int, default=None, help="Ograniczenie liczby części z pliku CSV")
    p.add_argument("--concurrency", type=int, default=4, help="Równoległe wątki dla pliku CSV")
    p.add_argument("-v", "--verbose", action="store_true", help="Włącz logi na poziomie INFO (widoczność co dokładnie się dzieje przy wyszukiwaniu/ekstrakcji)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Bezpośrednie narzędzia Docupedii
    if args.search_docupedia:
        client = DocupediaClient()
        if not client.is_configured:
            print("[ERROR] Docupedia nie jest skonfigurowana. Ustaw docupedia_token w agentic/config.yaml.")
            sys.exit(1)
        print(f"\n[DOCUPEDIA] Wyszukiwanie: \"{args.search_docupedia}\"...")
        pages = client.search_internal_knowledge_base(args.search_docupedia, limit=8)
        if not pages:
            print("Brak wyników w Docupedii.")
            return
        print(f"Znaleziono {len(pages)} dokumentów:")
        for p in pages:
            print(f" - [ID: {p['id']}] {p['title']} (Space: {p['space']})")
            print(f"   URL: {p['url']}")
        print()
        return

    if args.read_docupedia:
        client = DocupediaClient()
        if not client.is_configured:
            print("[ERROR] Docupedia nie jest skonfigurowana. Ustaw docupedia_token w agentic/config.yaml.")
            sys.exit(1)
        content = client.get_internal_document_content(args.read_docupedia)
        print("\n" + "=" * 65)
        print(content)
        print("=" * 65 + "\n")
        return

    # Wybór źródła danych: --mock / --docupedia / --web wygrywają z --source,
    # w przeciwnym razie używamy --source (domyślnie "hybrid").
    if args.mock:
        source = "mock"
    elif args.docupedia:
        source = "docupedia"
    elif args.web:
        source = "web"
    else:
        source = args.source

    use_mock = source == "mock"
    workflow = DimensionWorkflow(
        use_mock=use_mock,
        source=source,
        enable_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
    )

    # 1. Tryb pojedynczej części z parametrów CLI
    if args.pn:
        query = ProductQuery(
            part_number=args.pn,
            brand=args.brand,
            title=args.title or None,
            category=args.category or None,
        )
        res = workflow.process(query)

        if args.json:
            print(res.model_dump_json(indent=2))
            return

        print("\n" + "=" * 65)
        print(f"[PRODUCT] {res.query.brand} {res.query.part_number}")
        if res.query.title:
            print(f"   Title: {res.query.title}")
        print("=" * 65)

        if not res.found:
            print("[INFO] Dimensions not found or could not be extracted.")
            if res.notes:
                print(f"   Notes: {res.notes}")
            return

        print(f"[OK] STATUS: SUCCESS (Confidence: {res.confidence * 100:.1f}%)")
        print(f"   Source URL: {res.source_url}")
        if res.source_type:
            print(f"   Source Type: {res.source_type}")
        if res.raw and res.raw.extraction_method:
            tag = " (guessed, not from a real source - verify before trusting)" if res.raw.extraction_method == "estimated" else ""
            print(f"   Extraction Method: {res.raw.extraction_method}{tag}")
        if res.raw and res.raw.source_snippet:
            print(f"   Raw Snippet: \"{res.raw.source_snippet.strip()}\"")

        print("\n--- RAW EXTRACTED VALUES ---")
        if res.raw:
            print(f"   Length:   {res.raw.raw_length or 'N/A'}")
            print(f"   Width:    {res.raw.raw_width or 'N/A'}")
            print(f"   Height:   {res.raw.raw_height or 'N/A'}")
            print(f"   Diameter: {res.raw.raw_diameter or 'N/A'}")
            print(f"   Weight:   {res.raw.raw_weight or 'N/A'}")

        print("\n--- NORMALIZED METRIC VALUES (SI) ---")
        if res.metric:
            print(f"   Length:    {res.metric.length_mm} mm" if res.metric.length_mm is not None else "   Length:    N/A")
            print(f"   Width:     {res.metric.width_mm} mm" if res.metric.width_mm is not None else "   Width:     N/A")
            print(f"   Height:    {res.metric.height_mm} mm" if res.metric.height_mm is not None else "   Height:    N/A")
            if res.metric.diameter_mm is not None:
                print(f"   Diameter:  {res.metric.diameter_mm} mm")
            print(f"   Weight:    {res.metric.weight_g} g  ({res.metric.weight_kg} kg)" if res.metric.weight_g is not None else "   Weight:    N/A")
            if res.metric.volume_cm3 is not None:
                print(f"   Volume:    {res.metric.volume_cm3} cm³")
            if res.metric.dimension_string_mm:
                print(f"   Formatted: {res.metric.dimension_string_mm}")
        print("=" * 65 + "\n")
        return

    # 2. Tryb wsadowy (Batch CSV)
    if args.csv_in:
        if not args.csv_in.exists():
            print(f"Error: Plik {args.csv_in} nie istnieje.")
            sys.exit(1)

        with open(args.csv_in, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
        sep = ";" if ";" in first_line else ","
        # dtype=str jest tu krytyczne: numer części to często sam ciąg cyfr
        # (np. "0000000000"), a bez wymuszenia typu tekstowego pandas cicho
        # odczytuje taką kolumnę jako int64 i GUBI wiodące zera (np.
        # "0000000000" -> 0), co dalej prowadziło do pomylenia części.
        df = pd.read_csv(args.csv_in, sep=sep, low_memory=False, dtype=str)
        if args.limit:
            df = df.head(args.limit)

        pn_col = next((c for c in df.columns if "part" in c.lower() or "product" in c.lower() or "pn" in c.lower()), None)
        if not pn_col:
            print(f"Nie znaleziono kolumny z numerem części w pliku {args.csv_in}. Dostępne kolumny: {list(df.columns)}")
            sys.exit(1)

        title_col = next((c for c in df.columns if "desc" in c.lower() or "title" in c.lower()), None)

        queries = [
            ProductQuery(
                part_number=str(row[pn_col]).strip(),
                brand=args.brand,
                title=str(row[title_col]).strip() if title_col else None,
            )
            for _, row in df.iterrows()
        ]

        print(f"Rozpoczynam przetwarzanie wsadowe {len(queries)} części (wątki={args.concurrency})...")
        results = workflow.process_batch(queries, concurrency=args.concurrency)

        # Wzbogacenie DataFrame
        df["length_mm"] = [r.metric.length_mm if r.metric else None for r in results]
        df["width_mm"] = [r.metric.width_mm if r.metric else None for r in results]
        df["height_mm"] = [r.metric.height_mm if r.metric else None for r in results]
        df["weight_g"] = [r.metric.weight_g if r.metric else None for r in results]
        df["dim_confidence"] = [r.confidence for r in results]
        df["source_type"] = [r.source_type or "" for r in results]
        df["extraction_method"] = [r.raw.extraction_method if r.raw and r.raw.extraction_method else "" for r in results]

        out_path = args.csv_out or args.csv_in.parent / f"enriched_{args.csv_in.name}"
        df.to_csv(out_path, sep=";", index=False, encoding="utf-8")
        print(f"Zapisano wzbogacone dane do: {out_path}")
        n_estimated = sum(1 for r in results if r.raw and r.raw.extraction_method == "estimated")
        if n_estimated:
            print(f"[UWAGA] {n_estimated} z {len(results)} wierszy ma extraction_method='estimated' (zgadnięte przez LLM, brak realnego źródła) - warto je zweryfikować ręcznie przed użyciem np. do klasyfikacji taryfowej.")
        return

    print("Podaj parametr --pn <part_number> lub --csv-in <plik.csv>. Użyj --help, aby zobaczyć opcje.")


if __name__ == "__main__":
    main()
