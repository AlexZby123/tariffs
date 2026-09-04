# -*- coding: utf-8 -*-
"""
Wiersz poleceń (CLI) dla modułu search_agent.

Przykłady użycia:
  python cli.py --pn 0265005303 --mock
  python cli.py --pn 0445110189 --brand Bosch --title "Common rail injector" --mock
  python cli.py --pn 0265005303 --live
  python cli.py --csv-in ../data_sample.csv --csv-out enriched.csv --mock
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from models import ProductQuery
from pipeline import DimensionWorkflow


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agentic Product Dimension & Weight Extractor")
    p.add_argument("--pn", "--part-number", type=str, help="Product Number / Part Number")
    p.add_argument("--brand", type=str, default="Bosch", help="Marka (domyślnie Bosch)")
    p.add_argument("--title", type=str, default="", help="Tytuł / krótki opis materiału")
    p.add_argument("--category", type=str, default="", help="Kategoria części (np. SENSOR, VALVE)")
    p.add_argument("--mock", action="store_true", default=False, help="Wymuś tryb offline (Mock)")
    p.add_argument("--live", action="store_true", default=False, help="Użyj rzeczywistego wyszukiwania online")
    p.add_argument("--json", action="store_true", help="Zwróć wynik jako czysty JSON")
    p.add_argument("--csv-in", type=Path, help="Ścieżka do pliku CSV z listą części do wzbogacenia")
    p.add_argument("--csv-out", type=Path, help="Ścieżka zapisu wzbogaconego pliku CSV")
    p.add_argument("--limit", type=int, default=None, help="Ograniczenie liczby części z pliku CSV")
    p.add_argument("--concurrency", type=int, default=4, help="Równoległe wątki dla pliku CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    use_mock = args.mock or (not args.live)
    workflow = DimensionWorkflow(use_mock=use_mock)

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

        sep = ";" if ";" in open(args.csv_in, "r", encoding="utf-8", errors="ignore").readline() else ","
        df = pd.read_csv(args.csv_in, sep=sep, low_memory=False)
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

        out_path = args.csv_out or args.csv_in.parent / f"enriched_{args.csv_in.name}"
        df.to_csv(out_path, sep=";", index=False, encoding="utf-8")
        print(f"Zapisano wzbogacone dane do: {out_path}")
        return

    print("Podaj parametr --pn <part_number> lub --csv-in <plik.csv>. Użyj --help, aby zobaczyć opcje.")


if __name__ == "__main__":
    main()
