# -*- coding: utf-8 -*-
"""
Sprawdza, czy BIEZACY interpreter Pythona ma pakiety potrzebne do projektu.

Uruchom tym interpreterem, ktorego naprawde uzywasz, np.:
    C:\\Users\\ZBA1WZ\\.conda\\envs\\pandas_excel\\python.exe sprawdz_srodowisko.py

Skrypt nie instaluje niczego - tylko raportuje braki i podaje gotowe polecenia.
"""
from __future__ import annotations

import importlib
import sys

# (modul, nazwa w pip, do czego potrzebny, czy wymagany)
PAKIETY = [
    ("numpy",                 "numpy",                 "rdzen obliczen",                         True),
    ("pandas",                "pandas",                "wczytywanie i deduplikacja danych",      True),
    ("scipy",                 "scipy",                 "metryki (Hungarian), macierze rzadkie",  True),
    ("sklearn",               "scikit-learn",          "TF-IDF, SVD, LinearSVC, klastrowanie",   True),
    ("yaml",                  "PyYAML",                "config.yaml, overrides.yaml",            True),
    ("openpyxl",              "openpyxl",              "etykiety Rudolfa z .xlsx",               True),
    ("torch",                 "torch",                 "--encoder tfidf+supcon (siec metryczna)", False),
    ("sentence_transformers", "sentence-transformers", "--encoder minilm / bge (HuggingFace)",   False),
    ("openai",                "openai",                "tor chmurowy + --nazywaj llm",           False),
]


def main() -> int:
    print("=" * 70)
    print("  SPRAWDZENIE SRODOWISKA")
    print("=" * 70)
    print(f"interpreter : {sys.executable}")
    print(f"wersja      : {sys.version.split()[0]}\n")

    brak_wymaganych: list[str] = []
    brak_opcjonalnych: list[tuple[str, str]] = []
    for modul, pip_name, po_co, wymagany in PAKIETY:
        try:
            m = importlib.import_module(modul)
            wersja = getattr(m, "__version__", "?")
            print(f"  [OK]   {pip_name:22s} {wersja:12s} {po_co}")
        except Exception:
            znacznik = "BRAK " if wymagany else "opcj."
            print(f"  [{znacznik}] {pip_name:22s} {'-':12s} {po_co}")
            if wymagany:
                brak_wymaganych.append(pip_name)
            else:
                brak_opcjonalnych.append((pip_name, po_co))

    print()
    if brak_wymaganych:
        print("NIE URUCHOMISZ toru lokalnego - brakuje wymaganych pakietow:")
        print(f'  "{sys.executable}" -m pip install {" ".join(brak_wymaganych)}')
    else:
        print("Tor lokalny (--encoder tfidf) zadziala - wszystkie wymagane pakiety sa.")

    if brak_opcjonalnych:
        print("\nOpcjonalne (potrzebne tylko dla konkretnych trybow):")
        for pip_name, po_co in brak_opcjonalnych:
            print(f'  {po_co}\n    "{sys.executable}" -m pip install {pip_name}')

    print("\nUI uruchamia run_local.py interpreterem z PYTHON_EXECUTABLE.")
    print("Ustaw go na TEN interpreter, zeby UI nie trafil na inny Python:")
    print(f'  set PYTHON_EXECUTABLE={sys.executable}')
    return 1 if brak_wymaganych else 0


if __name__ == "__main__":
    raise SystemExit(main())
