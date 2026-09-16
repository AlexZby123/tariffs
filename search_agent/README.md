# 🔍 Search Agent: Product Dimension & Weight Extractor

Moduł do agentowego wyszukiwania kart katalogowych / specyfikacji technicznych online, ekstrakcji atrybutów fizycznych części (**długość, szerokość, wysokość, średnica, waga**) oraz deterministycznej normalizacji do standardowych jednostek metrycznych (SI: milimetry `mm`, gramy `g`, kilogramy `kg`, objętość `cm³`).

---

## 🎯 Rola w procesie taryfikacji i klastrowania

W procesach celnych i taryfikacyjnych (klasyfikacja kodów HS oraz grupowanie części Bosch):
1. **Reguły taryfowe HS**: Wiele pozycji taryfowych zależy bezpośrednio od wymiarów fizycznych lub masy (np. śruby powyżej/poniżej 6 mm średnicy, silniki wg mocy i gabarytów, łożyska wg średnicy zewnętrznej/wewnętrznej, uszczelki).
2. **Klastrowanie funkcjonalne**: Znajomość gabarytów i wagi pozwala rozróżnić miniaturowe czujniki/podkładki od masywnych korpusów pomp czy bloków hydraulicznych, nawet gdy opis tekstowy jest zwięzły lub niejednoznaczny.

---

## 📂 Struktura modułu

```text
search_agent/
├── __init__.py           <- Główny punkt wejścia pakietu
├── models.py             <- Modele danych Pydantic (ProductQuery, RawDimensions, MetricDimensions)
├── normalizer.py         <- Deterministyczny silnik przeliczania jednostek (in, ft, cm, lbs, oz -> mm, g, kg)
├── search_utils.py       <- Wspólne narzędzia: warianty numeru części, ocena trafności, "inteligentne" obcinanie tekstu
├── search_provider.py    <- Dostawcy kart katalogowych (Web / Catalog / Mock / Cached / Hybrid)
├── docupedia_provider.py <- Konektor do wewnętrznej bazy wiedzy Bosch Docupedia (Confluence API + Markdown)
├── extractor_agent.py    <- Agent LLM (Bosch Model Farm / Heurystyka Regex)
├── pipeline.py           <- Orkiestrator workflow (pojedyncze rekordy oraz batch)
├── cli.py                <- Interfejs wiersza poleceń (CLI)
├── test_agent.py         <- Testy jednostkowe i integracyjne
├── cache/                <- Bufor wyników na dysku (JSON, z TTL - tworzony automatycznie)
├── downloads/             <- Tymczasowe załączniki z Docupedii (czyszczone po każdym zapytaniu)
└── README.md             <- Dokumentacja modułu
```

### Zmiany w logice wyszukiwania (jeśli aktualizujesz z wcześniejszej wersji)

- **`HybridDatasheetProvider` faktycznie zatrzymuje się na pierwszym źródle z danymi** (Docupedia -> katalogi -> web), zamiast pytać wszystkie źródła zawsze - zgodnie z tym, co zawsze mówił jego docstring.
- **Warianty numeru części** (`search_utils.generate_pn_variants`) są teraz wspólne dla Docupedii/Web/Catalog i obejmują obie konwencje grupowania (1-3-3-3 i 4-3-3).
- **Cache ma TTL** (domyślnie 14 dni) i można go pominąć/odświeżyć flagami `--no-cache` / `--refresh-cache`.
- Każdy wynik ma teraz `source_type` (który dostawca odpowiedział) i `raw.extraction_method` (`extracted` / `estimated` / `regex_fallback`) - odróżnia realną ekstrakcję od domysłu LLM lub prostego parsera regexowego.

---

## 📐 Obsługiwane jednostki i normalizacja

Normalizator (`normalizer.py`) automatycznie rozpoznaje i przelicza:
- **Wymiary liniowe (długość / szerokość / wysokość / średnica)**:
  - Metryczne: `mm`, `cm`, `m`, `µm` -> przeliczane na `mm`
  - Calowe/Imperialne: `in`, `"`, `inch`, `inches`, `cal`, `cale` -> `mm` (1 in = 25.4 mm)
  - Ułamki calowe: `1 1/2 in`, `3/8"`, `1/4 in`
  - Stopy i jardy: `ft`, `'`, `feet`, `yd` -> `mm`
  - Złożone formaty gabarytowe: `182 x 35 x 30 mm`, `10 x 5 x 2 in`
- **Masa / Waga**:
  - Metryczne: `mg`, `g`, `kg`, `t` (tony) -> przeliczane na `g` i `kg`
  - Imperialne: `lb`, `lbs`, `pound`, `oz`, `ounce`
  - Złożone formaty wagowe: np. `2 lbs 4 oz` -> automatyczna suma w gramach

---

## 🚀 Przykłady użycia

### 1. Wiersz poleceń (CLI):

```bash
# Ekstrakcja z wewnętrznej bazy Docupedia
python cli.py --pn 7802376302 --title "TORQUE SENSOR" --docupedia

# Bezpośrednie przeszukiwanie stron w Docupedii
python cli.py --search-docupedia "torque sensor"

# Podgląd treści strony z Docupedii po ID w formacie Markdown
python cli.py --read-docupedia 7087925333

# Tryb hybrydowy (Docupedia -> Web -> LLM Knowledge)
python cli.py --pn 7802376302 --title "TORQUE SENSOR" --live

# Test pojedynczej części (tryb Mock - offline, testowy)
python cli.py --pn 0265005303 --mock

# Wyjście w formacie JSON
python cli.py --pn 7802376302 --title "TORQUE SENSOR" --docupedia --json

# Wzbogacenie całego pliku CSV z częściami
python cli.py --csv-in ../dane.csv --csv-out ../dane_z_wymiarami.csv --concurrency 4

# Wymuś świeże dane (zignoruj cache) dla jednej części, z logami INFO
python cli.py --pn 7802376302 --live --refresh-cache -v

# Całkowicie wyłącz warstwę cache dla tego przebiegu
python cli.py --pn 7802376302 --live --no-cache
```

### 2. Wykorzystanie w kodzie Pythona:

```python
from search_agent import DimensionWorkflow, ProductQuery

# Inicjalizacja workflow
workflow = DimensionWorkflow(use_mock=True)

# Definicja zapytania
query = ProductQuery(
    part_number="0265005303",
    brand="Bosch",
    title="Wheel speed sensor"
)

# Wykonanie
result = workflow.process(query)

if result.found:
    print(f"Długość: {result.metric.length_mm} mm")
    print(f"Szerokość: {result.metric.width_mm} mm")
    print(f"Wysokość: {result.metric.height_mm} mm")
    print(f"Masa: {result.metric.weight_g} g ({result.metric.weight_kg} kg)")
    print(f"Pewność: {result.confidence * 100:.1f}%")
    print(f"Źródło: {result.source_url} (typ: {result.source_type})")
    if result.raw.extraction_method == "estimated":
        print("UWAGA: wartości zgadnięte przez LLM (brak realnego źródła) - zweryfikuj ręcznie.")
```

---

## 🧪 Uruchomienie testów

```bash
python -m unittest test_agent.py
```
Testy pokrywają konwersję jednostek, ułamki, formaty złożone oraz integrację end-to-end z mockami.
