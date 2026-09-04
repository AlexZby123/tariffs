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
├── search_provider.py    <- Dostawcy kart katalogowych (Web / Mock / Cached)
├── extractor_agent.py    <- Agent LLM (Bosch Model Farm / Heurystyka Regex)
├── pipeline.py           <- Orkiestrator workflow (pojedyncze rekordy oraz batch)
├── cli.py                <- Interfejs wiersza poleceń (CLI)
├── test_agent.py         <- Testy jednostkowe i integracyjne
└── README.md             <- Dokumentacja modułu
```

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
# Test pojedynczej części (tryb Mock - offline, testowy)
python cli.py --pn 0265005303 --mock

# Test z podaniem marki i opisu
python cli.py --pn 0445110189 --brand Bosch --title "Common rail injector" --mock

# Wyjście w formacie JSON
python cli.py --pn 0265005303 --mock --json

# Tryb Live (wyszukiwanie w internecie)
python cli.py --pn 0265005303 --live

# Wzbogacenie całego pliku CSV z częściami
python cli.py --csv-in ../dane.csv --csv-out ../dane_z_wymiarami.csv --mock --concurrency 4
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
    print(f"Źródło: {result.source_url}")
```

---

## 🧪 Uruchomienie testów

```bash
python -m unittest test_agent.py
```
Testy pokrywają konwersję jednostek, ułamki, formaty złożone oraz integrację end-to-end z mockami.
