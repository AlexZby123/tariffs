# Architektura Lokalnego Klastrowania Części

> Stan: **zaimplementowane i zmierzone**. Wszystkie liczby w tym dokumencie pochodzą
> z `to_cluster.csv` (3581 wierszy → 1025 PN, 962 PN z etykietą Rudolfa, 77 klas)
> i są liczone **out-of-fold** — nigdy na danych, na których model się uczył.
> Odtworzenie: `python run_local.py --porownaj --sim-nowe 0.2`

## 1. Kontekst i cele

- **Problem**: chmurowe klastrowanie LLM (Bosch Model Farm) kosztuje, jest wolne i niedeterministyczne.
- **Wymagania**:
  1. 100% lokalnie, szybko, bez ciężkich lokalnych LLM (komputer ma pracować cicho).
  2. Mechanizm **human-in-the-loop** — trwałe przypisanie części do klastra i douczanie na poprawkach.
  3. Wykrywanie **nowych typów** części, nie tylko przypisywanie do istniejących.

## 2. Co zmierzyliśmy, zanim napisaliśmy kod

Pierwotny plan zakładał czyste **cluster-then-label** (embedding → klastrowanie → nazwanie grup).
Pomiary pokazały, że ta ścieżka ma niski sufit, a prawdziwe wyniki leżą gdzie indziej:

| podejście | ARI | pair_f1 |
|---|---|---|
| czyste klastrowanie: TF-IDF + agglomerative(cosine, average) | **0.598** | 0.612 |
| chmurowy LLM (Gemini 3.5 Flash, wg README) | 0.885 | – |
| **klasyfikator do znanej taksonomii (LinearSVC, OOF)** | **0.954** | **0.956** |

Czyli: mając ~950 zaetykietowanych PN, zwykły klasyfikator liniowy bije chmurowy LLM
o ~7 pkt ARI, za zero złotych i w kilka sekund. Klastrowanie bez etykiet ma sufit ~0.60.

**Wniosek, który przebudował architekturę**: etykiety Rudolfa to najcenniejszy zasób w tym
projekcie. Architektura ma z nich korzystać wszędzie tam, gdzie taksonomia jest znana,
a klastrowanie rezerwować dla tego, do czego jest naprawdę potrzebne — **odkrywania typów,
których w taksonomii jeszcze nie ma**.

### 2a. Ablacja cech — mniej znaczy więcej

| tekst wejściowy | ARI (OOF) |
|---|---|
| sam `MATDESC` | **0.948** |
| + opis HS Code | 0.934 |
| + hierarchia produktowa (subclass, statgroup, pclass, BU) | 0.917 |

Dosypywanie kontekstu **rozcieńcza** główny sygnał. Dla klastrowania bez etykiet efekt jest
dramatyczny: 0.599 → 0.304. Dlatego domyślnie do enkodera idzie **tylko `MATDESC`**.
(Kody HS i pole materiałowe opisują *surowiec*, a klaster to *typ funkcjonalny* — to dwie
różne osie, więc mieszanie ich szkodzi.)

### 2b. Sieć neuronowa — gdzie się opłaca, a gdzie nie

Douczony enkoder metryczny (SupCon, PyTorch, uczony na etykietach eksperta) **nie obronił się**:

| enkoder warstwy 1 | trafność (OOF) | ARI (OOF) | pokrycie |
|---|---|---|---|
| `tfidf` | **0.956** | **0.954** | **80.5%** |
| `tfidf+supcon` | 0.946 | 0.937 | 51.7% |

Przy pierwszym pomiarze `+supcon` wyglądał świetnie (ARI 0.991), ale to był **przeciek**:
enkoder trenował się na etykietach, które potem przewidywał. Po wymuszeniu re-treningu
enkodera osobno w każdym foldzie przewaga zniknęła. Backend został w kodzie
(`--encoder tfidf+supcon`) — przy większym zbiorze (`cla.csv`, ~15 tys. wierszy) może się
opłacić — ale **domyślny jest `tfidf`**.

Osobna, ważna obserwacja: uczenie metryczne **nie przenosi się na typy niewidziane w treningu**
(ARI 0.910 → 0.342 na symulacji nowych klas). Dlatego warstwa 2 celowo używa embeddingu
**bazowego**, nie douczonego.

## 3. Zaimplementowana architektura

```mermaid
flowchart TD
    A["MATDESC (part card)"] --> B{"WARSTWA 0<br/>overrides.yaml — PN w słowniku eksperta?"}
    B -- tak --> Z["przypisanie deterministyczne<br/>źródło: override"]
    B -- nie --> C["enkoder (wymienny): tfidf | minilm | bge | +supcon"]
    C --> D{"WARSTWA 1<br/>LinearSVC — margines >= próg?"}
    D -- tak --> E["klaster ze znanej taksonomii<br/>źródło: model"]
    D -- nie --> F["WARSTWA 2 — agglomerative(cosine)<br/>na embeddingu BAZOWYM, próg odległości"]
    F --> G["NOWY_1, NOWY_2, ... / DO_PRZEGLADU<br/>źródło: odkryty"]
    G --> N["ETAP 2 — nazwanie grup<br/>c-TF-IDF offline albo 1 zapytanie LLM"]
    N --> H
    E --> H["WARSTWA 3 — ekspert poprawia"]
    G --> H
    H -- "run_local.py --override PN=KLASTER" --> B
```

Każda część dostaje w wyniku kolumnę `zrodlo` (`override` / `model` / `odkryty`) i `pewnosc`,
więc widać, skąd wzięła się każda decyzja.

### Warstwa 1 — klasyfikator, który wie, czego nie wie

Pewność to **margines** = odstęp między najlepszą a drugą klasą w `decision_function`.
Próg nie jest zgadywany, tylko **dobierany automatycznie** z predykcji out-of-fold tak, by
trafność przyjętych osiągnęła zadany cel (`--cel-trafnosci`). Zmierzony kompromis:

| cel | pokrycie | trafność przyjętych | nowe typy wykryte | fałszywy alarm |
|---|---|---|---|---|
| 0.970 | 96.9% | 0.971 | 18.0% | 2.9% |
| 0.980 | 94.8% | 0.981 | 39.3% | 3.2% |
| 0.990 | 91.3% | 0.990 | 52.9% | 3.8% |
| 0.995 | 88.1% | 0.996 | 62.7% | 4.0% |
| **0.999 (domyślne)** | **83.1%** | **1.000** | **79.1%** | 6.3% |

Przy domyślnym progu warstwa 1 **nie myli się na tym, co przyjmuje**, a do eksperta trafia
~17% części — w tym 4 na 5 faktycznie nowych typów. Wartości stabilne (±2 pp przez 5 seedów).

> Testowaliśmy też drugą bramkę — odległość do najbliższego centroidu klasy (klasyczny
> open-set). Dawała +1 pp wykrywalności nowości za 3× więcej fałszywych alarmów, więc
> jej **nie ma** w kodzie.

### Warstwa 2 — odkrywanie nowych typów

`AgglomerativeClustering(distance_threshold, metric="cosine", linkage="average")` — bez
zgadywania liczby klastrów z góry. Próg (`--prog-odkrywania`, domyślnie 0.60) świadomie
ustawiony na **rozdrobnienie zamiast sklejania**:

| próg | wykrytych grup (prawda: 16) | pair_precision |
|---|---|---|
| 0.30 | 82 | 0.991 |
| 0.60 | 55 | 0.985 |
| 0.80 | 32 | 0.963 |

Eksperta łatwiej poprosić o scalenie dwóch czystych grupek niż o rozplątanie jednej błędnie
sklejonej. Grupy poniżej `min_licznosc_nowego` dostają etykietę `DO_PRZEGLADU`.

### Etap 2 — nazwanie odkrytych grup (`naming.py`)

Warstwa 2 umie grupować, ale nie umie nazywać — grupy wychodzą jako `NOWY_1`, `NOWY_2`.
Etap 2 zamienia numery na typy funkcjonalne. Dwie ścieżki, obie zaimplementowane:

| metoda | jak działa | koszt | trafność nazw |
|---|---|---|---|
| `ctfidf` (domyślna) | class-based TF-IDF — terminy charakterystyczne dla grupy względem pozostałych grup | **0 zł, offline** | **77.2% części / 52.0% grup** |
| `llm` | z każdej grupy 5 części najbliższych centroidowi → **jedno** zapytanie na wszystkie grupy | 1 zapytanie | do zmierzenia na Model Farm |

„Trafność nazw" = odsetek części, których proponowana nazwa trafia w ich prawdziwy typ
wg Rudolfa (kryterium tokenowe: `BALL BEARING` trafia w `BALL`, `GASKET SEAL` w `SEAL`).
Liczone na symulacji ukrytych klas, więc model nie widział tych typów w treningu.
Wynik po grupach jest niższy, bo małe grupki liczą się tak samo jak duże.

**Koszt, zmierzony:** nazwanie 31 grup = **1 zapytanie**. Tor chmurowy klasyfikuje te same
dane w partiach po 25 PN, czyli ~41 zapytań plus odkrywanie i konsolidacja taksonomii.
W tokenach różnica jest jeszcze większa: etap 2 wysyła 5 krótkich próbek na grupę
(~155 linii) zamiast pełnych part card dla wszystkich 1025 PN.

Nazwy dostają prefiks `NOWY: `, żeby nie mieszały się z zatwierdzoną taksonomią — to
**propozycje**, które ekspert akceptuje przez `--override`. Gdy dwie grupy dostaną tę samą
nazwę, dopisywany jest numer (`NOWY: BUSHING (1)`, `(2)`) — dwie osobne grupy nie mogą
zniknąć w jednym klastrze tylko dlatego, że heurystyka nazwała je tak samo.

Ścieżkę `llm` można przetestować bez tokenu: `provider: mock` w `config.yaml` obsługuje
zadanie nazywania offline.

### Wynik end-to-end (symulacja: 16 klas / 244 PN ukryte przed modelem)

| miara | wynik |
|---|---|
| nowe typy skierowane do warstwy 2 | 79.1% |
| fałszywy alarm na znanych typach | 6.3% |
| trafność warstwy 1 na tym, co przyjęła | 0.996 |
| jakość grupowania nowych typów (pair_precision) | 0.837 |
| nazwy trafiające w prawdziwy typ (etap 2, `ctfidf`) | 77.2% części |

## 3a. Etap 3 — obsługa z UI (`workflow-ui`)

Zakładka **Local (hybrid)** daje pełną pętlę human-in-the-loop z przeglądarki. Interfejs
jest po angielsku (tak jak zakładka chmurowa), podobnie jak komunikaty w oknie logu.

**Widok domyślny jest celowo minimalny** — pole na klucz API, rozwijane „Advanced settings"
i przycisk uruchomienia. Dwie rzeczy są **zaszyte na stałe**, bo pomiary rozstrzygnęły,
co jest najlepsze:

| zaszyte | dlaczego |
|---|---|
| enkoder `tfidf` | wygrał pomiar: ARI 0.954 vs 0.937 dla `tfidf+supcon` |
| nazywanie przez LLM | 1 zapytanie na cały bieg, koszt pomijalny |

Pozostałe backendy **zostały w kodzie** i są dostępne z CLI (`run_local.py --encoder minilm`,
`--nazywaj ctfidf` itd.) — zniknęły tylko z UI.

### Co jest w „Advanced settings"

- **Review workload** (`--cel-trafnosci`) — jak trafny musi być klasyfikator na tym, co
  przypisuje sam. Bezpieczniej = mniej cichych pomyłek i lepsze wykrywanie nowych typów,
  ale więcej części w kolejce. Przy domyślnym 0.999 klasyfikator nie pomylił się ani raz
  na 83% części, które przyjął.
- **Discovery threshold** (`--prog-odkrywania`) — jak daleko od siebie mogą być dwie części,
  by trafić do jednej odkrytej grupy. Dotyczy **wyłącznie** kolejki do przeglądu, nigdy
  części przypisanych przez klasyfikator.
- **Part-number limit** (`--limit`) — tylko do szybkich testów. Bierze pierwsze N PN
  w kolejności z pliku, a plik jest pogrupowany rodzinami produktów, więc mała próbka jest
  mocno przekrzywiona (np. pierwsze 300 PN zawiera 18 sztuk VM-ESP, a w całości jest ich 386).
  Przy ustawionym limicie ewaluacja na wierszach jest pomijana, bo etykiety Rudolfa są
  wyrównane pozycyjnie z pełnym plikiem.

### Klucz API

Pole w zakładce zapisuje token do `agentic/config.yaml` (zakłada plik na bazie
`config.example.yaml`, jeśli go nie ma). Klucz **nigdy nie wraca do przeglądarki** — UI
dostaje tylko flagę `llmReady`. To ten sam plik, którego używa tor chmurowy.

**Nazywanie przez LLM nie może wywrócić biegu.** Brak `config.yaml`, wygasły token, padnięty
endpoint albo niepoprawny JSON — wszystko spada na offline'owe c-TF-IDF, a użytkownik dostaje
komplet wyników, tylko z gorszymi nazwami grup. Osłonięte jest całe wywołanie, razem z częścią
sieciową (zweryfikowane na realnym błędzie połączenia).

### Reszta zakładki

- **Kafelki metryk**: trafność OOF, ARI, udział przypisanych automatycznie, liczba części
  do przeglądu. To liczby out-of-fold, nie z danych treningowych.
- **Lista klastrów** z licznością; grupy zaproponowane przez etap 2 mają ikonę iskierki.
- **Tabela części** posortowana rosnąco po marginesie pewności — najbardziej wątpliwe są
  na górze. Kolumna `Assigned by` pokazuje, która warstwa podjęła decyzję
  (`model` / `discovered` / `override`). Filtr „review queue only" zawęża do kolejki eksperta.
- **Pin permanently**: w każdym wierszu pole z podpowiedziami istniejących klastrów
  (można też wpisać zupełnie nową nazwę). Korekty zbierają się w koszyku, a przycisk
  **„Save and retrain"** zapisuje je do `overrides.yaml` i od razu przelicza model.

**Uwaga projektowa**: korekta eksperta wchodzi do treningu **zawsze**, nawet jako jedyny
przykład swojej klasy (`min_probek_klasy` jej nie dotyczy). Bez tego przypięcie części do
zupełnie nowego klastra działałoby wyłącznie dla tego jednego PN, a podobne części dalej
lądowałyby gdzie indziej — czyli model niczego by się nie nauczył.

## 4. Użycie

```bash
cd agentic

python run_local.py                          # pełny przebieg + uczciwa ewaluacja
python run_local.py --porownaj --sim-nowe 0.2   # z porównaniem torów i symulacją nowych typów
python run_local.py --encoder tfidf+supcon   # enkoder neuronowy (PyTorch, offline)
python run_local.py --encoder minilm         # bi-encoder z HuggingFace
python run_local.py --cel-trafnosci 0.98     # więcej automatyzacji, mniej odkrywania
python run_local.py --nazywaj llm            # etap 2 przez LLM (1 zapytanie) zamiast c-TF-IDF
python run_local.py --nazywaj brak           # zostaw surowe NOWY_1, NOWY_2
python run_local.py --predict-only           # użyj zapisanego modelu

# WARSTWA 3 — poprawka eksperta (trafia do overrides.yaml i uczy model)
python run_local.py --override "0204X00136=RESERVOIR CAP"
```

Wyniki lądują w `wyniki/<data>_local_<enkoder>/`: `klastry_per_PN.csv`,
`klastry_per_wiersz.csv`, `legenda_klastrow.csv`, `podsumowanie.txt`, `ewaluacja.txt`.

> `ewaluacja.txt` liczy metryki na wierszach dla modelu dotrenowanego na **wszystkich**
> etykietach — te liczby są **zawyżone** i służą tylko do porównania z historycznymi
> przebiegami chmurowymi. Miarodajna jest sekcja A w `podsumowanie.txt` (out-of-fold).

## 5. Zależności

Rdzeń (`tfidf`) wymaga tylko `pandas`, `scikit-learn`, `scipy`, `PyYAML`, `openpyxl` —
wszystko już jest w `requirements.txt`. Opcjonalnie:

```bash
pip install torch                    # backend +supcon
pip install sentence-transformers    # backendy minilm / bge (pobiera model z HuggingFace)
```

## 6. Co dalej (roadmapa)

- **[Etap 2] ✅ ZROBIONE** — `naming.py`, opis wyżej.
- **[Etap 3] ✅ ZROBIONE** — zakładka „Lokalne (hybryda)" w `workflow-ui`, opis niżej.
- **[Etap 4] `search_agent` jako ratunek**: dla części z `DO_PRZEGLADU` o enigmatycznym opisie
  wyciągnąć wymiary i wagę z kart katalogowych i doklastrować detal.
- **Walidacja na `cla.csv`**: większy zbiór (~15 tys. wierszy) — tam `+supcon` może się obronić.
  Uwaga: `dataset_train.csv` / `dataset_test.csv` **nie nadają się** do uczciwej ewaluacji —
  1166 z 1363 PN testowych występuje też w treningu (przeciek), a pliki nie zawierają `MATDESC`.

## 7. Mapa plików

| plik | rola |
|---|---|
| `encoders.py` | wymienne enkodery: `tfidf`, `st:<model>`, `+supcon` (PyTorch) |
| `local_clustering.py` | warstwy 0–3, dobór progu, zapis/odczyt modelu |
| `naming.py` | etap 2: nazywanie grup (c-TF-IDF offline / 1 zapytanie LLM) + ocena nazw |
| `wyniki/_ostatni_lokalny.json` | ostatni wynik w formacie dla UI (stała ścieżka) |
| `../workflow-ui/src/lib/local.ts` | uruchamianie toru lokalnego i zapis korekt z UI |
| `../workflow-ui/src/app/LocalPanel.tsx` | zakładka „Lokalne (hybryda)" |
| `run_local.py` | CLI, uczciwa ewaluacja OOF, symulacja nowych typów, zapis wyników |
| `overrides.yaml` | słownik eksperta PN → klaster (warstwa 0 / 3) |
| `data_prep.py` | wczytanie danych, deduplikacja do PN, part card |
| `evaluate.py` | metryki (ARI, pair_f1, NMI, Hungarian) — wspólne z torem chmurowym |
