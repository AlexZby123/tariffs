# Tariff Part Clustering & Workflow Studio

Projekt służy do automatycznego klastrowania części motoryzacyjnych i przemysłowych (dla procesów celnych i logistycznych Bosch / Customs & Logistics) z wykorzystaniem agentowego podejścia opartego o modele językowe (LLM) z platformy **Bosch Model Farm** oraz interfejsu graficznego (**Workflow UI**).

---

## 📌 Spis treści
1. [Przegląd projektu i cel](#przegląd-projektu-i-cel)
2. [Struktura repozytorium](#struktura-repozytorium)
3. [Architektura agentowa (`agentic/`)](#architektura-agentowa-agentic)
4. [Lokalne klastrowanie bez chmury (`run_local.py`)](#lokalne-klastrowanie-bez-chmury-run_localpy)
5. [Ewaluacja i metryki (`evaluate.py`)](#ewaluacja-i-metryki-evaluatepy)
6. [Interfejs graficzny (`workflow-ui/`)](#interfejs-graficzny-workflow-ui)
7. [Środowisko i zależności](#środowisko-i-zależności)
8. [Konfiguracja (`config.yaml`)](#konfiguracja-configyaml)
9. [Uruchamianie z CLI](#uruchamianie-z-cli)
10. [Ściąga dla agentów AI / wznowienia sesji](#ściąga-dla-agentów-ai--wznowienia-sesji)

---

## 🎯 Przegląd projektu i cel

Celem systemu jest pogrupowanie dziesiątek tysięcy rekordów części w spójne, grube **typy funkcjonalne** (np. `BALL`, `SENSOR`, `SEAL`, `SCREW`, `MOTOR`, `SPRING`, `FILTER`, `BEARING`, `PISTON`).

### Główne zasady domeny:
- **Klaster = typ funkcjonalny części** (czym ta część fizycznie i funkcjonalnie jest, a nie z czego została wykonana).
- **Główny sygnał = opis materiałowy (`MATDESC`)**: LLM wnioskuje na podstawie wiedzy dziedzinowej o produktach.
- **Kody celne (HS Code) i pole materiałowe (Material Field)** opisują surowiec/materiał (np. guma, stal, aluminium) i stanowią jedynie informację pomocniczą.
- **Deduplikacja do poziomu Product Number (PN)**: Części deduplikowane są per PN (`Product Number ACDC`), klasyfikowane tylko raz na PN, po czym wynik propagowany jest na wszystkie wiersze rekordu. Ogranicza to liczbę zapytań do LLM i drastycznie redukuje koszty.
- **Punkt odniesienia (Ground Truth)**: Wyniki porównywane są z eksperckim podziałem "Rudolfa" (zbiór `to_cluster_rudolf.xlsx` lub kolumna `Cluster NAME` w `cla.csv`).

---

## 📂 Struktura repozytorium

```text
tariff/
├── README.md                      <- Ten plik (kompletny przewodnik po projekcie)
├── search_agent/                  <- Moduł ekstrakcji i normalizacji wymiarów i wagi części
│   ├── models.py                  <- Modele Pydantic (ProductQuery, MetricDimensions)
│   ├── normalizer.py              <- Silnik konwersji jednostek (in, lbs -> mm, g, kg)
│   ├── search_provider.py         <- Pobieranie kart katalogowych (Web, Mock, Cache)
│   ├── extractor_agent.py         <- Agent LLM / Heurystyka ekstrakcji wymiarów
│   ├── pipeline.py                <- Orkiestrator workflow (pojedynczy i batch)
│   ├── cli.py                     <- Interfejs wiersza poleceń CLI
│   └── test_agent.py              <- Testy jednostkowe i integracyjne
├── agentic/                       <- Pipeline klastrowania (chmurowy + lokalny)
│   ├── ARCHITEKTURA_LOKALNA.md    <- Architektura toru lokalnego + wszystkie pomiary
│   ├── encoders.py                <- Wymienne enkodery: tfidf / MiniLM / BGE / +supcon (PyTorch)
│   ├── local_clustering.py        <- Tor LOKALNY: warstwy override -> klasyfikator -> odkrywanie
│   ├── naming.py                  <- Etap 2: nazywanie odkrytych grup (c-TF-IDF / 1 zapytanie LLM)
│   ├── sprawdz_srodowisko.py      <- Sprawdza, czy dany interpreter ma potrzebne pakiety
│   ├── config.example.yaml        <- Wzór konfiguracji LLM (skopiuj do config.yaml)
│   ├── run_local.py               <- CLI toru lokalnego (ewaluacja OOF, symulacja nowych typów)
│   ├── overrides.yaml             <- Słownik eksperta PN -> klaster (human-in-the-loop)
│   ├── agents.py                  <- Agenci LLM: Discovery, Consolidation, Classifier
│   ├── config.example.yaml        <- Wzór konfiguracji (endpointy, modele, tokeny)
│   ├── data_prep.py               <- Wczytywanie danych, deduplikacja PN, budowa "part card"
│   ├── evaluate.py                <- Ewaluacja (ARI, NMI, Hungarian, metryki parowe bez względu na nazwy)
│   ├── llm_client.py              <- Klient Bosch Model Farm (Azure OpenAI spec) + tryb mock offline
│   ├── requirements.txt           <- Zależności Pythona
│   └── run_clustering.py          <- Główny orkiestrator CLI (batching, cache, wielowątkowość)
└── workflow-ui/                   <- Panel sterowania i monitorowania w Next.js
    ├── package.json               <- Zależności Node.js (Next 16, React 19, Tailwind 4)
    ├── src/
    │   ├── app/
    │   │   ├── api/run/route.ts      <- API uruchamiania procesu pythonowego i stanu
    │   │   ├── api/workflow/route.ts <- API odczytu/zapisu config.yaml (maskowanie tokenu)
    │   │   ├── page.tsx              <- Główny widok kokpitu (konfiguracja + konsola live)
    │   │   ├── layout.tsx            <- Layout aplikacji
    │   │   └── globals.css           <- Style CSS
    │   └── lib/
    │       └── workflow.ts           <- Logika zarządzania procesami i plikiem config.yaml
    └── tsconfig.json
```

---

## 🤖 Architektura agentowa (`agentic/`)

System opiera się na 3 współpracujących rolach agentowych zdefiniowanych w `agentic/agents.py`:

```
   [Próbka danych PN]
           │
           ▼
┌────────────────────────────────────────┐
│ 1. Agent Odkrywający (Discovery)       │ -> Proponuje klastry z porcji próbek (np. 40-90 typów)
└────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────┐
│ 2. Agent Konsolidujący (Consolidation) │ -> Scala synonimy (np. BOLTS + SCREWS), usuwa duplikaty
└────────────────────────────────────────┘
           │
           ▼ [Czysta Taksonomia]
┌────────────────────────────────────────┐
│ 3. Agent Klasyfikujący (Classifier)    │ -> Przypisuje partiami (np. po 25) każdy PN do klastra
└────────────────────────────────────────┘    (Opcjonalnie: NEW: <name> lub klasyfikacja z pewnością)
```

### Tryby pracy taksonomii (`taxonomy.mode`):
1. `discover`: Agent sam odkrywa i konsoliduje taksonomię na podstawie próbki (czysty nienadzorowany clustering).
2. `seed_from_rudolf`: Użycie gotowej listy unikatowych klastrów Rudolfa jako taksonomii wyjściowej (klasyfikacja nadzorowana).
3. `fixed_list`: Ręcznie podana lista klastrów w konfiguracji.

### Reprezentacja części ("Part Card"):
Funkcja `part_card()` w `data_prep.py` formuje zwarty rekord tekstowy dla modelu:
- `desc="..."` – skonsolidowane opisy materiałowe (kluczowy sygnał),
- `hs6=...` oraz `hs_text=...` – kod i opis taryfy celnej,
- `part_name="..."`, `part_type="..."`, `material_field="..."` – dane pobrane z CaPRI,
- `bu=...` – jednostka biznesowa.

---

## 💻 Lokalne klastrowanie bez chmury (`run_local.py`)

Drugi, **w pełni lokalny** tor: bez LLM, bez tokenów, bez sieci. Na tym zbiorze
**bije tor chmurowy** (ARI 0.954 vs 0.885) i liczy się w sekundy na zwykłym CPU.

Każda część przechodzi przez trzy warstwy i dostaje w wyniku kolumnę `zrodlo`:

| warstwa | mechanizm | źródło |
|---|---|---|
| 0 | `overrides.yaml` — słownik eksperta, 100% determinizm | `override` |
| 1 | `LinearSVC` do znanej taksonomii, jeśli margines pewności ≥ próg | `model` |
| 2 | reszta → `AgglomerativeClustering(cosine)` → `NOWY_1`, `NOWY_2`… | `odkryty` |
| 3 | poprawka eksperta wraca do warstwy 0 i uczy model | — |

Grupy z warstwy 2 dostają na końcu nazwy (**Etap 2**, `naming.py`): domyślnie **jednym**
zapytaniem do LLM na wszystkie grupy naraz (zmierzone: 1 zapytanie zamiast ~41), z
automatycznym zejściem na offline'ową metodę c-TF-IDF (0 zł, 77% nazw trafia w prawdziwy
typ części), gdy chmura jest niedostępna.

Próg warstwy 1 **dobiera się sam** (z predykcji out-of-fold) tak, by trafność przyjętych
osiągnęła `--cel-trafnosci`. Domyślnie warstwa 1 przyjmuje ~83% części **bez ani jednego
błędu**, a do eksperta trafia ~17% — w tym 4 na 5 faktycznie nowych typów części.

```bash
cd agentic
python run_local.py                            # pełny przebieg + uczciwa ewaluacja OOF
python run_local.py --porownaj --sim-nowe 0.2  # porównanie torów + symulacja nowych typów
python run_local.py --encoder minilm           # bi-encoder z HuggingFace zamiast TF-IDF
python run_local.py --nazywaj llm               # Etap 2: nazwij nowe grupy 1 zapytaniem
python run_local.py --override "0204X00136=RESERVOIR CAP"   # poprawka eksperta
```

### Obsługa z przeglądarki

Zakładka **Local (hybrid)** w `workflow-ui` daje całą pętlę bez CLI:

```bash
cd workflow-ui
npm install
npm run dev      # http://localhost:3000
```

Interfejs jest po angielsku, a widok domyślny celowo minimalny: pole na klucz API,
rozwijane „Advanced settings" i przycisk uruchomienia. Enkoder (`tfidf`) i nazywanie przez
LLM są **zaszyte na stałe** — pomiary rozstrzygnęły, co jest najlepsze. Pozostałe warianty
zostały w kodzie i są dostępne z CLI (`--encoder minilm`, `--nazywaj ctfidf` itd.).

**Klucz API wklejasz w UI** — zapisuje się do `agentic/config.yaml` (plik powstaje sam,
jeśli go nie ma) i nigdy nie wraca do przeglądarki. Nazywanie grup przez LLM to **jedno
zapytanie na cały bieg**, więc może działać przy każdym klastrowaniu. Gdy token wygaśnie
albo endpoint nie odpowie, bieg **nie przerywa się** — grupy dostają nazwy offline'ową
heurystyką, a w logu pojawia się informacja dlaczego.

Tor lokalny **nie wymaga `config.yaml`** — zakładka działa od razu po `npm run dev`.

W tabeli części każdy wiersz ma pole **„Pin permanently"** — wybierasz istniejący klaster
albo wpisujesz nowy, korekty zbierają się w koszyku, a **„Save and retrain"** zapisuje je
do `overrides.yaml` i od razu przelicza model. Część przypięta przez eksperta zawsze wchodzi
do treningu, więc podobne części idą za Twoją decyzją.

> [!NOTE]
> Na Windows z condą ustaw `PYTHON_EXECUTABLE` na interpreter swojego środowiska, np.
> `set PYTHON_EXECUTABLE=C:\Users\ZBA1WZ\.conda\envs\vm\python.exe` przed `npm run dev`
> (albo na stałe w pliku `workflow-ui\.env.local`). Bez tego UI sięga po Anacondę base.
> Którego interpretera używa, widać w pierwszej linii logu po uruchomieniu.

> Pełny opis architektury, wszystkie pomiary i uzasadnienie decyzji projektowych (m.in.
> dlaczego sieć neuronowa **nie** jest domyślnym enkoderem i dlaczego do modelu idzie
> **tylko** `MATDESC`): **[`agentic/ARCHITEKTURA_LOKALNA.md`](agentic/ARCHITEKTURA_LOKALNA.md)**.

## 📊 Ewaluacja i metryki (`evaluate.py`)

Podczas ewaluacji **nazwy klastrów są ignorowane** – sprawdzana jest wyłącznie jakość grupowania rekordów:
- Wszystkie klastry modelu są mapowane na identyfikatory literowe `A, B, C, ...` wg liczby przypisanych rekordów.
- **Współprzynależność par (Pair Metrics)**:
  - `pair_precision`: ile par połączonych przez model jest rzeczywiście razem w ground-truth.
  - `pair_recall`: ile par z ground-truth model utrzymał razem.
  - `pair_f1`: średnia harmoniczna precyzji i pełności par.
  - `rand_index` oraz `ARI` (Adjusted Rand Index).
- **Metryki informacyjne**: NMI, V-measure, Homogeneity, Completeness.
- **Zgodność Hungarian**: Optymalne przyporządkowanie 1:1 za pomocą algorytmu węgierskiego (`linear_sum_assignment`).

Wyniki zapisywane są w katalogu `agentic/wyniki/<data_czas>_<provider>/`:
- `klastry_per_PN.csv` – przypisanie per numer części,
- `klastry_per_wiersz.csv` – zmapowane rekordy źródłowe,
- `legenda_klastrow.csv` – mapowanie identyfikatora literowego na nazwę i liczność,
- `ewaluacja.txt` – kompletny raport liczbowy,
- `ewaluacja_rudolf_do_llm.csv` – analiza rozbicia poszczególnych grup referencyjnych.

---

## 🖥️ Interfejs graficzny (`workflow-ui/`)

Aplikacja oparta o **Next.js 16 (App Router)** pozwala na pełną kontrolę z poziomu przeglądarki:
- **Zarządzanie konfiguracją**: Edycja parametrów modelu, temperatury, limitu tokenów, trybu taksonomii oraz aktywnych pól "part card".
- **Bezpieczeństwo**: Token `api_key` nie jest wysyłany do przeglądarki (oznaczany jedynie jako boolean `hasApiKey`).
- **Uruchamianie i podgląd na żywo**: Wyzwalanie skryptu Pythona w tle (`child_process.spawn`) oraz streaming ostatnich 300 linii logów w oknie konsoli.

---

## ⚙️ Środowisko i zależności

### Python:
Wymagany Python 3.10+ oraz biblioteki:
```bash
pip install -r agentic/requirements.txt
```

**Najpierw sprawdź, czego brakuje w Twoim środowisku** — uruchom tym interpreterem,
którego naprawdę używasz:
```bash
cd agentic
"C:\Users\ZBA1WZ\.conda\envs\pandas_excel\python.exe" sprawdz_srodowisko.py
```
Skrypt wypisze, co jest, czego brakuje i gotowe polecenia `pip install`.

Tor lokalny wymaga tylko: `pandas`, `numpy`, `scikit-learn`, `scipy`, `PyYAML`, `openpyxl`.
Pakiety `torch` (dla `--encoder tfidf+supcon`), `sentence-transformers` (dla `minilm`/`bge`)
i `openai` (tor chmurowy, `--nazywaj llm`) są importowane **leniwie** — potrzebne dopiero
przy użyciu danego trybu. `lightgbm` **nie jest już wymagany**.
> [!NOTE]
> Na tej maszynie w pełni skonfigurowane środowisko conda zawierające wszystkie pakiety (`openai`, `pandas`, `scikit-learn`, `scipy`, `openpyxl`, `pyyaml`) to:
> `C:\Users\ZBA1WZ\.conda\envs\pandas_excel\python.exe`

### Node.js:
Wymagany Node.js v20+ (na maszynie zainstalowany v24.16.0):
```bash
cd workflow-ui
npm install
```

---

## 🔧 Konfiguracja (`config.yaml`)

Aby uruchomić pipeline, w katalogu `agentic/` należy utworzyć plik `config.yaml` (na bazie `config.example.yaml`):

```yaml
provider: bosch # 'bosch' (Model Farm) lub 'mock' (offline bez tokenu)
api_key: "TWÓJ_TOKEN_BOSCH_MODEL_FARM"
base_url: "https://aoai-farm.bosch-temp.com/api"
api_version: "2025-04-01-preview"

# Rekomendowany model: Gemini 3.5 Flash (szybki, tani, wysokie ARI ~0.885)
deployment: "gemini-3.5-flash"
model: "google/gemini-3.5-flash" # Dla OpenAI puste ""

temperature: 0.0
max_tokens: 8000
request_timeout: 90
max_retries: 5

run:
  limit: null            # Ograniczenie liczby PN do testów (null = całość)
  batch_size: 25         # Liczba części w jednym zapytaniu
  concurrency: 4         # Liczba równoległych wątków
  discovery_sample: 180  # Liczba PN do próbki odkrywania taksonomii

taxonomy:
  mode: discover         # discover | seed_from_rudolf | fixed_list
  allow_new: true

features:
  use_capri: true
  part_card: [desc, hs6, rbname, parttype, material_field, bu]
```

---

## 🚀 Uruchamianie z CLI

Wszystkie polecenia uruchamiamy z katalogu `agentic/` (z aktywnym środowiskiem Pythona):

```bash
cd agentic

# 1. Test połączenia z Bosch Model Farm
python run_clustering.py --check

# 2. Test na małej próbce (np. 60 PN, tanio i szybko)
python run_clustering.py --limit 60

# 3. Test z ewaluacją na N wierszach danych źródłowych
python run_clustering.py --rows 200

# 4. Uruchomienie na pełnym zbiorze
python run_clustering.py

# 5. Nadpisanie trybu taksonomii z wiersza poleceń
python run_clustering.py --taxonomy seed_from_rudolf --no-new

# 6. Ponowna ewaluacja istniejącego pliku z wynikami
python evaluate.py wyniki/<folder>/klastry_per_wiersz.csv
```

### Uruchomienie UI:
```bash
cd workflow-ui
npm run dev
```
Interfejs dostępny jest pod adresem: `http://localhost:3000`.

---

## 🧠 Ściąga dla agentów AI / wznowienia sesji

W razie ponownego uruchomienia asystenta AI w tym projekcie:
1. **Lokalizacja repozytorium**: `C:\Users\ZBA1WZ\Documents\clustering_vm\tariff`.
2. **Interpreter Pythona**:
   - Domyślny Anaconda base nie ma `pandas` i `openai`.
   - Środowisko z kompletem zainstalowanych pakietów ML to: `C:\Users\ZBA1WZ\.conda\envs\pandas_excel\python.exe`.
   - Przed pracą zweryfikuj je: `python sprawdz_srodowisko.py` (uruchom TYM interpreterem).
   - W `workflow-ui/src/lib/workflow.ts` można ustawić zmienną środowiskową `PYTHON_EXECUTABLE="C:\\Users\\ZBA1WZ\\.conda\\envs\\pandas_excel\\python.exe"`.
3. **Klucze i autoryzacja**:
   - Tokeny API nie są commitowane w gicie (są w `agentic/config.yaml`).
   - Bosch Model Farm używa nagłówka `Authorization: Bearer <token>` z endpointem w stylu Azure OpenAI.
4. **Dane wejściowe**:
   - Skrypty oczekują plików w katalogach nadrzędnych: `../data_to_cluster/to_cluster.csv` (lub `../cla.csv`) oraz opcjonalnych plików cache CaPRI w `../wyniki_to_cluster/`.
   - Jeśli danych brakuje lub testowany jest wyłącznie przepływ logiki/UI, należy użyć konfiguracji `provider: mock`.
