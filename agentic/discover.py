# -*- coding: utf-8 -*-
"""
TRYB ODKRYWCZY - buduje podzial OD ZERA, bez etykiet Rudolfa.

To lokalny odpowiednik toru chmurowego (run_clustering.py): nikt nie podaje
gotowej taksonomii, system sam ma zdecydowac, jakie typy czesci istnieja.
Rozni sie tym od local_clustering.py, ktory uczy sie decyzji Rudolfa i ich
NIE zastepuje, tylko powtarza na nowych czesciach.

Przeplyw:

  1. FRAZA TYPU - z MATDESC wycinamy sam typ czesci.
     Opisy maja strukture "opis ACDC | OPIS SCND; wariant/wymiar", a czlon SCND
     przed srednikiem to praktycznie gotowa nazwa typu:
         "helical spring | SPRING; IBO2"          -> SPRING
         "rubber gasket | SEPARATING SEAL; D 25.4" -> SEPARATING SEAL
     Z 962 czesci robi sie ~237 unikalnych fraz, wiec problem taksonomii
     zwija sie do pogrupowania 237 krotkich napisow.

  2. CECHY FIZYCZNE - waga, objetosc, wartosc i gestosc na sztuke.
     Niezalezny od tekstu sygnal, w danych wypelniony w 100%:
         O-RING 0.2 g | BALL 1.4 g | SRUBA 5.5 g | SENSOR 32 g | ECU 525 g
     Rozdziela typy, ktorych opis nie rozroznia.

  3. GRUPOWANIE FRAZ w typy funkcjonalne - dwa warianty:
     local  - aglomeracja na tekscie frazy + fizyce, 100% offline
     cloud  - JEDNO zapytanie do LLM: "pogrupuj te 237 fraz".
              Wiedza o produktach, ktorej nie da sie wyliczyc z tekstu
              (BRACKET + GUIDE RING to u Rudolfa jedna grupa).

  4. PRZYPISANIE czesci do grupy jej frazy + overrides eksperta.

Zmierzone na to_cluster.csv (962 PN, 77 klas Rudolfa, k=77, BEZ etykiet):

    sam opis (stary baseline)                 ARI 0.599
    + fraza typu                              ARI 0.650
    + fizyka (waga 0.3)                       ARI 0.735
    chmura, 45 zapytan agentowych             ARI 0.885

Waga fizyki 0.3 sprawdzona na rozlacznych polowach (0.715 / 0.712), wiec nie
jest dopasowana do zbioru oceny.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import hstack as sp_hstack
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

import data_prep as dp

# --------------------------------------------------------------------------- #
#  1. FRAZA TYPU
# --------------------------------------------------------------------------- #

#: Wariantowe kody, wymiary i wypelniacze - wszystko, co opisuje KONKRETNY
#: egzemplarz zamiast typu czesci.
WZORZEC_SMIECI = re.compile(r"""(
    \b[A-Z]{1,3}[-/][A-Z0-9./]*[0-9/][A-Z0-9./]* |  # kody wariantow: A-V/T/42/TG11
                                                 #  (musi miec cyfre lub '/', zeby
                                                 #   nie zjadac nazw typu O-RING)
    \b[LDWHO]\s*=?\s*[\d.,]+\s*MM?\b         |   # wymiary: L=24,5MM, D 25.4
    \bD?\d+[.,]?\d*\s*[xX]\s*\d+[.,]?\d*\b   |   # 12.5X1.5
    \b\d{4,}\b                               |   # dlugie numery katalogowe
    \b\d+[.,]\d+\b                           |   # liczby dziesietne
    _x000D_                                  |   # artefakt eksportu Excela
    \b(ASSY|ASSEMBLY|KOMPL|KOMPLETT|STD|SE\s+SIZE)\b
)""", re.X | re.I)

FRAZA_NIEZNANA = "UNKNOWN"


def fraza_typu(opis: str) -> str:
    """Wycina z opisu materialowego sama nazwe typu czesci.

    >>> fraza_typu("helical spring | SPRING; IBO2")
    'SPRING'
    >>> fraza_typu("valve body | VALVE BODY; A-V/T/42/TG11")
    'VALVE BODY'
    """
    tekst = str(opis)
    if "|" in tekst:
        tekst = tekst.split("|")[-1]      # czlon SCND niesie typ
    tekst = tekst.split(";")[0]           # przed pierwszym srednikiem
    tekst = WZORZEC_SMIECI.sub(" ", tekst)
    tekst = re.sub(r"[^A-Za-z\- ]", " ", tekst)
    tekst = re.sub(r"\s+", " ", tekst).strip().upper()
    return tekst or FRAZA_NIEZNANA


# --------------------------------------------------------------------------- #
#  2. CECHY FIZYCZNE
# --------------------------------------------------------------------------- #

DO_GRAMOW = {"G": 1.0, "KG": 1000.0, "MG": 0.001, "T": 1e6}
KOLUMNY_FIZYCZNE = ["waga_g", "objetosc_cm3", "wartosc_eur", "gestosc"]


def _na_liczbe(s: pd.Series) -> pd.Series:
    """Parsuje liczby w formacie europejskim ('0,015' -> 0.015)."""
    return pd.to_numeric(
        s.astype(str).str.strip().str.replace(",", ".", regex=False)
         .replace({"": None, "?": None, "nan": None, "None": None}),
        errors="coerce",
    )


def cechy_fizyczne(rekordy: pd.DataFrame) -> pd.DataFrame:
    """Waga / objetosc / wartosc / gestosc na sztuke, agregowane do PN.

    Mediana, a nie srednia - pojedyncza nietypowa wysylka nie ma przesuwac
    charakterystyki czesci. Ilosci sa w PCS w calym zbiorze, wiec dzielenie
    sum przez ilosc daje wielkosci na sztuke.
    """
    df = rekordy
    waga = _na_liczbe(df["Brutto Weight Material MARA"])
    jednostka = df["Weight UoM"].astype(str).str.strip().str.upper().map(DO_GRAMOW)
    ilosc = _na_liczbe(df["Sum_Quantity_SCND"]).replace(0, np.nan)

    pom = pd.DataFrame({
        "PN": df[dp.PN_KOL].astype(str),
        "waga_g": waga * jednostka,
        "objetosc_cm3": _na_liczbe(df["Sum_Volume_cbm_SCND"]) / ilosc * 1e6,
        "wartosc_eur": _na_liczbe(df["Value_Per_Piece_SCND"]),
    })
    agg = pom.groupby("PN")[["waga_g", "objetosc_cm3", "wartosc_eur"]].median()
    agg["gestosc"] = agg["waga_g"] / agg["objetosc_cm3"]
    return agg


def _macierz_fizyczna(pn_df: pd.DataFrame) -> np.ndarray:
    """Cechy fizyczne w skali logarytmicznej, ustandaryzowane i znormalizowane.

    Log, bo waga czesci rozciaga sie przez piec rzedow wielkosci (0.2 g - 2 kg)
    i liniowo agregat hydrauliczny przytlaczalby wszystko inne.
    """
    braki = [k for k in KOLUMNY_FIZYCZNE if k not in pn_df.columns]
    if braki:
        return np.zeros((len(pn_df), 1), dtype=np.float32)
    X = np.column_stack([
        np.log10(pn_df[k].astype(float).clip(lower=1e-4)) for k in KOLUMNY_FIZYCZNE
    ])
    X = np.where(np.isfinite(X), X, np.nan)
    if np.isnan(X).all():
        return np.zeros((len(pn_df), 1), dtype=np.float32)
    # brakujace wartosci -> mediana kolumny (neutralna, nie przyciaga do zera)
    mediany = np.nanmedian(X, axis=0)
    X = np.where(np.isnan(X), mediany, X)
    return normalize(StandardScaler().fit_transform(X)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  3. PRZESTRZEN CECH
# --------------------------------------------------------------------------- #

def _tfidf_svd(teksty: Sequence[str], dim: int = 200, seed: int = 0) -> np.ndarray:
    tc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                         max_features=20000, sublinear_tf=True)
    tw = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                         max_features=10000, sublinear_tf=True)
    X = sp_hstack([tc.fit_transform(teksty), tw.fit_transform(teksty)]).tocsr()
    wymiar = max(2, min(dim, min(X.shape) - 1))
    return normalize(TruncatedSVD(wymiar, random_state=seed).fit_transform(X)).astype(np.float32)


@dataclass
class KonfiguracjaOdkrywania:
    """Parametry trybu odkrywczego. Domyslne = zmierzone optimum."""

    waga_fizyki: float = 0.30
    """Udzial cech fizycznych w przestrzeni. Zmierzone na rozlacznych polowach:
    0.0 -> ARI 0.68, 0.2 -> 0.68, 0.3 -> 0.71, 0.4 -> 0.67, 0.5 -> 0.54."""

    waga_pelnego_opisu: float = 0.20
    """Udzial pelnego MATDESC obok samej frazy typu (ratuje czesci bez '|')."""

    n_klastrow: Optional[int] = None
    """Docelowa liczba typow. None -> dobierana progiem odleglosci."""

    prog_odleglosci: float = 0.75
    """Prog aglomeracji, gdy n_klastrow is None.

    Zmierzone na rozlacznych polowach (4 losowania), grupowanie lokalne:
        0.55 -> 148 typow, ARI 0.657    0.75 -> 102 typy, ARI 0.740
        0.70 -> 119 typow, ARI 0.706    0.85 ->  76 typow, ARI 0.721
    Ponizej 0.7 taksonomia rozpada sie na setki mikro-typow (przy 0.55 az 55
    grup jednoelementowych), co zaniza wynik i zasypuje eksperta."""

    grupowanie: str = "cloud"
    """'cloud' (kilka zapytan do LLM) albo 'local' (w pelni offline)."""

    docelowo_typow: tuple[int, int] = (60, 90)
    """Ile typow funkcjonalnych ma miec taksonomia (min, max).

    To decyzja produktowa o granulacji, nie parametr do strojenia: tor chmurowy
    w tym repo celuje w 40-90. Zawezone do 60-90, bo pierwszy przebieg na Model
    Farm wylazl na 43 typy - wyraznie za grubo, skoro ekspercki podzial ma 77.
    Zbyt grube scalanie kosztuje w ARI tyle samo co zbyt drobne rozbicie."""

    seed: int = 0


def przestrzen_cech(pn_df: pd.DataFrame, cfg: KonfiguracjaOdkrywania
                    ) -> tuple[np.ndarray, pd.Series]:
    """Buduje wektory czesci: fraza typu + pelny opis + fizyka.

    Zwraca (macierz, frazy).
    """
    frazy = pn_df["MATDESC"].map(fraza_typu)
    Z_fraza = _tfidf_svd(frazy.tolist(), seed=cfg.seed)
    Z_opis = _tfidf_svd(pn_df["MATDESC"].astype(str).tolist(), seed=cfg.seed)
    Z_fiz = _macierz_fizyczna(pn_df)

    w_fiz, w_opis = cfg.waga_fizyki, cfg.waga_pelnego_opisu
    w_fraza = max(0.0, 1.0 - w_fiz - w_opis)
    return np.hstack([Z_fraza * w_fraza, Z_opis * w_opis, Z_fiz * w_fiz]), frazy


# --------------------------------------------------------------------------- #
#  4. GRUPOWANIE FRAZ W TYPY
# --------------------------------------------------------------------------- #

SYSTEM_GRUPOWANIA = (
    "You build a parts taxonomy for a customs & logistics team at Bosch. "
    "You are given NUMBERED type phrases extracted from material descriptions of "
    "automotive/industrial parts, each with the median weight of the parts it covers. "
    "Assign every phrase to a COARSE FUNCTIONAL PART TYPE - what the part IS, not what it "
    "is made of, and not its size. Merge synonyms and closely related types into one group "
    "(e.g. SCREW + BOLT + NUT + STUD + PIN belong together; BRACKET + SUPPORT + GUIDE RING "
    "+ NEEDLE BEARING belong together; CLIP + CLAMP + CABLE TIE belong together). "
    "Use SHORT, UPPERCASE group names. Do not lump together parts that do a different "
    "job just because they sound similar - a SENSOR is not an ECU, a COVER is not a SEAL."
)

#: Ile fraz w jednym zapytaniu. Odpowiedz musi zmiescic sie w budzecie tokenow -
#: przy 274 frazach naraz model byl ucinany w polowie JSON-a.
PARTIA_FRAZ = 90


def _parsuj_przypisania(obj) -> dict[int, str]:
    """Wyciaga {numer: nazwa_grupy} z odpowiedzi modelu.

    Tolerancyjnie, bo odpowiedz bywa ucieta: gdy JSON nie parsuje sie w calosci,
    ratujemy kompletne pary "numer": "grupa" wyrazeniem regularnym. Lepiej
    odzyskac 80 przypisan niz stracic cala partie.
    """
    wynik: dict[int, str] = {}

    def dodaj(k, v) -> None:
        try:
            numer = int(str(k).strip())
        except (TypeError, ValueError):
            return
        nazwa = str(v).strip().strip('"').upper()
        if nazwa:
            wynik[numer] = nazwa

    if isinstance(obj, dict):
        zrodlo = obj.get("assignments", obj.get("groups", obj))
        if isinstance(zrodlo, dict):
            for k, v in zrodlo.items():
                dodaj(k, v)
        elif isinstance(zrodlo, list):
            for poz in zrodlo:
                if isinstance(poz, dict):
                    dodaj(poz.get("i", poz.get("id")), poz.get("g", poz.get("group", "")))
    elif isinstance(obj, str):
        for k, v in re.findall(r'"(\d+)"\s*:\s*"([^"]*)"', obj):
            dodaj(k, v)
    return wynik


def grupuj_frazy_llm(frazy_z_waga: dict[str, float], client,
                     rozmiar_partii: int = PARTIA_FRAZ, verbose: bool = True,
                     docelowo: tuple[int, int] = (60, 90)) -> dict[str, str]:
    """Grupuje frazy typu przez LLM, partiami, z odpornoscia na urwana odpowiedz.

    Format odpowiedzi jest celowo zwiezly - {"1": "GROUP"} zamiast przepisywania
    calych fraz w tablicach members. Echo wszystkich 274 fraz nie miescilo sie w
    budzecie tokenow i model byl ucinany w polowie JSON-a.

    Kolejne partie dostaja liste juz utworzonych grup, zeby ich nie duplikowac.
    Partia, ktora sie nie powiedzie, jest pomijana - wywolujacy dogrupuje te
    frazy lokalnie, reszta wyniku zostaje.

    Returns:
        {fraza: nazwa_grupy} - tylko dla fraz, ktore model faktycznie przypisal.
    """
    if not frazy_z_waga:
        return {}

    frazy = sorted(frazy_z_waga)
    mapa: dict[str, str] = {}
    utworzone: list[str] = []

    for start in range(0, len(frazy), rozmiar_partii):
        partia = frazy[start:start + rozmiar_partii]
        linie = []
        for i, f in enumerate(partia, start=1):
            w = frazy_z_waga[f]
            linie.append(f"{i}. {f}" + (f"  (~{w:.1f} g)" if np.isfinite(w) else ""))

        juz = ""
        if utworzone:
            juz = ("\nGroups you already created - reuse these names whenever they fit, "
                   "instead of inventing near-duplicates:\n"
                   + ", ".join(sorted(set(utworzone))) + "\n")

        # Cel granulacji przeliczony NA TE PARTIE. Model widzi tylko jej frazy,
        # wiec globalne "40-90 grup w calym zbiorze" jest dla niego nieweryfikowalne
        # i konczylo sie zbyt grubym scalaniem (43 typy zamiast ~75).
        udzial = len(partia) / len(frazy)
        cel_min = max(2, round(docelowo[0] * udzial))
        cel_max = max(cel_min + 1, round(docelowo[1] * udzial))
        nowych = max(0, cel_min - len(set(utworzone)))

        user = (
            f"Assign each of the {len(partia)} numbered phrases below to a coarse "
            "functional part type.\n"
            f"These phrases are one slice of a larger set. Across the whole set the "
            f"taxonomy should end up with {docelowo[0]}-{docelowo[1]} groups, so this "
            f"slice should use roughly {cel_min}-{cel_max} distinct group names"
            + (f" (about {nowych} of them new)." if nowych else ".") + "\n"
            'Return ONLY compact JSON mapping every number to a group name: '
            '{"assignments": {"1": "GROUP NAME", "2": "GROUP NAME", ...}}\n'
            "Do not repeat the phrases themselves - only the numbers.\n"
            f"{juz}\nPHRASES:\n" + "\n".join(linie)
        )

        try:
            przypisania = _parsuj_przypisania(client.chat_json(SYSTEM_GRUPOWANIA, user))
        except Exception as e:  # noqa: BLE001 - jedna partia nie psuje reszty
            if verbose:
                print(f"  [discover] batch {start // rozmiar_partii + 1} failed "
                      f"({type(e).__name__}: {str(e)[:80]}) - those phrases go local")
            continue

        for numer, grupa in przypisania.items():
            if 1 <= numer <= len(partia):
                mapa[partia[numer - 1]] = grupa
                utworzone.append(grupa)

    return mapa


def grupuj_frazy_lokalnie(frazy: Sequence[str], Z_fraz: np.ndarray,
                          cfg: KonfiguracjaOdkrywania) -> dict[str, str]:
    """Aglomeracja fraz bez zadnej chmury. Nazwa grupy = jej najkrotsza fraza.

    Najkrotsza, bo to zwykle ta najbardziej ogolna ("SPRING" a nie
    "COMPRESSION SPRING OUTER").
    """
    frazy = list(frazy)
    if len(frazy) < 2:
        return {f: f for f in frazy}
    if cfg.n_klastrow:
        model = AgglomerativeClustering(n_clusters=min(cfg.n_klastrow, len(frazy)),
                                        metric="cosine", linkage="average")
    else:
        model = AgglomerativeClustering(n_clusters=None, metric="cosine",
                                        linkage="average",
                                        distance_threshold=cfg.prog_odleglosci)
    etykiety = model.fit_predict(Z_fraz)
    mapa: dict[str, str] = {}
    for grupa in set(etykiety):
        czlonkowie = [frazy[i] for i in np.where(etykiety == grupa)[0]]
        nazwa = min(czlonkowie, key=lambda f: (len(f), f))
        for f in czlonkowie:
            mapa[f] = nazwa
    return mapa


# --------------------------------------------------------------------------- #
#  5. KLASTROWACZ ODKRYWCZY
# --------------------------------------------------------------------------- #

ZRODLO_ODKRYTE = "discovered"
ZRODLO_OVERRIDE = "override"
ETYKIETA_PRZEGLAD = "NEEDS_REVIEW"


@dataclass
class DiagnostykaOdkrywania:
    n_czesci: int = 0
    n_fraz: int = 0
    n_grup: int = 0
    grupowanie: str = ""
    n_fraz_z_llm: int = 0
    n_fraz_lokalnie: int = 0
    n_fraz_od_eksperta: int = 0
    n_zapytan: int = 0
    pewnosc_mediana: float = 0.0
    n_do_przegladu: int = 0


class KlastrowaczOdkrywczy:
    """Buduje taksonomie od zera i przypisuje do niej czesci.

    W przeciwienstwie do local_clustering.HybrydowyKlasyfikator NIE widzi
    etykiet Rudolfa. Jedyna wiedza ekspercka, z ktorej korzysta, to
    overrides.yaml - poprawki, ktore uzytkownik sam naniosl.
    """

    def __init__(self, cfg: Optional[KonfiguracjaOdkrywania] = None):
        self.cfg = cfg or KonfiguracjaOdkrywania()
        self.fraza_do_grupy: dict[str, str] = {}
        self.centroidy: dict[str, np.ndarray] = {}
        self.prog_pewnosci: float = 0.0
        self.diagnostyka = DiagnostykaOdkrywania()

    # ------------------------------------------------------------------ fit --
    def fit(self, pn_df: pd.DataFrame, client=None, overrides: Optional[dict] = None,
            verbose: bool = True) -> "KlastrowaczOdkrywczy":
        cfg = self.cfg
        pn_df = pn_df.reset_index(drop=True)
        Z, frazy = przestrzen_cech(pn_df, cfg)
        unikalne = sorted(set(frazy))
        if verbose:
            print(f"  [discover] {len(pn_df)} parts -> {len(unikalne)} unique type phrases")

        # --- grupowanie fraz ---
        mapa: dict[str, str] = {}
        if cfg.grupowanie == "cloud" and client is not None:
            wagi = self._mediany_wag(pn_df, frazy, unikalne)
            try:
                zapytan_przed = getattr(client, "calls", 0)
                mapa = grupuj_frazy_llm(wagi, client, verbose=verbose,
                                        docelowo=cfg.docelowo_typow)
                self.diagnostyka.n_fraz_z_llm = len(mapa)
                self.diagnostyka.n_zapytan = getattr(client, "calls", 0) - zapytan_przed
                if verbose:
                    n = self.diagnostyka.n_zapytan
                    print(f"  [discover] LLM grouped {len(mapa)}/{len(unikalne)} phrases "
                          f"into {len(set(mapa.values()))} types "
                          f"({n} request{'s' if n != 1 else ''})")
            except Exception as e:  # noqa: BLE001
                if verbose:
                    print(f"  [discover] cloud grouping failed "
                          f"({type(e).__name__}: {str(e)[:100]})")
                    print("  [discover] grouping locally instead, fully offline")
                mapa = {}

        # frazy pominiete przez LLM (albo tryb local) dogrupowujemy lokalnie
        brakujace = [f for f in unikalne if f not in mapa]
        if brakujace:
            Z_fraz = _tfidf_svd(brakujace, seed=cfg.seed)
            mapa.update(grupuj_frazy_lokalnie(brakujace, Z_fraz, cfg))
            self.diagnostyka.n_fraz_lokalnie = len(brakujace)
            if verbose and self.diagnostyka.n_fraz_z_llm:
                print(f"  [discover] {len(brakujace)} phrases grouped locally as a fallback")

        # --- WARSTWA 3: korekty eksperta ucza CALE frazy ---
        # Jesli ekspert przypial czesc do klastra, to czesci o tej samej frazie
        # typu prawie na pewno naleza tam samo. Jedna poprawka naprawia wiele
        # czesci - o to chodzi w douczaniu.
        if overrides:
            mapa, ile = self._zastosuj_korekty(mapa, pn_df, frazy, overrides)
            self.diagnostyka.n_fraz_od_eksperta = ile
            if verbose and ile:
                print(f"  [discover] {ile} phrases remapped by expert corrections")

        self.fraza_do_grupy = mapa
        grupy = pn_df.index.map(lambda i: mapa.get(frazy.iloc[i], FRAZA_NIEZNANA))
        self._policz_centroidy(Z, np.asarray(grupy))

        pewnosc = self._pewnosc(Z, np.asarray(grupy))
        self.prog_pewnosci = float(np.quantile(pewnosc, 0.15)) if len(pewnosc) else 0.0

        d = self.diagnostyka
        d.n_czesci, d.n_fraz = len(pn_df), len(unikalne)
        d.n_grup = len(set(mapa.values()))
        d.grupowanie = cfg.grupowanie
        d.pewnosc_mediana = float(np.median(pewnosc)) if len(pewnosc) else 0.0
        if verbose:
            print(f"  [discover] taxonomy: {d.n_grup} functional part types")
        return self

    @staticmethod
    def _mediany_wag(pn_df, frazy, unikalne) -> dict[str, float]:
        """Mediana wagi na fraze - kontekst dla LLM (kulka 1 g vs agregat 650 g)."""
        if "waga_g" not in pn_df.columns:
            return {f: float("nan") for f in unikalne}
        pom = pd.DataFrame({"f": frazy.values, "w": pn_df["waga_g"].values})
        med = pom.groupby("f")["w"].median()
        return {f: float(med.get(f, float("nan"))) for f in unikalne}

    @staticmethod
    def _zastosuj_korekty(mapa, pn_df, frazy, overrides) -> tuple[dict, int]:
        """Przemapowuje fraze, gdy WSZYSTKIE korekty dla niej wskazuja ten sam klaster.

        Warunek zgodnosci jest celowy: gdy ekspert przypial dwie czesci o tej
        samej frazie do roznych klastrow, fraza jest wieloznaczna i nie wolno
        jej przemapowac globalnie - zostaja wtedy same przypiecia per PN.
        """
        pn_do_frazy = dict(zip(pn_df["PN"].astype(str), frazy))
        wg_frazy: dict[str, set] = {}
        for pn, klaster in overrides.items():
            f = pn_do_frazy.get(str(pn))
            if f:
                wg_frazy.setdefault(f, set()).add(str(klaster).strip())
        ile = 0
        for f, klastry in wg_frazy.items():
            if len(klastry) == 1:
                mapa[f] = next(iter(klastry))
                ile += 1
        return mapa, ile

    def _policz_centroidy(self, Z: np.ndarray, grupy: np.ndarray) -> None:
        self.centroidy = {}
        for g in set(grupy):
            srodek = Z[grupy == g].mean(axis=0)
            norma = np.linalg.norm(srodek)
            self.centroidy[g] = srodek / norma if norma else srodek

    def _pewnosc(self, Z: np.ndarray, grupy: np.ndarray) -> np.ndarray:
        """Podobienstwo kosinusowe do centroidu wlasnej grupy (0-1, wyzej = pewniej)."""
        out = np.zeros(len(Z))
        for i, g in enumerate(grupy):
            c = self.centroidy.get(g)
            if c is None:
                continue
            norma = np.linalg.norm(Z[i])
            out[i] = float(Z[i] @ c / norma) if norma else 0.0
        return out

    # -------------------------------------------------------------- predict --
    def predict(self, pn_df: pd.DataFrame, overrides: Optional[dict] = None,
                verbose: bool = True) -> pd.DataFrame:
        """Przypisuje klastry. Kolumny wyjsciowe jak w torze klasyfikujacym."""
        out = pn_df.reset_index(drop=True).copy()
        Z, frazy = przestrzen_cech(out, self.cfg)
        grupy = np.array([self.fraza_do_grupy.get(f, FRAZA_NIEZNANA) for f in frazy],
                         dtype=object)
        pewnosc = self._pewnosc(Z, grupy)
        zrodla = np.full(len(out), ZRODLO_ODKRYTE, dtype=object)

        # Czesc niepewna NIE traci swojej propozycji - dostaje tylko flage.
        # Wczesniej jej nazwa byla podmieniana na NEEDS_REVIEW, co szkodzilo
        # dwa razy: ekspert w kolejce widzial "NEEDS_REVIEW" zamiast "raczej
        # SEAL, ale slabo", a w ewaluacji 15% czesci ladowalo w jednym wielkim
        # worku, co zanizalo ARI o ponad 0.2 wzgledem toru chmurowego, ktory
        # zgaduje KAZDA czesc. Teraz i my zgadujemy kazda, a niepewnosc jest
        # osobna kolumna.
        do_przegladu = pewnosc < self.prog_pewnosci

        # WARSTWA 0 nadpisuje wszystko i zdejmuje flage przegladu
        if overrides:
            trafione = out["PN"].astype(str).map(overrides)
            ma = trafione.notna().values
            grupy[ma] = trafione[ma].values
            zrodla[ma] = ZRODLO_OVERRIDE
            pewnosc[ma] = 1.0
            do_przegladu[ma] = False

        self.diagnostyka.n_do_przegladu = int(do_przegladu.sum())
        if verbose:
            print(f"  [discover] flagged for review: {self.diagnostyka.n_do_przegladu} parts "
                  f"(confidence < {self.prog_pewnosci:.2f}) - they keep their proposed type")

        out["cluster_name"] = grupy
        out["source"] = zrodla
        out["confidence"] = pewnosc
        out["needs_review"] = do_przegladu
        out["type_phrase"] = frazy.values
        return out
