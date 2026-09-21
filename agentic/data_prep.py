# -*- coding: utf-8 -*-
"""
Przygotowanie danych dla agentow LLM.

- wczytuje data_to_cluster/to_cluster.csv
- (opcjonalnie) docza Material Field z CaPRI (cache pickle z wynikow analizy)
- deduplikuje do poziomu NUMERU PRODUKTU (jak klastrowal Rudolf)
- buduje zwiezle "part card" (opis + HS + Material Field + BU) na PN

Rudolf klastruje per Product Number, wiec klasyfikujemy raz na PN,
a wynik rozpropagowujemy na wszystkie wiersze -> mniej zapytan LLM.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KATALOG = Path(__file__).parent
BASE = KATALOG.parent
PLIK_CSV = BASE / "data_to_cluster" / "to_cluster.csv"
PLIK_RUDOLF = BASE / "data_to_cluster" / "to_cluster_rudolf.xlsx"
PLIK_CLA = BASE / "cla.csv"
CACHE_CAPRI = BASE / "wyniki_to_cluster" / "_cache_capri_pn.pkl"
CACHE_DESCPLUS = BASE / "wyniki_to_cluster" / "_cache_descplus_pn10.pkl"  # KROK 2 (build_descplus.py)

# Zrodla danych: nazwa -> (plik, kolumna z ground-truth w tym samym pliku lub None)
ZRODLA = {
    "to_cluster": (PLIK_CSV, None),        # etykiety osobno (to_cluster_rudolf.xlsx)
    "cla": (PLIK_CLA, "Cluster NAME"),     # etykiety w tym samym pliku (uzyte TYLKO do oceny)
}

PN_KOL = "Product Number ACDC"
PN10_KOL = "Product Number 10 ACDC"

# Kolumny doczytywane z cache CaPRI (per PN10). RB part name / Part Type sa
# w cache, ale w cla.csv sa PUSTE - dlatego bierzemy je z cache, nie z pliku.
CAPRI_KOLS = ["Matgrp", "Material group EN Text", "Material Field Desc",
              "Part Type", "RB part name"]

# Domyslny zestaw cech "part card" (zachowanie sprzed ablacji cech).
# Tokeny obslugiwane przez part_card(): desc, descplus, hs6, hstext, rbname,
# parttype, hierarchy (subclass+statgroup+pclass), material_field, bu.
DOMYSLNE_FEATURES = ("desc", "hs6", "material_field", "bu")


def _fillna(s: pd.Series) -> pd.Series:
    return s.astype(str).fillna("").replace({"nan": "", "None": "", "?": ""})


def _mode(s: pd.Series) -> str:
    s = _fillna(s)
    s = s[s.str.strip() != ""]
    return s.value_counts().index[0] if len(s) else ""


def wczytaj_rekordy(use_capri: bool = True, dataset: str = "to_cluster",
                    drop_tbd: bool = False, use_descplus: bool = False) -> pd.DataFrame:
    """Zwraca wszystkie wiersze wybranego zrodla z kolumnami pomocniczymi.

    dataset: 'to_cluster' (data_to_cluster/to_cluster.csv) albo 'cla' (cla.csv).
    Dla 'cla' dodaje kolumne 'GT' = Cluster NAME - uzywana WYLACZNIE do oceny,
    NIE trafia do promptu LLM. drop_tbd=True usuwa rekordy z etykieta 'tbd'.
    use_descplus=True docza kolumne DESCPLUS z cache (KROK 2, per PN10).
    """
    if dataset not in ZRODLA:
        raise ValueError(f"Nieznany dataset: {dataset} (dozwolone: {list(ZRODLA)})")
    plik, kol_gt = ZRODLA[dataset]
    df = pd.read_csv(plik, sep=";", encoding="cp1252", low_memory=False, dtype=str)
    df.columns = df.columns.str.strip()

    if kol_gt:
        df["GT"] = df[kol_gt].astype(str).str.strip()
        if drop_tbd:
            df = df[~df["GT"].str.lower().eq("tbd")].reset_index(drop=True)

    df["MATDESC"] = (_fillna(df["Material Description ACDC"]) + " | "
                     + _fillna(df["Material Description SCND"])).str.strip(" |")

    # cla.csv ma wlasne, PUSTE kolumny 'RB part name'/'Category' - usuwamy je,
    # by nie kolidowaly (merge_x/_y) z danymi z cache CaPRI.
    df = df.drop(columns=[c for c in CAPRI_KOLS if c in df.columns], errors="ignore")
    for kol in CAPRI_KOLS:
        df[kol] = ""
    if use_capri and CACHE_CAPRI.exists():
        capri = pd.read_pickle(CACHE_CAPRI)
        df["PN10"] = df[PN10_KOL].astype(str).str.strip()
        df = df.drop(columns=CAPRI_KOLS)
        df = df.merge(capri[["PN10"] + CAPRI_KOLS], on="PN10", how="left")
        df = df.drop(columns="PN10")  # klucz pomocniczy do merge - dalej niepotrzebny

    # KROK 2: dodatkowe warianty opisu z pelnego DALI (per PN10, juz oczyszczone)
    df["DESCPLUS"] = ""
    if use_descplus and CACHE_DESCPLUS.exists():
        dpc = pd.read_pickle(CACHE_DESCPLUS)  # dict: PN10 -> "term1 ; term2 ; ..."
        df["DESCPLUS"] = df[PN10_KOL].astype(str).str.strip().map(dpc).fillna("")
    return df


def deduplikuj_do_pn(df: pd.DataFrame) -> pd.DataFrame:
    """Jeden wiersz na Product Number - reprezentatywny 'part card'."""
    rek = []
    for pn, g in df.groupby(PN_KOL, sort=False):
        opisy = sorted({x for x in g["MATDESC"] if x and x.strip()})
        def m(kol: str) -> str:
            return _mode(g[kol]) if kol in g else ""
        rek.append({
            "PN": pn,
            "MATDESC": " ; ".join(opisy)[:400] if opisy else "(brak opisu)",
            "HS6": _mode(g["HS Code First 6 ACDC"]),
            "HS6_text": m("HS Code First 6 Text ACDC"),
            "HS_text": m("HS Code Text ACDC"),            # pelny opis HS (~100% wypelnienia)
            "material_field": m("Material Field Desc"),   # CaPRI: pole materialowe
            "rb_part_name": m("RB part name"),            # CaPRI: czysta nazwa czesci
            "part_type": m("Part Type"),                  # CaPRI: typ czesci
            "subclass": m("PRDH_ProductSubclassDescription"),
            "pdcl": m("PDCL_Desc_SCND"),
            "descplus": m("DESCPLUS"),                     # KROK 2: warianty opisu z DALI
            "BU": _mode(g["BU SCND"]),
            "statgroup": m("PRDH_StatisticGroupDescription"),
            "n_wierszy": len(g),
        })
    return pd.DataFrame(rek).reset_index(drop=True)


def part_card(row: pd.Series, use_capri: bool = True, feats=None) -> str:
    """Zwiezly opis czesci dla LLM. Klucz: opis materialu (glowny sygnal).

    feats: iterowalne tokenow cech do wlaczenia (patrz DOMYSLNE_FEATURES).
    None -> zachowanie domyslne. 'desc' trzymamy zawsze jako pierwsze pole,
    bo agents.klasyfikuj_partie rozpoznaje karte po prefiksie 'desc='.
    """
    feats = set(DOMYSLNE_FEATURES if feats is None else feats)
    czesci: list[str] = []
    if "desc" in feats:
        czesci.append(f'desc="{row.get("MATDESC", "")}"')
    if "descplus" in feats and row.get("descplus"):
        czesci.append(f'alt_desc="{str(row["descplus"])[:160]}"')
    if "rbname" in feats and row.get("rb_part_name"):
        czesci.append(f'part_name="{row["rb_part_name"]}"')
    if "parttype" in feats and row.get("part_type"):
        czesci.append(f'part_type="{row["part_type"]}"')
    if "hs6" in feats and row.get("HS6"):
        hs = f'hs6={row["HS6"]}'
        if row.get("HS6_text"):
            hs += f' ({str(row["HS6_text"])[:40]})'
        czesci.append(hs)
    if "hstext" in feats and row.get("HS_text"):
        czesci.append(f'hs_text="{str(row["HS_text"]).strip()[:60]}"')
    if "hierarchy" in feats:
        if row.get("subclass"):
            czesci.append(f'subclass="{row["subclass"]}"')
        if row.get("statgroup"):
            czesci.append(f'statgroup="{row["statgroup"]}"')
        if row.get("pdcl"):
            czesci.append(f'pclass="{row["pdcl"]}"')
    if "material_field" in feats and use_capri and row.get("material_field"):
        czesci.append(f'material_field="{row["material_field"]}"')
    if "bu" in feats and row.get("BU"):
        czesci.append(f'bu={row["BU"]}')
    if not czesci:  # awaryjnie zawsze cos zwroc
        czesci.append(f'desc="{row.get("MATDESC", "")}"')
    return " | ".join(czesci)


def wczytaj_etykiety_rudolfa() -> pd.Series:
    """Etykieta Rudolfa per wiersz (TYLKO do ewaluacji - nie podawac LLM)."""
    r = pd.read_excel(PLIK_RUDOLF, sheet_name="default_1", dtype=str)
    r.columns = r.columns.str.strip()
    return r["Cluster NAME"].astype(str).str.strip()


#: Kolumny, ktorych pipeline dotyka po wczytaniu. Brakujace zakladamy puste -
#: bez nich dziala, tylko slabiej (np. bez wagi nie ma cech fizycznych).
KOLUMNY_OPCJONALNE = [
    "MAT_LANE_YM", "HS Code First 6 ACDC", "HS Code First 6 Text ACDC",
    "HS Code Text ACDC", "BU SCND", "PRDH_ProductSubclassDescription",
    "PRDH_StatisticGroupDescription", "PDCL_Desc_SCND",
    "Brutto Weight Material MARA", "Weight UoM", "Sum_Quantity_SCND",
    "Sum_Volume_cbm_SCND", "Value_Per_Piece_SCND",
]

#: Proby wczytania CSV: rozne separatory i kodowania spotykane w eksportach.
WARIANTY_CSV = [
    {"sep": ";", "encoding": "cp1252"},
    {"sep": ";", "encoding": "utf-8-sig"},
    {"sep": ",", "encoding": "utf-8-sig"},
    {"sep": "\t", "encoding": "utf-8-sig"},
]


def wczytaj_plik_wejsciowy(sciezka: Path) -> pd.DataFrame:
    """Wczytuje DOWOLNY plik z czesciami do poklastrowania (.csv / .xlsx).

    Wymaga tylko numeru czesci i choc jednego opisu materialowego - cala reszta
    kolumn jest opcjonalna i zakladana pusta, zeby plik z innego eksportu nie
    wywracal calego przebiegu. Etykiety NIE sa potrzebne: to plik do
    poklastrowania, a nie do uczenia.
    """
    sciezka = Path(sciezka)
    if not sciezka.exists():
        raise FileNotFoundError(f"Nie ma pliku: {sciezka}")

    if sciezka.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(sciezka, sheet_name=0, dtype=str)
    else:
        df, awaryjny = None, None
        for wariant in WARIANTY_CSV:
            try:
                kand = pd.read_csv(sciezka, dtype=str, low_memory=False, **wariant)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            awaryjny = awaryjny if awaryjny is not None else kand
            # jedna kolumna = najpewniej zly separator, probujemy dalej
            if kand.shape[1] > 1:
                df = kand
                break
        # Zaden wariant nie dal wielu kolumn: bierzemy pierwszy, ktory sie
        # sparsowal. Dzieki temu sprawdzenie kolumn nizej powie, CZEGO brakuje,
        # zamiast zwracac mylace "nie udalo sie wczytac".
        df = df if df is not None else awaryjny
        if df is None:
            raise ValueError(
                f"Nie udalo sie wczytac {sciezka.name}. Obslugiwane: .xlsx albo CSV "
                f"z separatorem ';', ',' lub tabulatorem."
            )

    df.columns = df.columns.str.strip()
    if PN_KOL not in df.columns:
        raise ValueError(
            f"Plik {sciezka.name} nie ma kolumny '{PN_KOL}'. "
            f"Znalezione kolumny: {list(df.columns)[:10]}"
        )
    opisy = [k for k in ("Material Description ACDC", "Material Description SCND")
             if k in df.columns]
    if not opisy:
        raise ValueError(
            f"Plik {sciezka.name} nie ma zadnej kolumny z opisem materialowym "
            f"('Material Description ACDC' albo 'Material Description SCND')."
        )

    df["MATDESC"] = _fillna(df[opisy[0]])
    if len(opisy) == 2:
        df["MATDESC"] = (df["MATDESC"] + " | " + _fillna(df[opisy[1]])).str.strip(" |")

    for kol in KOLUMNY_OPCJONALNE + CAPRI_KOLS + ["DESCPLUS"]:
        if kol not in df.columns:
            df[kol] = ""
    return df.reset_index(drop=True)


#: Dodatkowy korpus Rudolfa: etykiety w tym samym pliku (kolumna Cluster NAME).
#: 1615 PN / 106 klas, z czego 1112 PN i ~36 nazw klastrow nie wystepuje w
#: to_cluster.csv. cla 1.xlsx i cla_weryfikacja.xlsx maja identyczna zawartosc.
PLIK_CLA_XLSX = KATALOG / "cla_bez_tbd.xlsx"
ARKUSZ_CLA_XLSX = "default_1"


def wczytaj_dodatkowe_etykiety(sciezka: Path | None = None) -> pd.DataFrame:
    """Dodatkowy zbior czesci opisanych przez Rudolfa, zagregowany do PN.

    Zwraca ramke z kolumnami PN / MATDESC / y. Pusta ramka, gdy pliku nie ma -
    brak dodatkowego korpusu nie jest bledem, tylko mniejszym pokryciem.
    """
    sciezka = sciezka or PLIK_CLA_XLSX
    if not sciezka.exists():
        return pd.DataFrame(columns=["PN", "MATDESC", "y"])

    df = pd.read_excel(sciezka, sheet_name=ARKUSZ_CLA_XLSX, dtype=str)
    df.columns = df.columns.str.strip()
    df["y"] = df["Cluster NAME"].astype(str).str.strip()
    df = df[~df["y"].str.lower().isin(["tbd", "nan", "none", ""])]
    df["MATDESC"] = (_fillna(df["Material Description ACDC"]) + " | "
                     + _fillna(df["Material Description SCND"])).str.strip(" |")

    agg = df.groupby(df[PN_KOL].astype(str)).agg(
        MATDESC=("MATDESC", lambda s: " ; ".join(sorted({x for x in s if x.strip()}))[:400]),
        y=("y", lambda s: s.value_counts().index[0]),
    ).reset_index().rename(columns={PN_KOL: "PN"})
    return agg[["PN", "MATDESC", "y"]]


def wczytaj_etykiety_cla() -> list[str]:
    """Unikatowe nazwy klastrow z cla.csv (Cluster NAME, bez 'tbd').

    Uzywane w trybie taksonomii 'seed_from_rudolf' dla datasetu 'cla' - LLM
    klasyfikuje do gotowej listy nazw (de facto klasyfikacja nadzorowana =
    sufit zgodnosci). Etykiety trafiaja do promptu jako TAKSONOMIA.
    """
    plik, kol = ZRODLA["cla"]
    df = pd.read_csv(plik, sep=";", encoding="cp1252", low_memory=False, dtype=str)
    df.columns = df.columns.str.strip()
    vals = df[kol].astype(str).str.strip()
    return sorted({v for v in vals if v and v.lower() not in ("tbd", "nan", "none")})
