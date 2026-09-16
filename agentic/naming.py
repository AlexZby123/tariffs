# -*- coding: utf-8 -*-
"""
ETAP 2: nazywanie grup odkrytych przez warstwe 2 (NOWY_1, NOWY_2, ...).

Warstwa 2 w local_clustering.py grupuje czesci, ktorych warstwa 1 nie umiala
pewnie przypisac, ale nie potrafi nadac im nazwy - dostaja numery. Ten modul
zamienia numery na czytelne nazwy typu funkcjonalnego.

Dwie sciezki:

  ctfidf  (domyslna, 100% offline, bez LLM)
      Class-based TF-IDF: opisy czesci z jednej grupy sklejane sa w jeden
      dokument, a nastepnie wazone wzgledem pozostalych grup. Terminy
      charakterystyczne DLA TEJ grupy (a nie czeste wszedzie) daja nazwe.
      Zero kosztow, deterministyczne, dziala bez sieci.

  llm     (jedno zbiorcze zapytanie dla WSZYSTKICH grup naraz)
      Z kazdej grupy bierzemy kilka czesci najblizszych centroidowi i wysylamy
      je w JEDNYM promptcie. Dla ~30 grup to jedno zapytanie zamiast tysiecy -
      redukcja kosztow API o >99% wzgledem klasyfikowania kazdej czesci osobno.

Nazwy sa PROPOZYCJAMI - dostaja prefiks "NOWY: ", zeby nie mylily sie z
zatwierdzona taksonomia. Ekspert akceptuje je przez run_local.py --override.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PREFIKS_PROPOZYCJI = "NOWY: "

#: Slowa zbyt ogolne, by odroznic jeden typ czesci od drugiego.
STOP_TECHNICZNE = {
    "assy", "assembly", "part", "parts", "komplett", "kompl", "set", "kit",
    "for", "with", "and", "the", "of", "type", "size", "new", "std", "version",
    "left", "right", "front", "rear", "upper", "lower", "inner", "outer",
    "black", "white", "red", "blue", "green", "grey", "gray",
    "mm", "cm", "kg", "pcs", "dia", "nom", "max", "min", "ref",
}

#: Token musi miec >=3 znaki i zawierac litere - odsiewa kody typu "A-V/T/42".
WZORZEC_TOKENU = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _tokenizuj(tekst: str) -> list[str]:
    """Slowa znaczace z opisu czesci (bez kodow, wymiarow i ogolnikow)."""
    return [t.lower() for t in WZORZEC_TOKENU.findall(str(tekst))
            if t.lower() not in STOP_TECHNICZNE]


# --------------------------------------------------------------------------- #
#  Sciezka 1: c-TF-IDF (offline)
# --------------------------------------------------------------------------- #

def nazwij_ctfidf(grupy: dict[str, Sequence[str]], n_slow: int = 2) -> dict[str, str]:
    """Nadaje nazwy grupom metoda class-based TF-IDF.

    Args:
        grupy: {identyfikator_grupy: lista opisow czesci}
        n_slow: ile terminow skleic w nazwe

    Returns:
        {identyfikator_grupy: "NOWY: NAZWA"}
    """
    if not grupy:
        return {}

    licznosci = {g: Counter(t for opis in opisy for t in _tokenizuj(opis))
                 for g, opisy in grupy.items()}
    # w ilu grupach wystepuje dany termin (df na poziomie GRUPY, nie dokumentu)
    df_terminu: Counter = Counter()
    for c in licznosci.values():
        df_terminu.update(set(c))

    n_grup = len(grupy)
    nazwy: dict[str, str] = {}
    for g, c in licznosci.items():
        suma = sum(c.values()) or 1
        # c-TF-IDF wg BERTopic: tf w klasie * log(1 + N_klas / df_terminu)
        punkty = {t: (n / suma) * np.log1p(n_grup / df_terminu[t]) for t, n in c.items()}
        # jeden wyraz opisujacy typ waze wiecej niz powtorzony wariant tego samego rdzenia
        najlepsze: list[str] = []
        for termin, _ in sorted(punkty.items(), key=lambda kv: -kv[1]):
            if any(termin.startswith(w[:4]) or w.startswith(termin[:4]) for w in najlepsze):
                continue  # ten sam rdzen, np. "sealing" po "seal"
            najlepsze.append(termin)
            if len(najlepsze) >= n_slow:
                break
        nazwy[g] = PREFIKS_PROPOZYCJI + (" ".join(najlepsze).upper() or "NIEROZPOZNANE")
    return nazwy


# --------------------------------------------------------------------------- #
#  Sciezka 2: jedno zbiorcze zapytanie do LLM
# --------------------------------------------------------------------------- #

SYSTEM_NAZYWANIA = (
    "You name clusters of automotive/industrial parts for a customs & logistics team at Bosch. "
    "A cluster name = the FUNCTIONAL PART TYPE (what the part IS), e.g. BALL, SENSOR, SEAL, "
    "SCREW, MOTOR, SPRING, FILTER, PISTON, BEARING. "
    "Judge from the material DESCRIPTION using real-world product knowledge. "
    "Use SHORT, UPPERCASE names (1-3 words). Prefer COARSE functional types over sub-types. "
    "Never invent a name that is not supported by the sample descriptions."
)


def probki_przy_centroidzie(opisy: Sequence[str], Z: np.ndarray,
                            n: int = 5) -> list[str]:
    """Wybiera n opisow najblizszych centroidowi grupy - najbardziej typowe czesci."""
    if len(opisy) <= n:
        return list(opisy)
    centroid = Z.mean(axis=0)
    norma = np.linalg.norm(centroid)
    if norma == 0:
        return list(opisy[:n])
    podobienstwo = Z @ (centroid / norma)
    return [opisy[i] for i in np.argsort(-podobienstwo)[:n]]


def nazwij_llm(grupy: dict[str, Sequence[str]], client,
               taksonomia: Optional[Sequence[str]] = None) -> dict[str, str]:
    """Nadaje nazwy wszystkim grupom w JEDNYM zapytaniu do LLM.

    Args:
        grupy: {identyfikator_grupy: lista reprezentatywnych opisow}
        client: llm_client.LLMClient
        taksonomia: istniejace nazwy klastrow - model ma ich NIE powielac

    Returns:
        {identyfikator_grupy: "NOWY: NAZWA"}. Grupy, na ktore model nie
        odpowiedzial, sa pomijane - wywolujacy zostawia dla nich numer.
    """
    if not grupy:
        return {}

    bloki = []
    for g, opisy in grupy.items():
        prob = "\n".join(f"    - {str(o)[:160]}" for o in opisy)
        bloki.append(f'  GROUP "{g}" ({len(opisy)} sample parts):\n{prob}')

    unikaj = ""
    if taksonomia:
        unikaj = ("\nThese cluster names ALREADY EXIST. If a group clearly belongs to one of "
                  "them, reuse that exact name; otherwise invent a new short name:\n"
                  + "\n".join(f"  - {t}" for t in taksonomia) + "\n")

    user = (
        "Name each group of parts below with its functional part type.\n"
        'Return JSON: {"names": [{"group": "<GROUP ID>", "name": "<NAME>"}, ...]} '
        "with exactly one entry per group.\n"
        f"{unikaj}\n"
        "GROUPS:\n" + "\n\n".join(bloki)
    )
    obj = client.chat_json(SYSTEM_NAZYWANIA, user)

    nazwy: dict[str, str] = {}
    for wpis in (obj.get("names") or []):
        if not isinstance(wpis, dict):
            continue
        g = str(wpis.get("group", "")).strip()
        n = str(wpis.get("name", "")).strip().strip('"').upper()
        if g in grupy and n:
            nazwy[g] = PREFIKS_PROPOZYCJI + n
    return nazwy


def rozroznij_kolizje(nazwy: dict[str, str], licznosci: Optional[dict[str, int]] = None
                      ) -> dict[str, str]:
    """Dodaje numer, gdy kilka grup dostalo te sama nazwe.

    Warstwa 2 celowo woli rozdrobnic niz blednie skleic, wiec dwie osobne grupy
    NIE moga zniknac w jednym klastrze tylko dlatego, ze heurystyka (albo model)
    nazwala je tak samo. Ekspert widzi "NOWY: BUSHING (1)" i "NOWY: BUSHING (2)"
    i sam decyduje, czy je scalic.

    Numeracja idzie od najliczniejszej grupy, zeby "(1)" bylo tym glownym wariantem.
    """
    wg_nazwy: dict[str, list[str]] = {}
    for grupa, nazwa in nazwy.items():
        wg_nazwy.setdefault(nazwa, []).append(grupa)

    wynik: dict[str, str] = {}
    for nazwa, grupy in wg_nazwy.items():
        if len(grupy) == 1:
            wynik[grupy[0]] = nazwa
            continue
        kolejnosc = sorted(grupy, key=lambda g: -(licznosci or {}).get(g, 0))
        for i, g in enumerate(kolejnosc, start=1):
            wynik[g] = f"{nazwa} ({i})"
    return wynik


# --------------------------------------------------------------------------- #
#  Dyspozytor
# --------------------------------------------------------------------------- #

def nazwij_grupy(wynik: pd.DataFrame, kolumna_grupy: str, kolumna_opisu: str,
                 metoda: str = "ctfidf", Z: Optional[np.ndarray] = None,
                 taksonomia: Optional[Sequence[str]] = None,
                 client=None, n_probek: int = 5,
                 tylko_prefiks: Optional[str] = None) -> dict[str, str]:
    """Nadaje nazwy grupom w ramce wynikowej.

    Args:
        wynik: ramka z kolumnami grupy i opisu
        kolumna_grupy: nazwa kolumny z identyfikatorem grupy (np. 'cluster_name')
        kolumna_opisu: nazwa kolumny z opisem czesci (np. 'MATDESC')
        metoda: 'ctfidf' | 'llm' | 'brak'
        Z: embeddingi wierszy `wynik` - wymagane dla 'llm' (wybor probek)
        taksonomia: istniejace nazwy klastrow (tylko 'llm')
        client: LLMClient (tylko 'llm')
        tylko_prefiks: nazywaj wylacznie grupy o identyfikatorze z tym prefiksem

    Returns:
        {stara_nazwa_grupy: nowa_nazwa}
    """
    if metoda == "brak":
        return {}

    ids = wynik[kolumna_grupy].astype(str)
    docelowe = sorted(set(ids[ids.str.startswith(tylko_prefiks)] if tylko_prefiks else ids))
    if not docelowe:
        return {}

    licznosci = {g: int((ids == g).sum()) for g in docelowe}

    if metoda == "ctfidf":
        grupy = {g: wynik.loc[(ids == g).values, kolumna_opisu].astype(str).tolist()
                 for g in docelowe}
        return rozroznij_kolizje(nazwij_ctfidf(grupy), licznosci)

    if metoda == "llm":
        if client is None:
            raise ValueError("metoda='llm' wymaga argumentu client (LLMClient).")
        grupy = {}
        for g in docelowe:
            maska = (ids == g).values
            opisy = wynik.loc[maska, kolumna_opisu].astype(str).tolist()
            grupy[g] = (probki_przy_centroidzie(opisy, Z[maska], n_probek)
                        if Z is not None else opisy[:n_probek])
        nazwy = nazwij_llm(grupy, client, taksonomia)
        # grupy pominiete przez model dobieramy heurystyka, zeby nie zostawic numeru
        brakujace = {g: wynik.loc[(ids == g).values, kolumna_opisu].astype(str).tolist()
                     for g in docelowe if g not in nazwy}
        nazwy.update(nazwij_ctfidf(brakujace))
        return rozroznij_kolizje(nazwy, licznosci)

    raise ValueError(f"Nieznana metoda nazywania: {metoda!r} (ctfidf | llm | brak)")


# --------------------------------------------------------------------------- #
#  Ewaluacja jakosci nazw
# --------------------------------------------------------------------------- #

def _rdzenie(nazwa: str) -> set[str]:
    """Tokeny znaczace nazwy klastra, skrocone do 4 znakow (prosty stemming)."""
    czysta = str(nazwa)
    if czysta.startswith(PREFIKS_PROPOZYCJI):
        czysta = czysta[len(PREFIKS_PROPOZYCJI):]
    return {t[:4] for t in _tokenizuj(czysta)}


def nazwa_trafiona(proponowana: str, prawdziwa: str) -> bool:
    """Czy proponowana nazwa trafia w prawdziwy typ czesci?

    Kryterium celowo lagodne - liczy sie, czy ekspert rozpozna typ, a nie czy
    nazwa jest znak w znak taka sama. 'BALL BEARING' trafia w 'BALL', a
    'GASKET SEAL' w 'SEAL'.
    """
    a, b = _rdzenie(proponowana), _rdzenie(prawdziwa)
    return bool(a & b)


def ocen_nazwy(grupy_pn: Sequence[str], nazwy_prop: Sequence[str],
               nazwy_prawdziwe: Sequence[str]) -> dict:
    """Jakosc nazw nadanych grupom, wazona liczba czesci.

    Args:
        grupy_pn: identyfikator grupy dla kazdej czesci
        nazwy_prop: proponowana nazwa dla kazdej czesci
        nazwy_prawdziwe: prawdziwa etykieta (ground truth) dla kazdej czesci

    Returns:
        trafnosc_czesci - odsetek CZESCI, ktore dostaly nazwe trafiajaca w ich
                          dominujacy prawdziwy typ,
        trafnosc_grup   - to samo liczone po GRUPACH (bez wagi licznosci),
        szczegoly       - ramka: grupa, n, nazwa proponowana, dominujaca prawdziwa, trafiona
    """
    d = pd.DataFrame({"grupa": list(grupy_pn), "prop": list(nazwy_prop),
                      "prawda": list(nazwy_prawdziwe)})
    wiersze = []
    for g, sub in d.groupby("grupa"):
        dominujaca = sub["prawda"].value_counts().index[0]
        prop = sub["prop"].iloc[0]
        wiersze.append({"grupa": g, "n": len(sub), "proponowana": prop,
                        "dominujaca_prawdziwa": dominujaca,
                        "trafiona": nazwa_trafiona(prop, dominujaca)})
    szcz = pd.DataFrame(wiersze).sort_values("n", ascending=False)
    if szcz.empty:
        return {"trafnosc_czesci": 0.0, "trafnosc_grup": 0.0, "szczegoly": szcz}
    return {
        "trafnosc_czesci": float((szcz["trafiona"] * szcz["n"]).sum() / szcz["n"].sum()),
        "trafnosc_grup": float(szcz["trafiona"].mean()),
        "szczegoly": szcz,
    }
