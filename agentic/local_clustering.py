# -*- coding: utf-8 -*-
"""
Lokalne klastrowanie czesci - architektura hybrydowa, dwutorowa.

Zamiast jednego mechanizmu na wszystko, kazda czesc trafia do jednego z trzech
torow (w tej kolejnosci):

  WARSTWA 0  override eksperta (overrides.yaml)   -> zrodlo "override"
  WARSTWA 1  klasyfikator do ZNANEJ taksonomii,   -> zrodlo "model"
             jesli margines pewnosci >= prog
  WARSTWA 2  reszta (niepewne / nieznane typy):   -> zrodlo "odkryty"
             klastrowanie aglomeracyjne na SUROWYM embeddingu -> NOWY_1, NOWY_2...

  WARSTWA 3  feedback: zapisz_override() dopisuje decyzje eksperta do
             overrides.yaml, kolejny fit() uczy sie juz na nich.

Dlaczego tak, a nie "czysty cluster-then-label" z ARCHITEKTURA_LOKALNA.md
(pomiary na to_cluster.csv, 962 PN z etykieta, 77 klas Rudolfa):

  czyste klastrowanie TF-IDF + agglomerative(cosine)   ARI 0.599
  to samo + douczony enkoder neuronowy (SupCon)        ARI 0.753
  klasyfikator do znanej taksonomii (LinearSVC, OOF)   ARI 0.954   <- tu sa wyniki
  chmurowy LLM (Gemini, wg README)                     ARI 0.885

  Dodatkowo margines LinearSVC swietnie wykrywa wlasne bledy: przy domyslnym
  progu warstwa 1 przyjmuje 83% czesci z trafnoscia 1.000, a odrzucone 17%
  zawieraja 79% czesci nalezacych do typow, ktorych model w ogole nie zna.
  Te 17% oddajemy do WARSTWY 2 zamiast zgadywac.

Ograniczenie, ktore ksztaltuje warstwe 2: douczony enkoder (+supcon) jest
SWIETNY na znanych typach, ale slaby na typach niewidzianych w treningu
(ARI 0.910 -> 0.342 na symulacji nowych klas). Dlatego warstwa 2 celowo uzywa
embeddingu bazowego, nie douczonego.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from encoders import BaseEncoder, SupConEncoder, zbuduj_enkoder

KATALOG = Path(__file__).parent
OVERRIDES_PATH = KATALOG / "overrides.yaml"
MODELS_DIR = KATALOG / "models"
MODEL_PATH = MODELS_DIR / "hybryda.pkl"

ZRODLO_OVERRIDE = "override"
ZRODLO_MODEL = "model"
ZRODLO_ODKRYTY = "odkryty"
PREFIKS_NOWY = "NOWY_"
ETYKIETA_PRZEGLAD = "DO_PRZEGLADU"


# --------------------------------------------------------------------------- #
#  WARSTWA 0 / 3 - twardy slownik eksperta (human-in-the-loop)
# --------------------------------------------------------------------------- #

def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, str]:
    """Wczytaj PN -> klaster ze slownika eksperckiego (YAML)."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


def zapisz_override(przypisania: dict[str, str], path: Path = OVERRIDES_PATH) -> int:
    """WARSTWA 3: dopisz decyzje eksperta do overrides.yaml (merge, nie nadpisanie).

    Zwraca liczbe wpisow w slowniku po zapisie. Komentarz-naglowek pliku jest
    odtwarzany, bo yaml.safe_dump go nie zachowuje.
    """
    biezace = load_overrides(path)
    biezace.update({str(k).strip(): str(v).strip() for k, v in przypisania.items()})
    naglowek = (
        "# Twardy slownik przypisan: PN -> klaster\n"
        "# Czesci wpisane tutaj ZAWSZE trafia do wskazanego klastra (100% determinizm).\n"
        "# Uzywaj do korekt eksperckich (human-in-the-loop).\n"
        "# Plik jest dopisywany przez local_clustering.zapisz_override().\n\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(naglowek)
        if biezace:
            yaml.safe_dump(biezace, f, allow_unicode=True, sort_keys=True,
                           default_flow_style=False)
    return len(biezace)


# --------------------------------------------------------------------------- #
#  Tekst wejsciowy dla enkodera
# --------------------------------------------------------------------------- #

#: Zmierzona ablacja cech (OOF, LinearSVC): sam MATDESC = 0.948 ARI,
#: +HS text = 0.934, +hierarchia produktowa = 0.917. Dodatkowy kontekst
#: ROZCIENCZA glowny sygnal, wiec domyslnie bierzemy tylko opis.
DOMYSLNE_POLA = ("MATDESC",)
PUSTE = {"", "nan", "none", "?", "(brak opisu)"}


def buduj_tekst(row: pd.Series, pola: Sequence[str] = DOMYSLNE_POLA) -> str:
    """Sklada tekst wejsciowy enkodera z wybranych kolumn part card."""
    czesci = []
    for kol in pola:
        val = str(row.get(kol, "")).strip()
        if val.lower() not in PUSTE:
            czesci.append(val)
    return " | ".join(czesci) if czesci else "unknown part"


def buduj_teksty(df: pd.DataFrame, pola: Sequence[str] = DOMYSLNE_POLA) -> list[str]:
    return [buduj_tekst(r, pola) for _, r in df.iterrows()]


# --------------------------------------------------------------------------- #
#  Konfiguracja
# --------------------------------------------------------------------------- #

@dataclass
class KonfiguracjaHybrydy:
    """Parametry pipeline'u. Wartosci domyslne = zmierzone optimum na to_cluster."""

    encoder: str = "tfidf"
    """Spec enkodera: 'tfidf' | 'minilm' | 'bge' | 'st:<model>' (+'+supcon')."""

    pola: tuple[str, ...] = DOMYSLNE_POLA
    """Kolumny part card sklejane w tekst wejsciowy."""

    cel_trafnosci: float = 0.999
    """WARSTWA 1: prog marginesu dobierany tak, by trafnosc przyjetych >= tej wartosci.

    Steruje kompromisem pokrycie <-> wykrywanie nowych typow. Zmierzone
    (to_cluster, 20% klas ukrytych jako "nowe", srednia z 5 seedow):

      cel     pokrycie  trafnosc przyjetych  nowe typy wykryte  falszywy alarm
      0.970     96.9%          0.971               18.0%             2.9%
      0.980     94.8%          0.981               39.3%             3.2%
      0.990     91.3%          0.990               52.9%             3.8%
      0.995     88.1%          0.996               62.7%             4.0%
      0.999     83.1%          1.000               79.1%             6.3%

    Domyslne 0.999: warstwa 1 nie popelnia bledu na tym, co przyjmuje, a do
    eksperta trafia ~17% czesci - w tym 4 na 5 faktycznie nowych typow.
    Obniz do 0.98, jesli zalezy Ci na maksymalnej automatyzacji.
    """

    prog_pewnosci: Optional[float] = None
    """Sztywny prog marginesu; None = dobierz automatycznie z OOF."""

    prog_odkrywania: float = 0.60
    """WARSTWA 2: prog odleglosci kosinusowej w klastrowaniu aglomeracyjnym.

    Nizszy = wiecej, czystszych grupek. Zmierzone na symulacji nowych typow:
    0.30 -> 82 grupy, pair_precision 0.991; 0.60 -> 55 grup, 0.985;
    0.80 -> 32 grupy, 0.963. Celowo preferujemy rozdrobnienie nad blednym
    sklejeniem - eksperta latwiej poprosic o scalenie niz o rozbicie grupy.
    """

    min_licznosc_nowego: int = 2
    """Grupy mniejsze niz tyle PN dostaja etykiete DO_PRZEGLADU zamiast NOWY_n."""

    min_probek_klasy: int = 2
    """Klasy o mniejszej licznosci nie wchodza do treningu warstwy 1."""

    C: float = 1.0
    """Regularyzacja LinearSVC."""

    seed: int = 0

    def __post_init__(self):
        self.pola = tuple(self.pola)


# --------------------------------------------------------------------------- #
#  WARSTWA 1 - klasyfikator + margines pewnosci
# --------------------------------------------------------------------------- #

def _margines(decyzja: np.ndarray) -> np.ndarray:
    """Pewnosc = odstep miedzy najlepsza a druga klasa w decision_function.

    Dla problemu 2-klasowego sklearn zwraca wektor 1-D - wtedy marginesem jest
    wartosc bezwzgledna.
    """
    if decyzja.ndim == 1:
        return np.abs(decyzja)
    posortowane = np.sort(decyzja, axis=1)
    return posortowane[:, -1] - posortowane[:, -2]


def dobierz_prog_pewnosci(marginesy: np.ndarray, trafne: np.ndarray,
                          cel: float = 0.98) -> float:
    """Najnizszy prog, przy ktorym trafnosc przyjetych predykcji osiaga `cel`.

    Liczone na predykcjach out-of-fold, wiec prog nie jest dopasowany do
    danych, na ktorych model byl trenowany.
    """
    if len(marginesy) == 0:
        return 0.0
    kolejnosc = np.argsort(-marginesy)          # od najpewniejszych
    skumulowana = np.cumsum(trafne[kolejnosc]) / np.arange(1, len(kolejnosc) + 1)
    spelnia = np.where(skumulowana >= cel)[0]
    if len(spelnia) == 0:
        return float(marginesy.max())           # nie da sie osiagnac celu
    # ostatnia pozycja, na ktorej cel jest jeszcze spelniony
    return float(marginesy[kolejnosc[spelnia[-1]]])


def ocena_oof(y: np.ndarray, n_folds: int = 5, C: float = 1.0, seed: int = 0,
              min_probek: int = 5, Z: Optional[np.ndarray] = None,
              teksty: Optional[Sequence[str]] = None,
              enkoder_fn: Optional[Callable[[], BaseEncoder]] = None,
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uczciwa walidacja out-of-fold warstwy 1.

    Klasy rzadsze niz `min_probek` nie daja sie stratyfikowac - trafiaja do
    KAZDEGO zbioru treningowego (model musi je znac), ale sa wykluczone ze
    scoringu. Zwraca (predykcje, marginesy, maska_ocenianych).

    Dwa tryby, zaleznie od tego, czy enkoder uczy sie na etykietach:

      Z=...                     enkoder bezetykietowy (tfidf, bi-encoder) -
                                embedding liczony raz, poza petla.
      teksty=..., enkoder_fn=...  enkoder uczony na etykietach (+supcon) -
                                MUSI byc trenowany od nowa w kazdym foldzie,
                                wylacznie na czesci treningowej. Inaczej
                                embedding widzi etykiety walidacyjne i wynik
                                jest zawyzony.
    """
    if Z is None and (teksty is None or enkoder_fn is None):
        raise ValueError("Podaj Z albo (teksty + enkoder_fn).")

    licznosc = pd.Series(y).value_counts()
    rzadkie = np.array([licznosc[v] < min_probek for v in y])
    idx_ocen = np.where(~rzadkie)[0]
    idx_rzadkie = np.where(rzadkie)[0]

    predykcje = np.array(y, dtype=object).copy()
    marginesy = np.zeros(len(y), dtype=float)
    teksty = None if teksty is None else np.asarray(list(teksty), dtype=object)

    n_folds = max(2, min(n_folds, int(pd.Series(y[idx_ocen]).value_counts().min())))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for tr, va in skf.split(idx_ocen, y[idx_ocen]):
        i_tr = np.concatenate([idx_ocen[tr], idx_rzadkie])
        i_va = idx_ocen[va]
        if Z is not None:
            Z_tr, Z_va = Z[i_tr], Z[i_va]
        else:
            enk = enkoder_fn()
            enk.fit(teksty[i_tr].tolist(), y[i_tr])
            Z_tr = enk.transform(teksty[i_tr].tolist())
            Z_va = enk.transform(teksty[i_va].tolist())
        clf = LinearSVC(C=C, random_state=seed).fit(Z_tr, y[i_tr])
        predykcje[i_va] = clf.predict(Z_va)
        marginesy[i_va] = _margines(clf.decision_function(Z_va))

    maska = np.zeros(len(y), dtype=bool)
    maska[idx_ocen] = True
    return predykcje, marginesy, maska


# --------------------------------------------------------------------------- #
#  Pipeline hybrydowy
# --------------------------------------------------------------------------- #

@dataclass
class WynikDopasowania:
    """Diagnostyka z fit() - do raportu, nie do dzialania modelu."""
    n_treningowych: int = 0
    n_klas: int = 0
    prog_pewnosci: float = 0.0
    trafnosc_oof: float = 0.0
    trafnosc_przyjetych: float = 0.0
    udzial_przyjetych: float = 0.0
    oof: dict = field(default_factory=dict)


class HybrydowyKlasyfikator:
    """Warstwy 0-2 w jednym obiekcie. fit() na etykietach, predict() na dowolnym df."""

    def __init__(self, cfg: Optional[KonfiguracjaHybrydy] = None):
        self.cfg = cfg or KonfiguracjaHybrydy()
        self.enkoder: Optional[BaseEncoder] = None
        self.enkoder_bazowy: Optional[BaseEncoder] = None   # do WARSTWY 2
        self.klasyfikator: Optional[LinearSVC] = None
        self.prog: float = 0.0
        self.klasy_: np.ndarray = np.array([])
        self.diagnostyka = WynikDopasowania()

    # ---------------------------------------------------------------- fit ---
    def fit(self, df: pd.DataFrame, y: np.ndarray,
            n_folds: int = 5, verbose: bool = True) -> "HybrydowyKlasyfikator":
        """Trenuje warstwe 1. `df` i `y` musza byc juz przefiltrowane do PN z etykieta.

        Overrides (warstwa 0) sa wciagane do treningu jako dodatkowe etykiety -
        dzieki temu korekta eksperta uczy model, a nie tylko nadpisuje wynik.
        """
        cfg = self.cfg
        df = df.reset_index(drop=True)
        y = np.asarray(y, dtype=object).copy()

        # WARSTWA 0 wchodzi do treningu (feedback loop)
        overrides = load_overrides()
        if overrides and "PN" in df.columns:
            trafione = df["PN"].astype(str).map(overrides)
            ma = trafione.notna().values
            y[ma] = trafione[ma].values
            if verbose and ma.sum():
                print(f"  [warstwa 0] {int(ma.sum())} PN z overrides.yaml wlaczonych do treningu")

        # klasy zbyt rzadkie, by czegokolwiek nauczyc
        licznosc = pd.Series(y).value_counts()
        dosc = np.array([licznosc[v] >= cfg.min_probek_klasy for v in y])
        if dosc.sum() < len(y) and verbose:
            print(f"  [warstwa 1] pomijam {int((~dosc).sum())} PN z klas o licznosci "
                  f"< {cfg.min_probek_klasy}")
        df_t, y_t = df[dosc].reset_index(drop=True), y[dosc]

        teksty = buduj_teksty(df_t, cfg.pola)
        if verbose:
            print(f"  [enkoder] {cfg.encoder} na {len(teksty)} tekstach...")
        self.enkoder = zbuduj_enkoder(cfg.encoder, seed=cfg.seed)
        Z = (self.enkoder.fit_transform(teksty, y_t)
             if self.enkoder.wymaga_etykiet else self.enkoder.fit_transform(teksty))
        # WARSTWA 2 zawsze na embeddingu bazowym - douczony gubi nowe typy
        self.enkoder_bazowy = (self.enkoder.base if isinstance(self.enkoder, SupConEncoder)
                               else self.enkoder)

        # uczciwy OOF: trafnosc + dobor progu pewnosci.
        # Enkoder uczony na etykietach przechodzi pelny re-trening w kazdym
        # foldzie - inaczej embedding widzialby etykiety walidacyjne.
        if self.enkoder.wymaga_etykiet:
            if verbose:
                print(f"  [warstwa 1] OOF z re-treningiem enkodera w {n_folds} foldach "
                      f"(wolniejsze, ale bez przecieku etykiet)...")
            pred, marg, maska = ocena_oof(
                y_t, n_folds=n_folds, C=cfg.C, seed=cfg.seed, teksty=teksty,
                enkoder_fn=lambda: zbuduj_enkoder(cfg.encoder, seed=cfg.seed))
        else:
            pred, marg, maska = ocena_oof(y_t, n_folds=n_folds, C=cfg.C,
                                          seed=cfg.seed, Z=Z)
        trafne = (pred == y_t)
        prog_oof = (cfg.prog_pewnosci if cfg.prog_pewnosci is not None
                    else dobierz_prog_pewnosci(marg[maska], trafne[maska], cfg.cel_trafnosci))
        przyjete = marg[maska] >= prog_oof

        self.klasyfikator = LinearSVC(C=cfg.C, random_state=cfg.seed).fit(Z, y_t)
        self.klasy_ = self.klasyfikator.classes_

        self.prog = prog_oof
        if self.enkoder.wymaga_etykiet and cfg.prog_pewnosci is None:
            # Enkoder z kazdego foldu ma wlasna skale marginesu, wiec prog z OOF
            # nie przenosi sie wprost na model finalny. Przenosimy POKRYCIE:
            # bierzemy ten sam kwantyl marginesow, tyle ze finalnego modelu.
            marg_final = _margines(self.klasyfikator.decision_function(Z))
            self.prog = float(np.quantile(marg_final, 1.0 - max(przyjete.mean(), 1e-9)))
            if verbose:
                print(f"  [warstwa 1] prog przeniesiony przez kwantyl pokrycia "
                      f"({przyjete.mean():.1%}): {prog_oof:.3f} -> {self.prog:.3f}")

        self.diagnostyka = WynikDopasowania(
            n_treningowych=len(y_t),
            n_klas=int(pd.Series(y_t).nunique()),
            prog_pewnosci=self.prog,
            trafnosc_oof=float(trafne[maska].mean()),
            trafnosc_przyjetych=float(trafne[maska][przyjete].mean()) if przyjete.any() else 0.0,
            udzial_przyjetych=float(przyjete.mean()),
            oof={"pred": pred, "margines": marg, "maska": maska, "y": y_t, "df": df_t},
        )
        if verbose:
            d = self.diagnostyka
            print(f"  [warstwa 1] OOF trafnosc={d.trafnosc_oof:.3f} | prog marginesu="
                  f"{d.prog_pewnosci:.3f} -> przyjmuje {d.udzial_przyjetych:.1%} czesci "
                  f"z trafnoscia {d.trafnosc_przyjetych:.3f}")
        return self

    # ------------------------------------------------------------ predict ---
    def predict(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        """Przypisuje klastry. Zwraca df z kolumnami cluster_name / zrodlo / pewnosc.

        WARSTWA 2 jest transduktywna - klastruje caly przekazany zbior reszty
        naraz, wiec wynik zalezy od tego, co jeszcze jest w batchu.
        """
        assert self.klasyfikator is not None, "Najpierw fit()."
        cfg = self.cfg
        out = df.reset_index(drop=True).copy()
        n = len(out)
        nazwy = np.array([None] * n, dtype=object)
        zrodla = np.array([None] * n, dtype=object)
        pewnosc = np.zeros(n)

        # --- WARSTWA 0 ---
        overrides = load_overrides()
        if overrides and "PN" in out.columns:
            trafione = out["PN"].astype(str).map(overrides)
            ma = trafione.notna().values
            nazwy[ma] = trafione[ma].values
            zrodla[ma] = ZRODLO_OVERRIDE
            pewnosc[ma] = 1.0
        else:
            ma = np.zeros(n, dtype=bool)

        pozostale = np.where(~ma)[0]
        if len(pozostale) == 0:
            return self._zloz(out, nazwy, zrodla, pewnosc)

        # --- WARSTWA 1 ---
        teksty = buduj_teksty(out.iloc[pozostale], cfg.pola)
        Z = self.enkoder.transform(teksty)
        marg = _margines(self.klasyfikator.decision_function(Z))
        pred = self.klasyfikator.predict(Z)
        pewne = marg >= self.prog

        idx_pewne = pozostale[pewne]
        nazwy[idx_pewne] = pred[pewne]
        zrodla[idx_pewne] = ZRODLO_MODEL
        pewnosc[idx_pewne] = marg[pewne]

        # --- WARSTWA 2 ---
        idx_reszta = pozostale[~pewne]
        if len(idx_reszta):
            tekst_reszty = buduj_teksty(out.iloc[idx_reszta], cfg.pola)
            Zb = self.enkoder_bazowy.transform(tekst_reszty)
            grupy = self._odkryj(Zb)
            licz = pd.Series(grupy).value_counts()
            # numeracja NOWY_n od najliczniejszej grupy
            duze = [g for g in licz.index if licz[g] >= cfg.min_licznosc_nowego]
            mapa = {g: f"{PREFIKS_NOWY}{i + 1}" for i, g in enumerate(duze)}
            nazwy[idx_reszta] = [mapa.get(g, ETYKIETA_PRZEGLAD) for g in grupy]
            zrodla[idx_reszta] = ZRODLO_ODKRYTY
            pewnosc[idx_reszta] = marg[~pewne]
            if verbose:
                print(f"  [warstwa 2] {len(idx_reszta)} niepewnych czesci -> "
                      f"{len(duze)} grup kandydackich + "
                      f"{int((np.array(nazwy[idx_reszta]) == ETYKIETA_PRZEGLAD).sum())} "
                      f"pojedynczych do przegladu")
        return self._zloz(out, nazwy, zrodla, pewnosc)

    def _odkryj(self, Z: np.ndarray) -> np.ndarray:
        """Aglomeracyjne klastrowanie reszty progiem odleglosci (k nieznane z gory)."""
        if len(Z) == 1:
            return np.array([0])
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.cfg.prog_odkrywania,
            metric="cosine", linkage="average",
        ).fit_predict(Z)

    @staticmethod
    def _zloz(out, nazwy, zrodla, pewnosc) -> pd.DataFrame:
        out["cluster_name"] = nazwy
        out["zrodlo"] = zrodla
        out["pewnosc"] = pewnosc
        return out

    # --------------------------------------------------------------- IO ----
    def zapisz(self, path: Path = MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        diag = self.diagnostyka
        lekka = WynikDopasowania(diag.n_treningowych, diag.n_klas, diag.prog_pewnosci,
                                 diag.trafnosc_oof, diag.trafnosc_przyjetych,
                                 diag.udzial_przyjetych)  # bez surowych tablic OOF
        with open(path, "wb") as f:
            pickle.dump({"cfg": self.cfg, "enkoder": self.enkoder,
                         "enkoder_bazowy": self.enkoder_bazowy,
                         "klasyfikator": self.klasyfikator, "prog": self.prog,
                         "diagnostyka": lekka}, f)
        return path

    @classmethod
    def wczytaj(cls, path: Path = MODEL_PATH) -> "HybrydowyKlasyfikator":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(d["cfg"])
        obj.enkoder, obj.enkoder_bazowy = d["enkoder"], d["enkoder_bazowy"]
        obj.klasyfikator, obj.prog = d["klasyfikator"], d["prog"]
        obj.klasy_ = obj.klasyfikator.classes_
        obj.diagnostyka = d["diagnostyka"]
        return obj
