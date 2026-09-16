# -*- coding: utf-8 -*-
"""
CLI lokalnego klastrowania hybrydowego (bez chmury, bez LLM).

  python run_local.py                          # pelny przebieg + uczciwa ewaluacja
  python run_local.py --encoder tfidf+supcon   # z douczonym enkoderem neuronowym
  python run_local.py --encoder minilm         # bi-encoder z HuggingFace
  python run_local.py --sim-nowe 0.2           # symulacja: 20% klas jako "nowe typy"
  python run_local.py --porownaj               # tabela porownawcza torow
  python run_local.py --predict-only           # uzyj zapisanego modelu
  python run_local.py --override 0204X00136=RESERVOIR CAP   # feedback eksperta

Ewaluacja jest liczona na predykcjach OUT-OF-FOLD. Nie raportujemy trafnosci
modelu na danych, na ktorych sie uczyl.
"""
from __future__ import annotations

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import data_prep as dp
import evaluate
import local_clustering as lc

warnings.filterwarnings("ignore", category=FutureWarning)

KATALOG = Path(__file__).parent
WYNIKI_DIR = KATALOG / "wyniki"


# --------------------------------------------------------------------------- #
#  Dane
# --------------------------------------------------------------------------- #

def etykiety_per_pn(rekordy: pd.DataFrame, rudolf: pd.Series) -> dict[str, str]:
    """Etykieta Rudolfa na poziomie PN = dominujaca etykieta nie-'tbd' wsrod wierszy."""
    rek = rekordy.copy()
    rek["_RUD"] = rudolf.values[:len(rek)]
    mapa: dict[str, str] = {}
    for pn, g in rek.groupby(dp.PN_KOL, sort=False):
        v = g["_RUD"].astype(str).str.strip()
        znane = v[~v.str.lower().str.contains("tbd")]
        mapa[str(pn)] = znane.value_counts().index[0] if len(znane) else "tbd"
    return mapa


def wczytaj(args) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, pd.Series]:
    """Zwraca (rekordy, pn_df, y, maska_z_etykieta, rudolf_per_wiersz)."""
    rekordy = dp.wczytaj_rekordy(use_capri=not args.bez_capri, dataset=args.dataset)
    if args.rows:
        rekordy = rekordy.head(int(args.rows)).copy()
    rudolf = dp.wczytaj_etykiety_rudolfa()

    pn_df = dp.deduplikuj_do_pn(rekordy)
    if args.limit:
        pn_df = pn_df.head(int(args.limit)).reset_index(drop=True)
        rekordy = rekordy[rekordy[dp.PN_KOL].isin(set(pn_df["PN"]))].reset_index(drop=True)
        rudolf = rudolf.iloc[:0]  # przy --limit numeracja wierszy sie rozjezdza

    mapa = etykiety_per_pn(rekordy, dp.wczytaj_etykiety_rudolfa())
    y = np.array([mapa.get(str(pn), "tbd") for pn in pn_df["PN"]], dtype=object)
    maska = y != "tbd"
    return rekordy, pn_df, y, maska, rudolf


# --------------------------------------------------------------------------- #
#  Raporty
# --------------------------------------------------------------------------- #

def raport_oof(model: lc.HybrydowyKlasyfikator) -> dict:
    """Metryki warstwy 1 liczone na predykcjach out-of-fold (uczciwe)."""
    d = model.diagnostyka.oof
    y, pred, maska = d["y"], d["pred"], d["maska"]
    met = evaluate.metryki(y[maska], pred[maska])
    met["trafnosc"] = float((pred[maska] == y[maska]).mean())
    met["n_ocenianych"] = int(maska.sum())
    return met


def raport_nienadzorowany(model: lc.HybrydowyKlasyfikator, pn_df: pd.DataFrame,
                          y: np.ndarray, maska: np.ndarray) -> dict:
    """Czyste klastrowanie (tor 'odkrywanie') na wszystkich PN z etykieta - punkt odniesienia."""
    from sklearn.cluster import AgglomerativeClustering
    teksty = lc.buduj_teksty(pn_df[maska], model.cfg.pola)
    Z = model.enkoder_bazowy.transform(teksty)
    k = int(pd.Series(y[maska]).nunique())
    pred = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                   linkage="average").fit_predict(Z)
    return evaluate.metryki(y[maska], pred)


def symulacja_nowych_typow(cfg: lc.KonfiguracjaHybrydy, pn_df: pd.DataFrame,
                           y: np.ndarray, maska: np.ndarray, frakcja: float,
                           seed: int = 0) -> dict:
    """Czy hybryda rozpoznaje typy, ktorych NIE widziala w treningu?

    Co n-ta klasa (wg licznosci) jest w calosci ukrywana przed warstwa 1.
    Sprawdzamy, ile jej czesci warstwa 1 slusznie odrzucila do warstwy 2.
    """
    P = pn_df[maska].reset_index(drop=True)
    yy = y[maska]
    klasy = pd.Series(yy).value_counts().index.tolist()
    krok = max(2, int(round(1 / max(frakcja, 1e-9))))
    ukryte = set(klasy[::krok])
    m_znane = np.array([c not in ukryte for c in yy])
    if m_znane.sum() < 20 or (~m_znane).sum() < 5:
        return {}

    model = lc.HybrydowyKlasyfikator(cfg).fit(P[m_znane].reset_index(drop=True),
                                              yy[m_znane], verbose=False)
    wynik = model.predict(P, verbose=False)
    do_w2 = (wynik["zrodlo"] == lc.ZRODLO_ODKRYTY).values

    nowe = ~m_znane
    met = {
        "n_klas_ukrytych": len(ukryte),
        "n_pn_nowych": int(nowe.sum()),
        "wykryte_jako_nowe": float(do_w2[nowe].mean()),          # recall nowosci
        "falszywy_alarm": float(do_w2[m_znane].mean()),          # znane wyslane do w2
    }
    # jakosc grupowania nowych typow w warstwie 2
    w2_nowe = nowe & do_w2
    if w2_nowe.sum() >= 5 and pd.Series(yy[w2_nowe]).nunique() >= 2:
        m2 = evaluate.metryki(yy[w2_nowe], wynik.loc[w2_nowe, "cluster_name"].values)
        met["w2_pair_precision"] = m2["pair_precision"]
        met["w2_pair_f1"] = m2["pair_f1"]
        met["w2_ari"] = m2["ari"]
    # trafnosc warstwy 1 na czesciach, ktore przyjela
    przyjete = m_znane & ~do_w2
    if przyjete.sum():
        met["w1_trafnosc_przyjetych"] = float(
            (wynik.loc[przyjete, "cluster_name"].values == yy[przyjete]).mean())
    return met


# --------------------------------------------------------------------------- #
#  Zapis wynikow
# --------------------------------------------------------------------------- #

def zapisz_wyniki(out: Path, wynik_pn: pd.DataFrame, rekordy: pd.DataFrame,
                  rudolf: pd.Series) -> dict:
    """Zapisuje pliki w formacie zgodnym z reszta projektu i liczy metryki na wierszach."""
    out.mkdir(parents=True, exist_ok=True)
    mapa_nazw = dict(zip(wynik_pn["PN"].astype(str), wynik_pn["cluster_name"]))
    mapa_zrodel = dict(zip(wynik_pn["PN"].astype(str), wynik_pn["zrodlo"]))

    rek = rekordy.copy()
    rek["cluster_name"] = rek[dp.PN_KOL].astype(str).map(mapa_nazw).fillna(lc.ETYKIETA_PRZEGLAD)
    rek["zrodlo"] = rek[dp.PN_KOL].astype(str).map(mapa_zrodel).fillna("brak")

    kolejnosc = rek["cluster_name"].value_counts().index.tolist()
    mapa_id = {n: evaluate.litera(i) for i, n in enumerate(kolejnosc)}
    rek["cluster"] = rek["cluster_name"].map(mapa_id)
    wynik_pn = wynik_pn.copy()
    wynik_pn["cluster"] = wynik_pn["cluster_name"].map(mapa_id)

    wynik_pn.to_csv(out / "klastry_per_PN.csv", sep=";", index=False, encoding="utf-8")
    kols = [c for c in ["MAT_LANE_YM", dp.PN_KOL, "MATDESC", "HS Code First 6 ACDC",
                        "BU SCND", "cluster", "cluster_name", "zrodlo"] if c in rek.columns]
    rek[kols].to_csv(out / "klastry_per_wiersz.csv", sep=";", index=False, encoding="utf-8")
    (rek.groupby(["cluster", "cluster_name", "zrodlo"]).size()
        .reset_index(name="n_rekordow").sort_values("n_rekordow", ascending=False)
        .to_csv(out / "legenda_klastrow.csv", sep=";", index=False, encoding="utf-8"))

    if len(rudolf) >= len(rek):
        rek["RUDOLF"] = rudolf.values[:len(rek)]
        return evaluate.ocen(rek, out, gt_label="Rudolf")
    return {}


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lokalne klastrowanie hybrydowe (bez LLM)")
    p.add_argument("--encoder", default="tfidf",
                   help="tfidf | minilm | bge | st:<model>, opcjonalnie +supcon")
    p.add_argument("--dataset", default="to_cluster", choices=list(dp.ZRODLA))
    p.add_argument("--limit", type=int, default=None, help="ogranicz do N PN (test)")
    p.add_argument("--rows", type=int, default=None, help="ogranicz do N wierszy zrodla")
    p.add_argument("--folds", type=int, default=5, help="liczba foldow OOF (domyslnie 5)")
    p.add_argument("--cel-trafnosci", type=float, default=0.999,
                   help="docelowa trafnosc warstwy 1 (0.999 dom.); nizej -> wieksze "
                        "pokrycie, ale slabsze wykrywanie nowych typow")
    p.add_argument("--prog-pewnosci", type=float, default=None,
                   help="sztywny prog marginesu (domyslnie dobierany automatycznie)")
    p.add_argument("--prog-odkrywania", type=float, default=0.60,
                   help="prog odleglosci kosinusowej w warstwie 2")
    p.add_argument("--pola", default="MATDESC",
                   help="kolumny part card sklejane w tekst, po przecinku")
    p.add_argument("--sim-nowe", type=float, default=None, metavar="FRAKCJA",
                   help="symulacja nowych typow: ukryj te czesc klas przed warstwa 1")
    p.add_argument("--porownaj", action="store_true",
                   help="tabela: klastrowanie vs klasyfikator vs hybryda")
    p.add_argument("--bez-capri", action="store_true", help="nie doczytuj cache CaPRI")
    p.add_argument("--predict-only", action="store_true",
                   help="uzyj zapisanego modelu zamiast trenowac")
    p.add_argument("--override", action="append", default=[], metavar="PN=KLASTER",
                   help="dopisz korekte eksperta do overrides.yaml i zakoncz")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- WARSTWA 3: feedback eksperta ---
    if args.override:
        pary = {}
        for wpis in args.override:
            if "=" not in wpis:
                raise SystemExit(f"Zly format --override: {wpis!r} (oczekiwano PN=KLASTER)")
            pn, klaster = wpis.split("=", 1)
            pary[pn.strip()] = klaster.strip()
        n = lc.zapisz_override(pary)
        print(f"Zapisano {len(pary)} korekt do {lc.OVERRIDES_PATH.name} "
              f"(lacznie {n} wpisow).")
        print("Uruchom ponownie bez --override, aby model nauczyl sie na poprawkach.")
        return

    print("=" * 70)
    print("  LOKALNE KLASTROWANIE HYBRYDOWE  (warstwy: override / klasyfikator / odkrywanie)")
    print("=" * 70)

    cfg = lc.KonfiguracjaHybrydy(
        encoder=args.encoder,
        pola=tuple(s.strip() for s in args.pola.split(",") if s.strip()),
        cel_trafnosci=args.cel_trafnosci,
        prog_pewnosci=args.prog_pewnosci,
        prog_odkrywania=args.prog_odkrywania,
    )

    print("\n[0] Dane...")
    try:
        rekordy, pn_df, y, maska, rudolf = wczytaj(args)
    except FileNotFoundError as e:
        raise SystemExit(f"  BLAD: {e}\n  Sprawdz pliki w data_to_cluster/.")
    print(f"  wierszy={len(rekordy)}  PN={len(pn_df)}  "
          f"PN z etykieta={int(maska.sum())}  klas={pd.Series(y[maska]).nunique()}")
    if maska.sum() < 20:
        raise SystemExit("  BLAD: za malo oznaczonych PN.")

    # --- predict-only ---
    if args.predict_only:
        if not lc.MODEL_PATH.exists():
            raise SystemExit(f"  BLAD: brak zapisanego modelu ({lc.MODEL_PATH}).")
        print(f"\n[*] Wczytuje model z {lc.MODEL_PATH.name}")
        model = lc.HybrydowyKlasyfikator.wczytaj()
        wynik_pn = model.predict(pn_df)
        out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_local_predict"
        zapisz_wyniki(out, wynik_pn, rekordy, rudolf)
        print(f"  Wyniki -> {out.relative_to(KATALOG)}")
        return

    print(f"\n[1] Trening warstwy 1 (enkoder: {cfg.encoder})...")
    model = lc.HybrydowyKlasyfikator(cfg).fit(
        pn_df[maska].reset_index(drop=True), y[maska], n_folds=args.folds)

    print("\n[2] Uczciwa ewaluacja (out-of-fold)...")
    oof = raport_oof(model)
    print(f"  warstwa 1 (OOF, {oof['n_ocenianych']} PN): trafnosc={oof['trafnosc']:.3f}  "
          f"ARI={oof['ari']:.3f}  pair_f1={oof['pair_f1']:.3f}  NMI={oof['nmi']:.3f}")

    tabela = [("warstwa 1: klasyfikator (OOF)", oof["ari"], oof["pair_f1"])]
    if args.porownaj:
        nien = raport_nienadzorowany(model, pn_df, y, maska)
        print(f"  czyste klastrowanie (bez etykiet):      ARI={nien['ari']:.3f}  "
              f"pair_f1={nien['pair_f1']:.3f}")
        tabela.insert(0, ("czyste klastrowanie (bez etykiet)", nien["ari"], nien["pair_f1"]))

    sim = {}
    if args.sim_nowe:
        print(f"\n[3] Symulacja nowych typow (ukrywam ~{args.sim_nowe:.0%} klas)...")
        sim = symulacja_nowych_typow(cfg, pn_df, y, maska, args.sim_nowe, cfg.seed)
        if sim:
            print(f"  ukrytych klas={sim['n_klas_ukrytych']} ({sim['n_pn_nowych']} PN)")
            print(f"  nowe typy skierowane do warstwy 2: {sim['wykryte_jako_nowe']:.1%}")
            print(f"  falszywy alarm na znanych typach:  {sim['falszywy_alarm']:.1%}")
            if "w1_trafnosc_przyjetych" in sim:
                print(f"  trafnosc warstwy 1 na przyjetych:  "
                      f"{sim['w1_trafnosc_przyjetych']:.3f}")
            if "w2_pair_precision" in sim:
                print(f"  warstwa 2 na nowych typach: pair_precision="
                      f"{sim['w2_pair_precision']:.3f}  pair_f1={sim['w2_pair_f1']:.3f}")
        else:
            print("  (za malo danych na sensowna symulacje)")

    print("\n[4] Przypisanie wszystkich PN + zapis...")
    wynik_pn = model.predict(pn_df)
    rozklad = wynik_pn["zrodlo"].value_counts().to_dict()
    print(f"  zrodla przypisan: {rozklad}")
    print(f"  klastrow lacznie: {wynik_pn['cluster_name'].nunique()}")

    out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_local_{cfg.encoder.replace(':', '-')}"
    met_wiersze = zapisz_wyniki(out, wynik_pn, rekordy, rudolf)
    model.zapisz()

    zapisz_podsumowanie(out / "podsumowanie.txt", cfg, model, oof, tabela, sim,
                        met_wiersze, rozklad)
    print(f"  wyniki -> {out.relative_to(KATALOG)}")
    if met_wiersze:
        print(f"\n  UWAGA: metryki w ewaluacja.txt licza sie na wierszach dla modelu "
              f"dotrenowanego\n  na WSZYSTKICH etykietach - sa zawyzone. Miarodajne sa "
              f"liczby OOF powyzej.")
    print("\nGotowe.")


def zapisz_podsumowanie(sciezka: Path, cfg, model, oof, tabela, sim, met_wiersze, rozklad):
    d = model.diagnostyka
    with open(sciezka, "w", encoding="utf-8") as f:
        f.write("LOKALNE KLASTROWANIE HYBRYDOWE\n" + "=" * 60 + "\n\n")
        f.write(f"enkoder            : {cfg.encoder}\n")
        f.write(f"pola part card     : {', '.join(cfg.pola)}\n")
        f.write(f"PN treningowych    : {d.n_treningowych}\n")
        f.write(f"klas w taksonomii  : {d.n_klas}\n")
        f.write(f"prog marginesu     : {d.prog_pewnosci:.3f} (cel trafnosci "
                f"{cfg.cel_trafnosci:.3f})\n")
        f.write(f"prog odkrywania    : {cfg.prog_odkrywania}\n\n")

        f.write("A) WARSTWA 1 - out-of-fold (uczciwe)\n")
        for k in ["trafnosc", "ari", "pair_f1", "pair_precision", "pair_recall", "nmi",
                  "zgodnosc"]:
            if k in oof:
                f.write(f"   {k:16s} = {oof[k]:.3f}\n")
        f.write(f"   przyjmuje {d.udzial_przyjetych:.1%} czesci, trafnosc przyjetych "
                f"{d.trafnosc_przyjetych:.3f}\n\n")

        if len(tabela) > 1:
            f.write("B) POROWNANIE TOROW (ARI / pair_f1)\n")
            for nazwa, ari, f1 in tabela:
                f.write(f"   {nazwa:36s} {ari:.3f} / {f1:.3f}\n")
            f.write(f"   {'chmurowy LLM (Gemini, wg README)':36s} 0.885 /   -\n\n")

        if sim:
            f.write("C) SYMULACJA NOWYCH TYPOW (klasy ukryte przed warstwa 1)\n")
            for k, v in sim.items():
                f.write(f"   {k:24s} = {v:.3f}\n" if isinstance(v, float)
                        else f"   {k:24s} = {v}\n")
            f.write("\n")

        f.write(f"D) ZRODLA PRZYPISAN (wszystkie PN)\n   {rozklad}\n\n")
        if met_wiersze:
            f.write("E) METRYKI NA WIERSZACH - model dotrenowany na wszystkich\n"
                    "   etykietach, wiec ZAWYZONE. Do porownan uzywaj sekcji A.\n")
            for k in ["ari", "pair_f1", "nmi", "zgodnosc"]:
                if k in met_wiersze:
                    f.write(f"   {k:16s} = {met_wiersze[k]:.3f}\n")


if __name__ == "__main__":
    main()
