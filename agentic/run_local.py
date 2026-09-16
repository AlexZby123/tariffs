# -*- coding: utf-8 -*-
"""
CLI lokalnego klastrowania hybrydowego (bez chmury, bez LLM).

  python run_local.py                          # pelny przebieg + uczciwa ewaluacja
  python run_local.py --encoder tfidf+supcon   # z douczonym enkoderem neuronowym
  python run_local.py --encoder minilm         # bi-encoder z HuggingFace
  python run_local.py --sim-nowe 0.2           # symulacja: 20% klas jako "nowe typy"
  python run_local.py --nazywaj llm            # ETAP 2: nazwij nowe grupy jednym zapytaniem
  python run_local.py --porownaj               # tabela porownawcza torow
  python run_local.py --predict-only           # uzyj zapisanego modelu
  python run_local.py --override 0204X00136=RESERVOIR CAP   # feedback eksperta

Ewaluacja jest liczona na predykcjach OUT-OF-FOLD. Nie raportujemy trafnosci
modelu na danych, na ktorych sie uczyl.
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import data_prep as dp
import evaluate
import local_clustering as lc
import naming

warnings.filterwarnings("ignore", category=FutureWarning)

KATALOG = Path(__file__).parent
WYNIKI_DIR = KATALOG / "wyniki"
#: Stala sciezka z ostatnim wynikiem - stad czyta workflow-ui (nie musi
#: zgadywac nazwy katalogu z timestampem ani parsowac CSV).
OSTATNI_JSON = WYNIKI_DIR / "_ostatni_lokalny.json"


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


def raport_oof_z_modelu(model: lc.HybrydowyKlasyfikator) -> dict:
    """Metryki OOF dla modelu wczytanego z dysku (surowe tablice nie sa zapisywane)."""
    d = model.diagnostyka
    return {"trafnosc": d.trafnosc_oof, "n_ocenianych": d.n_treningowych}


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
                           seed: int = 0, metoda_nazywania: str = "brak") -> dict:
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
    do_w2 = (wynik["source"] == lc.ZRODLO_ODKRYTY).values

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

        # ETAP 2: czy nadane nazwy trafiaja w prawdziwy typ czesci?
        if metoda_nazywania != "brak":
            nazwane = nazwij_odkryte(model, wynik, metoda_nazywania, verbose=False)
            on = naming.ocen_nazwy(wynik.loc[w2_nowe, "cluster_name"].values,
                                   nazwane.loc[w2_nowe, "cluster_name"].values,
                                   yy[w2_nowe])
            met["nazwy_trafnosc_czesci"] = on["trafnosc_czesci"]
            met["nazwy_trafnosc_grup"] = on["trafnosc_grup"]
            met["_nazwy_szczegoly"] = on["szczegoly"]
    # trafnosc warstwy 1 na czesciach, ktore przyjela
    przyjete = m_znane & ~do_w2
    if przyjete.sum():
        met["w1_trafnosc_przyjetych"] = float(
            (wynik.loc[przyjete, "cluster_name"].values == yy[przyjete]).mean())
    return met


def nazwij_odkryte(model: lc.HybrydowyKlasyfikator, wynik: pd.DataFrame,
                   metoda: str, verbose: bool = True) -> pd.DataFrame:
    """STAGE 2: turns NEW_1, NEW_2... into proposed functional-type names."""
    if metoda == "brak":
        return wynik
    do_nazwania = wynik["cluster_name"].astype(str).str.startswith(lc.PREFIKS_NOWY)
    if not do_nazwania.any():
        return wynik

    def _nazwij(metoda_: str, Z_=None, client_=None) -> dict[str, str]:
        return naming.nazwij_grupy(
            wynik, kolumna_grupy="cluster_name", kolumna_opisu="MATDESC",
            metoda=metoda_, Z=Z_, client=client_,
            taksonomia=list(model.klasy_), tylko_prefiks=lc.PREFIKS_NOWY,
        )

    client = None
    if metoda == "llm":
        # Nazywanie przez LLM jest podpiete na stale, wiec NIE MOZE wywrocic
        # biegu. Brak config.yaml, wygasly token, padniety endpoint, zly JSON -
        # wszystko spada na c-TF-IDF, ktore dziala offline. Uzytkownik dostaje
        # komplet wynikow, tylko z gorszymi nazwami grup. Osłaniamy CALA probe,
        # razem z samym wywolaniem sieciowym.
        try:
            from llm_client import LLMClient, wczytaj_config
            client = LLMClient(wczytaj_config())
            Z = model.enkoder_bazowy.transform(lc.buduj_teksty(wynik, model.cfg.pola))
            mapa = _nazwij("llm", Z, client)
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [stage 2] cloud naming failed ({type(e).__name__}: "
                      f"{str(e)[:110]})")
                print("  [stage 2] falling back to offline c-TF-IDF naming")
            metoda, client = "ctfidf", None
            mapa = _nazwij("ctfidf")
    else:
        mapa = _nazwij(metoda)
    if verbose and mapa:
        print(f"  [stage 2] named {len(mapa)} groups using '{metoda}':")
        licz = wynik["cluster_name"].value_counts()
        for stara, nowa in sorted(mapa.items(), key=lambda kv: -licz.get(kv[0], 0))[:10]:
            print(f"      {stara:10s} (n={licz.get(stara, 0):3d})  ->  {nowa}")
        if len(mapa) > 10:
            print(f"      ... and {len(mapa) - 10} more")
        if client is not None:
            print(f"  [stage 2] LLM cost: {client.podsumowanie_kosztow()}")

    wynik = wynik.copy()
    wynik["cluster_name"] = wynik["cluster_name"].replace(mapa)
    return wynik


# --------------------------------------------------------------------------- #
#  Zapis wynikow
# --------------------------------------------------------------------------- #

def zapisz_wyniki(out: Path, wynik_pn: pd.DataFrame, rekordy: pd.DataFrame,
                  rudolf: pd.Series) -> dict:
    """Zapisuje pliki w formacie zgodnym z reszta projektu i liczy metryki na wierszach."""
    out.mkdir(parents=True, exist_ok=True)
    mapa_nazw = dict(zip(wynik_pn["PN"].astype(str), wynik_pn["cluster_name"]))
    mapa_zrodel = dict(zip(wynik_pn["PN"].astype(str), wynik_pn["source"]))

    rek = rekordy.copy()
    rek["cluster_name"] = rek[dp.PN_KOL].astype(str).map(mapa_nazw).fillna(lc.ETYKIETA_PRZEGLAD)
    rek["source"] = rek[dp.PN_KOL].astype(str).map(mapa_zrodel).fillna("none")

    kolejnosc = rek["cluster_name"].value_counts().index.tolist()
    mapa_id = {n: evaluate.litera(i) for i, n in enumerate(kolejnosc)}
    rek["cluster"] = rek["cluster_name"].map(mapa_id)
    wynik_pn = wynik_pn.copy()
    wynik_pn["cluster"] = wynik_pn["cluster_name"].map(mapa_id)

    wynik_pn.to_csv(out / "klastry_per_PN.csv", sep=";", index=False, encoding="utf-8")
    kols = [c for c in ["MAT_LANE_YM", dp.PN_KOL, "MATDESC", "HS Code First 6 ACDC",
                        "BU SCND", "cluster", "cluster_name", "source"] if c in rek.columns]
    rek[kols].to_csv(out / "klastry_per_wiersz.csv", sep=";", index=False, encoding="utf-8")
    (rek.groupby(["cluster", "cluster_name", "source"]).size()
        .reset_index(name="n_rekordow").sort_values("n_rekordow", ascending=False)
        .to_csv(out / "legenda_klastrow.csv", sep=";", index=False, encoding="utf-8"))

    if len(rudolf) >= len(rek):
        rek["RUDOLF"] = rudolf.values[:len(rek)]
        return evaluate.ocen(rek, out, gt_label="Rudolf")
    return {}


def zapisz_json(out: Path, wynik_pn: pd.DataFrame, rekordy: pd.DataFrame,
                model: lc.HybrydowyKlasyfikator, oof: dict, args,
                sim: Optional[dict] = None) -> Path:
    """Zapisuje wynik w formacie czytanym przez workflow-ui.

    Trafia w dwa miejsca: do katalogu przebiegu (archiwum) i pod stala sciezke
    OSTATNI_JSON, ktora UI odpytuje bez znajomosci timestampu.
    """
    d = model.diagnostyka
    wiersze_na_pn = (rekordy.groupby(rekordy[dp.PN_KOL].astype(str)).size().to_dict()
                     if dp.PN_KOL in rekordy.columns else {})

    czesci = []
    for _, r in wynik_pn.iterrows():
        pn = str(r["PN"])
        czesci.append({
            "pn": pn,
            "description": str(r.get("MATDESC", ""))[:300],
            "cluster": str(r["cluster_name"]),
            "source": str(r["source"]),
            "confidence": round(float(r["confidence"]), 4),
            "hs6": str(r.get("HS6", "")),
            "bu": str(r.get("BU", "")),
            "n_rows": int(wiersze_na_pn.get(pn, 0)),
        })

    agg = (wynik_pn.groupby("cluster_name")
           .agg(n_pn=("PN", "size"), mean_confidence=("confidence", "mean"))
           .reset_index())
    zrodla_na_klaster = (wynik_pn.groupby(["cluster_name", "source"]).size()
                         .unstack(fill_value=0).to_dict(orient="index"))
    klastry = [{
        "name": str(r["cluster_name"]),
        "n_pn": int(r["n_pn"]),
        "n_rows": int(sum(wiersze_na_pn.get(c["pn"], 0) for c in czesci
                          if c["cluster"] == r["cluster_name"])),
        "mean_confidence": round(float(r["mean_confidence"]), 4),
        "sources": {k: int(v) for k, v in zrodla_na_klaster.get(r["cluster_name"], {}).items()},
        "proposed": str(r["cluster_name"]).startswith(naming.PREFIKS_PROPOZYCJI)
                    or str(r["cluster_name"]).startswith(lc.PREFIKS_NOWY),
    } for _, r in agg.sort_values("n_pn", ascending=False).iterrows()]

    dane = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "directory": out.name,
        "encoder": model.cfg.encoder,
        "naming": args.nazywaj,
        "taxonomy": sorted(str(k) for k in model.klasy_),
        "metrics": {
            "oof_accuracy": round(oof.get("trafnosc", 0.0), 4),
            "ari": round(oof.get("ari", 0.0), 4),
            "pair_f1": round(oof.get("pair_f1", 0.0), 4),
            "nmi": round(oof.get("nmi", 0.0), 4),
            "n_evaluated": int(oof.get("n_ocenianych", 0)),
            "confidence_threshold": round(d.prog_pewnosci, 4),
            "auto_assigned_share": round(d.udzial_przyjetych, 4),
            "auto_assigned_accuracy": round(d.trafnosc_przyjetych, 4),
            "n_classes": d.n_klas,
            "accuracy_target": model.cfg.cel_trafnosci,
        },
        "simulation": {k: v for k, v in (sim or {}).items() if not k.startswith("_")},
        "clusters": klastry,
        "parts": czesci,
        "review_label": lc.ETYKIETA_PRZEGLAD,
    }
    tresc = json.dumps(dane, ensure_ascii=False, indent=1)
    (out / "wynik.json").write_text(tresc, encoding="utf-8")
    OSTATNI_JSON.parent.mkdir(parents=True, exist_ok=True)
    OSTATNI_JSON.write_text(tresc, encoding="utf-8")
    return OSTATNI_JSON


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
    p.add_argument("--nazywaj", default="ctfidf", choices=["ctfidf", "llm", "brak"],
                   help="ETAP 2: jak nazwac grupy z warstwy 2. ctfidf = offline (dom.), "
                        "llm = jedno zbiorcze zapytanie, brak = zostaw NOWY_n")
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
                raise SystemExit(f"Bad --override format: {wpis!r} (expected PN=CLUSTER)")
            pn, klaster = wpis.split("=", 1)
            pary[pn.strip()] = klaster.strip()
        n = lc.zapisz_override(pary)
        print(f"Saved {len(pary)} correction(s) to {lc.OVERRIDES_PATH.name} "
              f"({n} entries total).")
        print("Run again without --override so the model learns from them.")
        return

    print("=" * 70)
    print("  LOCAL HYBRID CLUSTERING  (layers: override / classifier / discovery)")
    print("=" * 70)

    cfg = lc.KonfiguracjaHybrydy(
        encoder=args.encoder,
        pola=tuple(s.strip() for s in args.pola.split(",") if s.strip()),
        cel_trafnosci=args.cel_trafnosci,
        prog_pewnosci=args.prog_pewnosci,
        prog_odkrywania=args.prog_odkrywania,
    )

    print("\n[0] Loading data...")
    try:
        rekordy, pn_df, y, maska, rudolf = wczytaj(args)
    except FileNotFoundError as e:
        raise SystemExit(f"  ERROR: {e}\n  Check the files in data_to_cluster/.")
    print(f"  rows={len(rekordy)}  PN={len(pn_df)}  "
          f"labelled PN={int(maska.sum())}  classes={pd.Series(y[maska]).nunique()}")
    if maska.sum() < 20:
        raise SystemExit("  ERROR: too few labelled PN.")

    # --- predict-only ---
    if args.predict_only:
        if not lc.MODEL_PATH.exists():
            raise SystemExit(f"  ERROR: no saved model found ({lc.MODEL_PATH}).")
        print(f"\n[*] Loading model from {lc.MODEL_PATH.name}")
        model = lc.HybrydowyKlasyfikator.wczytaj()
        wynik_pn = nazwij_odkryte(model, model.predict(pn_df), args.nazywaj)
        out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_local_predict"
        zapisz_wyniki(out, wynik_pn, rekordy, rudolf)
        zapisz_json(out, wynik_pn, rekordy, model, raport_oof_z_modelu(model), args)
        print(f"  Results -> {out.relative_to(KATALOG)}")
        return

    print(f"\n[1] Training layer 1 (encoder: {cfg.encoder})...")
    model = lc.HybrydowyKlasyfikator(cfg).fit(
        pn_df[maska].reset_index(drop=True), y[maska], n_folds=args.folds)

    print("\n[2] Honest evaluation (out-of-fold)...")
    oof = raport_oof(model)
    print(f"  layer 1 (OOF, {oof['n_ocenianych']} PN): accuracy={oof['trafnosc']:.3f}  "
          f"ARI={oof['ari']:.3f}  pair_f1={oof['pair_f1']:.3f}  NMI={oof['nmi']:.3f}")

    tabela = [("layer 1: classifier (OOF)", oof["ari"], oof["pair_f1"])]
    if args.porownaj:
        nien = raport_nienadzorowany(model, pn_df, y, maska)
        print(f"  pure clustering (no labels):            ARI={nien['ari']:.3f}  "
              f"pair_f1={nien['pair_f1']:.3f}")
        tabela.insert(0, ("pure clustering (no labels)", nien["ari"], nien["pair_f1"]))

    sim = {}
    if args.sim_nowe:
        print(f"\n[3] New-type simulation (hiding ~{args.sim_nowe:.0%} of classes)...")
        sim = symulacja_nowych_typow(cfg, pn_df, y, maska, args.sim_nowe, cfg.seed,
                                     metoda_nazywania=args.nazywaj)
        if sim:
            print(f"  hidden classes={sim['n_klas_ukrytych']} ({sim['n_pn_nowych']} PN)")
            print(f"  new types routed to layer 2:      {sim['wykryte_jako_nowe']:.1%}")
            print(f"  false alarm on known types:      {sim['falszywy_alarm']:.1%}")
            if "w1_trafnosc_przyjetych" in sim:
                print(f"  layer 1 accuracy on accepted:    "
                      f"{sim['w1_trafnosc_przyjetych']:.3f}")
            if "w2_pair_precision" in sim:
                print(f"  layer 2 on new types: pair_precision="
                      f"{sim['w2_pair_precision']:.3f}  pair_f1={sim['w2_pair_f1']:.3f}")
            if "nazwy_trafnosc_czesci" in sim:
                print(f"  stage 2 - names matching the true type: "
                      f"{sim['nazwy_trafnosc_czesci']:.1%} of parts / "
                      f"{sim['nazwy_trafnosc_grup']:.1%} of groups")
        else:
            print("  (not enough data for a meaningful simulation)")

    print("\n[4] Assigning all PN + saving...")
    wynik_pn = model.predict(pn_df)
    wynik_pn = nazwij_odkryte(model, wynik_pn, args.nazywaj)
    rozklad = wynik_pn["source"].value_counts().to_dict()
    print(f"  assignment sources: {rozklad}")
    print(f"  clusters in total: {wynik_pn['cluster_name'].nunique()}")

    out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_local_{cfg.encoder.replace(':', '-')}"
    met_wiersze = zapisz_wyniki(out, wynik_pn, rekordy, rudolf)
    zapisz_json(out, wynik_pn, rekordy, model, oof, args, sim)
    model.zapisz()

    zapisz_podsumowanie(out / "podsumowanie.txt", cfg, model, oof, tabela, sim,
                        met_wiersze, rozklad)
    print(f"  results -> {out.relative_to(KATALOG)}")
    if met_wiersze:
        print("\n  NOTE: metrics in ewaluacja.txt are computed on rows for a model "
              "refitted\n  on ALL labels - they are inflated. The OOF numbers above "
              "are the reliable ones.")
    print("\nDone.")


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
                if k.startswith("_"):
                    continue
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
