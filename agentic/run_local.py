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
import discover as dk
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


def tryb_odkrywczy(args, rekordy: pd.DataFrame, pn_df: pd.DataFrame,
                   y: np.ndarray, maska: np.ndarray, rudolf: pd.Series) -> None:
    """Buduje podzial OD ZERA - lokalny odpowiednik toru chmurowego.

    Etykiety Rudolfa NIE sa uzywane do budowania podzialu. Sluza wylacznie do
    zewnetrznej oceny na koncu, zeby bylo wiadomo, ile ten podzial jest wart.
    """
    cfg = dk.KonfiguracjaOdkrywania(
        waga_fizyki=args.waga_fizyki,
        grupowanie=args.grupowanie,
        n_klastrow=args.n_typow,
    )
    pn_df = pn_df.join(dk.cechy_fizyczne(rekordy), on="PN")

    client = None
    if cfg.grupowanie == "cloud":
        try:
            from llm_client import LLMClient, wczytaj_config
            client = LLMClient(wczytaj_config())
        except Exception as e:  # noqa: BLE001
            print(f"  [discover] cloud unavailable ({type(e).__name__}: {str(e)[:90]})")
            print("  [discover] grouping locally instead, fully offline")

    print(f"\n[1] Building taxonomy from scratch (grouping: {cfg.grupowanie})...")
    overrides = lc.load_overrides()
    if overrides:
        print(f"  [layer 0] {len(overrides)} expert corrections in overrides.yaml")
    model = dk.KlastrowaczOdkrywczy(cfg).fit(pn_df, client=client, overrides=overrides)
    if client is not None and client.calls:
        print(f"  [discover] LLM cost: {client.podsumowanie_kosztow()}")

    print("\n[2] Assigning parts...")
    wynik_pn = model.predict(pn_df, overrides=overrides)

    out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_discover_{cfg.grupowanie}"
    met_wiersze = zapisz_wyniki(out, wynik_pn, rekordy, rudolf)

    print("\n[3] External check against Rudolf (labels NOT used to build this)...")
    met = _ocen_odkrycie(wynik_pn, y, maska)
    # Dwie liczby, bo mierza co innego i mylenie ich prowadzilo do zlych wnioskow.
    # Porownywalna z torem chmurowym jest TA PIERWSZA: te same wiersze, to samo
    # pokrycie (kolejka do przegladu wliczona jako wlasny klaster), ten sam
    # evaluate.ocen. W trybie discover nie jest zawyzona - zadna etykieta nie
    # brala udzialu w budowaniu podzialu.
    if met_wiersze:
        print(f"  COMPARABLE with the cloud track (all rows, full coverage):")
        print(f"    ARI={met_wiersze['ari']:.3f}  pair_f1={met_wiersze['pair_f1']:.3f}  "
              f"NMI={met_wiersze['nmi']:.3f}")
        print(f"    cloud agent pipeline on the same measure: ARI 0.885 (~45 requests)")
    if met:
        print(f"  narrower view (per part number, only the ones it was confident about, "
              f"n={met['n']}):")
        print(f"    ARI={met['ari']:.3f}  pair_f1={met['pair_f1']:.3f}  NMI={met['nmi']:.3f}")
    d = model.diagnostyka
    miary = {
        # porownywalne z chmura - to jest liczba, ktora sie liczy
        "ari": (met_wiersze or {}).get("ari", 0.0),
        "pair_f1": (met_wiersze or {}).get("pair_f1", 0.0),
        "nmi": (met_wiersze or {}).get("nmi", 0.0),
        # wezszy widok: per PN, tylko czesci pewne
        "ari_pewne": (met or {}).get("ari", 0.0),
        "pair_f1_pewne": (met or {}).get("pair_f1", 0.0),
        "n_ocenianych": (met or {}).get("n", 0),
    }
    zapisz_json_odkrycie(out, wynik_pn, rekordy, model, miary, args)
    print(f"\n  types discovered: {d.n_grup} | parts for review: {d.n_do_przegladu}")
    print(f"  results -> {out.relative_to(KATALOG)}")
    print("\nDone.")


def _ocen_odkrycie(wynik_pn: pd.DataFrame, y: np.ndarray, maska: np.ndarray
                   ) -> Optional[dict]:
    """ARI/NMI na czesciach, ktorych model byl pewny - wezszy widok pomocniczy."""
    pewne = ~wynik_pn.get("needs_review", pd.Series(False, index=wynik_pn.index)).values
    ma_etykiete = maska & pewne
    if ma_etykiete.sum() < 20:
        return None
    met = evaluate.metryki(y[ma_etykiete], wynik_pn.loc[ma_etykiete, "cluster_name"].values)
    met["n"] = int(ma_etykiete.sum())
    return met


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


def _czesci_i_klastry(wynik_pn: pd.DataFrame, rekordy: pd.DataFrame
                      ) -> tuple[list[dict], list[dict]]:
    """Wspolna dla obu trybow zamiana ramki wynikowej na strukture dla UI."""
    wiersze_na_pn = (rekordy.groupby(rekordy[dp.PN_KOL].astype(str)).size().to_dict()
                     if dp.PN_KOL in rekordy.columns else {})
    czesci = [{
        "pn": str(r["PN"]),
        # surowy opis zostaje (podpowiedz po najechaniu), ale domyslnie
        # pokazujemy wersje bez numerow katalogowych i kodow wariantow
        "description": dp.opis_czytelny(r.get("MATDESC", ""), r["PN"])[:220],
        "description_raw": str(r.get("MATDESC", ""))[:300],
        # cechy fizyczne - te same, ktorych uzywa warstwa 2 przy grupowaniu
        # nieznanych typow; pokazujemy je, zeby bylo widac na czym system pracuje
        **{klucz: (round(float(r[kol]), 2)
                   if kol in wynik_pn.columns and pd.notna(r.get(kol)) else None)
           for klucz, kol in (("weight_g", "waga_g"),
                              ("volume_cm3", "objetosc_cm3"),
                              ("value_eur", "wartosc_eur"))},
        "cluster": str(r["cluster_name"]),
        "source": str(r["source"]),
        "confidence": round(float(r["confidence"]), 4),
        "hs6": str(r.get("HS6", "")),
        "bu": str(r.get("BU", "")),
        "n_rows": int(wiersze_na_pn.get(str(r["PN"]), 0)),
        "type_phrase": str(r.get("type_phrase", "")),
        "needs_review": bool(r.get("needs_review", False)),
    } for _, r in wynik_pn.iterrows()]

    agg = (wynik_pn.groupby("cluster_name")
           .agg(n_pn=("PN", "size"), mean_confidence=("confidence", "mean"))
           .reset_index())
    zrodla = (wynik_pn.groupby(["cluster_name", "source"]).size()
              .unstack(fill_value=0).to_dict(orient="index"))
    klastry = [{
        "name": str(r["cluster_name"]),
        "n_pn": int(r["n_pn"]),
        "n_rows": int(sum(wiersze_na_pn.get(c["pn"], 0) for c in czesci
                          if c["cluster"] == r["cluster_name"])),
        "mean_confidence": round(float(r["mean_confidence"]), 4),
        "sources": {k: int(v) for k, v in zrodla.get(r["cluster_name"], {}).items()},
        "proposed": str(r["cluster_name"]).startswith(naming.PREFIKS_PROPOZYCJI)
                    or str(r["cluster_name"]).startswith(lc.PREFIKS_NOWY),
    } for _, r in agg.sort_values("n_pn", ascending=False).iterrows()]
    return czesci, klastry


def _zapisz_dane(out: Path, dane: dict) -> Path:
    tresc = json.dumps(dane, ensure_ascii=False, indent=1)
    (out / "wynik.json").write_text(tresc, encoding="utf-8")
    OSTATNI_JSON.parent.mkdir(parents=True, exist_ok=True)
    OSTATNI_JSON.write_text(tresc, encoding="utf-8")
    return OSTATNI_JSON


def zapisz_json(out: Path, wynik_pn: pd.DataFrame, rekordy: pd.DataFrame,
                model: lc.HybrydowyKlasyfikator, oof: dict, args,
                sim: Optional[dict] = None) -> Path:
    """Wynik trybu klasyfikujacego w formacie czytanym przez workflow-ui."""
    d = model.diagnostyka
    czesci, klastry = _czesci_i_klastry(wynik_pn, rekordy)
    # Pokrycie liczone na TYM przebiegu. Diagnostyka modelu (udzial_przyjetych)
    # mowi o danych treningowych, wiec kafelek "assigned automatically" pokazywal
    # inna liczbe niz to, co uzytkownik przed chwila policzyl.
    zrodla_biegu = wynik_pn["source"].value_counts()
    auto_w_biegu = float(
        (zrodla_biegu.get("model", 0) + zrodla_biegu.get("override", 0))
        / max(len(wynik_pn), 1))
    return _zapisz_dane(out, {
        "mode": "classify",
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
            "auto_assigned_share": round(auto_w_biegu, 4),
            "auto_assigned_share_validation": round(d.udzial_przyjetych, 4),
            "auto_assigned_accuracy": round(d.trafnosc_przyjetych, 4),
            "n_classes": d.n_klas,
            "accuracy_target": model.cfg.cel_trafnosci,
        },
        "simulation": {k: v for k, v in (sim or {}).items() if not k.startswith("_")},
        "clusters": klastry,
        "parts": czesci,
        "review_label": lc.ETYKIETA_PRZEGLAD,
    })


def zapisz_json_odkrycie(out: Path, wynik_pn: pd.DataFrame, rekordy: pd.DataFrame,
                         model, met: dict, args) -> Path:
    """Wynik trybu odkrywczego. Ten sam ksztalt, zeby UI dzialalo bez zmian.

    Metryki maja inne znaczenie niz w trybie klasyfikujacym: ARI jest tu
    ZEWNETRZNYM sprawdzianem (etykiety nie budowaly podzialu), a nie miara
    trafnosci modelu na swoim zadaniu.
    """
    d = model.diagnostyka
    czesci, klastry = _czesci_i_klastry(wynik_pn, rekordy)
    return _zapisz_dane(out, {
        "mode": "discover",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "directory": out.name,
        "encoder": f"discover/{d.grupowanie}",
        "naming": d.grupowanie,
        "taxonomy": sorted({str(c["name"]) for c in klastry}),
        "metrics": {
            "ari": round(met.get("ari", 0.0), 4),
            "pair_f1": round(met.get("pair_f1", 0.0), 4),
            "nmi": round(met.get("nmi", 0.0), 4),
            "ari_confident_only": round(met.get("ari_pewne", 0.0), 4),
            "pair_f1_confident_only": round(met.get("pair_f1_pewne", 0.0), 4),
            "n_evaluated": int(met.get("n_ocenianych", 0)),
            "n_classes": d.n_grup,
            "n_type_phrases": d.n_fraz,
            "phrases_grouped_by_llm": d.n_fraz_z_llm,
            "phrases_grouped_locally": d.n_fraz_lokalnie,
            "phrases_from_expert": d.n_fraz_od_eksperta,
            "llm_requests": d.n_zapytan,
            "physics_weight": model.cfg.waga_fizyki,
            "confidence_threshold": round(model.prog_pewnosci, 4),
        },
        "simulation": {},
        "clusters": klastry,
        "parts": czesci,
        "review_label": dk.ETYKIETA_PRZEGLAD,
    })


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lokalne klastrowanie hybrydowe (bez LLM)")
    p.add_argument("--tryb", default="classify", choices=["classify", "discover"],
                   help="classify (dom.) = ucz sie na czesciach juz opisanych przez "
                        "eksperta i przypisuj nowe; niepewne i nieznane typy ida do "
                        "warstwy odkrywczej. discover = zbuduj podzial OD ZERA, bez "
                        "zadnych etykiet - tylko na zimny start.")
    p.add_argument("--grupowanie", default="cloud", choices=["cloud", "local"],
                   help="tryb discover: jak pogrupowac frazy typu. cloud = 1 zapytanie "
                        "do LLM (dom.), local = w pelni offline")
    p.add_argument("--waga-fizyki", type=float, default=0.30,
                   help="tryb discover: udzial cech fizycznych (waga/objetosc/wartosc)")
    p.add_argument("--n-typow", type=int, default=None,
                   help="tryb discover, grupowanie local: docelowa liczba typow")
    p.add_argument("--encoder", default="tfidf",
                   help="tfidf | minilm | bge | st:<model>, opcjonalnie +supcon")
    p.add_argument("--plik", default=None, metavar="SCIEZKA",
                   help="plik z czesciami DO POKLASTROWANIA (.csv / .xlsx). Model uczy "
                        "sie na czesciach juz opisanych, a ten plik tylko przypisuje - "
                        "nie musi miec zadnych etykiet.")
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
    p.add_argument("--bez-dodatkowych", action="store_true",
                   help="tryb classify: NIE doczytuj dodatkowego korpusu Rudolfa "
                        "(cla_bez_tbd.xlsx). Mniejsze pokrycie taksonomii.")
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
    naglowek = ("LOCAL DISCOVERY  (builds the taxonomy from scratch, no Rudolf labels)"
                if args.tryb == "discover"
                else "LOCAL HYBRID CLUSTERING  (layers: override / classifier / discovery)")
    print(f"  {naglowek}")
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
    if args.tryb == "discover":
        # podzial budowany bez etykiet - brak etykiet nie jest przeszkoda
        tryb_odkrywczy(args, rekordy, pn_df, y, maska, rudolf)
        return

    if maska.sum() < 20:
        raise SystemExit("  ERROR: too few labelled PN.")

    # WARSTWY 1 i 2 pracuja w przestrzeni z cechami fizycznymi - musza byc w ramce
    pn_df = pn_df.join(dk.cechy_fizyczne(rekordy), on="PN")

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

    df_tren = pn_df[maska].reset_index(drop=True)
    y_tren = y[maska]
    if not args.bez_dodatkowych:
        extra = dp.wczytaj_dodatkowe_etykiety()
        # konflikt rozstrzyga to_cluster - to on jest punktem odniesienia oceny
        extra = extra[~extra["PN"].astype(str).isin(set(df_tren["PN"].astype(str)))]
        if len(extra):
            print(f"  [data] +{len(extra)} PN from the extra labelled corpus "
                  f"({extra['y'].nunique()} classes)")
            df_tren = pd.concat([df_tren, extra[["PN", "MATDESC"]]], ignore_index=True)
            y_tren = np.concatenate([y_tren, extra["y"].values])

    print(f"\n[1] Training layer 1 (encoder: {cfg.encoder})...")
    model = lc.HybrydowyKlasyfikator(cfg).fit(df_tren, y_tren, n_folds=args.folds)

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

    # Plik wskazany przez uzytkownika zastepuje dane DO PRZYPISANIA, ale nie
    # dane treningowe - model zostaje ten, ktory wyuczyl sie na czesciach juz
    # opisanych przez eksperta.
    rekordy_wyj, pn_wyj, rudolf_wyj = rekordy, pn_df, rudolf
    if args.plik:
        print(f"\n[3b] Loading parts to cluster from {Path(args.plik).name}...")
        rekordy_wyj = dp.wczytaj_plik_wejsciowy(Path(args.plik))
        pn_wyj = dp.deduplikuj_do_pn(rekordy_wyj).join(
            dk.cechy_fizyczne(rekordy_wyj), on="PN")
        rudolf_wyj = pd.Series(dtype=str)  # brak etykiet -> brak ewaluacji
        print(f"  rows={len(rekordy_wyj)}  PN={len(pn_wyj)}")

    print("\n[4] Assigning all PN + saving...")
    wynik_pn = model.predict(pn_wyj)
    wynik_pn = nazwij_odkryte(model, wynik_pn, args.nazywaj)
    rozklad = wynik_pn["source"].value_counts().to_dict()
    print(f"  assignment sources: {rozklad}")
    print(f"  clusters in total: {wynik_pn['cluster_name'].nunique()}")

    znacznik = Path(args.plik).stem[:24] if args.plik else cfg.encoder.replace(":", "-")
    out = WYNIKI_DIR / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_local_{znacznik}"
    met_wiersze = zapisz_wyniki(out, wynik_pn, rekordy_wyj, rudolf_wyj)
    zapisz_json(out, wynik_pn, rekordy_wyj, model, oof, args, sim)
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
