# -*- coding: utf-8 -*-
"""
Orkiestrator agentowego klastrowania LLM (Bosch Model Farm).

Przeplyw:
  0. (--check) test polaczenia z Model Farm
  1. Wczytanie danych + dedup do poziomu Product Number (jak Rudolf)
  2. Taksonomia:
       discover        -> AgentOdkrywajacy(probka) -> AgentKonsolidujacy
       seed_from_rudolf-> nazwy Rudolfa (tryb klasyfikacji)
       fixed_list      -> lista z config
  3. AgentKlasyfikujacy przypisuje kazdy PN (partiami, rownolegle, z cache)
  4. Rozpropagowanie PN->wiersze, zapis wynikow
  5. Ewaluacja vs Rudolf (chyba ze --no-eval)

Uruchomienie:
  python run_clustering.py --check
  python run_clustering.py --limit 60      # test na 60 PN (tanio)
  python run_clustering.py                 # pelny zbior
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

import agents
import data_prep as dp
import evaluate
from llm_client import LLMClient, wczytaj_config

KATALOG = Path(__file__).parent
CACHE_DIR = KATALOG / "cache"
WYNIKI_DIR = KATALOG / "wyniki"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=KATALOG / "config.yaml")
    p.add_argument("--check", action="store_true", help="tylko test polaczenia")
    p.add_argument("--dataset", choices=["to_cluster", "cla"], default="to_cluster",
                   help="zrodlo danych: to_cluster.csv lub cla.csv (etykiety tylko do oceny)")
    p.add_argument("--limit", type=int, default=None, help="ogranicz do N PN (test)")
    p.add_argument("--rows", type=int, default=None,
                   help="ogranicz do N pierwszych REKORDOW (wierszy) + ewaluacja na nich")
    p.add_argument("--no-eval", action="store_true", help="pomin ewaluacje")
    p.add_argument("--no-cache", action="store_true", help="ignoruj cache klasyfikacji")
    p.add_argument("--taxonomy", choices=["discover", "seed_from_rudolf", "fixed_list"],
                   default=None, help="nadpisz taxonomy.mode z config.yaml")
    p.add_argument("--no-new", action="store_true",
                   help="zabron LLM proponowania NEW: (musi wybrac z taksonomii)")
    p.add_argument("--features", default=None,
                   help="lista cech part_card po przecinku (nadpisuje features.part_card z config)")
    return p.parse_args()


def _cache_path(cfg: dict, feats=None) -> Path:
    # tryb taksonomii W KLUCZU cache: discover i seed daja rozne etykiety dla
    # tego samego PN, wiec nie moga wspoldzielic pliku cache. Podobnie ZESTAW
    # cech part_card zmienia etykiety -> osobny plik (sygnatura cech w tagu).
    tryb = cfg.get("taxonomy", {}).get("mode", "discover")
    tag = f"{cfg.get('provider','mock')}_{str(cfg.get('model','')).replace('/','-')}_{tryb}"
    if feats:
        sig = hashlib.md5("-".join(sorted(feats)).encode("utf-8")).hexdigest()[:8]
        tag += f"_f{sig}"
    return CACHE_DIR / f"klasyfikacje_{tag}.json"


def zbuduj_taksonomie(client: LLMClient, cfg: dict, pn_df: pd.DataFrame,
                      use_capri: bool, dataset: str = "to_cluster",
                      feats=None) -> list[str]:
    tryb = cfg.get("taxonomy", {}).get("mode", "discover")
    if tryb == "fixed_list":
        etykiety = agents._oczysc_liste(cfg["taxonomy"].get("labels", []))
        if not etykiety:
            raise ValueError("taxonomy.mode=fixed_list ale 'labels' jest puste")
        print(f"  Taksonomia (fixed_list): {len(etykiety)} klastrow")
        return etykiety
    if tryb == "seed_from_rudolf":
        # seed z gotowej listy nazw ground-truth (klasyfikacja nadzorowana).
        # dla 'cla' bierzemy Cluster NAME z cla.csv, inaczej nazwy Rudolfa.
        if dataset == "cla":
            surowe = dp.wczytaj_etykiety_cla()
            zrodlo = "Cluster NAME z cla.csv"
        else:
            surowe = [x for x in dp.wczytaj_etykiety_rudolfa().unique()]
            zrodlo = "nazwy Rudolfa"
        etykiety = agents._oczysc_liste(
            [x for x in surowe if "tbd" not in str(x).lower()])
        print(f"  Taksonomia (seed_from_rudolf, {zrodlo}): {len(etykiety)} klastrow")
        return etykiety

    # discover
    n = int(cfg.get("run", {}).get("discovery_sample", 180))
    probka = pn_df.sample(min(n, len(pn_df)), random_state=42)
    karty = [dp.part_card(r, use_capri, feats) for _, r in probka.iterrows()]
    # dzielimy probke na porcje, kazda daje propozycje, potem konsolidacja
    porcje = [karty[i:i + 60] for i in range(0, len(karty), 60)]
    propozycje: list[str] = []
    for i, porcja in enumerate(porcje):
        czesc = agents.odkryj_taksonomie(client, porcja)
        print(f"  [discovery {i+1}/{len(porcje)}] +{len(czesc)} propozycji")
        propozycje += czesc
    taksonomia = agents.skonsoliduj_taksonomie(client, sorted(set(propozycje)))
    print(f"  Taksonomia (discover->consolidate): {len(taksonomia)} klastrow")
    return taksonomia


def klasyfikuj_wszystko(client: LLMClient, cfg: dict, pn_df: pd.DataFrame,
                        taksonomia: list[str], use_capri: bool,
                        cache: dict, use_cache: bool, feats=None) -> dict[str, str]:
    allow_new = bool(cfg.get("taxonomy", {}).get("allow_new", True))
    batch = int(cfg.get("run", {}).get("batch_size", 25))
    workers = int(cfg.get("run", {}).get("concurrency", 4))

    do_zrobienia = pn_df[~pn_df["PN"].isin(cache)] if use_cache else pn_df
    print(f"  Do sklasyfikowania: {len(do_zrobienia)} PN "
          f"(z cache: {len(pn_df) - len(do_zrobienia)})")

    partie = [do_zrobienia.iloc[i:i + batch] for i in range(0, len(do_zrobienia), batch)]

    def zrob_partie(part_df: pd.DataFrame) -> dict[str, str]:
        idx_pn = {i: r["PN"] for i, (_, r) in enumerate(part_df.iterrows())}
        wsad = [(i, dp.part_card(r, use_capri, feats)) for i, (_, r) in enumerate(part_df.iterrows())]
        przyp = agents.klasyfikuj_partie(client, wsad, taksonomia, allow_new)
        return {idx_pn[i]: kl for i, kl in przyp.items() if i in idx_pn}

    # wynik trzyma TYLKO realne etykiety (cache + odpowiedzi LLM) - to zapisujemy do cache.
    # Braki (nieudana partia / brak id w odpowiedzi) NIE sa cache'owane jako 'INNE', wiec
    # kolejny bieg je ponowi. Do wyniku/oceny 'INNE' dokleja pozniej .fillna() w main().
    wynik = dict(cache)
    if partie:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(zrob_partie, p): j for j, p in enumerate(partie)}
            for done, fut in enumerate(as_completed(futs), 1):
                try:
                    wynik.update(fut.result())
                except Exception as e:  # noqa: BLE001 - jedna partia nie wywala calosci
                    print(f"    [partia nieudana] {str(e)[:120]} -> te PN dostana INNE")
                if done % 5 == 0 or done == len(partie):
                    print(f"    partii {done}/{len(partie)} gotowe")
    return wynik


def main() -> None:
    args = parse_args()
    try:
        cfg = wczytaj_config(args.config)
        client = LLMClient(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(f"[KONFIGURACJA] {e}")
        print("  -> Uzupelnij agentic/config.yaml (api_key + deployment) "
              "lub ustaw provider: mock, aby przetestowac bez tokenu.")
        return

    if args.check:
        print("Test polaczenia Model Farm...")
        print(" ", client.check())
        return

    # nadpisania z CLI (bez ruszania config.yaml, w ktorym jest token)
    if args.taxonomy:
        cfg.setdefault("taxonomy", {})["mode"] = args.taxonomy
    if args.no_new:
        cfg.setdefault("taxonomy", {})["allow_new"] = False

    use_capri = bool(cfg.get("features", {}).get("use_capri", True))
    # cechy part_card: CLI > config.features.part_card > None (=DOMYSLNE_FEATURES)
    feats = ([f.strip() for f in args.features.split(",") if f.strip()] if args.features
             else cfg.get("features", {}).get("part_card")) or None
    limit = args.limit if args.limit is not None else cfg.get("run", {}).get("limit")

    print(f"[1] Dane... (dataset={args.dataset})")
    # cla.csv ma etykiety w pliku -> uzywamy ich TYLKO do oceny, wykluczamy tbd
    rekordy = dp.wczytaj_rekordy(use_capri=use_capri, dataset=args.dataset,
                                 drop_tbd=(args.dataset == "cla"))
    if not use_capri or not dp.CACHE_CAPRI.exists():
        print("  (CaPRI wylaczone lub brak cache - part card bez material_field)")
        use_capri = use_capri and dp.CACHE_CAPRI.exists()

    # --rows: ogranicz do N pierwszych REKORDOW (wiersze wyrownane z Rudolfem
    # pozycyjnie), deduplikuj do PN tylko w tym podzbiorze -> ewaluacja na tych rekordach.
    if args.rows:
        rekordy = rekordy.head(int(args.rows)).copy()
    pn_df = dp.deduplikuj_do_pn(rekordy)
    if limit and not args.rows:
        pn_df = pn_df.head(int(limit))
    print(f"  Wierszy: {len(rekordy)} | unikatowych PN: {len(pn_df)}")

    if feats:
        print(f"  Cechy part_card: {feats}")
    print("[2] Taksonomia...")
    taksonomia = zbuduj_taksonomie(client, cfg, pn_df, use_capri, dataset=args.dataset, feats=feats)

    print("[3] Klasyfikacja PN...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_p = _cache_path(cfg, feats)
    cache = json.loads(cache_p.read_text(encoding="utf-8")) if (cache_p.exists() and not args.no_cache) else {}
    pn_label = klasyfikuj_wszystko(client, cfg, pn_df, taksonomia, use_capri, cache, not args.no_cache, feats=feats)
    cache_p.write_text(json.dumps(pn_label, ensure_ascii=False, indent=0), encoding="utf-8")

    # normalizacja NEW: -> nazwa
    pn_label = {pn: (kl.split("NEW:", 1)[-1].strip() if str(kl).upper().startswith("NEW:") else kl)
                for pn, kl in pn_label.items()}

    print("[4] Zapis wynikow...")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = WYNIKI_DIR / f"{ts}_{cfg.get('provider','mock')}"
    out.mkdir(parents=True, exist_ok=True)

    # Grupowanie WAZNIEJSZE niz nazwa: nadajemy klastrom ID literowe A,B,C,...
    # (wg liczby rekordow). Nazwa zaproponowana przez LLM zostaje jako
    # 'cluster_name' - do ewentualnego nazwania klastra pozniej.
    pn_df = pn_df.copy()
    rekordy = rekordy.copy()
    pn_df["cluster_name"] = pn_df["PN"].map(pn_label).fillna("INNE")
    rekordy["cluster_name"] = rekordy[dp.PN_KOL].map(pn_label).fillna("INNE")

    kolejnosc = rekordy["cluster_name"].value_counts().index.tolist()
    mapa_id = {name: evaluate.litera(i) for i, name in enumerate(kolejnosc)}
    for d in (pn_df, rekordy):
        d["cluster_id"] = d["cluster_name"].map(mapa_id)
        d["cluster"] = d["cluster_id"]  # kolumna do ewaluacji = samo grupowanie

    legenda = (rekordy.groupby(["cluster_id", "cluster_name"]).size()
               .reset_index(name="n_rekordow").sort_values("n_rekordow", ascending=False))
    legenda.to_csv(out / "legenda_klastrow.csv", sep=";", index=False, encoding="utf-8")

    pn_df.to_csv(out / "klastry_per_PN.csv", sep=";", index=False, encoding="utf-8")
    kols = ["MAT_LANE_YM", dp.PN_KOL, "MATDESC", "HS Code First 6 ACDC", "BU SCND",
            "cluster", "cluster_name"]
    rekordy[kols].to_csv(out / "klastry_per_wiersz.csv", sep=";", index=False, encoding="utf-8")

    n_klastrow = pn_df["cluster_id"].nunique()
    print(f"  Klastrow (A,B,C...): {n_klastrow} | wynik -> {out.relative_to(KATALOG.parent)}")
    print(f"  Koszt LLM: {client.podsumowanie_kosztow()}")

    # Ewaluacja: mozliwa gdy klasyfikowalismy WSZYSTKIE wiersze w rekordy
    # (pelny zbior lub --rows). Przy --limit (po PN) reszta wierszy = INNE -> pomijamy.
    if not args.no_eval and not (limit and not args.rows):
        gt_label = "Cluster NAME" if args.dataset == "cla" else "Rudolf"
        print(f"[5] Ewaluacja vs {gt_label}...")
        if args.dataset == "cla":
            rekordy["RUDOLF"] = rekordy["GT"].values           # etykieta z pliku cla
        else:
            rudolf_full = dp.wczytaj_etykiety_rudolfa().values  # z pliku Rudolfa (pozycyjnie)
            rekordy["RUDOLF"] = rudolf_full[rekordy.index.to_numpy()]
        met = evaluate.ocen(rekordy, out, gt_label=gt_label)
        print(f"  Rekordow ocenianych: {len(rekordy)}")
        print(f"  Porownanie po PARACH (nazwy ignorowane): "
              f"pair_f1={met['pair_f1']:.3f}  rand={met['rand_index']:.3f}")
        print(f"  ARI={met['ari']:.3f}  NMI={met['nmi']:.3f}  zgodnosc={met['zgodnosc']:.1%}")
    elif limit:
        print("[5] Ewaluacje pominieto (--limit po PN). Uzyj --rows N dla oceny na podzbiorze.")


if __name__ == "__main__":
    main()
