# -*- coding: utf-8 -*-
"""
CLI: Lokalne klastrowanie części (MiniLM + LightGBM).

Uruchomienie:
  python run_local.py                    # pełny pipeline z cross-validation
  python run_local.py --limit 100        # test na 100 PN
  python run_local.py --folds 10         # 10-fold CV
  python run_local.py --no-eval          # bez ewaluacji (tylko trening + zapis)
  python run_local.py --predict-only     # użyj wytrenowanego modelu do predykcji
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# lokalne moduły
import data_prep as dp
import evaluate
import local_clustering as lc

KATALOG = Path(__file__).parent
WYNIKI_DIR = KATALOG / "wyniki"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lokalne klastrowanie (MiniLM + LightGBM)")
    p.add_argument("--limit", type=int, default=None,
                   help="ogranicz do N PN (test)")
    p.add_argument("--rows", type=int, default=None,
                   help="ogranicz do N pierwszych wierszy")
    p.add_argument("--folds", type=int, default=5,
                   help="liczba foldów cross-validation (domyślnie 5)")
    p.add_argument("--model", default="all-MiniLM-L6-v2",
                   help="model sentence-transformers")
    p.add_argument("--no-eval", action="store_true",
                   help="pomiń ewaluację")
    p.add_argument("--predict-only", action="store_true",
                   help="użyj zapisanego modelu do predykcji (bez retreningu)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 64)
    print("  LOKALNE KLASTROWANIE CZĘŚCI (MiniLM + LightGBM)")
    print("=" * 64)

    # ── Wczytanie danych ──
    print("\n[0] Wczytywanie danych...")
    try:
        rekordy = dp.wczytaj_rekordy(use_capri=True, dataset="to_cluster")
        use_capri = dp.CACHE_CAPRI.exists()
        if not use_capri:
            print("  (CaPRI cache niedostępny — bez material_field/rb_part_name)")
    except FileNotFoundError as e:
        print(f"  BŁĄD: {e}")
        print("  Upewnij się, że pliki danych są w tariff/data_to_cluster/")
        return

    if args.rows:
        rekordy = rekordy.head(int(args.rows)).copy()

    pn_df = dp.deduplikuj_do_pn(rekordy)

    if args.limit:
        pn_df = pn_df.head(int(args.limit))

    print(f"  Wierszy (rekordów): {len(rekordy)}")
    print(f"  Unikatowych PN: {len(pn_df)}")

    # ── Etykiety Rudolfa (ground truth) ──
    print("\n[GT] Wczytywanie etykiet Rudolfa...")
    try:
        rudolf_labels = dp.wczytaj_etykiety_rudolfa()
    except FileNotFoundError:
        print("  BŁĄD: Brak pliku to_cluster_rudolf.xlsx")
        return

    # Mapowanie etykiet na poziom PN
    # Rudolf ma etykiety per wiersz (pozycyjnie), więc mapujemy:
    # dla każdego PN bierzemy modę etykiet z wierszy tego PN
    rekordy_full = dp.wczytaj_rekordy(use_capri=False, dataset="to_cluster")
    rekordy_full["RUDOLF"] = rudolf_labels.values[:len(rekordy_full)]

    pn_labels = {}
    pn_is_tbd = {}
    for pn, g in rekordy_full.groupby(dp.PN_KOL, sort=False):
        labels = g["RUDOLF"].astype(str).str.strip()
        non_tbd = labels[~labels.str.lower().str.contains("tbd")]
        if len(non_tbd) > 0:
            pn_labels[pn] = non_tbd.value_counts().index[0]
            pn_is_tbd[pn] = False
        else:
            pn_labels[pn] = "tbd"
            pn_is_tbd[pn] = True

    y_labels = np.array([pn_labels.get(pn, "tbd") for pn in pn_df["PN"]])
    mask_eval = np.array([not pn_is_tbd.get(pn, True) for pn in pn_df["PN"]])

    n_labeled = mask_eval.sum()
    n_classes = len(set(y_labels[mask_eval]))
    print(f"  PN z etykietą: {n_labeled} / {len(pn_df)}")
    print(f"  Unikatowych klastrów: {n_classes}")

    if n_labeled < 10:
        print("  BŁĄD: Za mało oznaczonych PN do treningu!")
        return

    # ── Predict-only mode ──
    if args.predict_only:
        import pickle
        model_path = lc.MODELS_DIR / "lgbm_clustering.pkl"
        if not model_path.exists():
            print(f"  BŁĄD: Brak zapisanego modelu ({model_path})")
            return
        print(f"\n[PREDICT] Ładuję model z {model_path.name}...")
        with open(model_path, "rb") as f:
            saved = pickle.load(f)

        texts = [lc._build_text_for_embedding(row) for _, row in pn_df.iterrows()]
        embeddings = lc.compute_embeddings(texts, model_name=saved["model_name"])
        X = saved["feature_builder"].transform(pn_df, embeddings)
        pred_labels, confidence = lc.predict_with_confidence(
            saved["model"], X, saved["label_encoder"]
        )
        pn_df = pn_df.copy()
        pn_df["cluster_name"] = pred_labels
        pn_df["confidence"] = confidence
        _save_results(pn_df, rekordy, "local_predict", args)
        return

    # ── Główny pipeline ──
    results = lc.run_pipeline(
        pn_df=pn_df,
        y_labels=y_labels,
        mask_eval=mask_eval,
        model_name=args.model,
        n_folds=args.folds,
        save_model=True,
    )

    # ── Zapis wyników ──
    cv = results["cv_results"]
    _save_results_with_eval(pn_df, rekordy, results, rudolf_labels, args, cv)


def _save_results_with_eval(pn_df, rekordy, results, rudolf_labels, args, cv):
    """Zapis wyników + ewaluacja na pełnym zbiorze."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = WYNIKI_DIR / f"{ts}_local"
    out.mkdir(parents=True, exist_ok=True)

    # Mapowanie predykcji na wiersze
    X_df = results["X_df"]
    pred_labels = results["predictions"]
    confidence = results["confidence"]

    pn_label_map = dict(zip(X_df["PN"], pred_labels))
    pn_conf_map = dict(zip(X_df["PN"], confidence))

    # Overrides nadpisują
    overrides = lc.load_overrides()
    for pn, cl in overrides.items():
        pn_label_map[pn] = cl
        pn_conf_map[pn] = 1.0

    pn_df = pn_df.copy()
    rekordy = rekordy.copy()
    pn_df["cluster_name"] = pn_df["PN"].map(pn_label_map).fillna("INNE")
    pn_df["confidence"] = pn_df["PN"].map(pn_conf_map).fillna(0.0)
    rekordy["cluster_name"] = rekordy[dp.PN_KOL].map(pn_label_map).fillna("INNE")

    # ID literowe A, B, C... wg wielkości (kompatybilne z evaluate.py)
    kolejnosc = rekordy["cluster_name"].value_counts().index.tolist()
    mapa_id = {name: evaluate.litera(i) for i, name in enumerate(kolejnosc)}
    for d in (pn_df, rekordy):
        d["cluster_id"] = d["cluster_name"].map(mapa_id)
        d["cluster"] = d["cluster_id"]

    # Legenda
    legenda = (rekordy.groupby(["cluster_id", "cluster_name"]).size()
               .reset_index(name="n_rekordow").sort_values("n_rekordow", ascending=False))
    legenda.to_csv(out / "legenda_klastrow.csv", sep=";", index=False, encoding="utf-8")

    # Wyniki per PN
    pn_df.to_csv(out / "klastry_per_PN.csv", sep=";", index=False, encoding="utf-8")

    # Wyniki per wiersz
    kols = [c for c in ["MAT_LANE_YM", dp.PN_KOL, "MATDESC",
                         "HS Code First 6 ACDC", "BU SCND",
                         "cluster", "cluster_name"] if c in rekordy.columns]
    rekordy[kols].to_csv(out / "klastry_per_wiersz.csv", sep=";", index=False, encoding="utf-8")

    n_klastrow = pn_df["cluster_id"].nunique()
    print(f"\n[6] Wyniki zapisane → {out.relative_to(KATALOG)}")
    print(f"  Klastrów: {n_klastrow}")

    # Ewaluacja na wierszach
    if not args.no_eval:
        print("\n[7] Ewaluacja vs Rudolf...")
        rudolf_full = dp.wczytaj_etykiety_rudolfa().values
        rekordy["RUDOLF"] = rudolf_full[:len(rekordy)]
        met = evaluate.ocen(rekordy, out, gt_label="Rudolf")
        print(f"  pair_f1={met['pair_f1']:.3f}  ARI={met['ari']:.3f}  "
              f"NMI={met['nmi']:.3f}  zgodność={met['zgodnosc']:.1%}")

        # Zapisz podsumowanie CV + full eval
        with open(out / "podsumowanie.txt", "w", encoding="utf-8") as f:
            f.write("LOKALNE KLASTROWANIE (MiniLM + LightGBM)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model embeddingów: {args.model}\n")
            f.write(f"Foldów CV: {args.folds}\n")
            f.write(f"PN z etykietą: {len(results['X_df'])}\n")
            f.write(f"Klastrów: {n_klastrow}\n\n")

            f.write("── Cross-Validation (średnia) ──\n")
            mean = cv["mean"]
            for k in ["ari", "pair_f1", "pair_precision", "pair_recall",
                       "nmi", "zgodnosc", "homogeneity", "completeness"]:
                if k in mean:
                    f.write(f"  {k:16s} = {mean[k]:.3f}\n")

            f.write(f"\n── Ewaluacja na pełnym zbiorze (refit) ──\n")
            for k in ["ari", "pair_f1", "pair_precision", "pair_recall",
                       "nmi", "zgodnosc"]:
                f.write(f"  {k:16s} = {met[k]:.3f}\n")

            f.write(f"\n── Porównanie ──\n")
            f.write(f"  TF-IDF baseline             ARI ~0.60\n")
            f.write(f"  Ta metoda (CV)              ARI  {mean['ari']:.3f}\n")
            f.write(f"  Gemini 3.5 Flash (chmura)   ARI  0.885\n")

    print("\n✅ Gotowe!")


def _save_results(pn_df, rekordy, tag, args):
    """Uproszczony zapis (predict-only)."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = WYNIKI_DIR / f"{ts}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    pn_df.to_csv(out / "klastry_per_PN.csv", sep=";", index=False, encoding="utf-8")
    print(f"\n  Wyniki → {out.relative_to(KATALOG)}")


if __name__ == "__main__":
    main()
