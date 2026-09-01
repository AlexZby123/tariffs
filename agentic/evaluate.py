# -*- coding: utf-8 -*-
"""
Ewaluacja klastrowania LLM wzgledem recznego podzialu Rudolfa.

Te same metryki co reszta projektu (ARI/NMI/V/homog/compl + zgodnosc Hungarian),
liczone na wierszach ze znana etykieta (bez 'tbd'). Wiersze sa wyrownane
pozycyjnie z to_cluster.csv (zweryfikowane w kod/porownaj_z_rudolfem.py).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score, completeness_score, confusion_matrix,
    homogeneity_score, normalized_mutual_info_score, v_measure_score,
)
from sklearn.metrics.cluster import pair_confusion_matrix


def _is_tbd(x: str) -> bool:
    return "tbd" in str(x).lower()


def pary_metryki(y_true, y_pred) -> dict:
    """Porownanie po WSPOLPRZYNALEZNOSCI PAR - nazwy klastrow sa ignorowane.

    Dla kazdej pary rekordow patrzymy tylko: czy sa razem w tym samym klastrze?
      TP = para razem u Rudolfa I razem u kodu
      FP = para razem u kodu, ale osobno u Rudolfa
      FN = para razem u Rudolfa, ale osobno u kodu
    pair_precision = TP/(TP+FP)  - jak "czyste" sa grupy kodu
    pair_recall    = TP/(TP+FN)  - ile par Rudolfa kod utrzymal razem
    rand_index     = (TP+TN)/wszystkie_pary
    """
    (tn, fp), (fn, tp) = pair_confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    rand = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"pair_precision": prec, "pair_recall": rec, "pair_f1": f1,
            "rand_index": rand}


def _zgodnosc_hungarian(y_true, y_pred) -> float:
    yt = pd.factorize(y_true)[0]
    yp = pd.factorize(y_pred)[0]
    cm = confusion_matrix(yt, yp)
    w, k = linear_sum_assignment(-cm)
    return float(cm[w, k].sum()) / cm.sum()


def metryki(y_rudolf, y_pred) -> dict:
    m = {
        "n_klastrow_llm": len(set(y_pred)),
        "ari": adjusted_rand_score(y_rudolf, y_pred),
        "nmi": normalized_mutual_info_score(y_rudolf, y_pred),
        "homogeneity": homogeneity_score(y_rudolf, y_pred),
        "completeness": completeness_score(y_rudolf, y_pred),
        "v_measure": v_measure_score(y_rudolf, y_pred),
        "zgodnosc": _zgodnosc_hungarian(y_rudolf, y_pred),
    }
    m.update(pary_metryki(y_rudolf, y_pred))  # metryki parowe (bez nazw)
    return m


def ocen(rows_df: pd.DataFrame, out_dir: Path, gt_label: str = "Rudolf") -> dict:
    """rows_df musi miec kolumny 'RUDOLF' (ground-truth) i 'cluster' (grupowanie kodu).
    gt_label: nazwa zrodla etykiet do raportu (np. 'Rudolf' albo 'Cluster NAME')."""
    rud = rows_df["RUDOLF"].astype(str).values
    llm = rows_df["cluster"].astype(str).values
    mask = np.array([not _is_tbd(x) for x in rud])

    met = metryki(rud[mask], llm[mask])

    # mapa rozbieznosci: klaster Rudolfa -> klastry LLM
    d = rows_df[mask].copy()
    rek = []
    for cl, g in d.groupby("RUDOLF"):
        lic = Counter(g["cluster"])
        dom, dn = lic.most_common(1)[0]
        rek.append({"rudolf": cl, "n": len(g), "n_klastrow_llm": len(lic),
                    "czystosc": dn / len(g), "dominujacy_llm": dom,
                    "rozklad": "; ".join(f"{k}({v})" for k, v in lic.most_common(4))})
    r2l = pd.DataFrame(rek).sort_values("n", ascending=False)
    r2l.to_csv(out_dir / "ewaluacja_rudolf_do_llm.csv", sep=";", index=False, encoding="utf-8")

    with open(out_dir / "ewaluacja.txt", "w", encoding="utf-8") as f:
        f.write(f"EWALUACJA klastrowania vs {gt_label} (bez 'tbd')\n")
        f.write("NAZWY KLASTROW SA IGNOROWANE - porownujemy tylko GRUPOWANIE rekordow.\n")
        f.write("=" * 64 + "\n")
        f.write(f"Rekordow ocenianych: {int(mask.sum())}\n")
        f.write(f"Klastrow kodu (na ocenianych): {met['n_klastrow_llm']}\n\n")

        f.write("A) WSPOLPRZYNALEZNOSC PAR (czy dwa rekordy sa razem w klastrze):\n")
        f.write(f"  pair_precision = {met['pair_precision']:.3f}  "
                f"(z par razem u KODU - ile tez razem u {gt_label})\n")
        f.write(f"  pair_recall    = {met['pair_recall']:.3f}  "
                f"(z par razem u {gt_label} - ile tez razem u kodu)\n")
        f.write(f"  pair_f1        = {met['pair_f1']:.3f}\n")
        f.write(f"  rand_index     = {met['rand_index']:.3f}  "
                "(odsetek par sklasyfikowanych zgodnie: razem/osobno)\n")
        f.write(f"  ARI            = {met['ari']:.3f}  (Rand skorygowany o przypadek)\n\n")

        f.write("B) METRYKI INFORMACYJNE (tez niezalezne od nazw):\n")
        for k in ["nmi", "v_measure", "homogeneity", "completeness"]:
            f.write(f"  {k:14s} = {met[k]:.3f}\n")
        f.write(f"  zgodnosc(Hung) = {met['zgodnosc']:.3f}  "
                "(najlepsze dopasowanie klastrow, tez bez wzgledu na nazwy)\n")

        f.write("\nPorownanie metod (ARI / pair_f1):\n")
        f.write("  TF-IDF + opis + CaPRI          0.60 /  -\n")
        f.write(f"  ta metoda                      {met['ari']:.2f} / {met['pair_f1']:.2f}\n")

        f.write(f"\nGrupy {gt_label} najbardziej rozbite (na rozne klastry kodu):\n")
        for _, r in r2l[r2l["n"] >= 20].sort_values("czystosc").head(12).iterrows():
            f.write(f"  {str(r['rudolf'])[:40]:40s} n={r['n']:4d} czystosc={r['czystosc']:.2f} "
                    f"-> {r['rozklad']}\n")

    return met


# ============================================================
#  CLI: ponowna ewaluacja istniejacego pliku wynikowego
#  python evaluate.py <klastry_per_wiersz.csv>
#  (nazwy klastrow zamieniane na A,B,C - liczy sie tylko grupowanie)
#  UWAGA: GT to zawsze Rudolf (to_cluster_rudolf.xlsx), wyrownany pozycyjnie -
#  dziala dla wynikow datasetu 'to_cluster', nie 'cla'.
# ============================================================

def litera(n: int) -> str:
    """Numer klastra -> etykieta literowa A, B, ..., Z, AA, AB, ... (wg wielkosci)."""
    s = ""
    while True:
        s = chr(65 + n % 26) + s
        n = n // 26
        if n == 0:
            break
        n -= 1
    return s


if __name__ == "__main__":
    import sys
    import data_prep as dp

    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not csv_path or not csv_path.exists():
        raise SystemExit("Uzycie: python evaluate.py <sciezka/klastry_per_wiersz.csv>")

    df = pd.read_csv(csv_path, sep=";", dtype=str)
    if "cluster" not in df.columns:
        raise SystemExit("Plik nie ma kolumny 'cluster'.")

    # nazwy -> ID literowe A,B,C wg wielkosci (grupowanie wazniejsze niz nazwa)
    kolejnosc = df["cluster"].value_counts().index.tolist()
    mapa_id = {name: litera(i) for i, name in enumerate(kolejnosc)}
    df["cluster_name_proposed"] = df["cluster"]
    df["cluster"] = df["cluster"].map(mapa_id)

    # etykiety Rudolfa wyrownane pozycyjnie (plik zachowuje kolejnosc to_cluster.csv)
    rud = dp.wczytaj_etykiety_rudolfa().values
    if len(df) > len(rud):
        raise SystemExit(f"Plik ma {len(df)} wierszy > {len(rud)} w Rudolfie.")
    df["RUDOLF"] = rud[:len(df)]

    # zapis legendy id->nazwa + przeliczenie metryk
    legenda = (df.groupby(["cluster", "cluster_name_proposed"]).size()
               .reset_index(name="n_rekordow").sort_values("n_rekordow", ascending=False))
    legenda.to_csv(csv_path.parent / "legenda_klastrow.csv", sep=";", index=False, encoding="utf-8")
    df[["MAT_LANE_YM", "cluster", "cluster_name_proposed"]].to_csv(
        csv_path.parent / "klastry_ABC_per_wiersz.csv", sep=";", index=False, encoding="utf-8")

    met = ocen(df, csv_path.parent)
    print(f"Re-ewaluacja {csv_path.name} (klastry jako A,B,C, nazwy ignorowane):")
    print(f"  pair_precision={met['pair_precision']:.3f}  pair_recall={met['pair_recall']:.3f}  "
          f"pair_f1={met['pair_f1']:.3f}")
    print(f"  rand_index={met['rand_index']:.3f}  ARI={met['ari']:.3f}  NMI={met['nmi']:.3f}")
    print(f"  Legenda id->nazwa: {(csv_path.parent / 'legenda_klastrow.csv').name}")
