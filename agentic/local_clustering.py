# -*- coding: utf-8 -*-
"""
Lokalne klastrowanie części — Bi-Encoder (MiniLM) + LightGBM.

Pipeline:
  1. Twardy słownik overrides (PN → klaster, 100% determinizm)
  2. Sentence-Transformers: all-MiniLM-L6-v2 → 384-dim embeddingi semantyczne
  3. Fuzja cech: embeddingi + TF-IDF (char n-gram) + one-hot (HS6, BU)
  4. LightGBM supervised classifier trenowany na etykietach Rudolfa
  5. Cross-validation 5-fold → metryki ARI, pair_f1, Hungarian accuracy

Używa istniejącego data_prep.py do wczytywania danych i evaluate.py do ewaluacji.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize

import lightgbm as lgb

# ścieżki
KATALOG = Path(__file__).parent
OVERRIDES_PATH = KATALOG / "overrides.yaml"
MODELS_DIR = KATALOG / "models"
EMBEDDINGS_CACHE = MODELS_DIR / "embeddings_cache.npy"
EMBEDDINGS_PN_CACHE = MODELS_DIR / "embeddings_pn_cache.pkl"

# --------------------------------------------------------------------------- #
#  1. OVERRIDES — twardy słownik ekspercki
# --------------------------------------------------------------------------- #

def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, str]:
    """Wczytaj PN→klaster ze słownika eksperckiego (YAML)."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or not isinstance(data, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in data.items()}


# --------------------------------------------------------------------------- #
#  2. EMBEDDINGI — Sentence-Transformers (MiniLM)
# --------------------------------------------------------------------------- #

def _build_text_for_embedding(row: pd.Series) -> str:
    """Buduje tekst wejściowy do embeddingu z dostępnych pól opisu."""
    parts = []
    # Główny sygnał: opis materiałowy
    matdesc = str(row.get("MATDESC", "")).strip()
    if matdesc and matdesc != "(brak opisu)":
        parts.append(matdesc)
    # Dodatkowe konteksty (jeśli dostępne)
    for col in ["rb_part_name", "part_type", "HS6_text"]:
        val = str(row.get(col, "")).strip()
        if val and val not in ("", "nan", "None", "?"):
            parts.append(val)
    return " | ".join(parts) if parts else "unknown part"


def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    cache_path: Optional[Path] = None,
    cache_key: Optional[str] = None,
) -> np.ndarray:
    """Generuje embeddingi MiniLM na CPU. Opcjonalnie cache'uje do pliku .npy.

    Returns:
        np.ndarray o kształcie (len(texts), 384)
    """
    # sprawdź cache
    if cache_path and cache_path.exists() and cache_key:
        meta_path = cache_path.with_suffix(".meta")
        if meta_path.exists() and meta_path.read_text(encoding="utf-8").strip() == cache_key:
            print(f"  [embeddingi] wczytano z cache ({cache_path.name})")
            return np.load(cache_path)

    from sentence_transformers import SentenceTransformer
    print(f"  [embeddingi] ładuję model {model_name}...")
    model = SentenceTransformer(model_name)
    print(f"  [embeddingi] generuję wektory dla {len(texts)} tekstów...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalized → cosine = dot product
    )
    embeddings = np.array(embeddings, dtype=np.float32)

    # zapisz cache
    if cache_path and cache_key:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        cache_path.with_suffix(".meta").write_text(cache_key, encoding="utf-8")
        print(f"  [embeddingi] zapisano cache → {cache_path.name}")

    return embeddings


# --------------------------------------------------------------------------- #
#  3. FUZJA CECH — embeddingi + TF-IDF + one-hot
# --------------------------------------------------------------------------- #

class FeatureBuilder:
    """Buduje wektor cech: embedding + TF-IDF(SVD) + one-hot(HS6, BU)."""

    def __init__(self,
                 emb_weight: float = 0.65,
                 tfidf_weight: float = 0.15,
                 hs6_weight: float = 0.10,
                 bu_weight: float = 0.05,
                 other_weight: float = 0.05,
                 svd_dim: int = 64,
                 hs6_top_n: int = 50):
        self.emb_weight = emb_weight
        self.tfidf_weight = tfidf_weight
        self.hs6_weight = hs6_weight
        self.bu_weight = bu_weight
        self.other_weight = other_weight
        self.svd_dim = svd_dim
        self.hs6_top_n = hs6_top_n

        self.tfidf = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 6),
            max_features=5000, sublinear_tf=True,
        )
        self.tfidf_word = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2),
            max_features=3000, sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=svd_dim, random_state=42)
        self.le_hs6 = None  # fitted top-N HS6 codes
        self.le_bu = None   # fitted BU values
        self._fitted = False

    def fit_transform(self, pn_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """Fit na danych treningowych i zwróć macierz cech."""
        texts = pn_df["MATDESC"].fillna("").values

        # TF-IDF char + word → concat → SVD
        tfidf_char = self.tfidf.fit_transform(texts)
        tfidf_word = self.tfidf_word.fit_transform(texts)
        tfidf_combined = sp_hstack([tfidf_char, tfidf_word])
        tfidf_svd = self.svd.fit_transform(tfidf_combined)
        tfidf_svd = normalize(tfidf_svd)

        # HS6 one-hot (top-N)
        hs6_onehot = self._build_hs6_onehot(pn_df, fit=True)

        # BU one-hot
        bu_onehot = self._build_bu_onehot(pn_df, fit=True)

        self._fitted = True
        return self._concat_weighted(embeddings, tfidf_svd, hs6_onehot, bu_onehot)

    def transform(self, pn_df: pd.DataFrame, embeddings: np.ndarray) -> np.ndarray:
        """Transform na nowych danych (po fit)."""
        assert self._fitted, "Najpierw wywołaj fit_transform!"
        texts = pn_df["MATDESC"].fillna("").values

        tfidf_char = self.tfidf.transform(texts)
        tfidf_word = self.tfidf_word.transform(texts)
        tfidf_combined = sp_hstack([tfidf_char, tfidf_word])
        tfidf_svd = self.svd.transform(tfidf_combined)
        tfidf_svd = normalize(tfidf_svd)

        hs6_onehot = self._build_hs6_onehot(pn_df, fit=False)
        bu_onehot = self._build_bu_onehot(pn_df, fit=False)

        return self._concat_weighted(embeddings, tfidf_svd, hs6_onehot, bu_onehot)

    def _concat_weighted(self, emb, tfidf_svd, hs6, bu) -> np.ndarray:
        """Konkatenacja z wagami → normalizacja końcowa."""
        parts = [
            emb * self.emb_weight,
            tfidf_svd * self.tfidf_weight,
            hs6 * self.hs6_weight,
            bu * self.bu_weight,
        ]
        return np.hstack(parts).astype(np.float32)

    def _build_hs6_onehot(self, pn_df: pd.DataFrame, fit: bool) -> np.ndarray:
        hs6 = pn_df["HS6"].fillna("OTHER").astype(str).str.strip()
        if fit:
            top = hs6.value_counts().head(self.hs6_top_n).index.tolist()
            self.le_hs6 = {v: i for i, v in enumerate(top)}
        n_cols = len(self.le_hs6) + 1  # +1 for OTHER
        result = np.zeros((len(pn_df), n_cols), dtype=np.float32)
        for i, val in enumerate(hs6):
            idx = self.le_hs6.get(val, n_cols - 1)
            result[i, idx] = 1.0
        return result

    def _build_bu_onehot(self, pn_df: pd.DataFrame, fit: bool) -> np.ndarray:
        bu = pn_df["BU"].fillna("OTHER").astype(str).str.strip()
        if fit:
            vals = sorted(bu.unique())
            self.le_bu = {v: i for i, v in enumerate(vals)}
        n_cols = len(self.le_bu) + 1
        result = np.zeros((len(pn_df), n_cols), dtype=np.float32)
        for i, val in enumerate(bu):
            idx = self.le_bu.get(val, n_cols - 1)
            result[i, idx] = 1.0
        return result


# --------------------------------------------------------------------------- #
#  4. LIGHTGBM — trening + cross-validation
# --------------------------------------------------------------------------- #

def train_lgbm(X: np.ndarray, y: np.ndarray,
               params: Optional[dict] = None) -> lgb.LGBMClassifier:
    """Trenuje LightGBM na pełnym zbiorze. Zwraca wytrenowany model."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 8,
            "num_leaves": 63,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 5,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)
    return model


def cross_validate(X: np.ndarray, y: np.ndarray, label_encoder: LabelEncoder,
                   n_folds: int = 5) -> dict:
    """Stratified K-Fold CV. Zwraca metryki per fold + średnie."""
    from evaluate import metryki as eval_metryki

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = train_lgbm(X_train, y_train)
        y_pred = model.predict(X_val)

        # dekoduj z powrotem na nazwy klastrów (evaluate.py tego potrzebuje)
        y_true_names = label_encoder.inverse_transform(y_val)
        y_pred_names = label_encoder.inverse_transform(y_pred)

        met = eval_metryki(y_true_names, y_pred_names)
        met["fold"] = fold_i + 1
        fold_metrics.append(met)

        all_y_true.extend(y_true_names)
        all_y_pred.extend(y_pred_names)

        print(f"    Fold {fold_i+1}/{n_folds}: ARI={met['ari']:.3f}  "
              f"pair_f1={met['pair_f1']:.3f}  zgodnosc={met['zgodnosc']:.1%}")

    # metryki globalne (na wszystkich predykcjach CV)
    overall = eval_metryki(np.array(all_y_true), np.array(all_y_pred))
    overall["type"] = "overall_cv"

    # średnia per-fold
    avg = {k: np.mean([m[k] for m in fold_metrics])
           for k in fold_metrics[0] if k != "fold"}
    avg["type"] = "mean_fold"

    return {
        "folds": fold_metrics,
        "overall": overall,
        "mean": avg,
    }


def predict_with_confidence(model: lgb.LGBMClassifier, X: np.ndarray,
                            label_encoder: LabelEncoder
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Predykcja + confidence (max prawdopodobieństwo klasy)."""
    proba = model.predict_proba(X)
    pred_idx = np.argmax(proba, axis=1)
    confidence = np.max(proba, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_idx)
    return pred_labels, confidence


# --------------------------------------------------------------------------- #
#  5. PEŁNY PIPELINE
# --------------------------------------------------------------------------- #

def run_pipeline(
    pn_df: pd.DataFrame,
    y_labels: np.ndarray,
    mask_eval: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
    n_folds: int = 5,
    save_model: bool = True,
) -> dict:
    """Pełny pipeline: overrides → embeddingi → cechy → CV → train final.

    Args:
        pn_df: DataFrame z deduplikowanymi PN (z data_prep.deduplikuj_do_pn)
        y_labels: etykiety Rudolfa per PN (np.ndarray of str)
        mask_eval: maska boolean — True = rekord z etykietą (nie tbd)
        model_name: model sentence-transformers
        n_folds: liczba foldów cross-validation
        save_model: czy zapisać model do pliku
    """
    print("\n[1] Overrides...")
    overrides = load_overrides()
    n_overridden = sum(1 for pn in pn_df["PN"] if pn in overrides)
    print(f"  Znaleziono {n_overridden} PN w słowniku overrides "
          f"(z {len(overrides)} wpisów)")

    # Aplikuj overrides do etykiet (nadpisz tam gdzie override istnieje)
    for i, row in pn_df.iterrows():
        pn = row["PN"]
        if pn in overrides:
            y_labels[i] = overrides[pn]
            mask_eval[i] = True  # override = pewna etykieta

    # Filtruj do rekordów z etykietą
    X_df = pn_df[mask_eval].reset_index(drop=True)
    y = y_labels[mask_eval]
    print(f"  Rekordów z etykietą: {len(X_df)} (z {len(pn_df)} PN)")

    print("\n[2] Embeddingi (MiniLM)...")
    texts = [_build_text_for_embedding(row) for _, row in X_df.iterrows()]
    cache_key = hashlib.md5(
        ("_".join(texts[:10]) + str(len(texts))).encode()
    ).hexdigest()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = compute_embeddings(
        texts, model_name=model_name,
        cache_path=EMBEDDINGS_CACHE, cache_key=cache_key,
    )

    print(f"\n[3] Fuzja cech... (embeddings {embeddings.shape[1]}d + TF-IDF + HS6 + BU)")
    fb = FeatureBuilder()
    X = fb.fit_transform(X_df, embeddings)
    print(f"  Wymiar końcowy: {X.shape}")

    # Enkoduj etykiety
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)
    print(f"  Klas: {n_classes}")

    print(f"\n[4] Cross-validation ({n_folds}-fold)...")
    cv_results = cross_validate(X, y_encoded, le, n_folds=n_folds)

    mean = cv_results["mean"]
    print(f"\n  ── ŚREDNIA CV ──")
    print(f"  ARI       = {mean['ari']:.3f}")
    print(f"  pair_f1   = {mean['pair_f1']:.3f}")
    print(f"  pair_prec = {mean['pair_precision']:.3f}")
    print(f"  pair_rec  = {mean['pair_recall']:.3f}")
    print(f"  zgodność  = {mean['zgodnosc']:.1%}")
    print(f"  NMI       = {mean['nmi']:.3f}")

    print(f"\n[5] Trening finalnego modelu (pełny zbiór)...")
    final_model = train_lgbm(X, y_encoded)
    pred_labels, confidence = predict_with_confidence(final_model, X, le)

    # Statystyki confidence
    low_conf = (confidence < 0.7).sum()
    print(f"  Pewność: min={confidence.min():.2f}  "
          f"mean={confidence.mean():.2f}  "
          f"median={np.median(confidence):.2f}")
    print(f"  Niska pewność (<0.7): {low_conf} PN ({low_conf/len(X)*100:.1f}%)")

    if save_model:
        model_path = MODELS_DIR / "lgbm_clustering.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": final_model,
                "label_encoder": le,
                "feature_builder": fb,
                "model_name": model_name,
            }, f)
        print(f"  Model zapisany → {model_path.relative_to(KATALOG)}")

    return {
        "cv_results": cv_results,
        "final_model": final_model,
        "label_encoder": le,
        "feature_builder": fb,
        "predictions": pred_labels,
        "confidence": confidence,
        "X_df": X_df,
    }
