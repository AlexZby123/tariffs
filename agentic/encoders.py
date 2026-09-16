# -*- coding: utf-8 -*-
"""
Wymienne enkodery tekstu (part card -> wektor) dla lokalnego klastrowania.

Wszystkie enkodery zwracaja macierz float32 znormalizowana L2, wiec iloczyn
skalarny = podobienstwo kosinusowe.

Dostepne backendy (spec przekazywany do zbuduj_enkoder):
  "tfidf"                      - TF-IDF (char_wb 3-5 + word 1-2) -> SVD. Offline,
                                 deterministyczny, bez pobierania czegokolwiek.
  "st:<nazwa-modelu>"          - sentence-transformers (bi-encoder), np.
                                 "st:all-MiniLM-L6-v2", "st:BAAI/bge-small-en-v1.5".
                                 Wymaga dostepu do HuggingFace przy pierwszym uruchomieniu.
  "<dowolny>+supcon"           - na wierzchu powyzszego doklejana jest mala siec
                                 (PyTorch) uczona metryka kontrastywna na etykietach
                                 eksperta, np. "tfidf+supcon".

UWAGA (zmierzone na to_cluster.csv, 962 PN / 77 klas):
  +supcon wyraznie poprawia klastrowanie ZNANYCH typow (ARI 0.599 -> 0.753),
  ale POGARSZA wykrywanie typow NIEWIDZIANYCH w treningu (ARI 0.910 -> 0.342
  na symulacji nowych klas). Dlatego tor odkrywania nowych typow w
  local_clustering.py celowo uzywa embeddingu BAZOWEGO, nie douczonego.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from scipy.sparse import hstack as sp_hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

KATALOG = Path(__file__).parent
CACHE_DIR = KATALOG / "models" / "cache_embeddingow"


# --------------------------------------------------------------------------- #
#  Interfejs
# --------------------------------------------------------------------------- #

class BaseEncoder:
    """Wspolny interfejs enkodera tekstu."""

    spec: str = "base"
    #: czy fit() korzysta z etykiet (wtedy trzeba je podac)
    wymaga_etykiet: bool = False

    def fit(self, texts: Sequence[str], y: Optional[np.ndarray] = None) -> "BaseEncoder":
        raise NotImplementedError

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, texts: Sequence[str],
                      y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)

    @property
    def wymiar(self) -> int:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
#  TF-IDF + SVD (offline, bez pobierania modelu)
# --------------------------------------------------------------------------- #

class TfidfEncoder(BaseEncoder):
    """TF-IDF (znaki + slowa) -> TruncatedSVD -> L2.

    Odporny na literowki i sklejone tokeny w MATDESC dzieki n-gramom znakowym.
    """

    def __init__(self, svd_dim: int = 200,
                 char_ngram: tuple[int, int] = (3, 5),
                 word_ngram: tuple[int, int] = (1, 2),
                 max_char_features: int = 20000,
                 max_word_features: int = 10000,
                 random_state: int = 0):
        self.spec = "tfidf"
        self.svd_dim = svd_dim
        self.tfidf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram,
                                          max_features=max_char_features, sublinear_tf=True)
        self.tfidf_word = TfidfVectorizer(analyzer="word", ngram_range=word_ngram,
                                          max_features=max_word_features, sublinear_tf=True)
        self.svd: Optional[TruncatedSVD] = None
        self.random_state = random_state

    def _raw(self, texts: Sequence[str], fit: bool):
        f = (lambda v, t: v.fit_transform(t)) if fit else (lambda v, t: v.transform(t))
        return sp_hstack([f(self.tfidf_char, texts), f(self.tfidf_word, texts)]).tocsr()

    def fit(self, texts: Sequence[str], y: Optional[np.ndarray] = None) -> "TfidfEncoder":
        X = self._raw(list(texts), fit=True)
        # SVD nie moze miec wiecej skladowych niz min(n_probek, n_cech) - 1
        dim = max(2, min(self.svd_dim, min(X.shape) - 1))
        self.svd = TruncatedSVD(n_components=dim, random_state=self.random_state).fit(X)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        assert self.svd is not None, "Najpierw fit()."
        X = self._raw(list(texts), fit=False)
        return normalize(self.svd.transform(X)).astype(np.float32)

    @property
    def wymiar(self) -> int:
        return int(self.svd.n_components) if self.svd is not None else 0


# --------------------------------------------------------------------------- #
#  Bi-encoder neuronowy (sentence-transformers / HuggingFace)
# --------------------------------------------------------------------------- #

class SentenceTransformerEncoder(BaseEncoder):
    """Pretrenowany bi-encoder (MiniLM / BGE). Wymaga sentence-transformers.

    Model pobierany jest z HuggingFace przy pierwszym uzyciu i cache'owany przez
    samo sentence-transformers (~/.cache/huggingface). W srodowiskach bez dostepu
    do HF ten backend nie zadziala - uzyj "tfidf".
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 128,
                 cache_dir: Optional[Path] = CACHE_DIR):
        self.spec = f"st:{model_name}"
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._dim = 0

    def _lazy(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - zalezy od srodowiska
                raise ImportError(
                    "Backend 'st:' wymaga pakietu sentence-transformers.\n"
                    "  pip install sentence-transformers\n"
                    "Bez dostepu do HuggingFace uzyj --encoder tfidf."
                ) from e
            print(f"  [enkoder] laduje {self.model_name} ...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, texts: Sequence[str], y: Optional[np.ndarray] = None
            ) -> "SentenceTransformerEncoder":
        self._lazy()  # model jest pretrenowany - "fit" to tylko zaladowanie
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        klucz = _klucz_cache(self.spec, texts)
        plik = self.cache_dir / f"{klucz}.npy" if self.cache_dir else None
        if plik is not None and plik.exists():
            emb = np.load(plik)
            print(f"  [enkoder] embeddingi z cache ({plik.name})")
            self._dim = emb.shape[1]
            return emb

        model = self._lazy()
        emb = model.encode(texts, batch_size=self.batch_size, show_progress_bar=len(texts) > 500,
                           normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        self._dim = emb.shape[1]
        if plik is not None:
            plik.parent.mkdir(parents=True, exist_ok=True)
            np.save(plik, emb)
        return emb

    @property
    def wymiar(self) -> int:
        return self._dim


# --------------------------------------------------------------------------- #
#  Glowica metryczna (supervised contrastive) na wierzchu dowolnego enkodera
# --------------------------------------------------------------------------- #

class SupConEncoder(BaseEncoder):
    """Mala siec MLP uczona strata SupCon na etykietach eksperta.

    Zaciska razem czesci tego samego typu, rozpycha rozne typy. Uczy sie w
    kilkanascie sekund na CPU (962 PN). Wymaga PyTorch.
    """

    wymaga_etykiet = True

    def __init__(self, base: BaseEncoder, out_dim: int = 128, hidden: int = 512,
                 epochs: int = 120, batch_size: int = 256, lr: float = 1e-3,
                 dropout: float = 0.2, noise: float = 0.05, temperature: float = 0.1,
                 seed: int = 0):
        self.spec = f"{base.spec}+supcon"
        self.base = base
        self.out_dim, self.hidden = out_dim, hidden
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.dropout, self.noise, self.temperature = dropout, noise, temperature
        self.seed = seed
        self._net = None

    # --- siec ---
    def _zbuduj(self, d_in: int):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(d_in, self.hidden), nn.GELU(), nn.Dropout(self.dropout),
            nn.Linear(self.hidden, self.hidden // 2), nn.GELU(),
            nn.Linear(self.hidden // 2, self.out_dim),
        )

    @staticmethod
    def _supcon(emb, lab, t: float):
        """Supervised contrastive loss (Khosla i in., 2020) - wariant in-batch."""
        import torch
        sim = (emb @ emb.T) / t
        n = len(lab)
        eye = torch.eye(n, dtype=torch.bool, device=emb.device)
        pozytywy = (lab[:, None] == lab[None, :]) & ~eye
        sim = sim.masked_fill(eye, -1e9)
        logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        licz = pozytywy.sum(1)
        ma_pary = licz > 0
        if not bool(ma_pary.any()):
            return sim.sum() * 0.0  # batch bez ani jednej pary - pomijamy
        return -((logp * pozytywy).sum(1)[ma_pary] / licz[ma_pary]).mean()

    def fit(self, texts: Sequence[str], y: Optional[np.ndarray] = None) -> "SupConEncoder":
        if y is None:
            raise ValueError("SupConEncoder wymaga etykiet (y) do treningu.")
        try:
            import torch
        except ImportError as e:  # pragma: no cover - zalezy od srodowiska
            raise ImportError("Backend '+supcon' wymaga PyTorch: pip install torch") from e
        import pandas as pd

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        Z = self.base.fit_transform(list(texts))
        X = torch.tensor(Z)
        yk = torch.tensor(pd.factorize(np.asarray(y))[0])

        self._net = self._zbuduj(Z.shape[1])
        opt = torch.optim.AdamW(self._net.parameters(), lr=self.lr, weight_decay=1e-4)
        self._net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), self.batch_size):
                idx = perm[i:i + self.batch_size]
                if len(idx) < 16:
                    continue
                xb = X[idx] + self.noise * torch.randn(len(idx), X.shape[1])
                emb = torch.nn.functional.normalize(self._net(xb), dim=-1)
                strata = self._supcon(emb, yk[idx], self.temperature)
                opt.zero_grad()
                strata.backward()
                opt.step()
        self._net.eval()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        assert self._net is not None, "Najpierw fit()."
        Z = self.base.transform(list(texts))
        with torch.no_grad():
            emb = torch.nn.functional.normalize(self._net(torch.tensor(Z)), dim=-1)
        return emb.numpy().astype(np.float32)

    @property
    def wymiar(self) -> int:
        return self.out_dim


# --------------------------------------------------------------------------- #
#  Fabryka
# --------------------------------------------------------------------------- #

def _klucz_cache(spec: str, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(spec.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def zbuduj_enkoder(spec: str = "tfidf", **kw) -> BaseEncoder:
    """Tworzy enkoder z opisu tekstowego.

    >>> zbuduj_enkoder("tfidf")
    >>> zbuduj_enkoder("st:all-MiniLM-L6-v2")
    >>> zbuduj_enkoder("tfidf+supcon")
    """
    spec = (spec or "tfidf").strip()
    supcon = spec.endswith("+supcon")
    if supcon:
        spec = spec[: -len("+supcon")]

    if spec == "tfidf":
        base: BaseEncoder = TfidfEncoder(**{k: v for k, v in kw.items()
                                            if k in ("svd_dim", "random_state")})
    elif spec.startswith("st:"):
        base = SentenceTransformerEncoder(spec[3:])
    elif spec in ("minilm", "all-MiniLM-L6-v2"):
        base = SentenceTransformerEncoder("all-MiniLM-L6-v2")
    elif spec in ("bge", "bge-small"):
        base = SentenceTransformerEncoder("BAAI/bge-small-en-v1.5")
    else:
        raise ValueError(
            f"Nieznany enkoder: {spec!r}. Dozwolone: 'tfidf', 'minilm', 'bge', "
            f"'st:<model>', opcjonalnie z sufiksem '+supcon'."
        )
    return SupConEncoder(base, **{k: v for k, v in kw.items()
                                  if k in ("out_dim", "epochs", "seed")}) if supcon else base
