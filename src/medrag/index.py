"""Hybrid index (BM25 + dense, RRF with a lexical lean) plus cross-encoder reranking."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

_TOK = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class HybridIndex:
    def __init__(self, cfg: dict[str, Any]) -> None:
        r = cfg["retrieval"]
        self.cfg = r
        self.embedder = SentenceTransformer(r["embed_model"])
        self.reranker = CrossEncoder(r["rerank"]["model"]) if r["rerank"]["enabled"] else None
        self.df: pd.DataFrame | None = None
        self.bm25: BM25Okapi | None = None
        self.faiss: faiss.Index | None = None

    # ---- build / persist ---------------------------------------------------
    def build(self, corpus: pd.DataFrame) -> "HybridIndex":
        self.df = corpus.reset_index(drop=True)
        self.bm25 = BM25Okapi([tok(t) for t in self.df["text"]], **self.cfg["bm25"])
        emb = self.embedder.encode(self.df["text"].tolist(), batch_size=64, normalize_embeddings=True,
                                   show_progress_bar=True, convert_to_numpy=True).astype("float32")
        self.faiss = faiss.IndexFlatIP(emb.shape[1]); self.faiss.add(emb)
        return self

    def save(self, d: str | Path) -> None:
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss, str(d / "dense.faiss"))
        self.df.to_parquet(d / "passages.parquet", index=False)

    def load(self, d: str | Path) -> "HybridIndex":
        d = Path(d)
        self.df = pd.read_parquet(d / "passages.parquet")
        self.bm25 = BM25Okapi([tok(t) for t in self.df["text"]], **self.cfg["bm25"])
        self.faiss = faiss.read_index(str(d / "dense.faiss"))
        return self

    # ---- search --------------------------------------------------------------
    def _bm25(self, q: str, k: int) -> list[int]:
        s = self.bm25.get_scores(tok(q))
        top = np.argpartition(-s, min(k, len(s) - 1))[:k]
        return top[np.argsort(-s[top])].tolist()

    def _dense(self, q: str, k: int) -> list[int]:
        v = self.embedder.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        _, idx = self.faiss.search(v, k)
        return [i for i in idx[0].tolist() if i >= 0]

    def search(self, query: str) -> tuple[pd.DataFrame, dict]:
        """Return top passages (after fusion and optional rerank) plus a stage trace."""
        c = self.cfg
        k = c["candidate_k"]
        bm, dn = self._bm25(query, k), self._dense(query, k)
        fused: dict[int, float] = {}
        w = c["lexical_weight"]
        for r, i in enumerate(bm):
            fused[i] = fused.get(i, 0.0) + w / (c["rrf_k"] + r + 1)
        for r, i in enumerate(dn):
            fused[i] = fused.get(i, 0.0) + (1 - w) / (c["rrf_k"] + r + 1)
        order = sorted(fused, key=lambda i: -fused[i])
        n_in = c["rerank"]["top_n_in"] if self.reranker else c["rerank"]["top_n_out"]
        cand = self.df.iloc[order[:n_in]].copy()
        cand["rrf"] = [fused[i] for i in order[:n_in]]
        trace = {"bm25_top": [int(i) for i in bm[:5]], "dense_top": [int(i) for i in dn[:5]],
                 "overlap_top20": len(set(bm[:20]) & set(dn[:20])), "reranked": bool(self.reranker)}
        if self.reranker:
            scores = self.reranker.predict([(query, t) for t in cand["text"]])
            cand["score"] = scores
            cand = cand.sort_values("score", ascending=False).head(c["rerank"]["top_n_out"])
        else:
            cand["score"] = cand["rrf"]
            cand = cand.head(c["rerank"]["top_n_out"])
        trace["best_score"] = float(cand["score"].iloc[0]) if len(cand) else float("-inf")
        return cand.reset_index(drop=True), trace
