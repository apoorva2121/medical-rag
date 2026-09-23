"""End-to-end request handling: route -> input guard -> retrieve -> evidence gate -> generate -> output guard.

Every stage writes into `trace` so a reviewer can see exactly why a request was
answered, abstained, or refused. Nothing here calls an LLM unless llm.provider is
`anthropic` and a key is present; otherwise generation is extractive.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .guardrails import Guardrails
from .index import HybridIndex
from .llm import LLMClient
from .router import Router

CANNED = {
    "emergency": ("If this is an emergency, call your local emergency number now (112 in India, 911 in the US). "
                  "I can only provide general health information and cannot help in an emergency."),
    "personal_dosing": ("I can't advise on personal doses or on starting or stopping a medicine — that depends on "
                        "your history and needs a clinician or pharmacist. I can share general NIH information "
                        "about a condition or drug if you ask about it in general terms."),
    "chitchat": "Hi! Ask me a general question about a health condition, symptom, test or treatment.",
    "out_of_scope": "I only answer general health-information questions grounded in NIH sources, so I can't help with that.",
    "blocked_input": "I can't process that request. Please remove personal identifiers or instructions aimed at the system and ask a plain health question.",
    "insufficient": "INSUFFICIENT EVIDENCE — I couldn't find a passage in the NIH sources that answers this reliably, so I'd rather not guess.",
}


@dataclass
class Citation:
    n: int
    source: str
    focus: str
    url: str
    score: float
    text: str


@dataclass
class Answer:
    question: str
    route: str
    status: str                     # answered | abstained | refused | canned
    answer: str
    citations: list[Citation] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Pipeline:
    def __init__(self, cfg: dict[str, Any], index_dir: str | Path | None = None) -> None:
        self.cfg = cfg
        self.index = HybridIndex(cfg)
        self.index.load(index_dir or cfg["paths"]["index_dir"])
        self.router = Router(cfg)
        self.guard = Guardrails(cfg, self.index.embedder)
        self.llm = LLMClient(cfg, cfg["paths"]["cache"])

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        trace: dict[str, Any] = {"llm_mode": self.llm.startup_note}

        # 1. input guardrails (cheap, before anything else)
        ic = self.guard.check_input(question)
        trace["input_guard"] = ic.reasons
        if not ic.allowed:
            return self._done(question, "blocked", "refused", CANNED["blocked_input"], trace, t0)

        # 2. route
        r = self.router.route(question)
        trace["route"] = {"name": r.name, "confidence": round(r.confidence, 3), "reason": r.reason,
                          "scores": {k: round(v, 3) for k, v in r.scores.items()}}
        if r.name != "medical_info":
            return self._done(question, r.name, "canned", CANNED[r.name], trace, t0)

        # 3. retrieve + rerank
        hits, rtrace = self.index.search(question)
        trace["retrieval"] = rtrace
        if hits.empty or rtrace["best_score"] < self.cfg["retrieval"]["min_evidence_score"]:
            trace["evidence_gate"] = "failed"
            return self._done(question, r.name, "abstained", CANNED["insufficient"], trace, t0)
        trace["evidence_gate"] = "passed"
        passages = hits["text"].tolist()

        # 4. generate
        gen = self.llm.generate(question, passages)
        trace["generation"] = {"provider": gen.provider, "note": gen.note, "usd": gen.usd,
                               "tokens": [gen.input_tokens, gen.output_tokens]}
        if gen.text.strip().upper().startswith("INSUFFICIENT EVIDENCE"):
            return self._done(question, r.name, "abstained", CANNED["insufficient"], trace, t0)

        # 5. output guardrails: groundedness, forbidden advice, disclaimer
        oc = self.guard.check_output(gen.text, passages)
        trace["output_guard"] = {"reasons": oc.reasons, "grounded_fraction": round(oc.grounded_fraction, 3),
                                 "dropped": oc.dropped_sentences}
        if not oc.allowed:
            return self._done(question, r.name, "abstained", CANNED["insufficient"], trace, t0)

        cites = [Citation(i + 1, row.source, row.focus, row.url, round(float(row.score), 3), row.text)
                 for i, row in enumerate(hits.itertuples())]
        return self._done(question, r.name, "answered", oc.answer, trace, t0, cites)

    def _done(self, q, route, status, text, trace, t0, cites=None) -> Answer:
        return Answer(q, route, status, text, cites or [], trace, round((time.perf_counter() - t0) * 1000, 1))
