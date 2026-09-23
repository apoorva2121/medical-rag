"""Semantic router: decide what kind of request this is before spending retrieval or LLM budget.

Routes
  emergency      -> immediate safety message, no retrieval (keyword gate, checked first)
  medical_info   -> full RAG pipeline
  personal_dosing-> refuse personalised dosing / stop-start medication decisions, offer general info
  chitchat       -> brief canned reply, no retrieval
  out_of_scope   -> polite decline (below similarity floor for every route)

Cheap by design: one embedding call, cosine to the nearest exemplar of each route. No LLM.
Near-ties between medical_info and personal_dosing resolve to personal_dosing (safety first).
The medical_info bank mirrors MedQuAD's own question types (symptoms, causes, treatment,
inheritance, outlook, tests, prevention, susceptibility, research, frequency, information).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

EXEMPLARS: dict[str, list[str]] = {
    "medical_info": [
        "What are the symptoms of type 2 diabetes?", "How is glaucoma treated?",
        "What causes iron deficiency anemia?", "What causes gallstones?", "Is asthma hereditary?",
        "What is the outlook for someone with Parkinson's disease?",
        "How do I prevent high blood pressure?", "What tests diagnose celiac disease?",
        "What are the side effects of metformin?", "Who is at risk for osteoporosis?",
        "What is lupus?", "Can you tell me about migraines?", "Do you have information about hepatitis C?",
        "What does a cholesterol test measure?", "How common is sickle cell disease?",
        "What are the stages of chronic kidney disease?", "What are the complications of diabetes?",
        "What research is being done on multiple sclerosis?", "What are the treatments for depression?",
        "What is the difference between type 1 and type 2 diabetes?",
    ],
    "personal_dosing": [
        "How many mg of ibuprofen should I take for my back pain?",
        "Can I stop taking my blood pressure medication?",
        "Should I double my insulin dose tonight?", "Is it safe for me to take these two drugs together?",
        "Can I give my 3-year-old adult paracetamol?",
    ],
    "chitchat": [
        "Hello!", "Thanks, that was helpful", "Who made you?", "Tell me a joke", "Good morning",
    ],
}


@dataclass
class Route:
    name: str
    confidence: float
    scores: dict[str, float]
    reason: str


class Router:
    def __init__(self, cfg: dict[str, Any]) -> None:
        r = cfg["router"]
        self.min_sim = float(r["min_similarity"])
        self.emergency_kw = [k.lower() for k in r["emergency_keywords"]]
        self.model = SentenceTransformer(r["embed_model"])
        self.safety_margin = float(r.get("safety_margin", 0.05))
        self.exemplars = {name: self.model.encode(ex, normalize_embeddings=True, convert_to_numpy=True)
                          for name, ex in EXEMPLARS.items()}

    def route(self, query: str) -> Route:
        q = query.lower()
        hit = [k for k in self.emergency_kw if k in q]
        if hit:
            return Route("emergency", 1.0, {}, f"emergency keyword: {hit[0]!r}")
        v = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = {n: float((e @ v).max()) for n, e in self.exemplars.items()}   # nearest exemplar per route
        best = max(scores, key=scores.get)
        if scores[best] < self.min_sim:
            return Route("out_of_scope", scores[best], scores, f"best route {best} below floor {self.min_sim}")
        if best == "medical_info" and scores["personal_dosing"] >= scores[best] - self.safety_margin:
            return Route("personal_dosing", scores["personal_dosing"], scores, "near-tie resolved to safer route")
        return Route(best, scores[best], scores, "nearest exemplar")
