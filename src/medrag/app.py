"""FastAPI surface: the UI at /, POST /ask, GET /health, GET /metrics, GET /budget, POST|DELETE /config/key."""

from __future__ import annotations

import os
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .ingest import load_config
from .pipeline import Pipeline

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)


class AskResponse(BaseModel):
    question: str
    route: str
    status: str
    answer: str
    citations: list[dict[str, Any]]
    trace: dict[str, Any]
    latency_ms: float


class KeyRequest(BaseModel):
    provider: str = Field(..., pattern="^(anthropic|openai)$")
    key: str = Field(..., min_length=20, max_length=400)


app = FastAPI(title="medical-rag", version="0.1.0")
_state: dict[str, Any] = {"pipeline": None, "lock": threading.Lock(), "latencies": deque(maxlen=1000),
                          "status": Counter(), "routes": Counter(), "started": time.time()}


def get_pipeline() -> Pipeline:
    if _state["pipeline"] is None:
        with _state["lock"]:
            if _state["pipeline"] is None:
                _state["pipeline"] = Pipeline(load_config(os.environ.get("MEDRAG_CONFIG", "config.yaml")))
    return _state["pipeline"]


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, Any]:
    p = get_pipeline()
    return {"status": "ok", "index_loaded": True, "passages": int(len(p.index.df)),
            "llm_mode": p.llm.startup_note, "uptime_s": round(time.time() - _state["started"])}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    a = get_pipeline().ask(req.question)
    _state["latencies"].append(a.latency_ms)
    _state["status"][a.status] += 1
    _state["routes"][a.route] += 1
    return AskResponse(**a.to_dict())


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    lat = sorted(_state["latencies"])
    pct = (lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None)
    return {"requests": len(lat), "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "max": lat[-1] if lat else None},
            "by_status": dict(_state["status"]), "by_route": dict(_state["routes"])}


@app.get("/budget")
def budget() -> dict[str, Any]:
    p = get_pipeline()
    return {"mode": p.llm.startup_note, "used": p.llm.usage(), "caps": p.cfg["llm"]["budget"]}


@app.post("/config/key")
def set_key(req: KeyRequest) -> dict[str, str]:
    """Activate an LLM provider with a key supplied from the UI. Held in memory; never persisted or logged."""
    try:
        mode = get_pipeline().llm.set_key(req.provider, req.key)
    except Exception as e:  # bad key format, SDK missing
        raise HTTPException(400, f"{type(e).__name__}: {e}") from None
    return {"mode": mode}


@app.delete("/config/key")
def clear_key() -> dict[str, str]:
    get_pipeline().llm.clear_key()
    return {"mode": get_pipeline().llm.startup_note}
