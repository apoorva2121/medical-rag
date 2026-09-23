"""LLM client with a hard budget guard, on-disk cache, and a key-free extractive fallback.

Design rules
  * The provider is a config switch: `extractive` (default, no key), `anthropic`, or `openai`.
    A key can also be supplied at runtime through the UI / `POST /config/key`; it is kept in
    process memory only and is never written to disk or logged.
  * Every LLM call is checked against max_calls / max_input_tokens / max_output_tokens /
    max_usd before it is made, and usage is persisted in SQLite so the caps survive restarts.
  * Identical (provider, model, system, prompt) requests are served from the cache; a repeated
    question is never billed twice.
  * If the budget is exhausted or the API errors, the client degrades to extractive mode and
    says so in the trace instead of failing the request.
"""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROVIDERS = ("extractive", "anthropic", "openai")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    provider: str            # extractive | anthropic | openai | <provider>-cache
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    note: str = ""


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "usd": round(self.usd, 4)}


class BudgetGuard:
    """Persist cumulative spend in SQLite and refuse calls past any cap."""

    def __init__(self, cfg: dict[str, Any], db_path: str | Path) -> None:
        self.cfg = cfg
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # FastAPI serves requests from a threadpool, so one connection is shared across threads
        # behind a lock rather than opened per request.
        self.con = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock:
            self.con.execute("CREATE TABLE IF NOT EXISTS usage (k TEXT PRIMARY KEY, v REAL)")
            self.con.execute("CREATE TABLE IF NOT EXISTS cache (h TEXT PRIMARY KEY, response TEXT, ts REAL)")
            self.con.commit()

    def usage(self) -> Usage:
        with self.lock:
            rows = dict(self.con.execute("SELECT k, v FROM usage").fetchall())
        return Usage(int(rows.get("calls", 0)), int(rows.get("input_tokens", 0)),
                     int(rows.get("output_tokens", 0)), float(rows.get("usd", 0.0)))

    def price(self, model: str, in_tok: int, out_tok: int) -> float:
        p = self.cfg["price_per_mtok"].get(model, {"input": 15.0, "output": 75.0})  # unknown model -> assume expensive
        return in_tok * p["input"] / 1e6 + out_tok * p["output"] / 1e6

    def check(self, model: str, est_in: int, est_out: int) -> None:
        u = self.usage()
        c = self.cfg
        if u.calls + 1 > c["max_calls"]:
            raise BudgetExceeded(f"call cap {c['max_calls']} reached")
        if u.input_tokens + est_in > c["max_input_tokens"]:
            raise BudgetExceeded("input-token cap reached")
        if u.output_tokens + est_out > c["max_output_tokens"]:
            raise BudgetExceeded("output-token cap reached")
        if u.usd + self.price(model, est_in, est_out) > c["max_usd"]:
            raise BudgetExceeded(f"USD cap {c['max_usd']} would be exceeded")

    def record(self, model: str, in_tok: int, out_tok: int) -> float:
        usd = self.price(model, in_tok, out_tok)
        u = self.usage()
        with self.lock:
            for k, v in (("calls", u.calls + 1), ("input_tokens", u.input_tokens + in_tok),
                         ("output_tokens", u.output_tokens + out_tok), ("usd", u.usd + usd)):
                self.con.execute("INSERT OR REPLACE INTO usage VALUES (?, ?)", (k, v))
            self.con.commit()
        return usd

    def cache_get(self, h: str) -> str | None:
        with self.lock:
            row = self.con.execute("SELECT response FROM cache WHERE h=?", (h,)).fetchone()
        return row[0] if row else None

    def cache_put(self, h: str, response: str) -> None:
        with self.lock:
            self.con.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (h, response, time.time()))
            self.con.commit()


SYSTEM_PROMPT = (
    "You are a careful health-information assistant. Answer ONLY from the numbered evidence passages. "
    "Cite every claim with its passage number in square brackets, e.g. [2]. If the evidence does not "
    "answer the question, reply exactly: INSUFFICIENT EVIDENCE. Never give personalised dosing, never "
    "tell the user to start or stop a medicine, keep to 5 sentences or fewer, plain language."
)

BATCH_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    " You will receive several independent questions, each with its own evidence block. Answer each one "
    "under a line of the form '### Q<n>' in the same order, citing only that question's passages."
)


def build_prompt(question: str, passages: list[str]) -> str:
    ev = "\n\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))
    return f"Evidence:\n{ev}\n\nQuestion: {question}\n\nAnswer with citations:"


def build_batch_prompt(items: list[tuple[str, list[str]]]) -> str:
    """Several questions in one request — used by the evaluator to test many cases per call."""
    return "\n\n".join(f"### Q{n}\n{build_prompt(q, ps)}" for n, (q, ps) in enumerate(items, 1))


def split_batch_answer(text: str, n: int) -> list[str]:
    chunks = [c.strip() for c in re.split(r"###\s*Q\d+\s*\n?", text) if c.strip()]
    return (chunks + ["INSUFFICIENT EVIDENCE"] * n)[:n]


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 8


def extractive_answer(question: str, passages: list[str], max_sentences: int = 4) -> str:
    """Key-free generation: pick the sentences with most query-term overlap, keep source order, cite."""
    q_terms = set(re.findall(r"[a-z0-9]+", question.lower())) - {"what", "is", "are", "the", "of", "a", "an",
                                                                     "how", "do", "does", "for", "to", "in", "and"}
    scored: list[tuple[float, int, int, str]] = []
    for pi, p in enumerate(passages):
        for si, s in enumerate(re.split(r"(?<=[.!?])\s+", p)):
            toks = set(re.findall(r"[a-z0-9]+", s.lower()))
            if len(toks) < 4:
                continue
            shared = q_terms & toks
            if not shared:                      # no content term in common -> never quotable
                continue
            overlap = len(shared) / (len(q_terms) or 1)
            # passage rank is a prior: the reranker already ordered these
            scored.append((overlap + 0.15 / (pi + 1), pi, si, s.strip()))
    if not scored:
        return "INSUFFICIENT EVIDENCE"
    top = sorted(scored, key=lambda t: -t[0])[:max_sentences]
    if top[0][0] < 0.2:
        return "INSUFFICIENT EVIDENCE"
    top.sort(key=lambda t: (t[1], t[2]))
    return " ".join(f"{s} [{pi + 1}]" for _, pi, _, s in top)


class LLMClient:
    def __init__(self, cfg: dict[str, Any], cache_path: str | Path) -> None:
        self.cfg = cfg["llm"]
        self.guard = BudgetGuard(self.cfg["budget"], cache_path)
        self.provider = "extractive"
        self.model = ""
        self._client = None
        self._lock = threading.Lock()
        self.startup_note = "extractive mode (no key)"
        want = self.cfg.get("provider", "extractive")
        env_key = {"anthropic": os.environ.get("ANTHROPIC_API_KEY"), "openai": os.environ.get("OPENAI_API_KEY")}
        if want in ("anthropic", "openai") and env_key[want]:
            self.set_key(want, env_key[want])
        elif want != "extractive":
            self.startup_note = f"{want.upper()}_API_KEY not set -> extractive mode"

    # ---- runtime key handling (UI) ------------------------------------------
    def set_key(self, provider: str, key: str) -> str:
        """Bring the system to life with a key supplied at runtime. Memory only; never persisted."""
        if provider not in ("anthropic", "openai"):
            raise ValueError("provider must be anthropic or openai")
        key = key.strip()
        if len(key) < 20:
            raise ValueError("key looks too short")
        with self._lock:
            if provider == "anthropic":
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
            else:
                import openai
                self._client = openai.OpenAI(api_key=key)
            self.model = self.cfg["models"][provider]
            self.provider = provider
            self.startup_note = f"{provider}:{self.model}"
        return self.startup_note

    def clear_key(self) -> None:
        with self._lock:
            self._client, self.provider, self.model = None, "extractive", ""
            self.startup_note = "extractive mode (key cleared)"

    @property
    def judge_model(self) -> str:
        return self.cfg["judge_models"].get(self.provider, self.model)

    # ---- raw completion with cache + budget ----------------------------------
    def complete(self, system: str, prompt: str, max_tokens: int, model: str | None = None) -> LLMResult:
        model = model or self.model
        h = hashlib.sha256(f"{self.provider}|{model}|{system}|{prompt}".encode()).hexdigest()
        if self.cfg["budget"]["cache"]:
            hit = self.guard.cache_get(h)
            if hit is not None:
                return LLMResult(hit, f"{self.provider}-cache", note="cache hit, not billed")
        est_in = estimate_tokens(system) + estimate_tokens(prompt)
        self.guard.check(model, est_in, max_tokens)
        if self.provider == "anthropic":
            resp = self._client.messages.create(model=model, max_tokens=max_tokens, temperature=self.cfg["temperature"],
                                                system=system, messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
            in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
        else:
            resp = self._client.chat.completions.create(
                model=model, max_completion_tokens=max_tokens, temperature=self.cfg["temperature"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}])
            text = (resp.choices[0].message.content or "").strip()
            in_tok, out_tok = resp.usage.prompt_tokens, resp.usage.completion_tokens
        usd = self.guard.record(model, in_tok, out_tok)
        if self.cfg["budget"]["cache"]:
            self.guard.cache_put(h, text)
        return LLMResult(text, self.provider, in_tok, out_tok, usd)

    # ---- generation for one question ------------------------------------------
    def generate(self, question: str, passages: list[str]) -> LLMResult:
        if self.provider == "extractive":
            return LLMResult(extractive_answer(question, passages), "extractive")
        try:
            return self.complete(SYSTEM_PROMPT, build_prompt(question, passages), self.cfg["max_tokens"])
        except BudgetExceeded as e:
            return LLMResult(extractive_answer(question, passages), "extractive", note=f"budget: {e}")
        except Exception as e:  # network / API errors must not take the service down
            return LLMResult(extractive_answer(question, passages), "extractive", note=f"api error: {type(e).__name__}")

    # ---- batched generation (evaluation only; one call answers many questions) ----
    def generate_batch(self, items: list[tuple[str, list[str]]]) -> tuple[list[str], LLMResult | None]:
        if self.provider == "extractive":
            return [extractive_answer(q, ps) for q, ps in items], None
        try:
            res = self.complete(BATCH_SYSTEM_PROMPT, build_batch_prompt(items), self.cfg["max_tokens"] * len(items))
            return split_batch_answer(res.text, len(items)), res
        except Exception as e:  # includes BudgetExceeded
            return [extractive_answer(q, ps) for q, ps in items], LLMResult("", "extractive", note=f"{type(e).__name__}: {e}")

    def usage(self) -> dict[str, float]:
        return self.guard.usage().as_dict()

    def estimate_request_tokens(self, question: str, passages: list[str]) -> int:
        return estimate_tokens(SYSTEM_PROMPT) + estimate_tokens(build_prompt(question, passages))
