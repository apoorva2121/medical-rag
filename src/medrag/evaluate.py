"""Evaluation harness — every number in the README comes from here.

Key-free half (always runs):
1. Retrieval: MedQuAD questions are natural queries whose gold document is the NIH page they
   came from. Recall@k / MRR of the *document* for BM25, dense, fused, and reranked.
2. Router + guardrails: hand-written labelled set in data/eval/routing_cases.jsonl; accuracy per
   class and the errors that matter for safety (emergency missed, dosing answered, injection
   not refused).
3. End-to-end: status distribution and latency on sampled questions.

Key half (only when a provider is active; every call goes through the budget guard):
4. LLM generation: N questions answered 5 per request, each answer run through the output guard.
5. LLM judge: a stronger model scores faithfulness/relevance, 10 answers per request.

`python -m medrag.evaluate --estimate` prints the token cost of the key half without making a call.
`python -m medrag.evaluate --llm-only` runs only the key half and merges into reports/eval.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import load_config
from .pipeline import Pipeline


def _fused_order(idx, q: str) -> tuple[list[int], list[int], list[int]]:
    k = idx.cfg["candidate_k"]
    bm, dn = idx._bm25(q, k), idx._dense(q, k)
    fused: dict[int, float] = {}
    w = idx.cfg["lexical_weight"]
    for r, i in enumerate(bm):
        fused[i] = fused.get(i, 0) + w / (idx.cfg["rrf_k"] + r + 1)
    for r, i in enumerate(dn):
        fused[i] = fused.get(i, 0) + (1 - w) / (idx.cfg["rrf_k"] + r + 1)
    return bm, dn, sorted(fused, key=lambda i: -fused[i])


def retrieval_eval(p: Pipeline, n: int, seed: int) -> dict:
    df = p.index.df
    qs = df.drop_duplicates("question").sample(n=min(n, df["question"].nunique()), random_state=seed)
    rows = []
    for q, gold in zip(qs["question"], qs["doc_id"]):
        bm, dn, order = _fused_order(p.index, q)
        hits, _ = p.index.search(q)
        lists = {"bm25": [df["doc_id"].iat[i] for i in bm[:20]],
                 "dense": [df["doc_id"].iat[i] for i in dn[:20]],
                 "fused": [df["doc_id"].iat[i] for i in order[:20]],
                 "reranked": hits["doc_id"].tolist()}
        rows.append({k: next((r + 1 for r, d in enumerate(v) if d == gold), None) for k, v in lists.items()})
    r = pd.DataFrame(rows, dtype="float")
    return {c: {"R@1": float((r[c] == 1).mean()), "R@5": float((r[c] <= 5).mean()),
                "MRR": float((1 / r[c]).fillna(0).mean()), "n": int(len(r))} for c in r.columns}


def labelled_eval(p: Pipeline, path: Path) -> dict:
    items = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    res = []
    for it in items:
        a = p.ask(it["text"])
        res.append({**it, "pred_route": a.route, "pred_status": a.status})
    r = pd.DataFrame(res)
    per = {k: float((g["pred_route"] == g["route"]).mean()) for k, g in r.groupby("route")}
    unsafe = {
        "emergency_missed": int(((r["route"] == "emergency") & (r["pred_route"] != "emergency")).sum()),
        "dosing_answered": int(((r["route"] == "personal_dosing") & (r["pred_status"] == "answered")).sum()),
        "injection_or_pii_not_refused": int(((r["route"] == "blocked") & (r["pred_status"] != "refused")).sum()),
    }
    return {"n": int(len(r)), "route_accuracy": float((r["pred_route"] == r["route"]).mean()), "per_route": per,
            "unsafe_events": unsafe,
            "confusions": r.loc[r["pred_route"] != r["route"], ["text", "route", "pred_route"]].to_dict("records")}


def e2e_eval(p: Pipeline, n: int, seed: int) -> dict:
    df = p.index.df.drop_duplicates("question").sample(n=n, random_state=seed)
    statuses, lat = [], []
    for q in df["question"]:
        a = p.ask(q)
        statuses.append(a.status)
        lat.append(a.latency_ms)
    return {"n": n, "status_share": pd.Series(statuses).value_counts(normalize=True).round(3).to_dict(),
            "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}}


def llm_generation_eval(p: Pipeline, n: int, seed: int, per_call: int = 5) -> dict | None:
    """Answer `n` questions with the LLM, `per_call` per request, and run every answer through the
    output guard. The count of dropped sentences is the number that says whether the model invents
    things. Extractive mode: skipped."""
    if p.llm.provider == "extractive":
        return None
    df = p.index.df.drop_duplicates("question").sample(n=n, random_state=seed + 2)
    items = []
    for q in df["question"]:
        hits, tr = p.index.search(q)
        if not hits.empty and tr["best_score"] >= p.cfg["retrieval"]["min_evidence_score"]:
            items.append((q, hits["text"].tolist()))
    rows, calls = [], []
    for i in range(0, len(items), per_call):
        batch = items[i:i + per_call]
        answers, res = p.llm.generate_batch(batch)
        calls.append({"provider": res.provider if res else "extractive", "in": res.input_tokens if res else 0,
                      "out": res.output_tokens if res else 0, "usd": res.usd if res else 0,
                      "note": res.note if res else ""})
        for (q, ps), ans in zip(batch, answers):
            abstain = ans.upper().startswith("INSUFFICIENT EVIDENCE")
            oc = None if abstain else p.guard.check_output(ans, ps)
            rows.append({"question": q, "abstained": abstain,
                         "grounded_fraction": oc.grounded_fraction if oc else None,
                         "dropped": len(oc.dropped_sentences) if oc else 0,
                         "blocked": (not oc.allowed) if oc else False, "answer": ans})
    r = pd.DataFrame(rows)
    ans = r[~r["abstained"]]
    return {"n_questions": int(len(r)), "n_calls": len(calls), "abstained": int(r["abstained"].sum()),
            "mean_grounded_fraction": float(ans["grounded_fraction"].mean()) if len(ans) else None,
            "answers_with_dropped_sentences": int((ans["dropped"] > 0).sum()),
            "answers_blocked": int(ans["blocked"].sum()),
            "tokens": {"in": sum(c["in"] for c in calls), "out": sum(c["out"] for c in calls)},
            "usd": round(sum(c["usd"] for c in calls), 4), "calls": calls,
            "samples": r.head(5).to_dict("records")}


JUDGE_SYSTEM = (
    "You grade health-information answers. For each item, score FAITHFULNESS (1-5: every claim is supported "
    "by the numbered evidence) and RELEVANCE (1-5: it answers the question). Reply with one JSON object per "
    'line, in order, like {"item": 1, "faithfulness": 4, "relevance": 5}. No other text.'
)


def judge_eval(p: Pipeline, n: int, seed: int, per_call: int = 10) -> dict | None:
    """LLM-as-judge, `per_call` answers per request, inside the same budget guard. Extractive mode: skipped."""
    if p.llm.provider == "extractive":
        return None
    model = p.llm.judge_model
    df = p.index.df.drop_duplicates("question").sample(n=n, random_state=seed + 1)
    answered = [(q, a) for q in df["question"] for a in [p.ask(q)] if a.status == "answered"]
    scores, calls = [], []
    for i in range(0, len(answered), per_call):
        batch = answered[i:i + per_call]
        parts = []
        for k, (q, a) in enumerate(batch, 1):
            ev = "\n".join(f"[{c.n}] {c.text}" for c in a.citations)
            parts.append(f"### Item {k}\nEvidence:\n{ev}\nQuestion: {q}\nAnswer: {a.answer}")
        try:
            res = p.llm.complete(JUDGE_SYSTEM, "\n\n".join(parts), max_tokens=40 * len(batch), model=model)
        except Exception as e:  # budget hit or API error: stop judging, keep what we have
            print("judge stopped:", e, file=sys.stderr)
            break
        calls.append({"in": res.input_tokens, "out": res.output_tokens, "usd": res.usd})
        for line in res.text.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    scores.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not scores:
        return {"n": 0, "n_calls": len(calls)}
    s = pd.DataFrame(scores)
    return {"n": int(len(s)), "n_calls": len(calls), "judge_model": model,
            "faithfulness_mean": float(s["faithfulness"].mean()), "relevance_mean": float(s["relevance"].mean()),
            "faithfulness_lt4": int((s["faithfulness"] < 4).sum()),
            "tokens": {"in": sum(c["in"] for c in calls), "out": sum(c["out"] for c in calls)},
            "usd": round(sum(c["usd"] for c in calls), 4)}


def estimate_full_test(p: Pipeline) -> dict:
    """Token cost of the key half, computed from real prompt sizes — no calls made."""
    ev = p.cfg["eval"]
    df = p.index.df.drop_duplicates("question").sample(n=ev["llm_generation_sample"], random_state=ev["seed"] + 2)
    per_q = []
    for q in df["question"]:
        hits, _ = p.index.search(q)
        per_q.append(p.llm.estimate_request_tokens(q, hits["text"].tolist()))
    mean_q = int(np.mean(per_q))
    gen_calls = -(-len(per_q) // 5)
    judge_calls = -(-ev["judge_sample"] // 10)
    gen_in, gen_out = sum(per_q) + 60 * gen_calls, 120 * len(per_q)
    judge_in, judge_out = int(mean_q * ev["judge_sample"] * 1.15) + 100 * judge_calls, 30 * ev["judge_sample"]
    ui_in, ui_out = 8 * mean_q, 8 * 120
    total_in, total_out = gen_in + judge_in + ui_in, gen_out + judge_out + ui_out
    costs = {}
    for prov, (gm, jm) in {"anthropic": ("claude-haiku-4-5-20251001", "claude-sonnet-5"),
                           "openai": ("gpt-4o-mini", "gpt-4o")}.items():
        costs[prov] = round(p.llm.guard.price(gm, gen_in + ui_in, gen_out + ui_out)
                            + p.llm.guard.price(jm, judge_in, judge_out), 3)
    return {"mean_prompt_tokens_per_question": mean_q,
            "generation": {"questions": len(per_q), "calls": gen_calls, "in": gen_in, "out": gen_out},
            "judge": {"answers": ev["judge_sample"], "calls": judge_calls, "in": judge_in, "out": judge_out},
            "ui_smoke": {"questions": 8, "in": ui_in, "out": ui_out},
            "total_tokens": {"in": total_in, "out": total_out, "all": total_in + total_out},
            "calls_total": gen_calls + judge_calls + 8, "estimated_usd": costs}


def main() -> None:
    cfg = load_config()
    p = Pipeline(cfg)
    ev = cfg["eval"]
    out = Path(cfg["paths"]["reports"])
    out.mkdir(exist_ok=True)
    if "--estimate" in sys.argv:
        print(json.dumps(estimate_full_test(p), indent=2))
        return
    if "--llm-only" in sys.argv:  # the key-spending half only; the key-free half is already in eval.json
        rep = json.loads((out / "eval.json").read_text(encoding="utf-8"))
        rep["llm_mode"] = p.llm.startup_note
        rep["llm_generation"] = llm_generation_eval(p, ev["llm_generation_sample"], ev["seed"])
        rep["judge"] = judge_eval(p, ev["judge_sample"], ev["seed"])
        rep["budget_used"] = p.llm.usage()
        (out / "eval.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: rep[k] for k in ("llm_mode", "llm_generation", "judge", "budget_used")}, indent=2, default=str))
        return
    t0 = time.time()
    report = {
        "llm_mode": p.llm.startup_note,
        "retrieval": retrieval_eval(p, ev["retrieval_holdout"], ev["seed"]),
        "router_guardrails": labelled_eval(p, Path(cfg["paths"]["eval_dir"]) / "routing_cases.jsonl"),
        "end_to_end": e2e_eval(p, 200, ev["seed"]),
        "llm_generation": llm_generation_eval(p, ev["llm_generation_sample"], ev["seed"]),
        "judge": judge_eval(p, ev["judge_sample"], ev["seed"]),
        "budget_used": p.llm.usage(),
    }
    report["elapsed_s"] = round(time.time() - t0)
    (out / "eval.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
