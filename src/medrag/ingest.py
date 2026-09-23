"""Parse MedQuAD XML into a passage corpus with provenance metadata.

Each NIH page is a <Document>; each <QAPair> answer becomes one or more
passages. Passages keep: doc id, source (e.g. GARD, MedlinePlus), URL, focus
(the condition), the question type, and the original question — the question
is what makes MedQuAD an evaluation set as well as a corpus.
"""

from __future__ import annotations

import hashlib
import re
import defusedxml.ElementTree as ET  # hardened against XXE / entity-expansion
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def chunk(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = text.rfind(". ", start, end)
        cut = cut + 1 if (cut != -1 and cut > start + size // 2) else end
        out.append(text[start:cut].strip())
        if cut >= len(text):
            break
        start = max(cut - overlap, start + 1)
    return [c for c in out if c]


def parse_medquad(raw_dir: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    ic = cfg["ingest"]
    rows, seen = [], set()
    for xml_path in sorted(Path(raw_dir).rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        doc_id = root.attrib.get("id", xml_path.stem)
        source = root.attrib.get("source", xml_path.parent.name)
        url = root.attrib.get("url", "")
        focus = _norm(root.findtext("Focus") or "")
        for qa in root.iter("QAPair"):
            q_el, a_el = qa.find("Question"), qa.find("Answer")
            if q_el is None or a_el is None:
                continue
            question = _norm(q_el.text or "")
            qtype = q_el.attrib.get("qtype", "")
            answer = _norm(a_el.text or "")
            if len(answer) < ic["min_answer_chars"]:
                continue
            for j, piece in enumerate(chunk(answer, ic["chunk_chars"], ic["chunk_overlap"])):
                h = hashlib.sha1(piece.lower().encode()).hexdigest()[:16]
                if ic["dedup"] and h in seen:
                    continue
                seen.add(h)
                rows.append({"pid": f"{doc_id}#{q_el.attrib.get('qid', '')}#{j}", "doc_id": doc_id,
                             "source": source, "url": url, "focus": focus, "qtype": qtype,
                             "question": question, "text": piece, "hash": h})
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    df = parse_medquad(cfg["paths"]["raw"], cfg)
    Path(cfg["paths"]["corpus"]).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg["paths"]["corpus"], index=False)
    print(f"{len(df):,} passages from {df['doc_id'].nunique():,} documents, {df['source'].nunique()} sources -> {cfg['paths']['corpus']}")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
