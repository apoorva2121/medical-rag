"""Ingest MedQuAD and build the hybrid index. Run once: python scripts/build_index.py"""

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from medrag.index import HybridIndex  # noqa: E402
from medrag.ingest import load_config, main as ingest_main  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    if not Path(cfg["paths"]["corpus"]).exists():
        ingest_main()
    corpus = pd.read_parquet(cfg["paths"]["corpus"])
    t0 = time.time()
    HybridIndex(cfg).build(corpus).save(cfg["paths"]["index_dir"])
    print(f"indexed {len(corpus):,} passages in {time.time() - t0:.0f}s -> {cfg['paths']['index_dir']}")
