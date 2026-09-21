"""Offline retrieval-only benchmark: how good is the vector search without an LLM?

For every game in data/games.json we synthesize a natural-language query from
its genres/tags (same templates as remote/make_dataset.py), run a real ChromaDB
search and check whether the target game made it into the top-k. This measures
the RETRIEVAL half of the RAG pipeline only — no LLM involved, so it is fast,
GPU-free and fully reproducible.

Usage:
    .venv\Scripts\python.exe src\benchmark.py
    .venv\Scripts\python.exe src\benchmark.py --top-k 3          # top-1/3 only
    .venv\Scripts\python.exe src\benchmark.py --games data\games2.json   # alt veri
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "remote"))

import chromadb  # noqa: E402
from indexer import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    MODEL_NAME,
    load_games,
    _game_to_text,
)
from make_dataset import QUERY_TEMPLATES, topic_from_game, reason_for  # noqa: E402

MODEL = None


def get_model():
    global MODEL
    if MODEL is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading {MODEL_NAME} once ...")
        MODEL = SentenceTransformer(MODEL_NAME)
    return MODEL


def search_topk(query_text: str, top_k: int):
    """Silent vector search (reuses the real Chroma store) — no stdout spam."""
    model = get_model()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    qemb = model.encode([query_text]).tolist()
    res = collection.query(query_embeddings=qemb, n_results=top_k)
    return [m["title"] for m in res["metadatas"][0]]


def synthesize_query(game: dict) -> str:
    return QUERY_TEMPLATES[0].format(topic=topic_from_game(game))


def run(top_k: int) -> None:
    games = load_games()
    hits = {k: 0 for k in {1, 3, top_k}}
    for game in games:
        query = synthesize_query(game)
        titles = search_topk(query, top_k)
        for k in hits:
            if game["title"] in titles[:k]:
                hits[k] += 1

    total = len(games)
    print(f"\n=== Retrieval benchmark (CPU, no LLM) — {total} games ===")
    print(f"Query template: {QUERY_TEMPLATES[0]!r}")
    for k in sorted(hits):
        pct = hits[k] / total * 100
        print(f"  recall@{k}: {hits[k]:>2}/{total} = {pct:5.1f}%")
    r5 = hits.get(5, hits.get(top_k))
    print(f"\nNot: 'recall@5' gercek RAG kullaniminda kullanilan top_k=5'tir.",
          "Hedef: >=80% (retrieval kismi LLM'siz iyi calisiyor demektir).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()
    run(args.top_k)
