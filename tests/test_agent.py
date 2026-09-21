import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent import ask
from indexer import search


def test_retrieval_coop():
    results = search("co-op game for two players", top_k=3)
    titles = [m["title"] for m in results["metadatas"][0]]
    assert "It Takes Two" in titles


def test_retrieval_portal():
    results = search("puzzle game with portals and physics", top_k=3)
    titles = [m["title"] for m in results["metadatas"][0]]
    assert "Portal 2" in titles


def test_retrieval_relaxing():
    results = search("relaxing farming game in a small village", top_k=3)
    titles = [m["title"] for m in results["metadatas"][0]]
    assert "Stardew Valley" in titles


def test_ask_returns_recommendation():
    answer = ask("I want an atmospheric horror game, what do you recommend?")
    assert isinstance(answer, str)
    assert len(answer) > 20