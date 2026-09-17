import json
import sys
from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "games.json"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "games"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_games():
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def _game_to_text(game):
    return (
        f"{game['title']}. Türler: {', '.join(game['genres'])}. "
        f"Etiketler: {', '.join(game['tags'])}. "
        f"Açıklama: {game['description']}"
    )


def index():
    games = load_games()
    print(f"Model yükleniyor: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [_game_to_text(g) for g in games]
    ids = [str(i) for i in range(len(games))]
    metadatas = [
        {
            "title": g["title"],
            "genres": ", ".join(g["genres"]),
            "platforms": ", ".join(g["platforms"]),
            "tags": ", ".join(g["tags"]),
        }
        for g in games
    ]

    print(f"Embedding oluşturuluyor ({len(games)} oyun)...")
    embeddings = model.encode(texts, show_progress_bar=True)

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    print(f"OK: {len(games)} oyun indekslendi -> {CHROMA_DIR}")


def search(query_text, top_k=5):
    if not CHROMA_DIR.exists():
        print("Önce 'index' komutunu çalıştırın.")
        sys.exit(1)

    model = SentenceTransformer(MODEL_NAME)
    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    query_emb = model.encode([query_text]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    print(f"\nSorgu: '{query_text}'\n")
    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"][0], results["metadatas"][0], results["distances"][0]),
        start=1,
    ):
        title = meta["title"]
        print(f"{i}. {title} (benzerlik: {1 - dist:.3f})")
    return results


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:])
        search(query)
    else:
        index()