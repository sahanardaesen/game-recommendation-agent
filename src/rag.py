import sys
from pathlib import Path

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION_NAME = "games"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "qwen2.5:3b"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sen bir video oyunu uzmanısın. Kullanıcının isteğine en uygun oyunları "
            "SADECE verilen bağlamdaki oyunlardan seçerek (varsa) üç oyun öner. "
            "Her oyunu 1-2 cümleyle, kullanıcının isteğiyle ilişkilendirerek açıkla. "
            "Bağlamda olmayan oyunu önerme. Cevap Türkçe olacak.",
        ),
        (
            "human",
            "Kullanıcı isteği: {question}\n\n"
            "Bağlam (aday oyunlar): {context}\n\n"
            "Tavsiyen:",
        ),
    ]
)


def _load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.7)


def _build_context(results):
    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lines.append(
            f"- {meta['title']} (Türler: {meta['genres']}; Etiketler: {meta['tags']}): {doc.split('Açıklama: ')[-1]}"
        )
    return "\n".join(lines)


def recommend(question, top_k=5):
    embedder = SentenceTransformer(MODEL_NAME)
    query_emb = embedder.encode([question]).tolist()

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=query_emb, n_results=top_k)

    context = _build_context(results)

    llm = _load_llm()
    chain = PROMPT | llm
    answer = chain.invoke({"question": question, "context": context})
    return answer.content


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "rahatlatıcı ve hikaye odaklı bir oyun öner"
    print("Ollama modeli çalışıyor, cevap hazırlanıyor (ilk sefer 30-60 sn sürebilir)...\n")
    print(recommend(query))