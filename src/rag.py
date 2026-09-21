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
            "You are a video game expert. Recommend the most suitable games for the "
            "user's request, choosing ONLY from the games given in the context. "
            "Recommend up to three games. Briefly explain each one in 1-2 sentences, "
            "linking them to the user's request. Do NOT recommend any game that is not "
            "in the context. Answer in English.",
        ),
        (
            "human",
            "User request: {question}\n\n"
            "Context (candidate games): {context}\n\n"
            "Your recommendation:",
        ),
    ]
)


def _load_llm():
    return ChatOllama(model=LLM_MODEL, temperature=0.7)


def _build_context(results):
    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lines.append(
            f"- {meta['title']} (Genres: {meta['genres']}; Tags: {meta['tags']}): {doc.split('Description: ')[-1]}"
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
    query = " ".join(sys.argv[1:]) or "I want a relaxing farming game, any suggestions?"
    print("Ollama model is running, preparing your answer (first run may take 30-60s)...\n")
    print(recommend(query))