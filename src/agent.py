import sys
from pathlib import Path
from typing import TypedDict

from chromadb import PersistentClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from sentence_transformers import SentenceTransformer

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


class AgentState(TypedDict):
    question: str
    context: str
    answer: str


_embedder = None
_llm = None


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=LLM_MODEL, temperature=0.7)
    return _llm


def retrieve(state: AgentState) -> dict:
    embedder = get_embedder()
    query_emb = embedder.encode([state["question"]]).tolist()

    client = PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    results = collection.query(query_embeddings=query_emb, n_results=5)

    lines = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        lines.append(
            f"- {meta['title']} (Genres: {meta['genres']}; Tags: {meta['tags']}): {doc.split('Description: ')[-1]}"
        )
    return {"context": "\n".join(lines)}


def generate(state: AgentState) -> dict:
    llm = get_llm()
    chain = PROMPT | llm
    answer = chain.invoke({"question": state["question"], "context": state["context"]})
    return {"answer": answer.content}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(question: str) -> str:
    app = build_graph()
    result = app.invoke({"question": question, "context": "", "answer": ""})
    return result["answer"]


def interactive():
    app = build_graph()
    print("Game Recommendation Assistant (type 'exit' or 'quit' to stop)\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print("Agent: preparing... (first run may take a moment)")
        result = app.invoke({"question": question, "context": "", "answer": ""})
        print(f"\n{result['answer']}\n")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        print(ask(" ".join(sys.argv[1:])))
    else:
        interactive()