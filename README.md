# Game Recommendation Assistant

A RAG-based game recommendation assistant that answers natural language queries.

## What It Does

Turns a user request (e.g. "atmospheric horror-focused game?") into a vector, retrieves the most relevant games from a vector database, then generates a proper recommendation using an open-source LLM.

```
question → embedding → ChromaDB (top relevant games) → Qwen2.5 (Ollama) → recommendation
```

## Technologies

- **Python**
- **Hugging Face Transformers** (`sentence-transformers/all-MiniLM-L6-v2` — embedding model)
- **ChromaDB** (vector database)
- **LangChain** (RAG pipeline — prompt building and LLM call)
- **Ollama + Qwen2.5 3B** (open-source LLM — recommendation generation)
- **PEFT** (planned — optional fine-tuning)
- **LangGraph** (pipeline orchestration — stateful graph)

## Project Structure

```
├── data/
│   └── games.json          # Game data (genres, tags, description)
├── src/
│   ├── indexer.py          # Embedding + ChromaDB indexing and search
│   ├── rag.py              # RAG pipeline (retrieval + LLM recommendation)
│   └── agent.py            # LangGraph orchestration + interactive CLI
├── tests/
│   └── test_agent.py       # Pytest: retrieval + LLM smoke tests
├── remote/                 # GPU + PEFT: QLoRA fine-tuning, Colab demo, docs
│   ├── make_dataset.py     # Synthetic LoRA training data generator
│   ├── finetune.py         # QLoRA fine-tuning script (RTX 4060 / T4)
│   └── colab_setup.ipynb   # Remote GPU demo (free T4)
├── requirements.txt
└── README.md
```

## Remote GPU + Fine-tuning (PEFT)

RAG pipeline'i üreten bu projede büyük bir LLM'e ince ayar gerekmez; ancak beceri kanıtı olarak küçük bir **QLoRA** demosu eklenmiştir (bkz. `remote/`):

```powershell
# 1) Training verisi üret (CPU)
.venv\Scripts\python.exe remote\make_dataset.py

# 2) Veri hattını GPU olmadan doğrula
.venv\Scripts\python.exe remote\finetune.py --dryrun

# 3) Gerçek eğitim (RTX 4060 vb. 8+ GB GPU gerekir)
.venv\Scripts\python.exe remote\finetune.py --max-steps 100
```

## Testing

```powershell
# Run automated tests (retrieval + LLM smoke tests)
.venv\Scripts\python.exe -m pytest tests\ -v
```

## Setup and Usage

Prerequisite: [Ollama](https://ollama.com/download) installed with the `qwen2.5:3b` model pulled.

```powershell
# 1) Create virtual environment
python -m venv .venv

# 2) Install dependencies
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3) Index the games into the vector database
.venv\Scripts\python.exe src\indexer.py

# 4) Start the interactive assistant (LangGraph)
.venv\Scripts\python.exe src\agent.py

# 5) One-shot recommendation (without interactive loop)
.venv\Scripts\python.exe src\agent.py "recommend a horror themed game"

# (Optional) Raw vector search test
.venv\Scripts\python.exe src\indexer.py search "relaxing farming game"
```

## Progress

1. ✅ Project scaffold and dependencies
2. ✅ Game database
3. ✅ Vector embeddings and vector database
4. ✅ RAG pipeline (ChromaDB + LangChain + Qwen2.5)
5. ✅ LLM integration quality (English data + prompts)
6. ✅ LangGraph orchestration + Interactive CLI
7. ✅ Automated tests (pytest)