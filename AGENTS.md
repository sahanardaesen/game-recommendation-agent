# AGENTS.md

## Project
Game Recommendation Assistant — a RAG-based assistant that recommends PC games from a local vector database using an open-source LLM.

## Tech Stack
- Python 3.12, PyTorch (CPU), Hugging Face Transformers / sentence-transformers
- ChromaDB (vector DB), LangChain, LangGraph, Ollama + Qwen2.5 3B
- PEFT / QLoRA (fine-tuning demo), pytest

## Key Commands (run from project root)
```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt   # install deps
.venv\Scripts\python.exe src\indexer.py                       # index games into ChromaDB
.venv\Scripts\python.exe src\indexer.py search "query"        # raw vector search
.venv\Scripts\python.exe src\agent.py                         # interactive LangGraph CLI
.venv\Scripts\python.exe src\agent.py "query"                 # one-shot recommendation
.venv\Scripts\python.exe src\rag.py "query"                   # legacy RAG pipeline
.venv\Scripts\python.exe -m pytest tests\ -v                  # run tests
```

## QLoRA Fine-tuning (remote/ — needs a GPU, e.g. RTX 4060 / T4)
```powershell
.venv\Scripts\python.exe remote\make_dataset.py               # generate training data (CPU)
.venv\Scripts\python.exe remote\finetune.py --dryrun          # verify pipeline w/o GPU
.venv\Scripts\python.exe remote\finetune.py --max-steps 100   # actual training on GPU
```

## Conventions
- Respond to the user in Turkish.
- Code, data, prompts, and README stay in **English** (embedding model is all-MiniLM-L6-v2, English-focused).
- Base model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim). LLM: `qwen2.5:3b` via Ollama (Ollama binary is at `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`, not on PATH).
- Do not change data format of `data/games.json` (29 games; genres/tags/platforms/description).
- `.venv/`, `chroma_db/` (generated), `remote/qlora_out/` (LoRA output) and `.env` are git-ignored. Recreate locally as needed.
- Tests must keep passing before any commit.

## Git
- Remote: https://github.com/sahanardaesen/game-recommendation-agent (branch `master`)
- Commit messages use Turkish `Adım N: ...` style, e.g. `Adım 7: Otomatik testler`.
- Only commit and push when the user explicitly asks.

## Security
- Never write, commit, or log secrets/API keys/credentials.
- `remote/README.md` documents: Ollama must stay on localhost or behind an SSH tunnel — never expose it on `0.0.0.0`.
- Colab notebook tokens belong to the user's account only.

## Project Status (handoff)
- Steps 1-8 complete (scaffold → data → embeddings → RAG → English quality → LangGraph CLI → pytest → remote GPU + QLoRA scripts).
- Remaining user actions: run QLoRA on home RTX 4060; optional Colab T4 demo (`remote/colab_setup.ipynb`).
- See `HANDOFF.md` for the full session summary.