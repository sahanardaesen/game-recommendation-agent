# HANDOFF.md — Session Summary

This file lets a fresh opencode session on another machine pick up exactly where
the original conversation left off. Read this first, then AGENTS.md.

## What this project is
A RAG-based "Game Recommendation Assistant". The user originally asked for a
portfolio/skill demo covering: Python+ML, Hugging Face/PEFT, NLP+LLMs, RAG,
LangChain/LangGraph, Git, and working on remote GPUs. All concepts are exercised
through one small, complete project.

## Done so far (steps 1-8)
1. **Scaffold + deps** — `.venv`, `requirements.txt`, `.gitignore`, `src/__init__.py`.
2. **Data** — `data/games.json`: 29 games (title, genres, platforms, tags, description).
3. **Embeddings** — `src/indexer.py`: all-MiniLM-L6-v2 (384-dim) → ChromaDB (`chroma_db/`).
4. **RAG** — `src/rag.py`: ChromaDB retrieval + `ChatOllama` (`qwen2.5:3b`) recommendation.
5. **English quality** — converted data/prompts/outputs/README from Turkish to English
   so the minilm model retrieves well; re-indexed.
6. **LangGraph CLI** — `src/agent.py`: StateGraph (START → retrieve → generate → END)
   with interactive CLI (type `exit`/`quit`). Verified: Portal 2, Stardew Valley,
   It Takes Two recommendations correctly produced.
7. **Tests** — `tests/test_agent.py` (pytest): 3 retrieval asserts + 1 LLM smoke test. 4/4 pass.
8. **Remote GPU + QLoRA (PEFT demo)** — `remote/`: `make_dataset.py` (87 synthetic
   examples → `remote/data/train.jsonl`), `finetune.py` (4-bit QLoRA on
   `Qwen/Qwen2.5-3B-Instruct`, `--max-steps 100` ≈ 15-30 min on an RTX 4060,
   `--dryrun` verified working on CPU), `colab_setup.ipynb` (free T4 demo),
   `README.md` (GPU options + security rules). `.env` and `remote/qlora_out/`
   added to `.gitignore`.

## Git status
- Remote: `https://github.com/sahanardaesen/game-recommendation-agent`, branch `master`.
- All commits authored as `sahanardaesen <sahanardaesen@outlook.com>` (old company
  email was scrubbed from history and force-pushed).
- Latest commit: `Adım 8: Uzaktan GPU + QLoRA (PEFT) — eğitim betiği, sentetik veri,
  Colab T4 demosu, güvenlik kuralları` (e2c10a8).

## Where left off / next user actions
- The conversation ended with a question about continuing the opencode session on
  another PC. AGENTS.md + HANDOFF.md were created specifically so the new machine's
  session can continue without exporting the full JSON transcript.
- Remaining hands-on tasks are done by the user on another machine (not by the agent):
  1. `git pull` on the other PC.
  2. Verify `pytest` passes there too.
  3. (Optional) run QLoRA on home RTX 4060: `remote\make_dataset.py` →
     `remote\finetune.py --dryrun` → `remote\finetune.py --max-steps 100`.
  4. (Optional) Colab T4 demo: open + run `remote/colab_setup.ipynb`.

## Environment notes
- Windows host, PowerShell. `.venv\Scripts\python.exe` is the interpreter.
- Ollama binary is NOT on PATH: use `& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe"`.
- If the new machine lacks `.venv`/`chroma_db`, recreate: `python -m venv .venv`,
  install requirements, then `python src\indexer.py`.
- When the new session is opened, prefer to re-run `pytest` once before claiming
  "everything works" on that machine.

## Personality / tone
- Respond to the user in Turkish; code/data/docs in English.
- Work step-by-step with the user's approval; explain before doing; do not
  over-engineer (the project is intentionally small).