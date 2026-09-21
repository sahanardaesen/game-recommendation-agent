"""Generate a small synthetic training set for the QLoRA demo.

Reads data/games.json and writes remote/data/train.jsonl with instruction
pairs of the form:

    {"instruction": "a natural language game request", "response": "game(s) + reason"}

Deterministic (seeded) so the file is reproducible.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GAMES = ROOT / "data" / "games.json"
OUT = ROOT / "remote" / "data" / "train.jsonl"
SEED = 42
EXAMPLES_PER_GAME = 3

QUERY_TEMPLATES = [
    "I'm looking for a game that is {topic}. Any recommendations?",
    "Can you recommend something {topic} that is worth my time?",
    "What should I play next if I want {topic}?",
    "Suggest a game for someone who enjoys {topic}.",
    "I'd love a {topic} experience, what do you suggest?",
]

REASON_TEMPLATES = [
    "It matches the request with its {desc}.",
    "It is well known for {desc}.",
    "A top pick thanks to its {desc}.",
    "Strong choice because of its {desc}.",
]


def topic_from_game(game: dict) -> str:
    genres = [g.lower() for g in game.get("genres", [])]
    tags = [t.lower() for t in game.get("tags", [])]
    if tags:
        return tag_phrase(tags)
    if genres:
        return genre_phrase(genres)
    return "story driven"


def tag_phrase(tags: list) -> str:
    if len(tags) == 1:
        return tags[0]
    return f"{tags[0]} and {tags[1]}"


def genre_phrase(genres: list) -> str:
    if len(genres) == 1:
        return f"{genres[0]} game"
    return f"{genres[0]} and {genres[1]} game"


def reason_for(game: dict) -> str:
    genres = [g.lower() for g in game.get("genres", [])]
    tags = [t.lower() for t in game.get("tags", [])]
    parts = []
    if len(genres) >= 2:
        parts.append(f"{genres[0]} and {genres[1]} gameplay")
    elif genres:
        parts.append(f"{genres[0]} themes")
    if len(tags) >= 2:
        parts.append(f"{tags[0]} and {tags[1]}")
    elif tags:
        parts.append(tags[0])
    return ", ".join(parts) if parts else "great level design"


def build_response(game: dict, topic: str) -> str:
    reason = reason_for(game)
    return f"1. {game['title']} — {reason}. {random.choice(REASON_TEMPLATES).format(desc=reason)}."


def main() -> None:
    with open(GAMES, encoding="utf-8") as f:
        games = json.load(f)

    rng = random.Random(SEED)
    examples = []
    for game in games:
        topic = topic_from_game(game)
        for _ in range(EXAMPLES_PER_GAME):
            tmpl = rng.choice(QUERY_TEMPLATES)
            instruction = tmpl.format(topic=topic)
            response = build_response(game, topic)
            examples.append({"instruction": instruction, "response": response})

    rng.shuffle(examples)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples -> {OUT}")


if __name__ == "__main__":
    main()