import json
from math import sqrt
from pathlib import Path

import dotenv
from agents import function_tool
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI()

KNOWLEDGE_FILE = Path("knowledge.txt")
EMBEDDINGS_FILE = Path("knowledge_embeddings.json")
SIMILARITY_THRESHOLD = 0.40
TOP_K = 3


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sqrt(sum(a * a for a in vector_a))
    magnitude_b = sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def load_or_create_embeddings():
    # Step 1: Read the current knowledge file.
    with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    # Step 2: Check whether saved embeddings exist.
    if EMBEDDINGS_FILE.exists():
        with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)

        # Step 3: Compare saved text with current text.
        stored_sentences = [record["text"] for record in records]

        if stored_sentences == sentences:
            print("Knowledge file unchanged. Reusing stored embeddings...")
            return records

        print("Knowledge file changed. Regenerating embeddings...")

    else:
        print("Creating knowledge embeddings for the first time...")

    # Step 4: Generate embeddings for the current sentences.
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences,
    )

    records = [
        {"text": sentence, "embedding": item.embedding}
        for sentence, item in zip(sentences, response.data)
    ]

    # Step 5: Save the updated embeddings.
    with EMBEDDINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file)

    print(f"Saved embeddings for {len(records)} knowledge sentences.")

    return records


@function_tool
def read_knowledge(question: str) -> list[str]:
    """Find the most relevant lines in the knowledge file using semantic search."""

    print(f"\nSemantic search question: {question}")

    knowledge = load_or_create_embeddings()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question,
    )

    question_embedding = response.data[0].embedding

    ranked_results = []

    for record in knowledge:
        score = cosine_similarity(
            question_embedding,
            record["embedding"],
        )
        ranked_results.append((score, record["text"]))

    ranked_results.sort(
        key=lambda result: result[0],
        reverse=True,
    )

    if not ranked_results:
        return []

    best_score, _ = ranked_results[0]
    print(f"Best similarity score: {best_score:.4f}")

    if best_score < SIMILARITY_THRESHOLD:
        print("No sufficiently relevant knowledge found.")
        return []

    for score, sentence in ranked_results[:3]:
        print(f"{score:.4f} | {sentence}")

    results = [
        sentence
        for score, sentence in ranked_results[:TOP_K]
        if score >= SIMILARITY_THRESHOLD
    ]

    print(f"Retrieved knowledge: {results}")
    return results
