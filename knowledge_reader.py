import json
import re
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
CHUNK_SIZE = 15
CHUNK_OVERLAP = 5


def create_chunks(text: str) -> list[str]:
    """Split text into word-limited chunks with sentence-based overlap."""

    # Step 1: Split text into sentences.
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Step 2: Split any oversized sentence into smaller pieces.
    processed_sentences = []

    for sentence in sentences:
        words = sentence.split()

        if len(words) > CHUNK_SIZE:
            for i in range(0, len(words), CHUNK_SIZE):
                piece = " ".join(words[i : i + CHUNK_SIZE])
                processed_sentences.append(piece)
        else:
            processed_sentences.append(sentence)

    # Step 3: Build chunks.
    chunks = []
    current_sentences = []
    current_word_count = 0

    for sentence in processed_sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_word_count = len(sentence.split())

        # If adding this sentence exceeds the chunk limit,
        # save the current chunk and prepare overlap.
        if current_word_count + sentence_word_count > CHUNK_SIZE and current_sentences:
            chunks.append(" ".join(current_sentences))

            # Remove older sentences while preserving recent context.
            while current_sentences and current_word_count > CHUNK_OVERLAP:
                removed_sentence = current_sentences.pop(0)
                current_word_count -= len(removed_sentence.split())

        # Add the incoming sentence.
        current_sentences.append(sentence)
        current_word_count += sentence_word_count

    # Step 4: Save the final chunk.
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


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
        text = file.read()

    chunks = create_chunks(text)

    # Step 2: Check whether saved embeddings exist.
    if EMBEDDINGS_FILE.exists():
        with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
            records = json.load(file)

        # Step 3: Compare saved text with current text.
        stored_chunks = [record["text"] for record in records]

        if stored_chunks == chunks:
            print("Knowledge file unchanged. Reusing stored embeddings...")
            return records

        print("Knowledge file changed. Regenerating embeddings...")

    else:
        print("Creating knowledge embeddings for the first time...")

    # Step 4: Generate embeddings for the current sentences.
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks,
    )

    records = [
        {"text": chunk, "embedding": item.embedding}
        for chunk, item in zip(chunks, response.data)
    ]

    # Step 5: Save the updated embeddings.
    with EMBEDDINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file)

    print(f"Saved embeddings for {len(records)} knowledge chunks.")

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
