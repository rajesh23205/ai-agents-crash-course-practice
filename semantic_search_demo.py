import json
from math import sqrt
from pathlib import Path

import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI()

KNOWLEDGE_FILE = Path("knowledge.txt")
EMBEDDINGS_FILE = Path("knowledge_embeddings.json")


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sqrt(sum(a * a for a in vector_a))
    magnitude_b = sqrt(sum(b * b for b in vector_b))

    return dot_product / (magnitude_a * magnitude_b)


def load_or_create_embeddings():
    if EMBEDDINGS_FILE.exists():
        print("Loading stored knowledge embeddings...")

        with EMBEDDINGS_FILE.open("r", encoding="utf-8") as file:
            return json.load(file)

    print("Creating knowledge embeddings for the first time...")

    with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
        sentences = [line.strip() for line in file if line.strip()]

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences,
    )

    records = [
        {"text": sentence, "embedding": item.embedding}
        for sentence, item in zip(sentences, response.data)
    ]

    with EMBEDDINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(records, file)

    return records


def main():
    knowledge = load_or_create_embeddings()

    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

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

        print("\nMost relevant knowledge:")

        if not ranked_results:
            print("No knowledge available.")
        elif ranked_results[0][0] < 0.30:
            print("I couldn't find sufficiently relevant information.")
        else:
            for score, sentence in ranked_results[:3]:
                print(f"{score:.4f} | {sentence}")


if __name__ == "__main__":
    main()
