import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI()


def load_knowledge():
    with open("knowledge.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )

    return [item.embedding for item in response.data]


def main():
    knowledge_lines = load_knowledge()

    print(f"Loaded {len(knowledge_lines)} knowledge lines.")

    embeddings = generate_embeddings(knowledge_lines)

    for line, embedding in zip(knowledge_lines, embeddings):
        print(f"\nKnowledge: {line}")
        print(f"Vector dimensions: {len(embedding)}")
        print(f"First 5 numbers: {embedding[:5]}")


if __name__ == "__main__":
    main()
