from math import sqrt

import dotenv
from openai import OpenAI

dotenv.load_dotenv()
client = OpenAI()

sentences = [
    "Python uses indentation to define code blocks.",
    "Indentation determines which statements belong together in Python.",
    "The capital of France is Paris.",
]

# Generate embeddings
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=sentences,
)

embeddings = [item.embedding for item in response.data]


# Calculate cosine similarity between two vectors
def cosine_similarity(vector_a, vector_b):
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = sqrt(sum(a * a for a in vector_a))
    magnitude_b = sqrt(sum(b * b for b in vector_b))

    return dot_product / (magnitude_a * magnitude_b)


# Compare each pair of sentences
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = cosine_similarity(
            embeddings[i],
            embeddings[j],
        )

        print(f"\nSentence A: {sentences[i]}")
        print(f"Sentence B: {sentences[j]}")
        print(f"Similarity score: {score:.4f}")
