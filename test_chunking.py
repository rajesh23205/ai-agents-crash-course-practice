from knowledge_reader import KNOWLEDGE_FILE, create_chunks

sample_text = (
    "Python is a popular programming language used for "
    "web development data analysis automation machine learning "
    "artificial intelligence and many other applications."
)

print("=== Sample Text Test ===")
sample_chunks = create_chunks(sample_text)

for index, chunk in enumerate(sample_chunks, start=1):
    print(f"Chunk {index}: {chunk}")

print("\n=== Knowledge File Test ===")
with KNOWLEDGE_FILE.open("r", encoding="utf-8") as file:
    text = file.read()

chunks = create_chunks(text)

print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {index} ({len(chunk.split())} words) ---")
    print(chunk)
