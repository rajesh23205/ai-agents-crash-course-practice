from agents import function_tool

# Common English words that usually don't help with searching.
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "there",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


@function_tool
def read_knowledge(question: str) -> list[str]:
    """
    Search the knowledge file for information related to the user's question.
    """

    print(f"read_knowledge function called with question: {question}")

    # Convert the question into individual words
    words = question.lower().split()

    # Remove common words that are not useful for searching
    keywords = [
        word.strip(".,?!'\"")
        for word in words
        if word.strip(".,?!'\"") not in STOP_WORDS
    ]

    print(f"Keywords: {keywords}")

    matching_lines = []

    with open("knowledge.txt", "r", encoding="utf-8") as file:
        for line in file:
            line_lower = line.lower()

            for keyword in keywords:
                if keyword in line_lower:
                    matching_lines.append(line.strip())
                    break

    print(f"Retrieved knowledge: {matching_lines}")

    return matching_lines


if __name__ == "__main__":
    knowledge = read_knowledge("Why can't I change a tuple after creating it?")
    print(knowledge)
