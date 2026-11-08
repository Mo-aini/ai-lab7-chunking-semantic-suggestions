# Lab 7 - runs the chunking comparison and the semantic suggestion checks
import platform
import sys

from src import (lab7_chunks, lab7_guided_chunks, lab7_guided_embeddings,
                 lab7_semantic_check, lab7_suggestions)


def section(title):
    print("=" * 72)
    print(title)
    print("=" * 72)


def main():
    use_model = "--no-model" not in sys.argv
    print("AI Lab 7: Text Chunking & Semantic Suggestions | Rasikh Ali (42365)")
    print("Python %s, %s" % (platform.python_version(),
                             "with all-MiniLM-L6-v2 on CPU" if use_model else "TF-IDF only (--no-model)"))
    print()
    section("LAB 7 / 6.3  Guided example: chunking with source offsets")
    lab7_guided_chunks.main()
    print()
    section("LAB 7 / Task  Word windows vs sentence-aware chunks + boundary tests")
    lab7_chunks.main()
    print()
    section("LAB 7 / 6.7  Suggested questions from the FAQ list (TF-IDF)")
    lab7_suggestions.main("tfidf")
    if not use_model:
        return
    print()
    section("LAB 7 / 6.7  Suggested questions from the FAQ list (embeddings)")
    lab7_suggestions.main("embedding")
    print()
    section("LAB 7 / 6.2  Guided example: embeddings and cosine similarity")
    lab7_guided_embeddings.main()
    print()
    section("LAB 7 / Task  Semantic suggestions over chunks: TF-IDF vs embeddings")
    lab7_semantic_check.main()


if __name__ == "__main__":
    main()
