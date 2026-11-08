"""Lab 7 - guided example 6.7: recommend related questions from the fixed FAQ list.
Run: python src/lab7_suggestions.py [--backend tfidf|embedding]
"""
import argparse

try:
    from .lab7_faq import build_bot
except ImportError:                     # so it also runs as: python src/<file>.py
    from lab7_faq import build_bot


def main(backend="tfidf"):
    engine = build_bot(backend)
    for query in ["assignment upload", "forgot login", "blank"]:
        print(f"Suggested questions for {query!r} ({backend}):")
        for record, score in engine.search(query, k=3):
            # For TF-IDF hide zero scores (no shared words). This is NOT a semantic
            # cutoff, so for embeddings all top-3 candidates are shown with scores.
            if backend == "embedding" or score > 0:
                print("  ", record["id"], "|", record["question"], f"({score:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
    main(parser.parse_args().backend)
