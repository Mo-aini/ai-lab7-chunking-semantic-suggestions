"""Lab 7 - the FAQ list and chatbot from Lab 6 (manual example 6.6), reused for suggestions.
Run: python src/lab7_faq.py [--backend tfidf|embedding]
"""
import argparse
import sys

try:
    from .lab7_search import Retriever
except ImportError:                     # so it also runs as: python src/<file>.py
    from lab7_search import Retriever

FAQ = [
    {"id": "faq:password", "question": "How do I reset my password?",
     "answer": "Use the recovery link on the login page.",
     "source": "it_help.txt"},
    {"id": "faq:submit", "question": "Where do I submit my assignment?",
     "answer": "Use the assignment upload page in the learning portal.",
     "source": "assessment.txt"},
    {"id": "faq:late", "question": "Can illness justify a late submission?",
     "answer": "Request an extension with medical evidence before the deadline.",
     "source": "handbook.txt"},
    # Added in Lab 6 (synthetic, sources are the synthetic guides in corpus.json)
    {"id": "faq:wifi", "question": "How do I connect to campus Wi-Fi?",
     "answer": "Sign in to the campus network with your university account username and password.",
     "source": "it_password (corpus.json)"},
    {"id": "faq:printer", "question": "What should I do if the printer prints blank pages?",
     "answer": "Check ink levels, run the cleaning utility, then print a test page.",
     "source": "printer_help (corpus.json)"},
]


def build_bot(backend="tfidf"):
    records = [dict(r, text=r["question"] + " " + r["answer"]) for r in FAQ]
    return Retriever(records, backend)


def reply(engine, query):
    if not query.strip():
        return "Please enter a question."
    # Only exact canonical FAQ text is answered automatically here.
    for r in FAQ:
        if query.strip().casefold() == r["question"].casefold():
            return r["answer"] + " [source: " + r["source"] + "]"
    # a TF-IDF score of 0 means no shared words, so it is not a real suggestion
    hits = [(r, score) for r, score in engine.search(query) if score > 0]
    if not hits:
        return "No matching FAQ. Please contact the help desk."
    lines = ["Possible matches; choose or rephrase your question:"]
    for r, score in hits:
        lines.append(r["question"] + f" (similarity {score:.3f})")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["tfidf", "embedding"], default="tfidf")
    args = parser.parse_args()
    engine = build_bot(args.backend)
    print("FAQ assistant. Type exit to stop.")
    while True:
        try:
            query = input("You: ")
            if not sys.stdin.isatty():
                print(query)  # echo piped input so saved transcripts are readable
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() == "exit":
            break
        print("Bot:", reply(engine, query))


if __name__ == "__main__":
    main()
