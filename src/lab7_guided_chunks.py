"""Lab 7 - guided example 6.3: chunking with source offsets (uses lab7_chunks.py).
Run: python src/lab7_guided_chunks.py
"""
try:
    from .lab7_chunks import windows, sentence_chunks
except ImportError:                     # so it also runs as: python src/<file>.py
    from lab7_chunks import windows, sentence_chunks


def main():
    text = ("Late work is penalized. Illness may qualify for an extension "
            "when medical evidence is provided. Ask the coordinator first.")
    long_text = text + " " + " ".join(["This sentence keeps going without any full stop"] * 4) + "."
    for label, src, budget in [("12-word budget", text, 12), ("20-word budget", text, 20),
                               ("long sentence, 12-word budget", long_text, 12)]:
        print("=====", label, "| words in source:", len(src.split()))
        for name, spans in [("word windows", windows(src, size=budget, overlap=4)),
                            ("sentence aware", sentence_chunks(src, size=budget))]:
            emitted = sum(len(src[a:b].split()) for a, b in spans)
            print(name, "| chunks:", len(spans), "| duplicate words:", emitted - len(src.split()))
            for start, end in spans:
                print(" ", start, end, repr(src[start:end]))
                assert 0 <= start < end <= len(src)


if __name__ == "__main__":
    main()
