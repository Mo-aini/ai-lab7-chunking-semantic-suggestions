# AI Lab 7 - Text Chunking and Semantic Suggestions

Lab 7 for the AI lab course (CAI601400), NLP fundamentals part 2. I split documents into chunks in two ways, fixed word windows (40 words, overlap 0 and 10) and sentence-aware packing with a 40-word budget, keep the source id and character offsets on every chunk, and test the edge cases. Then I compare TF-IDF and sentence embeddings for semantic suggestions over those chunks.

## Files

- `main.py` runs everything
- `src/lab7_chunks.py` the chunking functions, boundary tests and the window vs sentence comparison
- `src/lab7_guided_chunks.py` guided example 6.3, chunking with offsets
- `src/lab7_guided_embeddings.py` guided example 6.2, embeddings and cosine similarity
- `src/lab7_semantic_check.py` semantic suggestions over the chunk records, TF-IDF vs embeddings on paraphrased questions
- `src/lab7_suggestions.py` guided example 6.7, related questions from the FAQ list
- `src/lab7_faq.py` the FAQ list from Lab 6, used by the suggestions
- `src/lab7_search.py` the `Retriever` class from the manual
- `src/lab7_dataset.py` writes `data/corpus.json` and `data/queries.json`
- `data/` the synthetic help-desk documents (6 documents, also used in Lab 8)
- `outputs/output.txt` output of `main.py` on my machine
- `report/` the written report (DOCX and PDF). Labs 6, 7 and 8 were written up in one report; this lab is section 3

## How to run

Python 3.10+ is needed. The embedding parts download `sentence-transformers/all-MiniLM-L6-v2` (about 90 MB) the first time and run on CPU.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                   # add --no-model for the TF-IDF parts only
```

`data/` is already in the repo. To rebuild it: `python src/lab7_dataset.py --force`.

## Results

Three documents (587 words), 40-word budget:

| Strategy | Chunks | Mean words | Duplicate words | Labeled spans kept whole |
|----------|--------|------------|-----------------|--------------------------|
| window, overlap 0 | 16 | 36.7 | 0 | 1/3 |
| window, overlap 10 | 20 | 37.9 | 170 | 0/3 |
| sentence-aware | 21 | 28.0 | 0 | 1/3 |

All boundary tests pass: empty text, a document shorter than the budget, four invalid overlap values, a 95-word sentence that falls back to 3 windows, and offsets that reproduce the source text exactly.

The 10-word overlap duplicated 170 words and still kept none of the labeled rule + exception spans whole, because they are longer than the overlap. The sentence splitter also cuts after `Dr.` and `e.g.`, so it produces a `'Dr.'` fragment.

Semantic suggestions: for "my printouts come out with nothing on them" TF-IDF ranks three wrong chunks first, while embeddings put the blank-pages chunk first (0.528), because there are almost no shared words with the document.

## Author

Rasikh Ali (42365)

MIT License, see LICENSE.
