"""Lab 7 - guided example 6.2: sentence embeddings and cosine similarity.
Run: python src/lab7_guided_embeddings.py
"""
import numpy as np
from sentence_transformers import SentenceTransformer


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Zero vector: cosine similarity is undefined")
    return float(a @ b / (na * nb))


def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    passages = ["Reset a forgotten password using the recovery link.",
                "Submit assignments through the learning portal.",
                "Check ink levels when a printer produces blank pages."]
    C = model.encode(passages, normalize_embeddings=True)
    assert np.allclose(np.linalg.norm(C, axis=1), 1, atol=1e-5)
    print("Embedding shape:", C.shape)
    for query in ["How can I recover my login?", "I do not need a password reset"]:
        Q = model.encode([query], normalize_embeddings=True)
        scores = (Q @ C.T)[0]
        print("\nQuery:", query)
        for i in np.argsort(-scores):
            print(i, round(float(scores[i]), 3), passages[i])
        # dot product of unit vectors equals cosine similarity
        assert abs(cosine(Q[0], C[0]) - float(scores[0])) < 1e-5
    try:
        cosine(np.zeros(3), np.ones(3))
    except ValueError as e:
        print("\nZero-vector check:", e)


if __name__ == "__main__":
    main()
