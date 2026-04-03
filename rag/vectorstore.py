import faiss
import numpy as np

class VectorStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, query_embedding, k=3):
        query_embedding = np.array(query_embedding).astype("float32")

        D, I = self.index.search(query_embedding, k)

        results = []
        for idx, dist in zip(I[0], D[0]):
            results.append({
                "text": self.texts[idx],
                "distance": float(dist)
            })

        return results