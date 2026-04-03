from rag.embedder import get_embeddings

def retrieve(query, vectorstore):
    query_emb = get_embeddings([query])
    results = vectorstore.search(query_emb)
    return results