import streamlit as st
from rag.loader import load_pdf
from rag.embedder import get_embeddings
from rag.vectorstore import VectorStore
from agents.agent import agent_answer

st.set_page_config(page_title="Multi-modal RAG Chatbot", layout="wide")

st.title("🧾 Multi-modal RAG + Agentic AI Chatbot")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Save file
    with open("data/sample.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # Load & embed
    texts = load_pdf("data/sample.pdf")
    embeddings = get_embeddings(texts)

    # Create vector store
    vs = VectorStore(len(embeddings[0]))
    vs.add(embeddings, texts)

    # Query input
    query = st.text_input("Ask a question")

    if query:
        # 👉 Get query embedding
        query_embedding = get_embeddings([query])

        # 👉 Search from FAISS
        results = vs.search(query_embedding, k=3)

        # 👉 Extract context
        context = "\n\n".join([r["text"] for r in results])

        # 👉 Generate answer
        answer = agent_answer(query, context)

        # 👉 Compute relevance score (FIXED)
        scores = [1 / (1 + r["distance"]) for r in results]
        relevance_score = round(max(scores), 3)

        # ================= UI =================
        st.subheader("📌 Answer")
        st.write(answer)

        st.subheader("📊 Relevance Score")
        st.success(relevance_score)

        st.subheader("📚 Retrieved Context")
        for i, r in enumerate(results):
            st.write(f"**Chunk {i+1}:**")
            st.write(r["text"][:300])
            st.write(f"Score: {round(1 / (1 + r['distance']), 3)}")
            st.markdown("---")