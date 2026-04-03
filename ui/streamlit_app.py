import streamlit as st
import requests

st.title("📄 Multimodal RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF")

if uploaded_file:
    files = {
        "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
    }

    res = requests.post("http://localhost:8000/upload/", files=files)

    if res.status_code == 200:
        st.success("File uploaded!")
    else:
        st.error("Upload failed")
        st.text(res.text)

# Query
query = st.text_input("Ask a question")

if query:
    res = requests.get(
        "http://localhost:8000/query/",
        params={"q": query}
    )

    # 🔍 Debug info
    st.write("Status:", res.status_code)

    try:
        data = res.json()
    except Exception:
        st.error("Invalid JSON from backend")
        st.text(res.text)
        st.stop()

    if "error" in data:
        st.error(data["error"])
    else:
        st.write("### Answer")
        st.write(data.get("answer", "No answer"))

        st.write("### Context")
        for c in data.get("context", []):
            st.write("-", c[:200])