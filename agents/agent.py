from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME

client = Groq(api_key=GROQ_API_KEY)

def agent_answer(query, context):
    prompt = f"""
You are an intelligent RAG assistant.

Use ONLY the given context to answer.

Context:
{context}

Question:
{query}

Give a clear, structured answer with bullet points if needed.
Also mention the source from context.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )

    return response.choices[0].message.content