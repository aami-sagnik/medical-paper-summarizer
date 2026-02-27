import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from embed import embed

def create_messages(text): 
  return [
    {"role": "system", "content": "You are a biomedical expert"},
    {"role": "user", "content": text},
  ]

def retrieve(query_text, faiss_index, chunks, k=5):
    q_vec = embed(query_text).numpy()
    D, I = faiss_index.search(q_vec, k)
    return [chunks[i] for i in I[0]]

def query(text, llm, model, faiss_index, chunks):
    context = " ".join(retrieve(text, faiss_index, chunks))
    prompt = f"""
    You are a biomedical question answering system.

    Answer the query using ONLY the information
    provided in the context below.

    Do NOT use prior knowledge.
    Do NOT make assumptions.
    Do NOT infer beyond the context.

    If the answer is not explicitly stated
    in the context, reply exactly with:
    "Insufficient information in context."

    Context:
    {context}

    Query:
    {text}

    Answer:
    """

    messages = create_messages(prompt)
    
    response = llm.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content