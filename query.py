import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_core.messages import HumanMessage, SystemMessage

def create_messages(text):
    system = """    
    You are a biomedical question answering system.

    Answer the query using ONLY the information
    provided in the context below.

    Do NOT use prior knowledge.
    Do NOT make assumptions.
    Do NOT infer beyond the context.

    If the answer is not explicitly stated
    in the context, reply exactly with:
    'Insufficient information in context.'
    """

    return [
      SystemMessage(content=system),
      HumanMessage(content=text)
    ]

def retrieve(query_text, vectorstore, k=5):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )
    results = retriever.invoke(query_text)
    return [ r.page_content for r in results ]

def query(text, llm, vectorstore):
    context = retrieve(text, vectorstore)
    context = " ".join(context)
    prompt = f"""
    Context:
    {context}

    Query:
    {text}

    Answer:
    """

    messages = create_messages(prompt)
    response = llm.invoke(messages)

    return response.content