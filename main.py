import torch
import sys
import faiss
from openai import OpenAI
from utils import fetch_content
from query import query
from embed import embed

llm_url = "http://localhost:1234/v1"
llm_model = "llama-3.2-3b-instruct"
llm = OpenAI(base_url=llm_url, api_key="")

text = "The patient was diagnosed with lung cancer and received chemotherapy."

if len(sys.argv) < 2:
    print("Usage python3 main.py <paper_url>")
    sys.exit(1)

paper_url = sys.argv[1]

content = fetch_content(paper_url)
chunks, embeddings = embed(content, chunk=True)
index = faiss.IndexFlatL2(768)
index.add(embeddings)

print("<--- Chat begins --->")

text = "Summarize the document"

while text != "exit":
    answer = query(text, llm=llm, model=llm_model, faiss_index=index, chunks=chunks)
    print(answer)
    text = input("Query (enter exit to quit): ").strip()
