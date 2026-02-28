import torch
import sys
from utils import fetch_content
from query import query
from embed import embed
from langchain_openai import ChatOpenAI

import os
os.environ["HF_HUB_DISABLE_UNAUTHENTICATED_WARNING"] = "1"

llm_url = "http://localhost:1234/v1"
llm_model = "llama-3.2-3b-instruct"
llm_key = ""
llm = ChatOpenAI(model=llm_model, openai_api_key=llm_key, base_url=llm_url)

text = "The patient was diagnosed with lung cancer and received chemotherapy."

if len(sys.argv) < 2:
    print("Usage python3 main.py <paper_url>")
    sys.exit(1)

paper_url = sys.argv[1]

content = fetch_content(paper_url)
vectorstore = embed(content)

print("<--- Chat begins --->")

text = "Summarize the document"

print("Summary of the paper\n")
while text != "exit":
    if text != "":
        answer = query(text, llm=llm, vectorstore=vectorstore)
        print(answer)
    text = input("\nQuery (enter exit to quit): ").strip()
