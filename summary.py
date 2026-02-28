from utils import fetch_content
from query import query
from embed import embed
from llm import llm

def summarize(paper_url):
    content = fetch_content(paper_url)
    vectorstore = embed(content)
    return query("Summarize the paper", llm=llm, vectorstore=vectorstore)