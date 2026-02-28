import os
from langchain_openai import ChatOpenAI

llm_url = os.environ["OPENAI_BASE_URL"]
llm_model = os.environ["OPENAI_MODEL"]
llm_key = os.environ["OPENAI_API_KEY"]

llm = ChatOpenAI(model=llm_model, openai_api_key=llm_key, base_url=llm_url)

