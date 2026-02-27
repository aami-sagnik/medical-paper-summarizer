# Medical Paper Summarizer
Summarize and query medical papers from PubMed using the PubMedBert encoder and an external OpenAI-compatible LLM API. 

## Instructions
1. Clone the repo and run `uv sync` to get all the dependencies.
2. Modify the llm model and llm url in `main.py`. The LLM API must be OpenAI-compatible (ollama and lm-studio endpoints should be fine).
3. To summarize a paper run `uv run main.py <PedMedURL>`.