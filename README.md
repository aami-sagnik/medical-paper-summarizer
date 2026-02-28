# Medical Paper Summarizer
Summarize and query medical papers from PubMed using the PubMedBert encoder and an external OpenAI-compatible LLM API. 

## Instructions
1. Clone the repo and run `uv sync` to get all the dependencies.
2. Set the `OPENAI_API_KEY`, `OPENAI_BASE_URL` and `OPENAI_MODEL` environment variables in your shell before launching the app. The LLM API must be OpenAI-compatible (ollama and lm-studio endpoints should be fine).
3. To launch the app, `uvicorn main:app --reload`. Now the app is available at `http://localhost:8000`
3. To summarize a paper make a `GET` request at `http://localhost:8000/summary/<PedMedURL>`.