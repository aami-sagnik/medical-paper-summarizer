FROM astral/uv:0.10-python3.13-trixie

WORKDIR /app

# HuggingFace cache location inside container
ENV HF_HOME=/hf-cache
ENV TRANSFORMERS_CACHE=/hf-cache
ENV HF_HUB_DISABLE_TELEMETRY=1

# System deps required by torch / transformers
RUN apt install -y git

# Copy dependency files first
COPY pyproject.toml uv.lock ./

# Install Python dependencies
RUN uv sync --frozen --no-dev

# Pre-download the model into the image
RUN uv run python -c "\
from transformers import AutoTokenizer, AutoModel; \
model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'; \
AutoTokenizer.from_pretrained(model_name); \
AutoModel.from_pretrained(model_name); \
print('Model downloaded successfully') \
"

# Copy rest of project
COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]