import torch
from transformers import AutoTokenizer, AutoModel

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def chunk_text(text, chunk_size=256, overlap=50):

    tokens = tokenizer.encode(text)

    chunks = []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        chunks.append(decoded)

    return chunks

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def embed(text, chunk=False):
    if chunk:
        chunks = chunk_text(text)
        return chunks, torch.cat([embed(c, chunk=False) for c in chunks]).numpy()
    
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True,
                       padding=True,
                       max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    return mean_pooling(outputs, inputs['attention_mask'])