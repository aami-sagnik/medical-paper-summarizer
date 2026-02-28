import torch
import torch.nn.functional as F
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModel
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class BiomedBERTEmbeddings(Embeddings):

    def embed_documents(self, texts):

        encoded_input = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.inference_mode():
            model_output = model(**encoded_input)

        embeddings = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().tolist()


    def embed_query(self, text):

        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = model(**encoded_input)

        embedding = mean_pooling(
            model_output,
            encoded_input["attention_mask"]
        )

        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()[0].tolist()

embedding_model = BiomedBERTEmbeddings()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80
)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

def embed(text, chunk=False):
    docs = splitter.split_documents([Document(page_content=text)])
    vectorstore = FAISS.from_documents(
        docs,
        embedding_model
    )
    return vectorstore
