from os import getenv
#from functools import lru_cache
#from fastembed import TextEmbedding
from beaver import BeaverDB
from .config import load
from .embeddings import Embedding
# def embed(text: str) -> list[float]:
#     model = get_text_embedding(getenv("EMBEDDING"))
#     return list(model.embed(text))[0].tolist()

# @lru_cache
# def get_text_embedding(model) -> TextEmbedding:
#     return TextEmbedding(model=model)

def embed(text: str) -> list[float]:
    config = load()
    embedding_model = getenv("EMBEDDING_MODEL","")
    api_key = config.llm.api_key
    base_url = config.llm.base_url
    
    embedding = Embedding(api_key=api_key, base_url=base_url, embedding_model=embedding_model, embedding_dimension=768)
    response = embedding.create([text])
    return response[0] if response else []

def get_db() -> BeaverDB:
    config = load()
    db = BeaverDB(config.db)
    return db