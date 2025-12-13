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
    
    embedding = Embedding(**config.embedding.model_dump())
    response = embedding.create([text])
    return response[0] if response else []

def get_db() -> BeaverDB:
    config = load()
    db = BeaverDB(config.db)
    return db