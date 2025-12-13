from beaver import BeaverDB
from .config import load
from .embeddings import Embedding

def embed(text: str) -> list[float]:
    config = load()
    
    embedding = Embedding(**config.embedding.model_dump())
    response = embedding.create([text])
    return response[0] if response else []

def get_db() -> BeaverDB:
    config = load()
    db = BeaverDB(config.db)
    return db