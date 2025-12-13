import yaml
import tiktoken
from typing import Any, List, Dict

# Tokenizador (puedes cambiar el encoding si usas otro modelo)
ENCODING = "cl100k_base"  # adecuado para gpt-3.5-turbo, gpt-4, etc.
tokenizer = tiktoken.get_encoding(ENCODING)

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def serialize_subtree(data: Any) -> str:
    return yaml.dump(data, allow_unicode=True, default_flow_style=False, indent=2)

def hierarchical_chunking(
    data: Any,
    path: str = "",
    max_tokens: int = 3000,
    chunks: List[Dict] = None
) -> List[Dict]:
    if chunks is None:
        chunks = []

    # Serializar el subárbol actual
    content_str = serialize_subtree(data)
    token_count = count_tokens(content_str)

    # Caso base: si el subárbol cabe en un chunk, lo añadimos
    if token_count <= max_tokens:
        chunks.append({
            "path": path or "(root)",
            "content": content_str,
            "token_count": token_count
        })
        return chunks

    # Caso recursivo: descomponer si es dict o list
    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            hierarchical_chunking(value, new_path, max_tokens, chunks)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            hierarchical_chunking(item, new_path, max_tokens, chunks)
    else:
        # Valor escalar que, por alguna razón, supera el límite (poco probable)
        chunks.append({
            "path": path or "(root)",
            "content": content_str,
            "token_count": token_count
        })

    return chunks

def chunk_yaml_file(file_path: str, max_tokens: int = 3000) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return hierarchical_chunking(data, max_tokens=max_tokens)

chunks = chunk_yaml_file("./data/yaml/IA Información comercial.yaml", max_tokens=3000)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} (tokens: {chunk['token_count']}) ---")
    print(f"Path: {chunk['path']}")
    print(chunk['content'])
    print()