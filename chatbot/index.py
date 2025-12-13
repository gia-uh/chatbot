import fitz
import yaml
import unicodedata
from pathlib import Path
from beaver import Document
from .utils import embed, get_db

db = get_db()


def yaml_handler(file_path: Path, max_child_tokens=512, max_parent_tokens=2048):    
    """Load and chunk a YAML file using hierarchical chunking.
    Args:
        file_path (Path): The path to the YAML file.
    Returns:
        List[Dict]: A list of chunks with their paths and content.
    """
    yaml_dict = yaml.safe_load(file_path.read_bytes())
    
    chunks_parent = []
    chunks_child = []
    
    for top_key, top_value in yaml_dict.items():
        parent_context = {
            'key': top_key,
            'depth': 0,
            'full_path': [top_key]
        }
        
        # Si es un valor simple, crear chunk directamente
        if not isinstance(top_value, dict):
            chunk = {
                'content': f"{top_key}: {top_value}",
                'metadata': parent_context,
                'type': 'leaf'
            }
            chunks_child.append(chunk)
        else:
            # Procesar estructura anidada
            parent_text = serialize_to_yaml(top_key, top_value)
            parent_chunk = {
                'content': parent_text,
                'metadata': parent_context,
                'type': 'parent',
                'children': []
            }
            
            # Generar chunks hijo respetando semántica YAML
            for child_key, child_value in recursive_traverse(top_value):
                child_chunk = {
                    'content': format_child_content(child_key, child_value),
                    'metadata': {
                        **parent_context,
                        'parent_id': id(parent_chunk),
                        'full_path': parent_context['full_path'] + [child_key]
                    },
                    'type': 'child'
                }
                chunks_child.append(child_chunk)
                parent_chunk['children'].append(id(child_chunk))
    
    return chunks_parent, chunks_child
 
 
def clean_text(text: str) -> str:
    """
    Clean and normalize text by handling encoding issues and normalizing Unicode characters.
    This function will:
    1. Normalize Unicode characters (e.g., 'acci'on' -> 'acción')
    2. Remove any invalid Unicode characters
    3. Normalize whitespace
    
    Args:
        text (str): The input text to clean
        
    Returns:
        str: The cleaned and normalized text
    """
    # Handle the specific case of Spanish acute accents
    if "'on" in text:
        text = text.replace("'on", "ón")
    if "'an" in text:
        text = text.replace("'an", "án")
    if "'en" in text:
        text = text.replace("'en", "én")
    if "'in" in text:
        text = text.replace("'in", "ín")
    if "'un" in text:
        text = text.replace("'un", "ún")
    
    # Normalize any remaining Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text


def pdf_handler(doc_path: Path):
    """Load and chunk a PDF document, stopping at the first page that contains images.
        Args:
            doc_path (Path): The path to the PDF document.
        Returns:
            List[str]: A list of text chunks extracted from the PDF.
    """
    doc = fitz.open(doc_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)

        # Verificar si la página contiene imágenes
        image_list = page.get_images(full=True)
        if image_list:
            # ¡Imagen encontrada! Detenerse y no incluir esta página
            break

        # Si no hay imágenes, extraer y añadir el texto
        text = page.get_text("text")
        full_text.append(text)

    doc.close()
    d = "\n".join(full_text).strip().split("¿")[1:]
    chunks = [clean_text(f"¿{c}") for c in d]
    return chunks
    
    
def index():
    data_dir = Path("./data")
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Collect available handlers (functions named "<foldername>_handler")
    handlers = {name: fn for name, fn in globals().items() if name.endswith("_handler")}
    documents = db.collection("documents")
    
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        
        handler_name = f"{folder.stem}_handler"
        handler = handlers.get(handler_name)
        
        if handler is None:
            print(f"No handler found for folder: {folder}")
            continue
        
        for file_path in sorted(folder.iterdir()):
            if not file_path.is_file():
        
                continue
            print(f"Processing {file_path} with {handler_name}...")
        
            try:
                chunks = handler(file_path)
            except Exception as e:
                print(f"Handler {handler_name} failed for {file_path}: {e}")
                continue
        
            if not chunks:
                print(f"No chunks returned for {file_path}")
                continue
        
            for i, chunk in enumerate(chunks):
                try:
                    embedding = embed(chunk)
                    documents.index(
                        Document(
                            body=chunk,
                            embedding=embedding,
                            # metadata={
                            #     "source_file": file_path.name,
                            #     "chunk_index": i,
                            # }
                        )
                    )
                except Exception as e:
                    print(f"Failed to index chunk {i} from {file_path}: {e}")
        
if __name__=="__main__":
    query = embed("Que es la tarjeta clasica") 
    docs = get_db().collection("documents").search(query, top_k=5)
    print([doc.body for doc, _ in docs])