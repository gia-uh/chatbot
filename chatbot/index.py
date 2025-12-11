import fitz
import unicodedata
import pandas as pd
from pathlib import Path
from markitdown import MarkItDown
from fastembed import TextEmbedding
from typing import Union, List, Dict, Any
from docx import Document
from os import getenv
from functools import lru_cache
from .config import load
from beaver import BeaverDB


# config = load()
# db = BeaverDB(config.db)


# Example extension sets (define these globally or in config)
_DOCUMENT_EXTENSIONS = {'.txt', '.md', '.pdf', '.docx', '.html', '.htm', '.rtf'}
_TABLE_EXTENSIONS = {'.csv', '.tsv', '.xlsx', '.xls', '.xlsm', '.ods', '.parquet', '.feather', '.json'}



def dataframe_to_chunked_json(df: pd.DataFrame, name: str ,chunk_size: int = 10) -> List[Dict[str, Any]]:
    """
    Convierte un DataFrame en una lista de chunks JSON serializables.
    Cada chunk incluye los encabezados y un subconjunto de filas.

    Args:
        df (pd.DataFrame): DataFrame de entrada.
        chunk_size (int): Número máximo de filas por chunk.

    Returns:
        List[Dict]: Lista de chunks, cada uno con:
            {
                "headers": ["Col1", "Col2", ...],
                "rows": [
                    ["Val1", "Val2", ...],
                    ...
                ]
            }
    """
    headers = df.columns.tolist()
    records = df.values.tolist()  # Lista de filas (cada fila es una lista)

    chunks = []
    for i in range(0, len(records), chunk_size):
        chunk_rows = records[i:i + chunk_size]
        chunks.append({
            "name": name,
            "headers": headers,
            "rows": chunk_rows
        })
    return chunks


def docx_handler(doc_path: Path):
    return "Hello from docx_handler"
    

def docx_table_handler(doc_path:Path):
    return "Hello from docx_table_handler"
    

def docx_tables_to_dataframes(docx_path: str) -> List[pd.DataFrame]:
    """
    Lee un archivo .docx y extrae todas las tablas como una lista de pandas DataFrames.
    
    Args:
        docx_path (str): Ruta al archivo .docx.
        
    Returns:
        List[pd.DataFrame]: Lista de DataFrames, uno por cada tabla en el documento.
                            Si no hay tablas, devuelve una lista vacía.
    """
    doc = Document(docx_path)
    dataframes = []

    for table in doc.tables:
        # Extraer filas como listas de texto
        data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            data.append(row_data)
        
        if data:
            # Detectar encabezados: asumimos que la primera fila es el encabezado
            headers = data[0]
            rows = data[1:]
            
            # Si hay más de una fila, creamos el DataFrame con encabezados
            # Si solo hay una fila, la tratamos como datos sin encabezado (índices 0,1,...)
            if len(data) > 1:
                df = pd.DataFrame(rows, columns=headers)
            else:
                df = pd.DataFrame(data)
            dataframes.append(df)

    return dataframes

def docx_text_table_handler(doc_path:Path):
    return "Hello from docx_text_table_hanlder"


def pdf_images_handler(doc_path: Path):
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
    

def tables_handler(doc_path: Path):
    """
    Carga todas las hojas de un archivo Excel (xls o xlsx) en un diccionario
    de pandas DataFrames.

    Args:
        file_path (str): Ruta al archivo Excel.

    Returns:
        Dict[str, pd.DataFrame]: Diccionario donde las claves son los nombres
                                 de las hojas y los valores son los DataFrames.
    """
    data = pd.read_excel(doc_path, sheet_name=None, dtype=str)
    for key, value in data.items():
        name = f"{doc_path.name}_{key}"
        print(name)
    return data
    
    

def index():
    for file_path in Path("./data").iterdir():
        name = f"{file_path.name}_handler"
        
        # Checking if there is a handler for a specific folder type 
        if file_path.is_dir() and name in globals():
            handler = globals()[name]  
            if not callable(handler):
                print(f"This handler {name} is not a function")
            
            # Iterating through each file and processing it   
            for doc in file_path.iterdir():
                chunks = handler(doc)
                #print(chunks)


        


def _process_document(file_path: Path):
    try:
        # Convert any supported document to Markdown
        markitdown = MarkItDown()
        result = markitdown.convert(str(file_path))
        markdown = clean_text(result.markdown)  # Extract the markdown string
        # Optionally, access metadata: result.metadata

        # Do something with the markdown (e.g., store, index, etc.)
        print(f"Converted {file_path.name} to Markdown ({len(markdown)} chars)")
        return markdown

    except Exception as e:
        print(f"Failed to convert {file_path}: {e}")
        return ""


def load_table_file(filepath: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Load a tabular data file into a pandas DataFrame, auto-detecting format by extension.

    Supported formats:
      - CSV/TSV: .csv, .tsv, .txt
      - Excel: .xls, .xlsx, .xlsm, .xlsb, .ods
      - Parquet: .parquet
      - Feather: .feather
      - JSON: .json
      - HTML: .html, .htm (loads first table by default)
      - XML: .xml (assumes flat table structure)

    Parameters:
        filepath (str or Path): Path to the file.
        **kwargs: Passed to the underlying pandas read_* function.

    Returns:
        pd.DataFrame: Loaded table.

    Raises:
        ValueError: If file extension is unsupported.
        FileNotFoundError: If file does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    suffix = filepath.suffix.lower().lstrip('.')

    # Normalize common aliases
    if suffix in ('csv', 'tsv'):
        sep = kwargs.pop('sep', ',' if suffix == 'csv' else '\t')
        return pd.read_csv(filepath, sep=sep, **kwargs)

    elif suffix in ('xls', 'xlsx', 'xlsm', 'xlsb'):
        # pandas uses openpyxl or xlrd (for .xls) under the hood
        return pd.read_excel(filepath, **kwargs)

    elif suffix == 'ods':
        # Requires 'odfpy'
        return pd.read_excel(filepath, engine='odf', **kwargs)

    elif suffix == 'parquet':
        return pd.read_parquet(filepath, **kwargs)

    elif suffix == 'feather':
        return pd.read_feather(filepath, **kwargs)

    elif suffix == 'json':
        return pd.read_json(filepath, **kwargs)

    elif suffix in ('html', 'htm'):
        dfs = pd.read_html(filepath, **kwargs)
        if not dfs:
            raise ValueError(f"No tables found in HTML file: {filepath}")
        return dfs[0]  # Return first table; modify if you need selection logic

    elif suffix == 'xml':
        return pd.read_xml(filepath, **kwargs)

    else:
        raise ValueError(
            f"Unsupported file format: '.{suffix}'. "
            "Supported: csv, tsv, txt, xls, xlsx, xlsm, xlsb, ods, "
            "parquet, feather, json, html, htm, xml."
        )    


def _process_data(df: pd.DataFrame, file_path: Path):
    print(f"Loaded table: {file_path.name} with shape {df.shape}")
    # Example: convert to dict, generate SQL INSERTs, etc.
    # records = df.to_dict(orient='records')    
    print(df)
    
def table2text(rows: list[str]):
    import re
    patterns = r"\| Unnamed: \d \||\| \-{3,} \||\|\s+\||\| NaN \|"
    new_rows = []
    for row in rows:
        if not re.search(patterns,row):
            new_rows.append(row)
    
    rows_name = new_rows[0].strip("|").split("|")
    end_lines = []
    for new_row  in new_rows[1:]:
        elemns  = new_row.strip("|").split("|")
        line = " ".join(f"{i.strip().strip("*")} {j.strip()}" for i,j in zip(rows_name,elemns))
        end_lines.append(line)
    return "\n".join(end_lines)


def markdown2tree(text: str) -> list[dict]:
    """
    Convert a markdown text to a tree structure.
    Returns a list of dictionaries representing the markdown hierarchy.
    Each node has:
    - level: int (header level, inf for text)
    - content: str (header text without # or text content)
    - children: list[dict] (child nodes)
    """
    if not text.strip():
        return []

    tree = []
    node_stack = []  # Stack to track parent nodes
    current_text = []
    
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        print(line)
        print()
        if line.startswith("#"):
            # If there's accumulated text, add it to the current node or root
            if current_text:
                text_node = {
                    "level": float("inf"),
                    "content": "\n".join(current_text).strip(),
                    "children": []
                }
                if node_stack:
                    node_stack[-1]["children"].append(text_node)
                else:
                    tree.append(text_node)
                current_text = []
            
            # Extract header level and content
            level = len(line) - len(line.lstrip("#"))
            content = line.lstrip("#").strip()
            node = {"level": level, "content": content, "children": []}

            # Find the appropriate parent node
            while node_stack and node_stack[-1]["level"] >= level:
                node_stack.pop()
            
            if node_stack:
                node_stack[-1]["children"].append(node)
            else:
                tree.append(node)
            
            node_stack.append(node)
        
        # Table start (opcional, puedes mantenerlo si lo necesitas)
        elif line.startswith("|") and line.endswith("|"):
            print("TABLE")
            j = i + 1
            table_lines = [line]
            # Find all the lines that are part of the current table
            for temp_line in lines[j:]:
                strip_line = temp_line.strip()
                if strip_line.startswith("|") and strip_line.endswith("|"):
                    j += 1
                    table_lines.append(strip_line)
                else:
                    break
            
            # Once the search ends, create the text from the table
            table_text = table2text(table_lines) 
            current_text.append(table_text)
            i = j - 1
            
        elif line:
            current_text.append(line)
            
        elif current_text:  # Empty line after text
            text_node = {
                "level": float("inf"),
                "content": "\n".join(current_text).strip(),
                "children": []
            }
            if node_stack:
                node_stack[-1]["children"].append(text_node)
            else:
                tree.append(text_node)
            current_text = []
        
        i += 1
    
    # Handle any remaining text (al final del archivo)
    if current_text:
        text_node = {
            "level": float("inf"),
            "content": "\n".join(current_text).strip(),
            "children": []
        }
        if node_stack:
            node_stack[-1]["children"].append(text_node)
        else:
            tree.append(text_node)
    
    return tree


def hierarchial_markdown_chunker(max_chunk_size: int = 2000):
    """
    Chunker function that splits text based on markdown hierarchy.
    Each chunk is a tuple of (header_path, content).
    - header_path: string with headers separated by newlines
    - content: text content of the chunk
    """
    def chunker(text: str) -> list[tuple[str, str]]:
        tree = markdown2tree(text)
        chunks = []
        
        def process_node(node: dict, current_header_path: str = ""):
            nonlocal chunks
            
            # Add current node's content
            if node["level"] != float("inf"):
                current_header = f"{current_header_path}\n{node['content']}" if current_header_path else node["content"]
                
                # Process children
                for child in node["children"]:
                    process_node(child, current_header)
            else:
                # Text node
                content = node["content"]
                words = content.split()
                current_chunk = []
                current_size = 0
                
                for word in words:
                    if current_size + len(word.split()) > max_chunk_size:
                        if current_chunk:  # Only append if we have content
                            chunks.append((current_header_path.strip(), " ".join(current_chunk).strip()))
                        current_chunk = [word]
                        current_size = len(word.split())
                    else:
                        current_chunk.append(word)
                        current_size += len(word.split())
                
                if current_chunk:  # Append any remaining content
                    chunks.append((current_header_path.strip(), " ".join(current_chunk).strip()))
        
        for node in tree:
            process_node(node, "")
        
        return chunks

    return chunker


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


def embed(text: str) -> list[float]:
    model = get_text_embedding(getenv("EMBEDDING"))
    return list(model.embed(text))[0].tolist()

@lru_cache
def get_text_embedding(model) -> TextEmbedding:
    return TextEmbedding(model=model)

if __name__ == "__main__":
    index()