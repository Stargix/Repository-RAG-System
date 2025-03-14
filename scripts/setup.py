import numpy as np
import typing
import dotenv
import json
import os
import re
import subprocess
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Get the files in the repository
def get_repo_files(url,directory):
    if not os.path.exists(repo_dir):
        subprocess.run(["git", "clone", url, directory])

    files_dict = {}
    for file_path in Path(directory).rglob("*"):
        if file_path.is_file():
            try:
                with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                    content = f.read()
                relative_path = file_path.relative_to(directory)
                files_dict[str(relative_path)] = content
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return files_dict

def get_chunks(doc, chunk_size=2500, chunk_overlap=50) -> typing.List[dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, # Size of each chunk in characters
        chunk_overlap=chunk_overlap, # Overlap between consecutive chunks
        length_function=len, # Function to compute the length of the text
        add_start_index=True, # Flag to add start index to each chunk
    )
    # Split document into smaller chunks using text splitter
    chunks = text_splitter.split_documents(doc)

    return chunks

def normalize_code(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    # Preserve indentation while normalizing the rest of the line
    lines = []
    for line in code.split('\n'):
        # Count initial spaces
        leading_space_count = len(line) - len(line.lstrip(' \t'))
        leading_space = line[:leading_space_count]   
        # Normalize the rest of the line
        rest_of_line = re.sub(r'[ \t]+', ' ', line[leading_space_count:])
        lines.append(leading_space + rest_of_line)
        
    code = '\n'.join(lines)
    # Eliminate consecutive blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)
    
    return code.strip()

from tqdm import tqdm
import chromadb

def add_chunks_to_chromadb(chunks, encoder_model, name="escrcpy_repo_embeddings",batch_size=128):
    
    # Delete the existing ChromaDB collection if it exists
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    chroma_collection = chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    # Process in batches
    for i in tqdm(range(0, len(chunks), batch_size), desc="Processing Chunks"):
        batch = chunks[i:i+batch_size]
        batch_texts = [normalize_code(chunk.page_content) for chunk in batch]
        batch_embeddings = encoder_model.encode(batch_texts)
        
        # Prepare batch data for ChromaDB
        batch_ids = [f"chunk_{i + j}" for j in range(len(batch))]
        batch_documents = [chunk.page_content for chunk in batch]
        batch_metadatas = [{
            "source": chunk.metadata["source"],
            "start_index": chunk.metadata.get("start_index", 0)
        } for chunk in batch]
        
        # Add entire batch at once
        chroma_collection.add(
            embeddings=[emb.tolist() for emb in batch_embeddings],
            documents=batch_documents,
            metadatas=batch_metadatas,
            ids=batch_ids
        )

if __name__ == '__main__':

    repo_url = "https://github.com/viarotel-org/escrcpy"
    repo_dir = "repo_temp"

    files_dict = get_repo_files(repo_url, repo_dir)

    chunked_files = [Document(page_content=file_content, metadata={"source": file_path})
                    for file_path, file_content in files_dict.items()]
    
    chunks = get_chunks(chunked_files, chunk_size=2500, chunk_overlap=50)

    encoder_model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual", model_kwargs={"torch_dtype": "float16"})

    add_chunks_to_chromadb(chunks, encoder_model)
