from sentence_transformers import SentenceTransformer
import chromadb
import re

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

def retrieve_repository(question, n_results=5, chroma_collection=None):

    encoder_model = SentenceTransformer("ibm-granite/granite-embedding-107m-multilingual", model_kwargs={"torch_dtype": "float16"})  
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    if chroma_collection is None:
        chroma_collection = chroma_client.get_collection("escrcpy_repo_embeddings")
    
    # Normalize the question text and generate embedding
    normalized_question = normalize_code(question)
    question_embedding = encoder_model.encode(normalized_question)
    
    combined_results = {}
    seen_sources = set()
    max_attempts = n_results * 3  # Avoid infinite loop
    offset = 0
    
    # Keep querying until we have n_results unique sources or reach max_attempts
    while len(combined_results) < n_results and offset < max_attempts:
        # Calculate how many more results we need
        remaining = n_results - len(combined_results)
        
        results = chroma_collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=remaining + offset,  # Get extra results to account for duplicates
            include=["documents", "metadatas", "distances"]
        )
        
        for i in range(len(results["documents"][0])):
            source = results["metadatas"][0][i]["source"]
            content = results["documents"][0][i]
            similarity = 1 - results["distances"][0][i]
            
            # Skip if we've already seen this source
            if source in seen_sources:
                combined_results[source]["content"] += "\n\n" + content
                combined_results[source]["similarity"] = max(combined_results[source]["similarity"], similarity)
            
            else:
                seen_sources.add(source)
                combined_results[source] = {
                    "content": content,
                    "source": source,
                    "similarity": similarity
                }   
        # Increase offset to get new results in the next query
        offset += remaining
    
    # Convert dictionary back to list and sort by similarity
    formatted_results = list(combined_results.values())
    formatted_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return formatted_results[:n_results]