# Code Repository RAG System

A Retrieval-Augmented Generation (RAG) system for searching and exploring code repositories using natural language queries. This system indexes code repositories and enables semantic search to find relevant files and code snippets based on their meaning rather than just keywords.

## Features

- **Repository Indexing**: Clone repositories or use GitHub API to access and index code files
- **Semantic Search**: Retrieve relevant code using natural language queries
- **Code Understanding**: Generate concise summaries and answers about repository functionality
- **Performance Evaluation**: Built-in evaluation capabilities using recall@10 metrics

## Requirements

All dependencies can be installed from the requirements.txt file:

```bash
pip install -r requirements.txt
```

You'll also need:
- Python 3.8 or higher
- Git (for repository cloning)
- GitHub API token (optional, for API-based repository access)
- Google genai API token (optional, for calling the LLM)

Create a `.env` file with your API keys:

```plaintext
GITHUB_TOKEN=your_github_token_here
LLM_API_KEY=your_llm_api_key_here
```

## Usage

### Setting Up and Indexing a Repository

You can use the provided setup script to clone and index a repository:

```bash
python setup.py
```

Or run the notebook `main.ipynb` and execute the indexing cells.

### Searching the Repository

Use the `retrieve_repository` function to search the indexed repository:

```python
from retrieve import retrieve_repository

# Search for relevant code related to a query
results = retrieve_repository("How does the app handle screen rotation?", n_results=5)

# Print the most relevant file
print(f"Most relevant file: {results[0]['source']}")
```

### Generating Answers with LLM

To generate answers based on retrieved code, use `llm.py` or the last chunk of the main notebook:

```python
from llm import generate_answer

# Example usage
answer = generate_answer("How is the sponsor dialog implemented?")
print(answer)
```

### Evaluating Performance

The system includes evaluation capabilities using recall metrics:

```bash
python evaluate.py
```

This will run evaluation on a test set and output the average recall@10 score based on a synthetic dataset from the [escrcpy repository](https://github.com/viarotel-org/escrcpy).

## Project Structure

- `setup.py`: Script for setting up and indexing a repository
- `retrieve.py`: Core retrieval functions for searching the indexed repository
- `evaluate.py`: Evaluation script to measure system performance
- `main.ipynb`: Jupyter notebook demonstrating the full workflow
- `requirements.txt`: List of required Python packages
- `chroma_db/`: Directory containing the indexed repository as a vector database
- `repo_temp/`: Temporal directory containing a clone from the target repository

## Technical Details

This RAG system uses:
- **Embedding Model**: IBM Granite Embedding 107M Multilingual model for semantic code understanding
- **Vector Store**: ChromaDB for efficient similarity search (cosine distance)
- **Text Chunking**: Recursive character text splitting for preserving code structure
- **Normalization**: Custom code normalization preserving indentation while cleaning whitespace
- **LLM Integration**: Optional Gemini 2.0 Flash model for generating answers (requires API key)
