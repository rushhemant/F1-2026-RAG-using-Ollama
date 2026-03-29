# Ollama RAG System - F1 Rules Q&A

A sophisticated **Retrieval Augmented Generation (RAG)** system that combines local LLM inference with vector similarity search and intelligent reranking to provide accurate answers about F1 racing rules.

## 🎯 Overview

This project implements a complete RAG pipeline that allows you to ask questions about Formula 1 rules using a local Ollama LLM. The system retrieves relevant information from PDF documents and uses advanced reranking to ensure the most relevant context is provided to the language model.

### Key Features

- **Local LLM Inference**: Uses Ollama with gemma3:4b model (no API keys needed)
- **Semantic Search**: Vector embeddings using SentenceTransformer
- **Intelligent Reranking**: CrossEncoder for improved relevance ranking
- **Persistent Storage**: ChromaDB for efficient vector storage and retrieval
- **PDF Support**: Automatic text extraction from PDF documents
- **Context-Aware Responses**: Grounds answers in retrieved document context

## 🏗️ Architecture

### RAG Pipeline Flow

```
PDF Document
    ↓
Text Extraction → Chunking → Embedding Generation
    ↓
Vector Database (ChromaDB)
    ↓
User Query → Query Embedding → Semantic Similarity Search (Top 10)
    ↓
Reranking (CrossEncoder) → Top 3 Relevant Chunks
    ↓
Context + Query → Ollama LLM → Answer
```

### Components

1. **PDF Loader**: Extracts text from PDF documents (using PyPDF2)
2. **Text Splitter**: Splits documents into overlapping chunks for better context preservation
3. **Embedder**: Generates vector embeddings (all-MiniLM-L6-v2 model)
4. **Vector DB**: Stores and retrieves embeddings (ChromaDB with persistence)
5. **Reranker**: Scores document relevance (BAAI/bge-reranker-v2-m3 model)
6. **LLM**: Generates contextual answers (Ollama with gemma3:4b)

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **Ollama**: Installed and running locally
- **RAM**: Minimum 8GB (16GB recommended for better performance)

### Install Ollama

1. Download from [ollama.ai](https://ollama.ai)
2. Install for your operating system (macOS, Linux, or Windows)
3. Start the Ollama service

### Verify Ollama Installation

```bash
ollama list  # Check available models
ollama pull gemma3:4b  # Download the required model
```

## 🚀 Installation

### 1. Clone or Setup Project

```bash
cd /path/to/your/project
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install ollama
pip install PyPDF2
pip install chromadb
pip install sentence-transformers
pip install langchain-text-splitters
pip install python-dotenv
```

### 4. Verify Installation

```python
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
print("All imports successful!")
```

## 🎯 Usage Guide

### Quick Start

1. **Open the Notebook**
   ```bash
   jupyter notebook RAGusingOllama.ipynb
   ```

2. **Run All Cells**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac) → "Run All Cells"
   - Or run cells sequentially from top to bottom

3. **Modify Your Query**
   - In the **Reranking** cell, change the `query` variable:
   ```python
   query = "Your F1 rules question here"
   ```

4. **View Results**
   - The system will display:
     - Retrieved chunks with scores
     - Final answer from the LLM

### Example Queries

```python
# Example 1: Safety car procedures
query = "Can a team pit when pit lane is closed during safety car?"

# Example 2: Overtaking rules
query = "What are the rules for overtaking under yellow flags?"

# Example 3: Specific regulations
query = "What is the maximum fuel tank capacity allowed?"
```

### Interactive Mode

Uncomment this line in the **Semantic Similarity Retrieval** cell:

```python
query = input("What do you want to know about F1 rules: ")
```

Then run the cell to get interactive prompts.

## ⚙️ Configuration

### Adjust Chunk Size

In the **Text Chunking** cell:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # Increase for longer context
    chunk_overlap=200,   # Increase for more overlap
)
```

- **Larger chunk_size**: Better context but slower retrieval
- **Larger chunk_overlap**: More context continuity but more redundancy

### Change Number of Retrieved Results

In the **Semantic Similarity Retrieval** cell:

```python
results_top10 = collection.query(
    query_embeddings=query_embedding,
    n_results=10  # Change this number
)
```

### Change Final Context Chunks

In the **Prepare Context** cell:

```python
top_chunks = ranked_chunks[:3]  # Change 3 to desired number
```

### Switch Models

#### Embedding Model
```python
embed_model = SentenceTransformer("any-mpnet-base-v2")  # Different model
```

#### Reranker Model
```python
model = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L-2-v2")  # Faster
```

#### LLM Model
```python
ol_chat = chat(
    model='mistral:latest',  # Or any other Ollama model
    messages=[{'role': 'user', 'content': user_prompt}],
    stream=False,
)
```

## 📊 Model Information

### Sentence Transformer (Embedding)
- **Model**: all-MiniLM-L6-v2
- **Dimensions**: 384
- **Use**: Converting text to dense vectors
- **Size**: ~61MB

### CrossEncoder (Reranker)
- **Model**: BAAI/bge-reranker-v2-m3
- **Use**: Scoring query-document relevance
- **Advantage**: More accurate than semantic similarity alone
- **Size**: ~259MB

### Ollama LLM
- **Model**: gemma3:4b
- **Parameters**: 4 billion
- **Context Window**: 8192 tokens
- **Size**: ~2.6GB
- **Use**: Generating contextual answers

## 🔧 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ollama'"

**Solution**: Install the Ollama package
```bash
pip install ollama
```

### Issue: "Connection refused" for Ollama

**Solution**: Ensure Ollama is running
```bash
# macOS/Linux
ollama serve

# Or check if running in background
ps aux | grep ollama
```

### Issue: Models not found

**Solution**: Download required models
```bash
ollama pull gemma3:4b
ollama pull nomic-embed-text  # If using different embedding model
```

### Issue: Slow performance

**Solutions**:
- Reduce `chunk_size` in text splitter
- Reduce `n_results` in vector retrieval
- Use smaller embedding/reranker models
- Increase available system RAM
- Check Ollama is using GPU acceleration

### Issue: Out of memory errors

**Solutions**:
- Reduce chunk sizes
- Fewer retrieved documents
- Use quantized smaller models
- Close other applications

## 💡 How It Works

### 1. Indexing Phase (One-time setup)
```
PDF → Extract Text → Split into Chunks → Generate Embeddings → Store in DB
```

### 2. Query Phase (Per question)
```
Query → Embedding → Vector Search → Initial Retrieval (top 10)
→ Reranking → Top 3 Chunks → LLM Context → Generate Answer
```

### Why Two Stages?

- **Semantic Search**: Fast retrieval from large corpus (embedding-based)
- **Reranking**: Accurate relevance scoring for final context (cross-encoder)
- **Combined**: Best of speed and accuracy

## 🎓 Learning Resources

### About RAG
- [RAG Introduction](https://arxiv.org/abs/2005.11401)
- [LLM Augmentation Techniques](https://towardsdatascience.com)

### Used Libraries
- [Ollama Documentation](https://ollama.ai)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [LangChain Text Splitters](https://python.langchain.com/)

## 📝 Future Enhancements

- [ ] Add support for multiple document sources
- [ ] Implement streaming responses
- [ ] Add conversation history/memory
- [ ] Web interface with Streamlit/Gradio
- [ ] Support for different languages
- [ ] Advanced prompt engineering
- [ ] Response quality evaluation metrics
- [ ] Multi-stage reranking

## 🤝 Contributing

To improve this project:

1. Test with different documents
2. Experiment with various models
3. Share results and optimizations
4. Report issues and suggestions

## 📄 License

This project is open source and available for educational and research purposes.

## 📧 Contact & Support

For questions or issues:
- Review the troubleshooting section
- Check model documentation
- Verify Ollama installation

---

**Last Updated**: March 2026
**Version**: 1.0
**Status**: Production Ready ✓
