# RAG Local Knowledge Search

A production-ready RAG (Retrieval-Augmented Generation) system using Ollama for local document search and question answering with automatic resource management.

## 🎯 What is this?

This system allows you to:
- **Upload your own documents** (PDF, Word, Text files)
- **Ask questions** about the content in natural language
- **Get accurate answers** based on your specific documents
- **Run everything locally** - no data leaves your machine
- **Automatically optimize** for your system's resources

## 🔧 How it Works

```
Your Documents → Vector Database → Semantic Search → LLM Response
    ↓               ↓                    ↓              ↓
 PDF/DOCX/TXT → Embeddings → Find Relevant Context → Answer Question
```

1. **Document Ingestion**: Upload documents, system splits them into chunks
2. **Vector Storage**: Creates semantic embeddings using `nomic-embed-text`
3. **Semantic Search**: Finds relevant content based on your question
4. **Answer Generation**: Uses LLM to generate response based on found context

## 📋 Prerequisites

### System Requirements
- **RAM:** Minimum 4GB (8GB+ recommended)
- **CPU:** 4+ cores recommended
- **Storage:** 2GB+ free space
- **OS:** Linux, macOS, or Windows
- **Python:** 3.8 or higher

### Required Software

1. **Install Ollama** (if not already installed):
   ```bash
   # Linux/macOS
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # Windows: Download from https://ollama.ai/download
   ```

2. **Start Ollama service**:
   ```bash
   ollama serve
   ```

3. **Download required models**:
   ```bash
   # Essential: Embedding model (required)
   ollama pull nomic-embed-text
   
   # Choose LLM based on your RAM:
   
   # Option 1: Fast and lightweight (1.5GB RAM)
   ollama pull llama3.2:1b
   
   # Option 2: Good balance (3.5GB RAM)  
   ollama pull llama3.2:3b
   
   # Option 3: High quality (4.5GB RAM)
   ollama pull codellama:7b
   
   # Option 4: Best quality (10GB RAM)
   ollama pull qwen3:14b
   ```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the RAG system files
cd rag/

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make scripts executable (Linux/macOS)
chmod +x cli.py resource_monitor.py model_config.py
```

### 2. Check Your System

```bash
# Check available resources and get model recommendations
python resource_monitor.py

# See model compatibility
python model_config.py
```

### 3. Add Your Documents

```bash
# Add individual documents
python cli.py add my_document.pdf
python cli.py add user_manual.docx
python cli.py add notes.txt

# Add multiple documents
python cli.py add document1.pdf
python cli.py add document2.docx
```

### 4. Start Asking Questions

```bash
# Simple query (auto-selects best model)
python cli.py query "What is the main topic?"

# Specific model (if you have enough RAM)
python cli.py query "Explain the process" --model codellama:7b

# Interactive mode for exploration
python cli.py interactive
```

## 📖 Detailed Usage

### Document Management

**Supported Formats:**
- **PDF files** (`.pdf`) - Extracts text content with OCR fallback
- **Word documents** (`.docx`) - Reads paragraphs, tables, headers, and footers
- **Text files** (`.txt`) - Plain text content

**Adding Documents:**
```bash
# Single document
python cli.py add /path/to/document.pdf

# Check what's in your database
python cli.py list
```

**Managing Documents:**
```bash
# List all documents in knowledge base
python cli.py list

# Remove specific document
python cli.py remove /path/to/document.pdf

# Clear all documents (with confirmation)
python cli.py clear
```

**Important:** Documents must be explicitly added to the vector database. Simply placing files in a folder doesn't automatically index them.

### Querying

**Basic Query:**
```bash
python cli.py query "What are the key points about machine learning?"
```

**Query Options:**
```bash
# Specify number of context chunks (default: 15)
python cli.py query "question" --results 20

# Enable debug mode to see retrieved chunks
python cli.py query "question" --debug

# Set custom timeout (default: 600 seconds / 10 minutes)
python cli.py query "question" --timeout 300

# Force specific model
python cli.py query "question" --model llama3.2:1b

# Combine all options
python cli.py query "question" --model codellama:7b --results 10 --debug --timeout 400
```

**Interactive Mode:**
```bash
python cli.py interactive

# In interactive mode:
> add new_document.pdf           # Add document
> remove old_document.pdf        # Remove document
> list                          # Show all documents
> clear_docs                    # Clear all documents
> reset                         # Clear conversation history
> history                       # Show conversation history
> What is Python?               # Ask question
> help                          # Show commands
> quit                          # Exit
```

### Model Selection

The system **automatically selects** the best model based on your available RAM:

| Your Available RAM | Auto-Selected Model | Quality | Speed |
|-------------------|---------------------|---------|-------|
| < 4GB | `llama3.2:1b` | Basic | Very Fast |
| 4-8GB | `llama3.2:3b` | Good | Fast |
| 8-12GB | `codellama:7b` | Very Good | Moderate |
| 12GB+ | `qwen3:14b` | Excellent | Slow |

**Manual Override:**
```bash
# Force specific model (if available)
python cli.py query "question" --model qwen3:14b

# Check which models you have installed
ollama list
```

## 🛠 System Monitoring

### Resource Monitoring

```bash
# Check current system status
python resource_monitor.py

# Output example:
# System Resources:
#   Memory: 8.2GB / 16.0GB (51.3%)
#   Available Memory: 7.8GB
#   CPU: 12.5% (8 cores)
# 
# Model Recommendations for 7.8GB available:
#   1. codellama:7b
#      Very good quality, moderate speed (uses ~4.5GB RAM)
```

### Performance Testing

```bash
# Test different models (if installed)
python resource_monitor.py
# Select 'y' when prompted to test performance
```

## 🎯 Examples

### Example 1: Research Paper Analysis

```bash
# Add research papers
python cli.py add research_paper1.pdf
python cli.py add research_paper2.pdf

# Ask questions
python cli.py query "What are the main findings?"
python cli.py query "What methodology was used?"
python cli.py query "What are the limitations mentioned?"
```

### Example 2: Technical Documentation

```bash
# Add manuals and documentation
python cli.py add user_manual.pdf
python cli.py add api_documentation.docx

# Query for specific information
python cli.py query "How do I configure the API settings?"
python cli.py query "What are the system requirements?"
```

### Example 3: Code Documentation

```bash
# Add code-related documents
python cli.py add architecture_guide.md
python cli.py add coding_standards.txt

# Use code-focused model
python cli.py query "What are the coding best practices?" --model codellama:7b
```

## ⚙️ Configuration

### Default Settings

```python
# System defaults (configured automatically)
EMBEDDING_MODEL = "nomic-embed-text"     # 274MB
LLM_MODEL = "auto"                       # Selected based on RAM
CHUNK_SIZE = 3000                        # tokens per chunk (increased for better context)
CHUNK_OVERLAP = 400                      # token overlap between chunks
VECTOR_DB_PATH = "./vector_db/"          # database location
DEFAULT_RESULTS = 15                     # chunks retrieved per query (increased)
RESPONSE_LIMIT = 1500                    # max tokens in response (increased)
TIMEOUT = 600                           # request timeout in seconds (10 minutes)
```

### Custom Configuration

**Change default model:**
```python
# Edit rag_system.py or pass parameter
rag = RAGSystem(llm_model="codellama:7b")
```

**Adjust chunking:**
```python
# Edit DocumentProcessor parameters
processor = DocumentProcessor(chunk_size=2000, chunk_overlap=300)
```

**Analyze document processing:**
```bash
# Check how documents are being chunked
python cli.py analyze path/to/document.docx
```

## 🚨 Troubleshooting

### Common Issues

**1. "Model not found" error:**
```bash
# Check installed models
ollama list

# Install missing model
ollama pull llama3.2:1b
```

**2. High memory usage:**
```bash
# Check current usage
python resource_monitor.py

# Use smaller model
python cli.py query "question" --model llama3.2:1b
```

**3. Slow responses:**
```bash
# Reduce context chunks
python cli.py query "question" --results 5

# Use faster model
python cli.py query "question" --model llama3.2:1b

# Reduce timeout for quicker failures
python cli.py query "question" --timeout 60
```

**4. "Document not found in responses":**
```bash
# Ensure document is added to database
python cli.py add path/to/document.pdf

# Verify it's indexed
python -c "from rag_system import RAGSystem; rag=RAGSystem(); print(f'Documents: {rag.vectordb.count()}')"
```

**5. Connection errors:**
```bash
# Ensure Ollama is running
ollama serve

# Check if accessible
curl http://localhost:11434/api/tags
```

### Performance Optimization

**For Low-Resource Systems:**
```bash
# Minimal resource usage
python cli.py query "question" --model llama3.2:1b --results 5 --timeout 60
```

**For High-Resource Systems:**
```bash
# Maximum quality
python cli.py query "question" --model qwen3:14b --results 25 --timeout 400
```

**Monitor During Usage:**
```bash
# In another terminal
htop  # or top on macOS
```

## 📁 File Structure

```
rag/
├── cli.py                 # Main command-line interface
├── rag_system.py         # Core RAG implementation
├── simple_vectordb.py    # Vector database implementation
├── model_config.py       # Model selection and resource management
├── resource_monitor.py   # System monitoring utilities
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── .gitignore           # Git ignore rules
├── vector_db/           # Vector database (auto-created)
├── test_docs/           # Example documents  
├── test_improvements.py # Test script for improvements
└── venv/               # Virtual environment (if created)
```

## 🔒 Privacy & Security

- **Fully Local:** All processing happens on your machine
- **No Internet Required:** Once models are downloaded
- **No Data Sharing:** Your documents never leave your system
- **Open Source:** All code is transparent and auditable

## 🤝 Contributing

This is a production-ready system. Key components:

- `rag_system.py`: Core RAG logic
- `cli.py`: User interface
- `model_config.py`: Resource optimization
- `simple_vectordb.py`: Custom vector database

## 📄 License

[Specify your license here]

---

## 🆘 Need Help?

1. **Check system resources:** `python resource_monitor.py`
2. **Verify model installation:** `ollama list`
3. **Test with simple query:** `python cli.py query "hello"`
4. **Use interactive mode:** `python cli.py interactive`

For optimal performance, ensure you have adequate RAM for your chosen model and consider using the auto-selection feature for best results.